import os
import json
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image

from jutils import exists

from patch_flow.correspondence import CorrespondenceAttentionLoss, DinoCorrespondenceTeacher
from patch_flow.attention_smoothing import attention_centers, masked_center_tv
from patch_flow.diagonal_gaussian import DiagonalGaussian
from patch_flow.log_utils import log_images
from patch_flow.trainer import LatentFlowTrainer, un_normalize_ims
from patch_flow.vae_features import encode_vae_pyramid
from patch_flow.vton_utils import compose_vton, masked_mean


class LatentVTONPatchForcingTrainer(LatentFlowTrainer):
    def __init__(
        self,
        *args,
        uncertainty_weight=0.01,
        outside_velocity_weight=0.01,
        detail_loss_weight=0.0,
        detail_edge_weight=5.0,
        detail_pure_noise_only=True,
        detail_min_time=0.3,
        detail_max_time=0.95,
        decoded_rgb_weight=0.0,
        decoded_edge_weight=0.0,
        decoded_min_time=0.3,
        decoded_max_time=0.95,
        decoded_max_samples=1,
        decoded_checkpoint=True,
        allow_new_garment_refiner=False,
        attention_tv_weight=0.0,
        garment_supervision_only=False,
        garment_token_min_coverage=0.8,
        garment_dropout_prob=0.1,
        correspondence_center_weight=0.25,
        correspondence_entropy_weight=0.05,
        correspondence_nll_weight=0.3,
        correspondence_nll_radius=0.05,
        correspondence_photometric_weight=1.0,
        correspondence_value_weight=0.0,
        correspondence_value_cosine_mix=0.5,
        correspondence_value_target_ema=0.999,
        correspondence_teacher_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        correspondence_teacher_input_size=(512, 384),
        correspondence_garment_grid=None,
        correspondence_min_similarity=0.35,
        correspondence_soft_target_temperature=0.0,
        correspondence_mutual=False,
        correspondence_cycle_tolerance=None,
        correspondence_weight_by_similarity=False,
        correspondence_scales=None,
        correspondence_warmup_steps=0,
        train_adapters_only=False,
        backbone_lr_multiplier=0.1,
        compute_validation_metrics=True,
        save_validation_previews=True,
        preview_every_n_validations=1,
        max_validation_previews=8,
        compute_garment_validation_metrics=False,
        **kwargs,
    ):
        super().__init__(*args, enable_metrics=compute_validation_metrics, **kwargs)
        self.uncertainty_weight = float(uncertainty_weight)
        self.outside_velocity_weight = float(outside_velocity_weight)
        self.detail_loss_weight = float(detail_loss_weight)
        self.detail_edge_weight = float(detail_edge_weight)
        self.detail_pure_noise_only = bool(detail_pure_noise_only)
        self.detail_min_time = float(detail_min_time)
        self.detail_max_time = float(detail_max_time)
        if not 0 <= self.detail_min_time <= self.detail_max_time <= 1:
            raise ValueError("Detail time window must satisfy 0 <= min <= max <= 1")
        self.decoded_rgb_weight = float(decoded_rgb_weight)
        self.decoded_edge_weight = float(decoded_edge_weight)
        self.decoded_min_time = float(decoded_min_time)
        self.decoded_max_time = float(decoded_max_time)
        self.decoded_max_samples = int(decoded_max_samples)
        self.decoded_checkpoint = bool(decoded_checkpoint)
        self.allow_new_garment_refiner = bool(allow_new_garment_refiner)
        self.attention_tv_weight = float(attention_tv_weight)
        if self.attention_tv_weight < 0:
            raise ValueError("attention_tv_weight must be non-negative")
        if min(self.decoded_rgb_weight, self.decoded_edge_weight) < 0:
            raise ValueError("Decoded loss weights must be non-negative")
        if not 0 < self.decoded_min_time <= self.decoded_max_time < 1:
            raise ValueError("Decoded time window must satisfy 0 < min <= max < 1")
        if self.decoded_max_samples < 1:
            raise ValueError("decoded_max_samples must be positive")
        self.garment_supervision_only = bool(garment_supervision_only)
        self.garment_token_min_coverage = float(garment_token_min_coverage)
        if not 0 < self.garment_token_min_coverage <= 1:
            raise ValueError("garment_token_min_coverage must be in (0, 1]")
        if self.detail_edge_weight < 0:
            raise ValueError("detail_edge_weight must be non-negative")
        self.garment_dropout_prob = float(garment_dropout_prob)
        self.use_vae_garment = bool(getattr(self.model, "use_vae_garment", False))
        self.use_multiscale_garment = bool(getattr(self.model, "use_multiscale_garment", False))
        if not (self.use_vae_garment or self.use_multiscale_garment):
            raise ValueError("At least one garment conditioning branch must be enabled")

        # CORAL-style correspondence supervision. The teacher is a training-time-only
        # frozen DINOv3; nothing it produces is fed to the network, so inference is
        # unchanged and the checkpointed conditioning path stays pure SD-VAE.
        self.correspondence_loss = CorrespondenceAttentionLoss(
            center_weight=correspondence_center_weight,
            entropy_weight=correspondence_entropy_weight,
            nll_weight=correspondence_nll_weight,
            nll_radius=correspondence_nll_radius,
            photometric_weight=correspondence_photometric_weight,
            value_weight=correspondence_value_weight,
            value_cosine_mix=correspondence_value_cosine_mix,
        )
        self.correspondence_min_similarity = float(correspondence_min_similarity)
        self.correspondence_soft_target_temperature = float(correspondence_soft_target_temperature)
        self.correspondence_mutual = bool(correspondence_mutual)
        self.correspondence_cycle_tolerance = correspondence_cycle_tolerance
        if correspondence_cycle_tolerance is not None and float(correspondence_cycle_tolerance) < 0:
            raise ValueError("correspondence_cycle_tolerance must be non-negative")
        self.correspondence_weight_by_similarity = bool(correspondence_weight_by_similarity)
        self.correspondence_scales = None if correspondence_scales is None else list(correspondence_scales)
        self.correspondence_value_target_ema = float(correspondence_value_target_ema)
        if not 0 <= self.correspondence_value_target_ema < 1:
            raise ValueError("correspondence_value_target_ema must be in [0, 1)")
        self.value_target_embedders = nn.ModuleDict()
        self.value_target_norms = nn.ModuleDict()
        if self.correspondence_loss.value_weight > 0:
            configured_scales = set(
                self.correspondence_scales
                if self.correspondence_scales is not None
                else self.model._enabled_scales()
            )
            for scale in self.model._enabled_scales():
                if scale not in configured_scales:
                    continue
                self.value_target_embedders[scale] = deepcopy(self._value_source_embedder(scale))
                source_norm = self._value_source_norm(scale)
                self.value_target_norms[scale] = (
                    nn.Identity() if source_norm is None else deepcopy(source_norm)
                )
            self.value_target_embedders.requires_grad_(False)
            self.value_target_norms.requires_grad_(False)
        self.correspondence_warmup_steps = int(correspondence_warmup_steps)
        if self.correspondence_warmup_steps < 0:
            raise ValueError("correspondence_warmup_steps must be non-negative")
        # Own optimizer-step counter. LightningModule.global_step is not usable under the
        # accelerate loop in train.py: depending on the Lightning version it either does
        # not exist until train.py assigns it after the first optimizer step, or it is a
        # read-only property pinned at 0 because no Trainer is ever attached.
        self._optimizer_steps = 0
        # The photometric and entropy terms need no teacher, so it is only constructed
        # when a position target is actually consumed.
        self.correspondence_teacher = None
        if self.correspondence_loss.needs_target:
            self.correspondence_teacher = DinoCorrespondenceTeacher(
                model_name=correspondence_teacher_name,
                input_size=correspondence_teacher_input_size,
                garment_grid=correspondence_garment_grid,
            )
        self.backbone_lr_multiplier = float(backbone_lr_multiplier)
        self.train_adapters_only = bool(train_adapters_only)
        self.compute_validation_metrics = bool(compute_validation_metrics)
        self.save_validation_previews = bool(save_validation_previews)
        self.preview_every_n_validations = int(preview_every_n_validations)
        self.max_validation_previews = int(max_validation_previews)
        if self.max_validation_previews < 1:
            raise ValueError("max_validation_previews must be positive")
        self.compute_garment_validation_metrics = bool(compute_garment_validation_metrics)
        self._garment_validation_totals = {}
        self._validation_rows = []
        if self.preview_every_n_validations < 1:
            raise ValueError("preview_every_n_validations must be positive")
        if train_adapters_only:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            for name, parameter in self.model.named_parameters():
                if "garment_" in name or name.startswith("x_embedder") or name.startswith("final_layer"):
                    parameter.requires_grad = True

    def configure_optimizers(self):
        adapter_parameters = []
        backbone_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "garment_" in name or ".x_embedder" in name:
                adapter_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        groups = []
        if adapter_parameters:
            groups.append({"params": adapter_parameters, "lr": self.lr})
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": self.lr * self.backbone_lr_multiplier})
        optimizer = torch.optim.AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        output = {"optimizer": optimizer}
        if exists(self.lr_scheduler_cfg):
            from jutils import load_partial_from_config

            output["lr_scheduler"] = load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
        return output

    @staticmethod
    def _gradient_norm(parameter, row_slice=None):
        if parameter is None or parameter.grad is None:
            return torch.zeros((), device=parameter.device if parameter is not None else "cpu")
        gradient = parameter.grad.detach()
        if row_slice is not None:
            gradient = gradient[row_slice]
        return torch.linalg.vector_norm(gradient.float())

    @torch.no_grad()
    def garment_gradient_norms(self):
        """Return pre-clipping gradient norms for every garment-conditioning path."""
        metrics = {}
        for block_index, block in enumerate(self.model.blocks, start=1):
            if not block.use_garment_cross_attention:
                continue
            attention = block.garment_cross_attention
            prefix = f"garment_grad/block_{block_index:02d}_{block.garment_scale}"
            metrics[f"{prefix}/out_proj"] = self._gradient_norm(attention.out_proj.weight)
            qkv_rows = attention.embed_dim
            metrics[f"{prefix}/q"] = self._gradient_norm(
                attention.in_proj_weight, slice(0, qkv_rows)
            )
            metrics[f"{prefix}/k"] = self._gradient_norm(
                attention.in_proj_weight, slice(qkv_rows, 2 * qkv_rows)
            )
            metrics[f"{prefix}/v"] = self._gradient_norm(
                attention.in_proj_weight, slice(2 * qkv_rows, 3 * qkv_rows)
            )

        embedders = {
            "coarse": self.model.garment_embedder.proj if self.model.garment_embedder is not None else None,
            "middle": self.model.garment_middle_embedder,
            "detail": self.model.garment_detail_embedder,
        }
        for name, embedder in embedders.items():
            if embedder is not None:
                metrics[f"garment_grad/embedder_{name}"] = self._gradient_norm(embedder.weight)
        refiner = getattr(self.model, "garment_refiner", None)
        if refiner is not None:
            for name in ("query_expand", "query", "key", "value", "output"):
                metrics[f"garment_grad/refiner/{name}"] = self._gradient_norm(getattr(refiner, name).weight)
        return metrics

    def _label(self, batch, batch_size, device):
        label = batch.get("label")
        if label is not None:
            return label.long().view(batch_size).to(device)
        return torch.full(
            (batch_size,),
            self.model.y_embedder.num_classes,
            device=device,
            dtype=torch.long,
        )

    def _encode_batch(self, batch):
        target = batch["image"]
        garment = batch["garment"]
        target_latent = batch.get("latent")
        target_middle = batch.get("target_middle")
        target_detail = batch.get("target_detail")
        supervised_scales = set(
            self.correspondence_scales
            if self.correspondence_scales is not None
            else getattr(self.model, "_enabled_scales")()
        )
        needs_target_pyramid = (
            self.training and self.correspondence_loss.value_weight > 0
            and self.use_multiscale_garment
            and bool(supervised_scales & {"middle", "detail"})
            and (target_middle is None or target_detail is None)
        )
        if needs_target_pyramid:
            pyramid_latent, target_middle, target_detail = encode_vae_pyramid(self.first_stage, target)
            if target_latent is None:
                target_latent = pyramid_latent
        if target_latent is None:
            target_latent = self.encode(target)
        # A single mask: the token grid is already the finest editable granularity, so the
        # generated region and the pixel-conditioned region coincide and no identity
        # evidence is thrown away.
        masks = self.flow.prepare_masks(
            batch["agnostic_mask"],
            target_latent.shape[-2:],
            target_latent.dtype,
        )
        agnostic_latent = self.encode(batch["person_agnostic"])
        person_context = agnostic_latent * (1 - masks.latent)

        garment_latent = batch.get("garment_latent")
        garment_middle = batch.get("garment_middle")
        garment_detail = batch.get("garment_detail")
        if self.use_multiscale_garment and (garment_middle is None or garment_detail is None):
            garment_latent, garment_middle, garment_detail = encode_vae_pyramid(self.first_stage, garment)
        elif self.use_vae_garment and garment_latent is None:
            garment_latent = self.encode(garment)
        if not self.use_vae_garment:
            garment_latent = None
        return {
            "target_image": target,
            "person_image": batch.get("person", target),
            "agnostic_image": batch["person_agnostic"],
            "target": target_latent,
            "target_middle": target_middle,
            "target_detail": target_detail,
            "person_context": person_context,
            "masks": masks,
            "garment": garment_latent,
            "garment_middle": garment_middle,
            "garment_detail": garment_detail,
        }

    def _garment_conditions(self, encoded):
        return {key: encoded[key] for key in ("garment", "garment_middle", "garment_detail")}

    def _drop_garment(self, conditions, garment_mask):
        """Classifier-free dropout. Also returns the per-sample keep flag, because a
        dropped sample has no garment to correspond to and must not be supervised."""
        if not self.training or self.garment_dropout_prob <= 0:
            return conditions, garment_mask, None
        reference = next((value for value in conditions.values() if value is not None), None)
        if reference is None:
            return conditions, garment_mask, None
        keep = (
            torch.rand(reference.shape[0], device=reference.device) >= self.garment_dropout_prob
        ).to(reference.dtype)
        keep_image = keep[:, None, None, None]
        conditions = {
            key: None if value is None else value * keep_image.to(value.dtype)
            for key, value in conditions.items()
        }
        if garment_mask is not None:
            garment_mask = garment_mask * keep_image.to(garment_mask.dtype)
        return conditions, garment_mask, keep

    def _training_step_count(self):
        """Optimizer steps taken so far.

        Uses Lightning's counter only when a real Trainer is attached. ``train.py`` adds
        a synthetic ``global_step`` that jumps to the checkpoint step after its first
        update; using it made a requested 1000-step restart warmup last only one batch.
        """
        if getattr(self, "_trainer", None) is not None:
            step = getattr(self, "global_step", None)
            if isinstance(step, int) and step > 0:
                return step
        return self._optimizer_steps

    def _correspondence_ramp(self):
        if self.correspondence_warmup_steps <= 0:
            return 1.0
        return min(1.0, self._training_step_count() / self.correspondence_warmup_steps)

    def _value_source_embedder(self, scale):
        if scale == "coarse":
            return self.model.garment_embedder
        return getattr(self.model, f"garment_{scale}_embedder")

    def _value_source_norm(self, scale):
        if self.model.garment_token_norms is None:
            return None
        return self.model.garment_token_norms[scale]

    @staticmethod
    @torch.no_grad()
    def _ema_module(target, source, decay):
        target_parameters = dict(target.named_parameters())
        source_parameters = dict(source.named_parameters())
        if target_parameters.keys() != source_parameters.keys():
            raise RuntimeError("EMA value target and source projector parameters do not match")
        for name, target_parameter in target_parameters.items():
            target_parameter.mul_(decay).add_(source_parameters[name], alpha=1 - decay)
        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())
        if target_buffers.keys() != source_buffers.keys():
            raise RuntimeError("EMA value target and source projector buffers do not match")
        for name, target_buffer in target_buffers.items():
            if target_buffer.is_floating_point():
                target_buffer.mul_(decay).add_(source_buffers[name], alpha=1 - decay)
            else:
                target_buffer.copy_(source_buffers[name])

    @torch.no_grad()
    def _update_value_target_projectors(self, decay=None):
        if not hasattr(self, "value_target_embedders"):
            return
        decay = self.correspondence_value_target_ema if decay is None else float(decay)
        for scale, target_embedder in self.value_target_embedders.items():
            self._ema_module(target_embedder, self._value_source_embedder(scale), decay)
            source_norm = self._value_source_norm(scale)
            if source_norm is not None:
                self._ema_module(self.value_target_norms[scale], source_norm, decay)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load legacy targets; optionally warm-start a wholly absent detail refiner.

        This does not migrate optimizer states. Old architectures require load_weights,
        not resume_checkpoint. Partially missing refiner weights remain strict errors.
        """
        target_prefixes = ("value_target_embedders.", "value_target_norms.")
        try:
            incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        except TypeError:  # PyTorch versions before the ``assign`` argument.
            incompatible = super().load_state_dict(state_dict, strict=False)
        missing_targets = [
            key for key in incompatible.missing_keys if key.startswith(target_prefixes)
        ]
        missing = [key for key in incompatible.missing_keys if key not in missing_targets]
        if self.allow_new_garment_refiner:
            expected = self.state_dict().keys()
            for prefix in ("model.garment_refiner.", "ema_model.garment_refiner."):
                branch_keys = {key for key in expected if key.startswith(prefix)}
                if branch_keys and not any(key.startswith(prefix) for key in state_dict):
                    missing = [key for key in missing if key not in branch_keys]
                    warnings.warn(
                        f"Warm-start: {prefix} is new and retains its zero-output initialization. "
                        "Use load_weights with a fresh optimizer for the old architecture.",
                        UserWarning,
                    )
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"Missing key(s): {missing}")
            if unexpected:
                details.append(f"Unexpected key(s): {unexpected}")
            raise RuntimeError("Error(s) in loading state_dict: " + "; ".join(details))
        if self.value_target_embedders and missing_targets:
            # This runs after the student was loaded, so an old checkpoint starts with a
            # genuinely frozen copy of its learned projectors, not constructor weights.
            self._update_value_target_projectors(decay=0.0)
        return type(incompatible)(missing, unexpected)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self._update_value_target_projectors()
        self._optimizer_steps += 1

    @torch.no_grad()
    def _supervision_pixels(self, batch):
        """Training targets only; never used as generation conditioning."""
        edit = batch["agnostic_mask"].float()
        if not self.garment_supervision_only:
            return edit
        if "person_garment_mask" not in batch:
            raise ValueError("garment_supervision_only requires dataset garment_parse_labels")
        return edit * batch["person_garment_mask"].float()

    def _supervision_weights(self, batch, encoded, edit_tokens, keep):
        if self.garment_supervision_only:
            coverage = F.adaptive_avg_pool2d(
                self._supervision_pixels(batch), self._person_token_grid(encoded["target"])
            ).flatten(1)
            weight = coverage * (coverage >= self.garment_token_min_coverage)
            weight = weight * edit_tokens.float()
        else:
            weight = edit_tokens.float()
        if keep is not None:
            weight = weight * keep[:, None].to(weight.dtype)
        paired = batch.get("has_ground_truth")
        if paired is not None:
            weight = weight * paired[:, None].to(weight.dtype)
        if batch.get("garment_mask") is not None:
            weight = weight * batch["garment_mask"].flatten(1).any(1)[:, None]
        return weight

    def _correspondence_targets(self, batch, encoded, edit_tokens, keep):
        """Frozen-DINOv3 person->garment matches, as (target uv, supervision weight)."""
        teacher = self.correspondence_teacher
        person_grid = (
            encoded["target"].shape[-2] // self.flow.patch_size,
            encoded["target"].shape[-1] // self.flow.patch_size,
        )
        supervision = self._supervision_weights(batch, encoded, edit_tokens, keep)
        target, weight, similarity = teacher.correspondence(
            encoded["person_image"],
            batch["garment"],
            person_grid=person_grid,
            garment_mask=batch.get("garment_mask"),
            person_valid=supervision > 0,
            min_similarity=self.correspondence_min_similarity,
            soft_target_temperature=self.correspondence_soft_target_temperature,
            mutual=self.correspondence_mutual,
            cycle_tolerance=self.correspondence_cycle_tolerance,
            weight_by_similarity=self.correspondence_weight_by_similarity,
        )
        weight = weight * supervision
        return target, weight, similarity

    def _person_token_grid(self, target_latent):
        return (
            target_latent.shape[-2] // self.flow.patch_size,
            target_latent.shape[-1] // self.flow.patch_size,
        )

    @torch.no_grad()
    def _appearance_targets(self, batch, encoded, edit_tokens, keep):
        """Ground-truth worn appearance per person token, plus the garment image.

        Pool only garment pixels at mixed boundaries. Coverage and pair/dropout gates
        apply independently of teacher confidence; skin is never a garment RGB target.
        """
        grid = self._person_token_grid(encoded["target"])
        pixels = self._supervision_pixels(batch)
        coverage = F.adaptive_avg_pool2d(pixels, grid)
        query = F.adaptive_avg_pool2d(encoded["target_image"].float() * pixels, grid)
        query = (query / coverage.clamp_min(1e-6)).flatten(2).transpose(1, 2)
        weight = self._supervision_weights(batch, encoded, edit_tokens, keep)
        return {"query": query, "garment": batch["garment"], "garment_mask": batch.get("garment_mask")}, weight

    @torch.no_grad()
    def _target_value_features(self, encoded):
        """Content-only target-person tokens in the same SD-VAE feature basis as garment.

        No positional embedding is added: position is already supervised by CORAL, while
        this target must retain the appearance direction needed by V and out_proj.
        """
        query_grid = self._person_token_grid(encoded["target"])
        configured = set(
            self.correspondence_scales
            if self.correspondence_scales is not None
            else self.model._enabled_scales()
        )
        targets = {}
        if "coarse" in configured:
            if "coarse" not in self.value_target_embedders:
                raise ValueError("Coarse value supervision requires the coarse garment embedder")
            tokens = self.value_target_embedders["coarse"](encoded["target"])
            targets["coarse"] = self.value_target_norms["coarse"](tokens)

        for scale in ("middle", "detail"):
            if scale not in configured:
                continue
            source = encoded[f"target_{scale}"]
            if source is None or scale not in self.value_target_embedders:
                raise ValueError(f"{scale.capitalize()} value supervision requires target SD-VAE features")
            embedded = self.value_target_embedders[scale](source)
            if embedded.shape[-2:] != query_grid:
                embedded = F.adaptive_avg_pool2d(embedded, query_grid)
            tokens = embedded.flatten(2).transpose(1, 2)
            targets[scale] = self.value_target_norms[scale](tokens)
        return targets

    @staticmethod
    def _detail_importance(target, mask, edge_weight=5.0):
        """Weight sparse latent edges (logos, text and seams) up to 6x by default."""
        horizontal = (target[:, :, :, 1:] - target[:, :, :, :-1]).abs().mean(1, keepdim=True)
        vertical = (target[:, :, 1:, :] - target[:, :, :-1, :]).abs().mean(1, keepdim=True)
        edge = F.pad(horizontal, (0, 1, 0, 0)) + F.pad(vertical, (0, 0, 0, 1))
        edge = edge.detach() * mask
        mean_edge = edge.sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True).clamp_min(1.0)
        strength = (edge / mean_edge.clamp_min(1e-6)).clamp(0, 1)
        return 1.0 + float(edge_weight) * strength

    @staticmethod
    def _pure_noise_samples(timesteps, edit_tokens):
        """Samples whose entire editable region is at t=0; outside tokens are t=1."""
        return ((timesteps == 0) | ~edit_tokens).all(dim=1) & edit_tokens.any(dim=1)

    def _detail_supervision_mask(self, batch, encoded, timesteps, masks, keep):
        coverage = F.adaptive_avg_pool2d(self._supervision_pixels(batch), encoded["target"].shape[-2:])
        spatial = coverage * (coverage >= self.garment_token_min_coverage) * masks.latent
        pure = self._pure_noise_samples(timesteps, masks.token)
        eligible = pure[:, None].expand_as(timesteps)
        if not self.detail_pure_noise_only:
            eligible = eligible | ((timesteps >= self.detail_min_time) & (timesteps <= self.detail_max_time))
        time_mask = self.flow._tokens_to_latent(
            eligible.float(), *encoded["target"].shape[-2:], spatial.dtype
        )
        if keep is not None:
            spatial = spatial * keep[:, None, None, None]
        if batch.get("has_ground_truth") is not None:
            spatial = spatial * batch["has_ground_truth"][:, None, None, None]
        return spatial * time_mask

    @staticmethod
    def _detail_loss(predicted, target, mask, importance=None):
        """L1 on first spatial differences: penalises washed-out high-frequency structure
        (logo edges, printed text, colour-block seams) that a plain MSE tolerates."""
        horizontal_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        vertical_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        if importance is not None:
            horizontal_mask = horizontal_mask * 0.5 * (
                importance[:, :, :, 1:] + importance[:, :, :, :-1]
            )
            vertical_mask = vertical_mask * 0.5 * (
                importance[:, :, 1:, :] + importance[:, :, :-1, :]
            )
        horizontal = (predicted[:, :, :, 1:] - predicted[:, :, :, :-1]) - (
            target[:, :, :, 1:] - target[:, :, :, :-1]
        )
        vertical = (predicted[:, :, 1:, :] - predicted[:, :, :-1, :]) - (
            target[:, :, 1:, :] - target[:, :, :-1, :]
        )
        return masked_mean(horizontal.abs(), horizontal_mask) + masked_mean(vertical.abs(), vertical_mask)

    def _attention_tv_loss(self, entries, batch, keep):
        pixels = self._supervision_pixels(batch)
        if keep is not None:
            pixels = pixels * keep[:, None, None, None]
        if batch.get("has_ground_truth") is not None:
            pixels = pixels * batch["has_ground_truth"][:, None, None, None]
        if batch.get("garment_mask") is not None:
            pixels = pixels * batch["garment_mask"].flatten(1).any(1)[:, None, None, None]
        per_scale = {}
        for entry in entries:
            grid = entry["query_grid"]
            coverage = F.adaptive_avg_pool2d(pixels, grid)
            mask = coverage * (coverage >= self.garment_token_min_coverage)
            centers = entry.get("centers")
            if centers is None:
                centers = attention_centers(entry["weights"], entry["grid"])
            value = masked_center_tv(centers, grid, mask)
            per_scale.setdefault(entry["scale"], []).append(value)
        means = {scale: torch.stack(values).mean() for scale, values in per_scale.items()}
        total = torch.stack(list(means.values())).mean() if means else pixels.sum() * 0
        metrics = {f"attention_tv/{scale}": value.detach() for scale, value in means.items()}
        metrics["attention_tv_loss"] = total.detach()
        return total, metrics

    def _decode_with_grad(self, latent):
        # jutils AutoencoderKL.decode ALSO has @no_grad. Reproduce its SD-VAE
        # normalization and modules explicitly; parameters remain frozen.
        vae = self.first_stage
        if not hasattr(vae, "post_quant_conv") or not hasattr(vae, "decoder"):
            raise TypeError("Decoded garment supervision requires the jutils SD AutoencoderKL")
        latent = latent / vae.scale + vae.shift
        return vae.decoder(vae.post_quant_conv(latent))

    def _decoded_garment_loss(self, predicted_clean, batch, encoded, timesteps, keep):
        """Pixel supervision on paired, retained garments at refinement timesteps.

        Decode the estimated clean latent, NOT velocity. Do not use self.decode:
        that inference helper has no_grad and would silently cut off this loss.
        Targets/masks are full-resolution; edges require both endpoints eligible.
        """
        pixels = self._supervision_pixels(batch)
        eligible = (timesteps >= self.decoded_min_time) & (timesteps <= self.decoded_max_time)
        time_mask = self.flow._tokens_to_latent(
            eligible.float(), *predicted_clean.shape[-2:], torch.float32
        )
        mask = pixels * F.interpolate(time_mask, pixels.shape[-2:], mode="nearest")
        if keep is not None:
            mask = mask * keep[:, None, None, None]
        if batch.get("has_ground_truth") is not None:
            mask = mask * batch["has_ground_truth"][:, None, None, None]
        if batch.get("garment_mask") is not None:
            has_garment = batch["garment_mask"].flatten(1).any(1)
            mask = mask * has_garment[:, None, None, None]
        candidates = mask.flatten(1).any(1).nonzero(as_tuple=True)[0]
        zero = predicted_clean.sum() * 0.0
        metrics = {"decoded_rgb_loss": zero.detach(), "decoded_edge_loss": zero.detach(),
                   "decoded_samples": zero.detach(), "decoded_supervised_fraction": zero.detach()}
        if candidates.numel() == 0:
            return zero, metrics
        # Random selection avoids always training the first person in each batch.
        selected = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:self.decoded_max_samples]]
        latent = predicted_clean[selected]
        if self.decoded_checkpoint and torch.is_grad_enabled():
            decoded = checkpoint(self._decode_with_grad, latent, use_reentrant=False)
        else:
            decoded = self._decode_with_grad(latent)
        target = encoded["target_image"][selected]
        if decoded.shape != target.shape:
            raise ValueError("Decoded prediction and person target must have identical image resolution")
        # Work in [0,1] units, but do not clamp predictions and lose out-of-range gradients.
        decoded, target = (decoded.float() + 1) * 0.5, (target.float() + 1) * 0.5
        mask = mask[selected]
        rgb = masked_mean((decoded - target).abs(), mask)
        edge = self._detail_loss(decoded, target, mask)
        metrics.update(decoded_rgb_loss=rgb.detach(), decoded_edge_loss=edge.detach(),
                       decoded_samples=rgb.new_tensor(selected.numel()),
                       decoded_supervised_fraction=mask.mean().detach())
        return self.decoded_rgb_weight * rgb + self.decoded_edge_weight * edge, metrics

    def forward(self, batch):
        encoded = self._encode_batch(batch)
        target = encoded["target"]
        masks = encoded["masks"]
        conditions, garment_mask, keep = self._drop_garment(
            self._garment_conditions(encoded), batch.get("garment_mask")
        )
        xt, ut, timesteps, masks = self.flow.get_interpolants(
            target,
            encoded["person_context"],
            batch["agnostic_mask"].float(),
            masks=masks,
        )
        label = self._label(batch, target.shape[0], target.device)
        supervise_correspondence = (
            self.training and self.correspondence_loss.enabled and self._correspondence_ramp() > 0
        )
        supervise_tv = self.training and self.attention_tv_weight > 0 and self._correspondence_ramp() > 0
        request_attention = supervise_correspondence or supervise_tv
        output = self.model(
            x=xt,
            t=timesteps,
            y=label,
            person_agnostic=encoded["person_context"],
            person_mask=masks.condition,
            edit_mask=masks.condition,
            garment_mask=garment_mask,
            return_uncertainty=True,
            return_garment_attention=request_attention,
            return_garment_centers=supervise_tv,
            garment_attention_scales=self.correspondence_scales,
            **conditions,
        )
        attention_maps = []
        if request_attention:
            velocity, logvar, attention_maps = output
        else:
            velocity, logvar = output

        flow_loss = masked_mean((velocity - ut).square(), masks.latent)
        outside = 1 - masks.latent
        outside_loss = masked_mean(velocity.square(), outside)
        distribution = DiagonalGaussian(mean=velocity.detach(), logvar=logvar)
        uncertainty_loss = masked_mean(distribution.nll(ut), masks.latent)
        loss = flow_loss + self.outside_velocity_weight * outside_loss + self.uncertainty_weight * uncertainty_loss
        metrics = {
            "flow_loss": flow_loss,
            "outside_velocity_loss": outside_loss,
            "sigma_loss": uncertainty_loss,
            "editable_fraction": masks.latent.mean(),
            "outside_fraction": outside.mean(),
            "zero_time_fraction": ((timesteps == 0) & masks.token).float().sum()
            / masks.token.float().sum().clamp_min(1),
            "high_time_fraction": ((timesteps > 0.9) & masks.token).float().sum()
            / masks.token.float().sum().clamp_min(1),
        }
        use_decoded = self.decoded_rgb_weight > 0 or self.decoded_edge_weight > 0
        if self.detail_loss_weight > 0 or use_decoded:
            time_latent = self.flow._tokens_to_latent(
                timesteps, target.shape[-2], target.shape[-1], target.dtype
            )
            predicted_clean = xt + (1 - time_latent) * velocity
        if self.detail_loss_weight > 0:
            detail_mask = self._detail_supervision_mask(batch, encoded, timesteps, masks, keep)
            detail_importance = self._detail_importance(
                target, detail_mask, edge_weight=self.detail_edge_weight
            )
            detail_loss = self._detail_loss(
                predicted_clean, target, detail_mask, importance=detail_importance
            )
            loss = loss + self.detail_loss_weight * detail_loss
            metrics["detail_loss"] = detail_loss
            metrics["detail_active_fraction"] = detail_mask.flatten(1).any(1).float().mean()
            metrics["detail_edge_importance"] = masked_mean(detail_importance, detail_mask)
            metrics["detail_supervised_fraction"] = detail_mask.sum() / masks.latent.sum().clamp_min(1)
        if supervise_tv:
            tv_loss, tv_metrics = self._attention_tv_loss(attention_maps, batch, keep)
            loss = loss + self.attention_tv_weight * self._correspondence_ramp() * tv_loss
            metrics.update(tv_metrics)
        attention_maps = [entry for entry in attention_maps if "weights" in entry]
        if attention_maps and supervise_correspondence:
            appearance, appearance_weight = self._appearance_targets(batch, encoded, masks.token, keep)
            value_targets = (
                self._target_value_features(encoded)
                if self.correspondence_loss.value_weight > 0
                else None
            )
            if value_targets is not None:
                # The projected targets are all the loss needs; release the much larger
                # raw 1/2- and 1/4-resolution target pyramids before CORAL backward.
                encoded["target_middle"] = None
                encoded["target_detail"] = None
            correspondence_target = correspondence_weight = similarity = None
            if self.correspondence_teacher is not None:
                correspondence_target, correspondence_weight, similarity = self._correspondence_targets(
                    batch, encoded, masks.token, keep
                )
            correspondence_loss, correspondence_metrics = self.correspondence_loss(
                attention_maps,
                correspondence_target,
                correspondence_weight,
                appearance=appearance,
                appearance_weight=appearance_weight,
                value_targets=value_targets,
            )
            ramp = self._correspondence_ramp()
            loss = loss + ramp * correspondence_loss
            metrics.update(correspondence_metrics)
            metrics["correspondence_loss"] = correspondence_loss.detach()
            metrics["correspondence_ramp"] = torch.as_tensor(ramp, device=loss.device)
            eligible = appearance_weight > 0
            editable = eligible.float().sum().clamp_min(1)
            metrics["garment_supervision_fraction"] = eligible.float().sum() / masks.token.float().sum().clamp_min(1)
            if correspondence_weight is not None:
                # How much of the editable region the teacher was confident enough to
                # supervise, and how strong those matches were. If coverage collapses,
                # min_similarity is too aggressive and the loss is silently doing nothing.
                metrics["correspondence_coverage"] = (correspondence_weight > 0).float().sum() / editable
                metrics["correspondence_similarity"] = (
                    similarity * eligible.float()
                ).sum() / editable
        if use_decoded:
            decoded_loss, decoded_metrics = self._decoded_garment_loss(
                predicted_clean, batch, encoded, timesteps, keep
            )
            loss = loss + decoded_loss
            metrics.update(decoded_metrics)
        return loss, metrics

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        encoded = self._encode_batch(batch)
        target = encoded["target"]
        masks = encoded["masks"]
        target_image = encoded["target_image"]
        person_image = encoded["person_image"]
        label = self._label(batch, target.shape[0], target.device)
        seeds = batch.get("validation_seed")
        if seeds is None:
            generator = self.generator.manual_seed(batch_idx + self.global_rank * 16102024)
            noise = torch.randn(target.shape, generator=generator, dtype=target.dtype).to(target.device)
        else:
            noise = torch.stack([
                torch.randn(target.shape[1:], generator=self.generator.manual_seed(int(seed)), dtype=target.dtype)
                for seed in seeds
            ]).to(target.device)
        sample_model = self.ema_model if exists(self.ema_model) else self.model
        samples = self.flow.generate(
            model=sample_model,
            x=noise,
            person_agnostic=encoded["person_context"],
            person_condition=encoded["person_context"],
            person_condition_mask=masks.condition,
            edit_mask=batch["agnostic_mask"].float(),
            garment_mask=batch.get("garment_mask"),
            y=label,
            **self._garment_conditions(encoded),
            **self.sample_kwargs,
        )
        generated = self.decode(samples)
        expanded = torch.nn.functional.interpolate(
            masks.latent, size=target_image.shape[-2:], mode="nearest"
        )
        composed = compose_vton(generated, person_image, expanded)
        has_ground_truth = batch.get("has_ground_truth")
        if self.compute_garment_validation_metrics:
            self._record_garment_validation(batch, generated, target_image)
        if self.compute_validation_metrics:
            if has_ground_truth is None:
                self.metric_tracker(target_image, composed)
            elif has_ground_truth.any():
                self.metric_tracker(target_image[has_ground_truth], composed[has_ground_truth])
        retained = 0 if self.val_images is None else len(self.val_images["target"])
        count = min(target.shape[0], self.max_validation_previews - retained)
        if count > 0:
            mask_preview = (expanded[:count].repeat(1, 3, 1, 1) * 255).clamp(0, 255).to(torch.uint8)
            images = {
                "target": un_normalize_ims(target_image[:count]),
                "person": un_normalize_ims(person_image[:count]),
                "agnostic": un_normalize_ims(encoded["agnostic_image"][:count]),
                "edit_mask": mask_preview,
                "garment": un_normalize_ims(batch["garment"][:count]),
                "tryon": un_normalize_ims(composed[:count]),
            }
            images = {key: value.cpu() for key, value in images.items()}
            self.val_images = images if self.val_images is None else {
                key: torch.cat((self.val_images[key], images[key])) for key in images
            }
            for index in range(count):
                self._validation_rows.append({
                    "group": batch.get("validation_group", ["paired"] * target.shape[0])[index],
                    "person": batch.get("person_name", [""] * target.shape[0])[index],
                    "garment": batch.get("garment_name", [""] * target.shape[0])[index],
                    "has_ground_truth": True if has_ground_truth is None else bool(has_ground_truth[index]),
                })

    def _record_garment_validation(self, batch, generated, target):
        if "person_garment_mask" not in batch:
            raise ValueError("Garment validation metrics require person_garment_mask")
        paired = batch.get("has_ground_truth", torch.ones(target.shape[0], device=target.device, dtype=torch.bool))
        groups = batch.get("validation_group", ["paired"] * target.shape[0])
        for index in range(target.shape[0]):
            if not bool(paired[index]):
                continue
            # Measure generated pixels before composition so copied pixels cannot
            # improve the score. Restrict to the requested garment editing region.
            mask = batch["person_garment_mask"][index:index + 1].float() * batch["agnostic_mask"][index:index + 1]
            if not bool(mask.any()):
                continue
            predicted = (generated[index:index + 1].float().clamp(-1, 1) + 1) / 2
            expected = (target[index:index + 1].float() + 1) / 2
            rgb = masked_mean((predicted - expected).abs(), mask)
            edge = self._detail_loss(predicted, expected, mask)
            values = torch.stack((rgb, edge, rgb.new_ones(()))).detach()
            group = groups[index]
            if group not in ("train_paired", "test_paired", "paired"):
                raise ValueError(f"Unknown paired validation group: {group}")
            self._garment_validation_totals[group] = self._garment_validation_totals.get(group, 0) + values

    def on_validation_epoch_end(self):
        if self.val_images is not None:
            for key, images in self.val_images.items():
                log_images(self.logger, images, f"val/{key}/samples", stack="row", split=4, step=self.global_step)
            if self.save_validation_previews and (self.val_epochs + 1) % self.preview_every_n_validations == 0:
                self._save_validation_preview()
            self.val_images = None
            self._validation_rows = []
        if self.compute_garment_validation_metrics:
            for group in ("train_paired", "test_paired", "paired"):
                values = self._garment_validation_totals.get(group, torch.zeros(3, device=self.device)).clone()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(values)
                if values[2] > 0:
                    self.log(f"val/{group}/garment_rgb_mae", values[0] / values[2])
                    self.log(f"val/{group}/garment_edge_mae", values[1] / values[2])
                    self.log(f"val/{group}/samples", values[2])
            self._garment_validation_totals.clear()
        if self.compute_validation_metrics:
            metrics = self.metric_tracker.aggregate()
            for key, value in metrics.items():
                self.log(f"val/{key}", value, sync_dist=True)
            self.metric_tracker.reset()
        self.val_epochs += 1

    def _save_validation_preview(self):
        log_dir = getattr(self.logger, "log_dir", None)
        if not isinstance(log_dir, (str, os.PathLike)) or self.val_images is None:
            return
        keys = ("target", "person", "agnostic", "edit_mask", "garment", "tryon")
        rows = torch.stack([self.val_images[key] for key in keys], dim=1)
        rows = rows.flatten(0, 1).float().cpu() / 255
        preview_dir = os.path.join(log_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        step_path = os.path.join(preview_dir, f"step{self.global_step:06d}.png")
        latest_path = os.path.join(preview_dir, "latest.png")
        save_image(rows, step_path, nrow=len(keys), padding=4, pad_value=1)
        save_image(rows, latest_path, nrow=len(keys), padding=4, pad_value=1)
        with open(os.path.join(preview_dir, f"step{self.global_step:06d}.json"), "w", encoding="utf-8") as handle:
            json.dump({"columns": keys, "rows": self._validation_rows,
                       "note": "Unpaired rows show the original person in the target column; no swap ground truth exists."}, handle, indent=2)
