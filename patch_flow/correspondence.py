"""CORAL-style correspondence supervision for the garment cross-attention.

A frozen DINOv3 teacher matches every editable person token to the garment token it
actually corresponds to, by cosine similarity of patch features. That match is treated as
ground truth for *where* the cross-attention should look, and two losses are applied
directly to the attention distribution:

``target mass`` (negative log-likelihood)
    the primary term. Maximises the attention mass that lands *within a radius of the
    matched key*, i.e. it constrains **which** keys are attended to.

``centre of mass``
    a secondary, smooth long-range pull on the attention barycentre. On its own it is not
    enough, and measurably so: a map can be sharp (3.8 effective keys out of 768), have a
    barycentre 1.7 tokens from the target, and still place its peak 4.5 tokens away with
    only 5% of its mass near the target. Sharp plus barycentre-correct plus displaced is a
    bimodal map straddling the target, and what it retrieves is a *blend* of two different
    fabrics -- which is exactly how a navy-and-lavender garment renders as uniform violet.
    Entropy does not catch this, because it penalises spread and not displacement.

``entropy``
    removes the remaining freedom to answer with a diffuse distribution.

``photometric``
    the mass-weighted garment appearance a person token retrieves must match the
    appearance that token actually has in the ground-truth worn image. This needs no
    teacher at all -- it is self-supervised from the paired data -- and it constrains
    routing in *appearance* space rather than in position space. That matters because
    DINOv3 correspondence is geometric: measured over 24 VITON-HD test pairs, a delta at
    the DINOv3 target retrieves the right colour to within 0.455 RGB, against 0.817 for
    the garment mean and 0.119 for the best available garment token. Geometry recovers
    about half of the colour signal; the rest has to be asked for directly.

The teacher is **not** a conditioning branch: no DINO feature ever reaches the network.
It runs under ``no_grad`` on the ground-truth person image, which exists only at training
time, and disappears entirely at inference.
"""

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _StableAttentionEntropy(torch.autograd.Function):
    """Entropy with a finite derivative for probabilities underflowed to zero.

    BF16 garment attention contains exact zeros on the 3072-key detail grid.
    ``torch.special.entr(0)`` is finite in the forward pass but has an infinite
    derivative, which makes softmax backward NaN. Work in key chunks as well, so the
    stability fix does not materialise another full (B,H,Q,K) tensor during forward.
    """

    @staticmethod
    def forward(ctx, probabilities, eps, chunk_size):
        ctx.save_for_backward(probabilities)
        ctx.eps = max(float(eps), float(torch.finfo(probabilities.dtype).tiny))
        ctx.chunk_size = int(chunk_size)
        entropy = torch.zeros(
            probabilities.shape[:-1], device=probabilities.device, dtype=torch.float32
        )
        for start in range(0, probabilities.shape[-1], ctx.chunk_size):
            values = probabilities[..., start : start + ctx.chunk_size]
            safe = values.clamp_min(ctx.eps)
            entropy.add_(-(values * safe.log()).sum(-1).float())
        return entropy

    @staticmethod
    def backward(ctx, grad_output):
        (probabilities,) = ctx.saved_tensors
        gradient = torch.empty_like(probabilities)
        upstream = grad_output.to(probabilities.dtype).unsqueeze(-1)
        for start in range(0, probabilities.shape[-1], ctx.chunk_size):
            values = probabilities[..., start : start + ctx.chunk_size]
            safe = values.clamp_min(ctx.eps)
            # d[-p log(clamp(p, eps))]/dp. Below eps, the clamp's derivative is
            # zero, leaving the finite linear derivative -log(eps).
            derivative = -safe.log() - (values >= ctx.eps).to(values.dtype)
            gradient[..., start : start + ctx.chunk_size] = upstream * derivative
        return gradient, None, None


def stable_attention_entropy(probabilities, eps=1e-6, chunk_size=256):
    """Per-head entropy reduced over keys with finite BF16 gradients."""
    if not probabilities.is_floating_point():
        raise TypeError("Attention probabilities must be floating point")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return _StableAttentionEntropy.apply(probabilities, float(eps), int(chunk_size))


def grid_coordinates(grid, device=None, dtype=torch.float32):
    """Normalised (u, v) centres of a row-major ``grid`` of tokens, shape (H*W, 2)."""
    height, width = int(grid[0]), int(grid[1])
    rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    columns = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    v, u = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1)


def mask_to_token_valid(mask, grid):
    """Max-pool a pixel mask onto ``grid``; a token is valid if any pixel inside it is set."""
    if mask.ndim == 3:
        mask = mask[:, None]
    valid = F.adaptive_max_pool2d(mask.float(), (int(grid[0]), int(grid[1]))).flatten(1) > 0.5
    empty = ~valid.any(dim=1)
    if empty.any():
        # An empty key set would make every similarity -inf; fall back to the full grid so
        # the sample degrades to "no usable target" via the confidence gate instead of NaN.
        valid = valid.clone()
        valid[empty] = True
    return valid


class DinoCorrespondenceTeacher(nn.Module):
    """Frozen DINOv3 patch-feature extractor used only to build correspondence targets."""

    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        input_size=None,
        garment_grid=None,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_name = str(model_name)
        self.input_size = None if input_size is None else (int(input_size[0]), int(input_size[1]))
        self.garment_grid = None if garment_grid is None else (int(garment_grid[0]), int(garment_grid[1]))
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.requires_grad_(False)
        self.model.eval()

        patch_size = getattr(self.model.config, "patch_size", None)
        if patch_size is None:
            raise ValueError(f"{self.model_name} does not expose a patch size; it cannot be used as a teacher")
        self.patch_size = int(patch_size)
        if self.input_size is not None and any(size % self.patch_size for size in self.input_size):
            raise ValueError(f"Teacher input size {self.input_size} must be divisible by patch size {self.patch_size}")
        self.feature_dim = int(self.model.config.hidden_size)
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        # Frozen teacher: never leaves eval, so dropout and stochastic depth stay off and
        # the target for a given (person, garment) pair is deterministic across steps.
        super().train(False)
        return self

    def resolve_input_size(self, images):
        """Teacher resolution for ``images``.

        Defaults to the incoming resolution rounded to the patch grid. That matters: the
        dataset letterboxes person and garment to a fixed aspect ratio, and squeezing them
        into a differently shaped teacher input would shear both images and skew every
        matched position.
        """
        if self.input_size is not None:
            return self.input_size
        patch = self.patch_size
        return tuple(max(patch, int(round(size / patch)) * patch) for size in images.shape[-2:])

    def _forward_features(self, images):
        size = self.resolve_input_size(images)
        pixels = F.interpolate(
            (images.float() + 1) / 2,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.image_mean) / self.image_std
        tokens = self.model(pixel_values=pixels).last_hidden_state
        height, width = size[0] // self.patch_size, size[1] // self.patch_size
        prefix = tokens.shape[1] - height * width
        if prefix < 0:
            raise RuntimeError(f"Expected at least {height * width} patch tokens, got {tokens.shape[1]}")
        # DINOv2 prepends one CLS token, DINOv3 prepends CLS plus register tokens; both are
        # positional-free summaries and carry no correspondence information.
        tokens = tokens[:, prefix:]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.feature_dim, height, width)

    @torch.no_grad()
    def features(self, images, grid=None):
        features = self._forward_features(images)
        if grid is not None and tuple(int(size) for size in grid) != features.shape[-2:]:
            features = F.interpolate(features, size=(int(grid[0]), int(grid[1])), mode="bilinear", align_corners=False)
        return features

    @torch.no_grad()
    def correspondence(
        self,
        person,
        garment,
        person_grid,
        garment_mask=None,
        person_valid=None,
        min_similarity=0.0,
        soft_target_temperature=0.0,
        mutual=False,
        weight_by_similarity=False,
        cycle_tolerance=None,
    ):
        """Match editable person tokens to garment tokens and return their positions.

        Returns ``(target, weight, similarity)`` where ``target`` is (B, Np, 2) normalised
        (u, v) garment coordinates, ``weight`` is (B, Np) supervision strength, and
        ``similarity`` is (B, Np) the raw best cosine similarity.
        """
        person_features = self.features(person, person_grid)
        garment_features = self.features(garment, self.garment_grid)
        garment_grid = tuple(garment_features.shape[-2:])
        return correspondence_targets(
            person_features,
            garment_features,
            garment_grid=garment_grid,
            person_valid=person_valid,
            garment_valid=None if garment_mask is None else mask_to_token_valid(garment_mask, garment_grid),
            min_similarity=min_similarity,
            soft_target_temperature=soft_target_temperature,
            mutual=mutual,
            weight_by_similarity=weight_by_similarity,
            cycle_tolerance=cycle_tolerance,
        )


@torch.no_grad()
def correspondence_targets(
    person_features,
    garment_features,
    garment_grid,
    person_valid=None,
    garment_valid=None,
    min_similarity=0.0,
    soft_target_temperature=0.0,
    mutual=False,
    weight_by_similarity=False,
    cycle_tolerance=None,
):
    """Cosine-similarity correspondence from person tokens to garment positions."""
    person = F.normalize(person_features.float().flatten(2).transpose(1, 2), dim=-1)
    garment = F.normalize(garment_features.float().flatten(2).transpose(1, 2), dim=-1)
    similarity = torch.einsum("bpc,bgc->bpg", person, garment)
    neginf = torch.finfo(similarity.dtype).min

    masked = similarity
    if garment_valid is not None:
        masked = masked.masked_fill(~garment_valid[:, None, :], neginf)
    best_similarity, best_index = masked.max(dim=-1)

    coordinates = grid_coordinates(garment_grid, device=similarity.device, dtype=similarity.dtype)
    if soft_target_temperature > 0:
        attention = torch.softmax(masked / float(soft_target_temperature), dim=-1)
        target = attention @ coordinates
    else:
        target = coordinates[best_index]

    valid = torch.ones_like(best_similarity, dtype=torch.bool) if person_valid is None else person_valid.clone()
    valid = valid & (best_similarity >= float(min_similarity))
    if cycle_tolerance is not None and float(cycle_tolerance) < 0:
        raise ValueError("cycle_tolerance must be non-negative or None")
    if mutual or cycle_tolerance is not None:
        # Cycle consistency: the garment token a person token chose must choose it back.
        backward = similarity
        if person_valid is not None:
            backward = backward.masked_fill(~person_valid[:, :, None], neginf)
        back_index = backward.argmax(dim=1)
        positions = torch.arange(similarity.shape[1], device=similarity.device)
        returned = back_index.gather(1, best_index)
        if mutual:
            valid = valid & (returned == positions[None, :])
        else:
            # Euclidean distance in PERSON-token units, preserving the aspect ratio.
            # Nearby returns survive local garment deformation, unlike exact mutual NN.
            width = person_features.shape[-1]
            dy = returned.div(width, rounding_mode="floor") - positions.div(width, rounding_mode="floor")[None]
            dx = returned.remainder(width) - positions.remainder(width)[None]
            valid = valid & (dy.square() + dx.square() <= float(cycle_tolerance) ** 2)

    weight = valid.to(similarity.dtype)
    if weight_by_similarity:
        weight = weight * best_similarity.clamp_min(0)
    return target, weight, best_similarity


def neighbourhood_mass(attention, target, grid, radius):
    """Attention mass landing within ``radius`` (normalised units) of each target.

    Only the window of keys around each target is gathered, never a dense
    ``(queries, keys)`` distance matrix: the detail branch has 3072 keys, and that matrix
    would be several times larger than the attention map it is measuring.
    """
    grid_height, grid_width = int(grid[0]), int(grid[1])
    device = attention.device
    span_y = max(1, int(math.ceil(radius * grid_height)))
    span_x = max(1, int(math.ceil(radius * grid_width)))
    offset_y, offset_x = torch.meshgrid(
        torch.arange(-span_y, span_y + 1, device=device),
        torch.arange(-span_x, span_x + 1, device=device),
        indexing="ij",
    )
    offset_y = offset_y.reshape(-1)
    offset_x = offset_x.reshape(-1)
    centre_y = (target[..., 1] * grid_height - 0.5).round().long()
    centre_x = (target[..., 0] * grid_width - 0.5).round().long()
    row = centre_y[..., None] + offset_y
    column = centre_x[..., None] + offset_x
    # Out-of-grid offsets are dropped rather than clamped, so no key is counted twice.
    inside = (row >= 0) & (row < grid_height) & (column >= 0) & (column < grid_width)
    row_safe = row.clamp(0, grid_height - 1)
    column_safe = column.clamp(0, grid_width - 1)
    key_u = (column_safe.to(attention.dtype) + 0.5) / grid_width
    key_v = (row_safe.to(attention.dtype) + 0.5) / grid_height
    near = (key_u - target[..., 0:1]).square() + (key_v - target[..., 1:2]).square() <= radius ** 2
    index = row_safe * grid_width + column_safe
    return (attention.gather(-1, index) * (inside & near).to(attention.dtype)).sum(-1)


class CorrespondenceAttentionLoss(nn.Module):
    """Position, routing-appearance and transported-value losses on garment attention."""

    def __init__(
        self,
        center_weight=0.25,
        entropy_weight=0.05,
        nll_weight=0.3,
        nll_radius=0.05,
        photometric_weight=1.0,
        value_weight=0.0,
        value_cosine_mix=0.5,
        entropy_eps=1e-8,
    ):
        super().__init__()
        self.center_weight = float(center_weight)
        self.entropy_weight = float(entropy_weight)
        self.nll_weight = float(nll_weight)
        if isinstance(nll_radius, Mapping):
            self.nll_radius = {str(scale): float(radius) for scale, radius in nll_radius.items()}
            if not self.nll_radius:
                raise ValueError("nll_radius mapping must contain at least one garment scale")
        else:
            self.nll_radius = float(nll_radius)
        self.photometric_weight = float(photometric_weight)
        self.value_weight = float(value_weight)
        self.value_cosine_mix = float(value_cosine_mix)
        self.entropy_eps = float(entropy_eps)
        if self.value_weight < 0:
            raise ValueError("value_weight must be non-negative")
        if not 0.0 <= self.value_cosine_mix <= 1.0:
            raise ValueError("value_cosine_mix must be in [0, 1]")
        radii = self.nll_radius.values() if isinstance(self.nll_radius, dict) else (self.nll_radius,)
        if any(not 0 < radius < 1 for radius in radii):
            raise ValueError("nll_radius values are in normalised garment units and must be in (0, 1)")

    def _nll_radius_for_scale(self, scale):
        if not isinstance(self.nll_radius, dict):
            return self.nll_radius
        try:
            return self.nll_radius[scale]
        except KeyError as error:
            configured = ", ".join(sorted(self.nll_radius))
            raise ValueError(
                f"No correspondence NLL radius configured for garment scale '{scale}'; "
                f"configured scales: {configured}"
            ) from error

    @property
    def needs_target(self):
        """Whether a correspondence teacher is required at all."""
        return self.center_weight > 0 or self.nll_weight > 0

    @property
    def enabled(self):
        return (
            self.needs_target
            or self.entropy_weight > 0
            or self.photometric_weight > 0
            or self.value_weight > 0
        )

    def forward(
        self,
        attention_maps,
        target=None,
        weight=None,
        appearance=None,
        appearance_weight=None,
        value_targets=None,
    ):
        """``attention_maps`` is the list returned by ``VTONPatchForcingDiT``.

        ``target``/``weight`` drive the teacher-based terms; ``appearance`` carries
        ``query`` (B, queries, C) ground-truth worn appearance and ``garment`` (B, C, H, W)
        for the photometric term, which needs no teacher.
        """
        reference = next((entry["weights"] for entry in attention_maps), None)
        if reference is None:
            device = target.device if target is not None else "cpu"
            return torch.zeros((), device=device, dtype=torch.float32), {}
        device = reference.device
        zero = torch.zeros((), device=device, dtype=torch.float32)

        use_target = target is not None and weight is not None and self.needs_target
        weight = None if weight is None else weight.float()
        appearance_weight = None if appearance_weight is None else appearance_weight.float()
        spread_weight = weight if weight is not None else appearance_weight
        use_photometric = (
            appearance is not None and appearance_weight is not None and self.photometric_weight > 0
        )
        use_value = value_targets is not None and appearance_weight is not None and self.value_weight > 0
        prepared_value_targets = (
            {scale: value.detach().float() for scale, value in value_targets.items()}
            if use_value
            else None
        )

        totals = {
            "center": zero,
            "entropy": zero,
            "nll": zero,
            "photometric": zero,
            "value": zero,
            "value_cosine": zero,
            "value_huber": zero,
            "mass": zero,
        }
        scale_totals = {}
        metrics = {}
        for entry in attention_maps:
            # Keep the full per-head map in its autocast dtype. At 512x384 a detail map
            # is (B,16,768,3072); eagerly copying every block to fp32 costs >1 GiB on a
            # 24-GiB GPU. Reductions below promote only their compact outputs to fp32.
            attention = entry["weights"]
            grid = entry["grid"]
            scale = entry["scale"]
            prefix = f"correspondence/block_{entry['block']:02d}_{scale}"
            if scale not in scale_totals:
                scale_totals[scale] = {
                    "center": zero,
                    "entropy": zero,
                    "nll": zero,
                    "photometric": zero,
                    "value": zero,
                    "value_cosine": zero,
                    "value_huber": zero,
                    "mass": zero,
                    "count": 0,
                }
            scale_totals[scale]["count"] += 1
            coordinates = grid_coordinates(grid, device=device, dtype=attention.dtype)
            if coordinates.shape[0] != attention.shape[-1]:
                raise ValueError(
                    f"Branch '{entry['scale']}' has {attention.shape[-1]} keys but grid {grid}"
                )
            if attention.ndim == 3:
                # Backward compatibility for external callers and old unit fixtures.
                attention_heads = attention[:, None]
            elif attention.ndim == 4:
                attention_heads = attention
            else:
                raise ValueError(
                    "Garment attention must have shape (B,Q,K) or (B,H,Q,K), got "
                    f"{tuple(attention.shape)}"
                )
            heads = attention_heads.shape[1]

            if use_target:
                if attention_heads.shape[-2] != target.shape[1]:
                    raise ValueError(
                        f"Attention has {attention_heads.shape[-2]} queries but the correspondence target has "
                        f"{target.shape[1]}; the query grid must be the person token grid"
                    )
                target_heads = target[:, None].expand(-1, heads, -1, -1)
                weight_heads = weight[:, None].expand(-1, heads, -1)
                denominator = weight_heads.sum().clamp_min(1e-6)
                center = attention_heads @ coordinates
                center = center.float()
                center_loss = (
                    (center - target_heads).square().sum(-1) * weight_heads
                ).sum() / denominator
                radius = self._nll_radius_for_scale(scale)
                mass = neighbourhood_mass(attention_heads, target_heads, grid, radius)
                mass = mass.float()
                # Mean of per-head NLL, not NLL of the mean attention. The latter is
                # satisfied by a few good heads and was the measured logo-routing loophole.
                nll_loss = (
                    -torch.log(mass.clamp_min(1e-6)) * weight_heads
                ).sum() / denominator
                totals["center"] = totals["center"] + center_loss
                totals["nll"] = totals["nll"] + nll_loss
                totals["mass"] = totals["mass"] + (mass * weight_heads).sum() / denominator
                scale_totals[scale]["center"] = scale_totals[scale]["center"] + center_loss
                scale_totals[scale]["nll"] = scale_totals[scale]["nll"] + nll_loss
                scale_totals[scale]["mass"] = (
                    scale_totals[scale]["mass"] + (mass * weight_heads).sum() / denominator
                )
                metrics[f"{prefix}/center"] = center_loss.detach()
                metrics[f"{prefix}/nll"] = nll_loss.detach()
                metrics[f"{prefix}/target_mass"] = (
                    (mass * weight_heads).sum() / denominator
                ).detach()

            if self.entropy_weight > 0 and spread_weight is not None:
                spread_heads = spread_weight[:, None].expand(-1, heads, -1)
                spread_denominator = spread_heads.sum().clamp_min(1e-6)
                entropy = stable_attention_entropy(
                    attention_heads, eps=self.entropy_eps, chunk_size=256
                )
                padding = entry.get("key_padding")
                keys = (
                    torch.full(
                        (attention_heads.shape[0], 1, 1),
                        float(attention_heads.shape[-1]),
                        device=device,
                    )
                    if padding is None else (~padding).sum(-1).float()[:, None, None]
                )
                # Normalise by the entropy of the uniform distribution over usable keys so the
                # 3072-key detail branch and the 768-key coarse branch contribute comparably.
                entropy = entropy / torch.log(keys.clamp_min(2.0))
                entropy_loss = (entropy * spread_heads).sum() / spread_denominator
                totals["entropy"] = totals["entropy"] + entropy_loss
                scale_totals[scale]["entropy"] = scale_totals[scale]["entropy"] + entropy_loss
                metrics[f"{prefix}/entropy"] = entropy_loss.detach()

            if use_photometric:
                # Colour routing remains a property of the ensemble of heads. Geometry
                # and confidence above are intentionally strict per head.
                routing_attention = attention_heads.mean(dim=1).float()
                photometric_denominator = appearance_weight.sum().clamp_min(1e-6)
                keys_appearance = (
                    appearance["garment"].float()
                )
                key_mask = appearance.get("garment_mask")
                if key_mask is not None:
                    key_coverage = F.adaptive_avg_pool2d(key_mask.float(), grid)
                    keys_appearance = F.adaptive_avg_pool2d(keys_appearance * key_mask, grid)
                    keys_appearance = keys_appearance / key_coverage.clamp_min(1e-6)
                else:
                    keys_appearance = F.adaptive_avg_pool2d(keys_appearance, grid)
                keys_appearance = keys_appearance.flatten(2).transpose(1, 2)
                retrieved = routing_attention @ keys_appearance
                error = (retrieved - appearance["query"].float()).square().sum(-1)
                photometric_loss = (error * appearance_weight).sum() / photometric_denominator
                totals["photometric"] = totals["photometric"] + photometric_loss
                scale_totals[scale]["photometric"] = (
                    scale_totals[scale]["photometric"] + photometric_loss
                )
                metrics[f"{prefix}/photometric"] = photometric_loss.detach()

            if use_value:
                transported = entry.get("output")
                if transported is None:
                    raise ValueError(
                        f"Branch '{scale}' did not return its transported value for value supervision"
                    )
                if scale not in prepared_value_targets:
                    raise ValueError(f"No target-person value feature was provided for scale '{scale}'")
                value_target = prepared_value_targets[scale]
                if value_target.device != device:
                    value_target = value_target.to(device)
                transported = transported.float()
                if transported.shape != value_target.shape:
                    raise ValueError(
                        f"Transported value shape {tuple(transported.shape)} does not match "
                        f"the '{scale}' target shape {tuple(value_target.shape)}"
                    )
                cosine_error = 1 - F.cosine_similarity(transported, value_target, dim=-1)
                huber_error = F.smooth_l1_loss(
                    transported, value_target, reduction="none"
                ).mean(-1)
                value_error = (
                    self.value_cosine_mix * cosine_error
                    + (1.0 - self.value_cosine_mix) * huber_error
                )
                value_denominator = appearance_weight.sum().clamp_min(1e-6)
                value_loss = (value_error * appearance_weight).sum() / value_denominator
                value_cosine = (cosine_error * appearance_weight).sum() / value_denominator
                value_huber = (huber_error * appearance_weight).sum() / value_denominator
                totals["value"] = totals["value"] + value_loss
                totals["value_cosine"] = totals["value_cosine"] + value_cosine
                totals["value_huber"] = totals["value_huber"] + value_huber
                scale_totals[scale]["value"] = scale_totals[scale]["value"] + value_loss
                scale_totals[scale]["value_cosine"] = (
                    scale_totals[scale]["value_cosine"] + value_cosine
                )
                scale_totals[scale]["value_huber"] = (
                    scale_totals[scale]["value_huber"] + value_huber
                )
                metrics[f"{prefix}/value"] = value_loss.detach()
                metrics[f"{prefix}/value_cosine"] = value_cosine.detach()
                metrics[f"{prefix}/value_huber"] = value_huber.detach()

        count = len(attention_maps)
        for key in totals:
            totals[key] = totals[key] / count
        for scale, scale_values in scale_totals.items():
            scale_count = scale_values.pop("count")
            averaged = {key: value / scale_count for key, value in scale_values.items()}
            metrics[f"correspondence/{scale}/nll"] = averaged["nll"].detach()
            metrics[f"correspondence/{scale}/center"] = averaged["center"].detach()
            metrics[f"correspondence/{scale}/entropy"] = averaged["entropy"].detach()
            metrics[f"correspondence/{scale}/photometric"] = averaged["photometric"].detach()
            metrics[f"correspondence/{scale}/value"] = averaged["value"].detach()
            metrics[f"correspondence/{scale}/value_cosine"] = averaged["value_cosine"].detach()
            metrics[f"correspondence/{scale}/value_huber"] = averaged["value_huber"].detach()
            metrics[f"correspondence/{scale}/target_mass"] = averaged["mass"].detach()
            metrics[f"correspondence/{scale}/appearance_error"] = (
                averaged["photometric"].detach().clamp_min(0).sqrt()
            )
        loss = (
            self.nll_weight * totals["nll"]
            + self.center_weight * totals["center"]
            + self.entropy_weight * totals["entropy"]
            + self.photometric_weight * totals["photometric"]
            + self.value_weight * totals["value"]
        )
        metrics["correspondence_nll"] = totals["nll"].detach()
        metrics["correspondence_center_loss"] = totals["center"].detach()
        metrics["correspondence_entropy"] = totals["entropy"].detach()
        metrics["correspondence_photometric"] = totals["photometric"].detach()
        metrics["correspondence_value"] = totals["value"].detach()
        metrics["correspondence_value_cosine"] = totals["value_cosine"].detach()
        metrics["correspondence_value_huber"] = totals["value_huber"].detach()
        # Directly interpretable: mass actually landing on the target, and the retrieved
        # appearance error in the same RGB units the diagnostics report.
        metrics["correspondence_target_mass"] = totals["mass"].detach()
        metrics["correspondence_appearance_error"] = totals["photometric"].detach().clamp_min(0).sqrt()
        return loss, metrics
