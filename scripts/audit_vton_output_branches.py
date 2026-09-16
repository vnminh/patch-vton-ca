#!/usr/bin/env python3
"""Ablate CFG and each final VTON detail carrier on one fixed paired sample."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from jutils import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from torchvision.utils import save_image

from patch_flow.vton_utils import masked_mean


def edge_error(prediction, target, mask):
    horizontal_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    vertical_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    horizontal = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]) - (
        target[:, :, :, 1:] - target[:, :, :, :-1]
    )
    vertical = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]) - (
        target[:, :, 1:, :] - target[:, :, :-1, :]
    )
    return 0.5 * (
        masked_mean(horizontal.abs(), horizontal_mask)
        + masked_mean(vertical.abs(), vertical_mask)
    )


def disable_training_only_teacher(config):
    params = config.trainer.params
    for name in (
        "correspondence_center_weight", "correspondence_entropy_weight",
        "correspondence_nll_weight", "correspondence_photometric_weight",
        "correspondence_value_weight", "fine_correspondence_weight",
        "fine_value_weight", "fine_rgb_weight", "fine_warp_coordinate_weight",
        "fine_warp_smoothness_weight", "fine_warp_mask_weight",
    ):
        params[name] = 0.0
    params.fine_teacher_forcing_start = 0.0
    params.fine_teacher_forcing_steps = 0
    params.compute_validation_metrics = False
    params.save_validation_previews = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--experiment", default="viton-pft-xl-512x384-detail-logo-hf")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", default="/tmp/vton_branch_ablation.png")
    parser.add_argument("--route-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for XL branch ablation")

    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        config = compose(config_name="config", overrides=[f"experiment={args.experiment}"])
    config.model.params.pretrained_ckpt = None
    config.autoencoder.params.ckpt_path = args.vae_checkpoint
    config.data.params.validation.params.root = args.data_root
    disable_training_only_teacher(config)
    module = instantiate_from_config(
        OmegaConf.to_container(config.trainer, resolve=True)
    ).eval()
    checkpoint = torch.load(
        Path(args.checkpoint).resolve(strict=True), map_location="cpu",
        weights_only=False, mmap=True,
    )
    expected = module.state_dict()
    state = {
        key: value for key, value in checkpoint["state_dict"].items()
        if key in expected and value.shape == expected[key].shape
    }
    missing = {
        key for key in set(expected) - set(state)
        if ".garment_refiner.support_head." not in key
    }
    if missing:
        raise RuntimeError(f"Checkpoint is missing inference tensors: {sorted(missing)}")
    module.load_state_dict(state, strict=True)
    checkpoint_step = int(checkpoint.get("global_step", 0))
    print(f"loaded_step={checkpoint_step} tensors={len(state)}", flush=True)
    del checkpoint, state, expected
    module = module.cuda().eval()

    dataset = instantiate_from_config(
        OmegaConf.to_container(config.data.params.validation, resolve=True)
    )
    batch = default_collate([dataset[0]])
    batch = {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = module._encode_batch(batch)
    target = (encoded["target_image"].float() + 1) * 0.5
    garment = (batch["garment"].float() + 1) * 0.5
    mask = batch["person_garment_mask"].float() * batch["agnostic_mask"].float()
    seed = int(batch["validation_seed"][0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(encoded["target"].shape, generator=generator, dtype=encoded["target"].dtype).cuda()
    label = module._label(batch, 1, noise.device)
    refiner = module.model.garment_refiner
    saved_output = refiner.output.weight.detach().clone()
    saved_latent = refiner.latent_fusion.weight.detach().clone()
    saved_hf = refiner.hf_fusion.weight.detach().clone()

    @torch.inference_mode()
    def audit_route():
        masks = encoded["masks"]
        xt = masks.latent * noise + (1 - masks.latent) * encoded["person_context"]
        token_times = torch.where(
            masks.token,
            torch.zeros_like(masks.token, dtype=xt.dtype),
            torch.ones_like(masks.token, dtype=xt.dtype),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, entries = module.model(
                x=xt, t=token_times, y=label,
                person_agnostic=encoded["person_context"],
                person_mask=masks.condition,
                dense_pose=encoded["dense_pose"],
                garment_high_frequency=encoded["garment_high_frequency"],
                edit_mask=batch["agnostic_mask"].float(),
                garment_mask=batch["garment_mask"],
                return_garment_attention=True,
                garment_attention_scales=(),
                return_refiner_supervision=True,
                **module._garment_conditions(encoded),
            )
            entry = next(value for value in entries if value.get("scale") == "refiner")
            warped = (module.decode(entry["warped_garment_latent"]).float().clamp(-1, 1) + 1) * 0.5
        height, width = entry["grid"]
        grid = entry["sampling_grid"].float()
        identity_y, identity_x = torch.meshgrid(
            (torch.arange(height, device=grid.device) + 0.5) / height * 2 - 1,
            (torch.arange(width, device=grid.device) + 0.5) / width * 2 - 1,
            indexing="ij",
        )
        identity = torch.stack((identity_x, identity_y), -1).reshape(1, 1, -1, 2)
        displacement = grid - identity
        garment_mask_latent = F.interpolate(
            batch["garment_mask"].float(), (height, width), mode="area"
        )
        sampled_mask = F.grid_sample(
            garment_mask_latent,
            grid[:, 0].reshape(-1, height, width, 2),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        route_rgb = torch.cat(
            (grid[:, 0].reshape(-1, height, width, 2).permute(0, 3, 1, 2).add(1).mul(0.5),
             sampled_mask), dim=1,
        ).clamp(0, 1)
        route_rgb = F.interpolate(route_rgb, target.shape[-2:], mode="nearest")
        mask_rgb = F.interpolate(sampled_mask, target.shape[-2:], mode="nearest").repeat(1, 3, 1, 1)
        route_output = str(Path(args.output).with_name(Path(args.output).stem + "_route.png"))
        save_image(torch.cat((target, garment, warped, route_rgb, mask_rgb)), route_output, nrow=5)
        route_report = {
            "mean_abs_displacement": float(displacement.abs().mean()),
            "rms_displacement": float(displacement.square().mean().sqrt()),
            "head_grid_std": float(grid.float().std(1).mean()),
            "sampled_valid_fraction": float((sampled_mask > 0.5).float().mean()),
        }
        with open(str(Path(route_output).with_suffix(".json")), "w", encoding="utf-8") as handle:
            json.dump(route_report, handle, indent=2)
        print("route", json.dumps(route_report), f"saved={route_output}", flush=True)

    @torch.no_grad()
    def set_paths(fine=True, latent=True, hf=True):
        refiner.output.weight.copy_(saved_output if fine else torch.zeros_like(saved_output))
        refiner.latent_fusion.weight.copy_(saved_latent if latent else torch.zeros_like(saved_latent))
        refiner.hf_fusion.weight.copy_(saved_hf if hf else torch.zeros_like(saved_hf))

    @torch.inference_mode()
    def generate(cfg_scale, fine=True, latent=True, hf=True):
        set_paths(fine=fine, latent=latent, hf=hf)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sample = module.flow.generate(
                model=module.model, x=noise.clone(),
                person_agnostic=encoded["person_context"],
                person_condition=encoded["person_context"],
                person_condition_mask=encoded["masks"].condition,
                dense_pose=encoded["dense_pose"],
                garment_high_frequency=encoded["garment_high_frequency"],
                edit_mask=batch["agnostic_mask"].float(),
                garment_mask=batch["garment_mask"], y=label,
                **module._garment_conditions(encoded),
                num_steps=args.steps, cfg_scale=cfg_scale, adaptive=False,
            )
            return (module.decode(sample).float().clamp(-1, 1) + 1) * 0.5

    audit_route()
    if args.route_only:
        return

    cases = (
        ("backbone_cfg1", dict(cfg_scale=1.0, fine=False)),
        ("full_cfg1", dict(cfg_scale=1.0)),
        ("full_cfg1.5", dict(cfg_scale=1.5)),
        ("no_direct_latent", dict(cfg_scale=1.0, latent=False)),
        ("no_hf", dict(cfg_scale=1.0, hf=False)),
        ("learned_v_only", dict(cfg_scale=1.0, latent=False, hf=False)),
    )
    images = [target, garment]
    report = {"checkpoint_step": checkpoint_step, "columns": ["target", "garment"]}
    denominator = mask.sum((2, 3), keepdim=True).clamp_min(1)
    target_mean = (target * mask).sum((2, 3), keepdim=True) / denominator
    for name, options in cases:
        prediction = generate(**options)
        prediction_mean = (prediction * mask).sum((2, 3), keepdim=True) / denominator
        report[name] = {
            "rgb_mae": float(masked_mean((prediction - target).abs(), mask)),
            "edge_mae": float(edge_error(prediction, target, mask)),
            "mean_rgb": [round(float(value), 6) for value in prediction_mean.flatten()],
            "target_mean_rgb": [round(float(value), 6) for value in target_mean.flatten()],
            "mean_bias": round(float((prediction_mean - target_mean).mean()), 6),
        }
        report["columns"].append(name)
        images.append(prediction)
        print(name, json.dumps(report[name]), flush=True)
    set_paths()
    save_image(torch.cat(images), args.output, nrow=len(images))
    with open(str(Path(args.output).with_suffix(".json")), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
