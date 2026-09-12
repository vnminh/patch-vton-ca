#!/usr/bin/env python3
"""CPU regression check for the checkpoint-compatible VTON deformable detail path."""
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from patch_flow.models.pf_transformer_vton import (
    VTONPatchForcingDiT,
    attention_sampling_grid,
    hard_attention_sampling_grid,
    sample_attention_heads,
    upsample_displacement_grid,
)
from patch_flow.trainer_vton import LatentVTONPatchForcingTrainer


def tiny_model(dense_pose_channels=4, high_frequency_channels=8):
    return VTONPatchForcingDiT(
        input_size=8,
        in_channels=4,
        hidden_size=32,
        depth=3,
        num_heads=4,
        num_classes=10,
        cross_attention_every=1,
        garment_middle_channels=8,
        garment_detail_channels=8,
        garment_scale_routes=["coarse", "middle", "detail"],
        garment_latent_refiner=True,
        garment_match_query_grid=True,
        garment_refiner_width=32,
        garment_refiner_heads=4,
        garment_refiner_qk_norm=True,
        gradient_checkpointing=True,
        dense_pose_channels=dense_pose_channels,
        garment_high_frequency_channels=high_frequency_channels,
    )


def model_inputs():
    return dict(
        x=torch.randn(2, 4, 8, 6),
        t=torch.full((2, 12), 0.5),
        person_agnostic=torch.randn(2, 4, 8, 6),
        person_mask=torch.ones(2, 1, 8, 6),
        dense_pose=torch.randn(2, 4, 8, 6),
        garment=torch.randn(2, 4, 8, 6),
        garment_middle=torch.randn(2, 8, 16, 12),
        garment_detail=torch.randn(2, 8, 32, 24),
        garment_mask=torch.ones(2, 1, 64, 48),
        garment_high_frequency=torch.randn(2, 8, 32, 24),
        return_garment_attention=True,
        return_refiner_supervision=True,
    )


def trainer(model):
    return LatentVTONPatchForcingTrainer(
        model=model,
        first_stage=torch.nn.Identity(),
        ema_rate=0,
        flow={"target": "patch_flow.flow_vton.VTONPatchFlowForcing", "params": {"patch_size": 2}},
        compute_validation_metrics=False,
        correspondence_center_weight=0,
        correspondence_nll_weight=0,
        correspondence_entropy_weight=0,
        correspondence_photometric_weight=0,
        allow_new_garment_refiner=True,
        allow_new_garment_high_frequency=True,
    )


def main():
    torch.manual_seed(23)
    query = torch.eye(4).reshape(1, 1, 4, 4) * 20
    key = query.clone()
    valid = torch.ones(1, 4, dtype=torch.bool)
    values = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]])
    grid = attention_sampling_grid(query, key, valid, 2, 2)
    sampled = sample_attention_heads(values, grid, valid, 2, 2)
    torch.testing.assert_close(sampled, values, rtol=0, atol=1e-5)

    model = tiny_model().train()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    torch.nn.init.normal_(model.garment_refiner.output.weight, std=0.02)
    torch.nn.init.normal_(model.garment_high_frequency_control.encoder.weight, std=0.02)
    captured = []
    handle = model.garment_high_frequency_control.register_forward_pre_hook(
        lambda _, args: captured.append(
            (args[1].requires_grad, args[2].requires_grad, args[6].requires_grad)
        )
    )
    velocity, maps = model(**model_inputs())
    handle.remove()
    fine = [entry for entry in maps if entry.get("scale") == "refiner"]
    assert len(fine) == 1 and fine[0]["sampling_grid"].shape == (2, 4, 48, 2)
    assert captured == [(False, False, False)]
    (velocity - torch.randn_like(velocity)).square().mean().backward()
    assert model.garment_refiner.state.weight.grad[:, 8:].abs().sum() > 0
    assert model.garment_refiner.warp_mix.weight.grad.abs().sum() > 0
    assert model.garment_high_frequency_control.warp_mix.weight.grad.abs().sum() > 0

    mixed = tiny_model().train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed_velocity, mixed_maps = mixed(**model_inputs())
    assert torch.isfinite(mixed_velocity).all()
    assert torch.isfinite(mixed_maps[-1]["sampling_grid"]).all()

    module = trainer(tiny_model())
    legacy = module.state_dict()
    state_key = "model.garment_refiner.state.weight"
    old_state = torch.randn_like(legacy[state_key][:, :8])
    legacy[state_key] = old_state
    for prefix in (
        "model.garment_refiner.warp_mix.",
        "model.garment_high_frequency_control.warp_mix.",
    ):
        for name in [key for key in legacy if key.startswith(prefix)]:
            del legacy[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.load_state_dict(legacy, strict=True)
    assert caught
    torch.testing.assert_close(module.model.garment_refiner.state.weight[:, :8], old_state)
    assert not module.model.garment_refiner.state.weight[:, 8:].any()
    assert not module.model.garment_refiner.warp_mix.weight.any()
    assert not module.model.garment_high_frequency_control.warp_mix.weight.any()

    module.fine_correspondence_weight = 0.05
    module.fine_rgb_weight = 0.2
    module.fine_warp_coordinate_weight = 0.1
    module.fine_warp_smoothness_weight = 0.02
    module.fine_warp_mask_weight = 0.05
    fine_query = torch.randn(2, 4, 16, 8, requires_grad=True)
    fine_key = torch.randn(2, 4, 16, 8, requires_grad=True)
    fine_valid = torch.ones(2, 16, dtype=torch.bool)
    fine_grid = attention_sampling_grid(fine_query, fine_key, fine_valid, 4, 4)
    coarse_query = torch.randn(2, 4, 4, 8, requires_grad=True)
    coarse_key = torch.randn(2, 4, 4, 8, requires_grad=True)
    coarse_valid = torch.ones(2, 4, dtype=torch.bool)
    coarse_grid = hard_attention_sampling_grid(
        coarse_query, coarse_key, coarse_valid, 2, 2
    )
    coarse_identity = module._identity_sampling_grid((2, 2), torch.device("cpu")).add(1).mul(0.5)
    image = torch.rand(2, 3, 16, 16).mul(2).sub(1)
    fine_batch = {
        "agnostic_mask": torch.ones(2, 1, 16, 16),
        "person_garment_mask": torch.ones(2, 1, 16, 16),
        "garment_mask": torch.ones(2, 1, 16, 16),
        "has_ground_truth": torch.ones(2, dtype=torch.bool),
        "garment": image.roll(2, -1),
    }
    fine_entry = {
        "query": fine_query,
        "key": fine_key,
        "key_valid": fine_valid,
        "sampling_grid": fine_grid,
        "output": torch.randn(2, 16, 32),
        "grid": (4, 4),
        "coarse_query": coarse_query,
        "coarse_key": coarse_key,
        "coarse_key_valid": coarse_valid,
        "coarse_sampling_grid": coarse_grid,
        "coarse_grid": (2, 2),
    }
    fine_loss, fine_metrics = module._fine_losses(
        fine_entry,
        coarse_identity[None].expand(2, -1, -1),
        torch.ones(2, 4),
        fine_batch,
        {"target": torch.zeros(2, 4, 4, 4), "target_image": image},
        None,
    )
    assert fine_loss > 0
    assert all(torch.isfinite(value) for value in fine_metrics.values())
    fine_loss.backward()
    assert fine_query.grad.abs().sum() > 0 and fine_key.grad.abs().sum() > 0

    identity = module._identity_sampling_grid((4, 4), torch.device("cpu"))
    routed = identity.reshape(1, 1, 16, 2).clone()
    routed[:, :, 5, 0] += 0.2
    routed.requires_grad_(True)
    smooth = module._fine_warp_bending_loss(routed, torch.ones(1, 16), (4, 4))
    assert smooth > 0
    source = torch.randn(1, 16, 3)
    warped = module._sample_fine_tokens(routed, source, (4, 4))
    (warped.square().mean() + smooth).backward()
    assert routed.grad is not None and torch.isfinite(routed.grad).all()

    print("PASS: hard coarse anchors, local coherent warp, propagated supervision, "
          "direct DensePose, detached shared HF grid and checkpoint migration")


if __name__ == "__main__":
    main()
