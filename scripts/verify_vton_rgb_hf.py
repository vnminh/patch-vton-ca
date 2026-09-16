#!/usr/bin/env python3
"""CPU smoke check for RGB-DoG features fused into the one RGB refiner head."""
from pathlib import Path
import sys
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from jutils.nn.kl_autoencoder import AutoencoderKL
from patch_flow.models.pf_transformer_vton import VTONPatchForcingDiT
from patch_flow.trainer_vton import LatentVTONPatchForcingTrainer


def main():
    torch.manual_seed(17)
    vae = AutoencoderKL(ddconfig={
        "attn_type": "vanilla", "double_z": True, "z_channels": 4,
        "resolution": 64, "in_channels": 3, "out_ch": 3, "ch": 32,
        "ch_mult": [1, 2, 2, 2], "num_res_blocks": 1,
        "attn_resolutions": [], "dropout": 0.0,
    })
    model = VTONPatchForcingDiT(
        input_size=8, in_channels=4, hidden_size=32, depth=3, num_heads=4,
        num_classes=10, compile=False, cross_attention_every=1,
        garment_middle_channels=64, garment_detail_channels=32,
        garment_scale_routes=["coarse", "middle", "detail"],
        garment_match_query_grid=True, garment_latent_refiner=True,
        garment_refiner_width=32, garment_refiner_heads=4,
        garment_refiner_qk_norm=True, garment_high_frequency_channels=64,
        garment_high_frequency_global_attention=False,
        garment_refiner_global_attention=False,
        garment_refiner_shared_sampling_grid=True,
        garment_refiner_velocity_max_backbone_ratio=0.1,
        garment_refiner_velocity_min_limit=0.05,
        garment_refiner_detail_activity_floor=0.25,
        garment_value_preserve_magnitude=True,
    )
    # Production warm-starts a trained nonzero RGB refiner. Reproduce that condition so
    # the new zero-HF feature gate receives a useful gradient on the first test update.
    torch.nn.init.normal_(model.garment_refiner.output.weight, std=0.01)
    module = LatentVTONPatchForcingTrainer(
        model=model, first_stage=vae, ema_rate=0,
        flow={"target": "patch_flow.flow_vton.VTONPatchFlowForcing", "params": {"patch_size": 2}},
        compute_validation_metrics=False, garment_supervision_only=True,
        garment_token_min_coverage=0.8, garment_dropout_prob=0.0,
        correspondence_center_weight=0.0, correspondence_nll_weight=0.0,
        correspondence_entropy_weight=0.0, correspondence_photometric_weight=0.0,
        detail_loss_weight=0.0, detail_pure_noise_only=False,
        hf_detail_loss_weight=0.0,
        hf_source_consistency_weight=0.25,
        hf_source_consistency_scale=2,
        hf_source_sparse_weight=1.0,
        hf_sparse_activity_threshold=0.04,
        hf_sparse_support_radius=1,
        hf_decoded_rgb_weight=1.0,
        hf_decoded_contrast_weight=1.0,
        hf_decoded_chroma_weight=1.0,
        hf_decoded_edge_weight=0.1,
        hf_decoded_max_samples=1,
        learnable_hf_condition_encoder=True,
        hf_condition_encoder_lr_multiplier=0.05,
        fine_velocity_regularization_weight=2.0,
        fine_velocity_max_backbone_ratio=0.1,
        fine_velocity_min_limit=0.05,
        garment_refiner_lr_multiplier=0.1,
        garment_high_frequency_lr_multiplier=0.1,
        garment_value_mix_lr_multiplier=1.0,
        garment_latent_fusion_lr_multiplier=1.0,
        adapter_lr_multiplier=0.1,
        decoded_rgb_weight=1.0,
        decoded_edge_weight=0.1,
        decoded_garment_rgb_weight=1.0,
        decoded_garment_low_frequency_weight=1.0,
        decoded_garment_mean_weight=2.0,
        decoded_max_samples=1,
        allow_new_garment_refiner=True,
    ).train()
    legacy = module.state_dict()
    del legacy["model.garment_refiner.latent_fusion.weight"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        module.load_state_dict(legacy, strict=True)
    assert not model.garment_refiner.latent_fusion.weight.any()
    module.flow.t_sampler = lambda shape, device, dtype: torch.full(
        shape, 0.5, device=device, dtype=dtype
    )
    image = torch.rand(1, 3, 64, 48).mul(2).sub(1)
    edit = torch.ones(1, 1, 64, 48)
    data = {
        "image": image, "person": image, "person_agnostic": image * (1 - edit),
        "agnostic_mask": edit, "person_garment_mask": edit,
        "garment": image.roll(3, -1), "garment_mask": edit,
        "has_ground_truth": torch.ones(1, dtype=torch.bool),
        "person_high_frequency": torch.rand(1, 6, 64, 48),
        "garment_high_frequency": torch.cat((
            torch.full((1, 3, 64, 48), 0.5), torch.zeros(1, 3, 64, 48)
        ), 1),
    }
    blank = module._encode_batch(data)["garment_high_frequency"]
    assert blank.abs().max() == 0, blank.abs().max()

    data["garment_high_frequency"] = torch.rand(1, 6, 64, 48)
    loss, metrics = module(data)
    assert torch.isfinite(loss) and metrics["fine_velocity_regularization"] >= 0
    assert metrics["hf_source_consistency_loss"] > 0
    assert metrics["hf_source_sparse_loss"] > 0
    assert metrics["hf_decoded_rgb_loss"] > 0
    assert metrics["hf_decoded_contrast_loss"] > 0
    assert metrics["hf_decoded_chroma_loss"] > 0
    assert metrics["hf_decoded_edge_loss"] > 0
    assert metrics["hf_decoded_samples"] == 1
    assert metrics["fine_velocity_dc_rms"] < 1e-5
    assert metrics["decoded_garment_rgb_loss"] > 0
    assert metrics["decoded_garment_low_frequency_loss"] > 0
    assert metrics["decoded_garment_mean_loss"] > 0
    assert metrics["garment_value_mix_detail"] == 0
    assert 0 <= metrics["fine_velocity_norm_gate"] <= 1
    assert 0 < metrics["fine_velocity_learned_gate"] < 2
    assert .25 <= metrics["fine_velocity_activity_gate"] < 1
    assert 0 < metrics["fine_velocity_effective_gate"] < 2
    assert not any(
        parameter.requires_grad
        for parameter in model.garment_high_frequency_control.warp_mix.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.garment_refiner.warp_mix.parameters()
    )
    loss.backward()
    value_mix_gradient = model.garment_value_mix["detail"].grad
    assert value_mix_gradient is not None and value_mix_gradient.abs() > 0
    fusion_gradient = model.garment_refiner.hf_fusion.weight.grad
    assert fusion_gradient is not None and fusion_gradient.abs().sum() > 0
    latent_gradient = model.garment_refiner.latent_fusion.weight.grad
    assert latent_gradient is not None and latent_gradient.abs().sum() > 0
    gradient = model.garment_high_frequency_control.encoder.weight.grad
    assert gradient is None or gradient.abs().sum() == 0
    assert model.garment_high_frequency_control.encoder.weight.detach().abs().sum() > 0
    optimizer = module.configure_optimizers()["optimizer"]
    parameter_lrs = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups for parameter in group["params"]
    }
    assert parameter_lrs[id(model.garment_refiner.hf_fusion.weight)] == module.lr * 0.1
    assert parameter_lrs[id(model.garment_refiner.latent_fusion.weight)] == module.lr
    assert parameter_lrs[id(model.garment_refiner.fine_gate.weight)] == module.lr * 0.1
    assert parameter_lrs[id(model.garment_high_frequency_control.encoder.weight)] == module.lr * 0.1
    assert parameter_lrs[id(module.hf_condition_encoder.conv_in.weight)] == module.lr * 0.05
    assert parameter_lrs[id(model.x_embedder.proj.weight)] == module.lr * 0.1
    assert parameter_lrs[id(model.garment_value_mix["detail"])] == module.lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss, _ = module(data)
    second_loss.backward()
    assert model.garment_high_frequency_control.encoder.weight.grad.abs().sum() > 0
    assert model.garment_high_frequency_control.feature_out.weight.grad.abs().sum() > 0
    assert module.hf_condition_encoder.conv_in.weight.grad.abs().sum() > 0

    # HF must alter the one RGB-refiner output, never appear as an independently added
    # four-channel velocity. The state condition itself remains backbone-only.
    model.eval()
    direct = dict(
        x=torch.randn(1, 4, 8, 6), t=torch.full((1, 12), 0.5),
        person_agnostic=torch.randn(1, 4, 8, 6), person_mask=torch.ones(1, 1, 8, 6),
        edit_mask=torch.ones(1, 1, 8, 6), garment=torch.randn(1, 4, 8, 6),
        garment_middle=torch.randn(1, 64, 16, 12),
        garment_detail=torch.randn(1, 32, 32, 24),
        garment_mask=torch.ones(1, 1, 64, 48),
    )
    conditions, hf_features = [], []
    handle = model.garment_refiner.velocity_condition.register_forward_pre_hook(
        lambda layer, args: conditions.append(args[0].detach().clone())
    )
    hf_handle = model.garment_high_frequency_control.register_forward_hook(
        lambda layer, args, value: hf_features.append(value.detach().clone())
    )
    with torch.no_grad():
        without_hf, maps_without = model(
            **direct, garment_high_frequency=torch.zeros(1, 64, 32, 24),
            return_garment_attention=True, return_refiner_supervision=True,
        )
        with_hf, maps_with = model(
            **direct, garment_high_frequency=torch.randn(1, 64, 32, 24),
            return_garment_attention=True, return_refiner_supervision=True,
        )
    handle.remove()
    hf_handle.remove()
    torch.testing.assert_close(conditions[0], conditions[1])
    assert hf_features[0].shape[1] == model.garment_refiner.width
    assert not hf_features[0].any() and hf_features[1].abs().sum() > 0
    fine_without = [entry for entry in maps_without if entry.get("scale") == "refiner"][0]
    fine_with = [entry for entry in maps_with if entry.get("scale") == "refiner"][0]
    assert fine_with["warped_garment_latent"].shape == (1, 4, 8, 6)
    assert not torch.allclose(fine_without["fine_velocity"], fine_with["fine_velocity"])
    torch.testing.assert_close(
        with_hf - without_hf,
        fine_with["fine_velocity"] - fine_without["fine_velocity"],
    )
    assert not torch.allclose(without_hf, with_hf)
    support = fine_with["fine_velocity_support"]
    torch.testing.assert_close(
        fine_with["fine_velocity"].sum((2, 3)),
        torch.zeros_like(fine_with["fine_velocity"].sum((2, 3))),
        atol=1e-5, rtol=0,
    )
    denominator = support.sum((1, 2, 3)) * fine_with["fine_velocity"].shape[1]
    fine_rms = (
        fine_with["fine_velocity"].float().square().sum((1, 2, 3)) / denominator
    ).sqrt()
    backbone_rms = (
        (fine_with["pre_refiner_velocity"].float().square() * support).sum((1, 2, 3))
        / denominator
    ).sqrt()
    limit = torch.maximum(backbone_rms * .1, backbone_rms.new_full((1,), .05))
    assert torch.all(fine_rms <= limit + 1e-5)
    assert 0 <= fine_with["fine_norm_gate_mean"] <= 1
    assert .25 <= fine_with["fine_activity_gate_mean"] < 1
    assert fine_with["fine_activity_gate_mean"] > fine_without["fine_activity_gate_mean"]
    grid = fine_with["sampling_grid"]
    torch.testing.assert_close(grid, grid[:, :1].expand_as(grid))
    rows = (torch.arange(8, dtype=torch.float32) + .5) * (2 / 8) - 1
    columns = (torch.arange(6, dtype=torch.float32) + .5) * (2 / 6) - 1
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    teacher_grid = torch.stack((xx, yy), -1).reshape(1, 48, 2)
    with torch.no_grad():
        _, teacher_maps = model(
            **direct,
            garment_high_frequency=torch.zeros(1, 64, 32, 24),
            garment_sampling_grid=teacher_grid,
            garment_sampling_mask=torch.ones(1, 48, dtype=torch.bool),
            return_garment_attention=True,
            return_refiner_supervision=True,
        )
    teacher_entry = [
        entry for entry in teacher_maps if entry.get("scale") == "refiner"
    ][0]
    torch.testing.assert_close(
        teacher_entry["transport_sampling_grid"],
        teacher_grid[:, None].expand(-1, 4, -1, -1),
    )
    assert teacher_entry["teacher_forcing_fraction"] == 1
    print(
        "PASS: blank VAE baseline is exactly zero; RGB-DoG/gradient features are "
        "64 channels in this tiny test; source consistency trains routing; fused decoded "
        "RGB/chroma losses open the zero-init bounded HF-to-RGB fusion; pretrained "
        "condition stem receives gradient; both global mixers are disabled; all heads "
        "share one deformation; detached HF activity and learned spatial gates are live; "
        "absolute/low-pass/mean garment colour losses are finite; final fine velocity "
        "is DC-free and hard-limited to 10% backbone authority; teacher-forced "
        "transport uses its supplied grid exactly"
    )


if __name__ == "__main__":
    main()
