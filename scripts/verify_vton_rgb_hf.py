#!/usr/bin/env python3
"""CPU smoke check for RGB-DoG features fused into the one RGB refiner head."""
from pathlib import Path
import sys
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
        hf_detail_loss_weight=0.5,
        hf_source_consistency_weight=0.25,
        hf_source_consistency_scale=2,
        hf_source_sparse_weight=1.0,
        hf_sparse_activity_threshold=0.04,
        hf_sparse_support_radius=1,
        hf_decoded_rgb_weight=0.5,
        hf_decoded_chroma_weight=0.5,
        hf_decoded_edge_weight=1.0,
        hf_decoded_max_samples=1,
        learnable_hf_condition_encoder=True,
        hf_condition_encoder_lr_multiplier=0.05,
    ).train()
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
    assert torch.isfinite(loss) and metrics["hf_detail_loss"] > 0
    assert metrics["hf_source_consistency_loss"] > 0
    assert metrics["hf_source_sparse_loss"] > 0
    assert metrics["hf_decoded_rgb_loss"] > 0
    assert metrics["hf_decoded_chroma_loss"] > 0
    assert metrics["hf_decoded_edge_loss"] > 0
    assert metrics["hf_decoded_samples"] == 1
    assert not any(
        parameter.requires_grad
        for parameter in model.garment_high_frequency_control.warp_mix.parameters()
    )
    hf_gradient, backbone_gradient, refiner_gradient = torch.autograd.grad(
        metrics["hf_detail_loss"],
        (
            model.garment_high_frequency_control.encoder.weight,
            model.final_layer.linear.weight,
            model.garment_refiner.output.weight,
        ),
        retain_graph=True,
        allow_unused=True,
    )
    assert hf_gradient is not None and hf_gradient.abs().sum() > 0
    assert backbone_gradient is None
    assert refiner_gradient is not None and refiner_gradient.abs().sum() > 0
    loss.backward()
    gradient = model.garment_high_frequency_control.encoder.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    assert model.garment_high_frequency_control.encoder.weight.detach().abs().sum() == 0
    optimizer = module.configure_optimizers()["optimizer"]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss, _ = module(data)
    second_loss.backward()
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
    conditions, fine_outputs, hf_features = [], [], []
    handle = model.garment_refiner.velocity_condition.register_forward_pre_hook(
        lambda layer, args: conditions.append(args[0].detach().clone())
    )
    fine_handle = model.garment_refiner.output.register_forward_hook(
        lambda layer, args, value: fine_outputs.append(value.detach().clone())
    )
    hf_handle = model.garment_high_frequency_control.register_forward_hook(
        lambda layer, args, value: hf_features.append(value.detach().clone())
    )
    with torch.no_grad():
        without_hf = model(
            **direct, garment_high_frequency=torch.zeros(1, 64, 32, 24)
        )
        with_hf = model(**direct, garment_high_frequency=torch.randn(1, 64, 32, 24))
    handle.remove()
    fine_handle.remove()
    hf_handle.remove()
    torch.testing.assert_close(conditions[0], conditions[1])
    assert hf_features[0].shape[1] == model.garment_refiner.width
    assert not hf_features[0].any() and hf_features[1].abs().sum() > 0
    assert not torch.allclose(fine_outputs[0], fine_outputs[1])
    torch.testing.assert_close(with_hf - without_hf, fine_outputs[1] - fine_outputs[0])
    assert not torch.allclose(without_hf, with_hf)
    print(
        "PASS: blank VAE baseline is exactly zero; RGB-DoG/gradient features are "
        "64 channels in this tiny test; source consistency trains routing; fused decoded "
        "RGB/chroma/edge losses reach both the zero-init HF encoder and RGB refiner; "
        "pretrained condition stem receives gradient; global HF mixing is disabled; "
        "HF changes the one fine velocity only through RGB+HF feature fusion"
    )


if __name__ == "__main__":
    main()
