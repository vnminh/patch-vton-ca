#!/usr/bin/env python3
"""CPU smoke check for RGB-DoG baseline subtraction and HF-only detail loss."""
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
    )
    module = LatentVTONPatchForcingTrainer(
        model=model, first_stage=vae, ema_rate=0,
        flow={"target": "patch_flow.flow_vton.VTONPatchFlowForcing", "params": {"patch_size": 2}},
        compute_validation_metrics=False, garment_supervision_only=True,
        garment_token_min_coverage=0.8, garment_dropout_prob=0.0,
        correspondence_center_weight=0.0, correspondence_nll_weight=0.0,
        correspondence_entropy_weight=0.0, correspondence_photometric_weight=0.0,
        detail_loss_weight=0.0, detail_pure_noise_only=False,
        hf_detail_loss_weight=0.5,
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
        "garment_high_frequency": torch.cat((
            torch.full((1, 3, 64, 48), 0.5), torch.zeros(1, 3, 64, 48)
        ), 1),
    }
    blank = module._encode_batch(data)["garment_high_frequency"]
    assert blank.abs().max() == 0, blank.abs().max()

    data["garment_high_frequency"] = torch.rand(1, 6, 64, 48)
    loss, metrics = module(data)
    assert torch.isfinite(loss) and metrics["hf_detail_loss"] > 0
    hf_gradient, backbone_gradient = torch.autograd.grad(
        metrics["hf_detail_loss"],
        (
            model.garment_high_frequency_control.encoder.weight,
            model.final_layer.linear.weight,
        ),
        retain_graph=True,
        allow_unused=True,
    )
    assert hf_gradient is not None and hf_gradient.abs().sum() > 0
    assert backbone_gradient is None
    loss.backward()
    gradient = model.garment_high_frequency_control.encoder.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    assert model.garment_high_frequency_control.encoder.weight.detach().abs().sum() == 0
    print(
        "PASS: blank VAE baseline is exactly zero; RGB-DoG/gradient features are "
        "64 channels in this tiny test; hf_detail_loss reaches the zero-init HF encoder"
    )


if __name__ == "__main__":
    main()
