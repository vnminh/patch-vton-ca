#!/usr/bin/env python3
"""CPU integration smoke test using real VITON images, SD-VAE and DINO weights.

The DiT is deliberately small and randomly initialized. This verifies loss wiring,
finite backward gradients, and validation bookkeeping; it does not measure quality.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from hydra import compose, initialize_config_dir
from jutils import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from patch_flow.vton_data import VTONValidationDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        config = compose(config_name="config", overrides=["experiment=viton-pft-xl-512x384-garment-fix"])
    params = config.model.params
    params.pretrained_ckpt = None
    params.hidden_size = 64
    params.depth = 3
    params.num_heads = 4
    params.cross_attention_every = 1
    params.garment_scale_routes = ["coarse", "middle", "detail"]
    OmegaConf.update(config, "model.params.compile", False, force_add=True)
    config.autoencoder.params.ckpt_path = args.vae_checkpoint
    config.trainer.params.correspondence_warmup_steps = 0
    config.trainer.params.garment_dropout_prob = 0
    config.trainer.params.sample_kwargs.num_steps = 2
    config.trainer.params.sample_kwargs.cfg_scale = 1
    # Preserve the same pixel tolerance on the 4x smaller smoke-test images.
    config.trainer.params.correspondence_nll_radius = {"coarse": .16, "middle": .12, "detail": .1}
    module = instantiate_from_config(OmegaConf.to_container(config.trainer, resolve=True)).cpu()
    module.flow.zero_time_probability = 0
    module.flow.high_time_probability = 0
    module.flow.t_sampler = lambda shape, device, dtype: torch.full(shape, .5, device=device, dtype=dtype)
    dataset = VTONValidationDataset(args.data_root, image_size=(128,96), preview_sample_id="00055_00.jpg")
    batch = default_collate([dataset[0], dataset[2]])
    module.train()
    optimizer = module.configure_optimizers()["optimizer"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = module(batch)
    assert torch.isfinite(loss), "Non-finite training loss"
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    for block in module.model.blocks:
        assert block.garment_cross_attention.in_proj_weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    print({key: float(metrics[key].detach()) for key in (
        "flow_loss", "detail_loss", "garment_supervision_fraction", "correspondence_coverage"
    )}, flush=True)
    # A source person's paired reconstruction and swap use the same initial noise.
    module.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        module.validation_step(default_collate([dataset[0], dataset[1]]), 0)
    assert module._garment_validation_totals["test_paired"][2] == 1
    assert "test_unpaired" not in module._garment_validation_totals
    assert len(module.val_images["tryon"]) == 2
    print("PASS: real-data BF16 backward, optimizer step, paired/swap generation, and metric isolation.", flush=True)


if __name__ == "__main__":
    main()
