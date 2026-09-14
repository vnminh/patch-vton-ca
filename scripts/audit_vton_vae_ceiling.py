#!/usr/bin/env python3
"""Measure the frozen SD-VAE reconstruction ceiling on a real VITON-HD pair."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from jutils import instantiate_from_config
from omegaconf import OmegaConf
from torchvision.utils import save_image

from patch_flow.vton_data import VTONHDDataset
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--output", default="/tmp/vton_vae_ceiling.png")
    parser.add_argument("--sample", default="00055_00.jpg")
    args = parser.parse_args()
    torch.set_num_threads(4)

    repo = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(repo / "configs/autoencoder/sd_ae.yaml")
    config.params.ckpt_path = args.vae_checkpoint
    vae = instantiate_from_config(OmegaConf.to_container(config, resolve=True)).eval()
    dataset = VTONHDDataset(
        args.data_root, split="test", paired=True, image_size=(512, 384),
        garment_parse_labels=(5, 6, 7), preview_sample_id=args.sample,
    )
    sample = dataset[0]
    person = sample["person"][None]
    garment = sample["garment"][None]
    person_mask = sample["person_garment_mask"][None]
    garment_mask = sample["garment_mask"][None]
    with torch.inference_mode():
        person_reconstruction = vae.decode(vae.encode(person))
        garment_reconstruction = vae.decode(vae.encode(garment))

    def report(name, prediction, target, mask):
        prediction_01 = (prediction.float().clamp(-1, 1) + 1) * 0.5
        target_01 = (target.float() + 1) * 0.5
        print(
            f"{name}: rgb_mae={masked_mean((prediction_01-target_01).abs(), mask):.6f} "
            f"edge_mae={edge_error(prediction_01,target_01,mask):.6f}",
            flush=True,
        )

    report("person_garment", person_reconstruction, person, person_mask)
    report("inshop_garment", garment_reconstruction, garment, garment_mask)
    panel = torch.cat((person, person_reconstruction, garment, garment_reconstruction))
    save_image((panel.float().clamp(-1, 1) + 1) * 0.5, args.output, nrow=4)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
