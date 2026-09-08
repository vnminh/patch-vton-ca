#!/usr/bin/env python3
"""Read-only CPU checkpoint probe: compare raw and bounded fine attention, not image quality."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from jutils import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from patch_flow.vton_data import VTONValidationDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = OmegaConf.load(args.config)
    cfg.model.params.pretrained_ckpt = None
    cfg.autoencoder.params.ckpt_path = args.vae_checkpoint
    module = instantiate_from_config(OmegaConf.to_container(cfg.trainer, resolve=True)).cpu().eval()
    saved = torch.load(Path(args.checkpoint).resolve(strict=True), map_location='cpu', weights_only=False, mmap=True)
    module.load_state_dict(saved['state_dict'], strict=True)
    print(json.dumps({'checkpoint_step': saved['global_step'], 'strict_load': True}), flush=True)
    del saved
    data = default_collate([VTONValidationDataset(
        args.data_root, image_size=(512,384), preview_sample_id='00055_00.jpg',
        garment_parse_labels=[5,6,7],
    )[0]])
    captured = []
    with torch.no_grad():
        encoded = module._encode_batch(data)
        refiner = module.model.garment_refiner
        handle = refiner.register_forward_pre_hook(lambda mod, inputs: captured.append(inputs[:6]))
        target = encoded['target']
        module.model(
            x=.5 * target + .5 * torch.randn_like(target),
            t=torch.full((1, target.shape[-2] * target.shape[-1] // 4), .5),
            person_agnostic=encoded['person_context'], person_mask=encoded['masks'].condition,
            edit_mask=encoded['masks'].condition, garment_mask=data['garment_mask'],
            **module._garment_conditions(encoded),
        )
        handle.remove()
        uv, weight, _ = module._correspondence_targets(data, encoded, encoded['masks'].token, None)
        uv, weight = module._fine_targets(uv, weight, data, encoded, None)
        indices = (weight[0] > 0).nonzero().flatten()[::4]
        for normalized in (False, True):
            refiner.qk_norm = normalized
            _, entry = refiner(*captured[0], return_supervision=True)
            q, k = entry['query'][:, :, indices], entry['key']
            logits = q @ k.transpose(-1,-2) / math.sqrt(q.shape[-1])
            logits = logits.masked_fill(~entry['key_valid'][:,None,None], -torch.inf)
            attention = logits.softmax(-1)
            nll, mass, correct, count = module._fine_correspondence_chunk(
                q, k, uv[:,indices], weight[:,indices], entry['key_valid'], *entry['grid'],
            )
            denominator = count.clamp_min(1) * q.shape[1]
            print(json.dumps({
                'qk_norm': normalized, 'queries': len(indices),
                'q_rms': float(q.square().mean().sqrt()), 'k_rms': float(k.square().mean().sqrt()),
                'score_absmax': float(logits[logits.isfinite()].abs().max()),
                'entropy': float(-(attention * attention.clamp_min(1e-30).log()).sum(-1).mean()),
                'top_key_mass': float(attention.max(-1).values.mean()),
                'fine_nll': float(nll / denominator), 'target_mass': float(mass / denominator),
                'fine_top1': float(correct / denominator),
            }), flush=True)


if __name__ == '__main__':
    main()
