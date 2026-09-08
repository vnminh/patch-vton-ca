#!/usr/bin/env python3
"""CPU-only real-data/SD-VAE/DINO detail smoke test; never starts an XL training job."""
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
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--checkpoint', help='Also audit the full XL checkpoint on CPU/meta, without optimizer loading')
    parser.add_argument('--experiment', default='viton-pft-xl-512x384-detail')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(repo/'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=[f'experiment={args.experiment}'])
    cfg.model.params.pretrained_ckpt = None
    if args.checkpoint:
        # Resolve the completed file behind last.ckpt before opening it. mmap avoids
        # materialising several GB of optimizer history; this never loads it to GPU.
        path = Path(args.checkpoint).resolve(strict=True)
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        with torch.device('meta'):
            full = instantiate_from_config(OmegaConf.to_container(cfg.model, resolve=True))
        expected = full.state_dict()
        current = {key[6:]:value for key,value in saved['state_dict'].items() if key.startswith('model.')}
        missing = set(expected)-set(current)
        unexpected = set(current)-set(expected)
        obsolete = {key for key in unexpected if key.startswith(
            ('garment_refiner.condition.','garment_refiner.state.')
        ) or key in ('garment_refiner.local.1.bias','garment_refiner.local.3.bias')}
        assert unexpected == obsolete, unexpected - obsolete
        assert all(key.startswith('garment_refiner.') for key in missing), missing
        assert all(current[key].shape == expected[key].shape for key in set(current) & set(expected))
        print(f"CHECKPOINT PASS: step={saved['global_step']}, matching={len(set(current)&set(expected))}, "
              f"new_refiner_tensors={len(missing)}, obsolete_shortcut_tensors={len(obsolete)}", flush=True)
        del saved, current, expected, full
    cfg.model.params.hidden_size = 64
    cfg.model.params.depth = 3
    cfg.model.params.num_heads = 4
    cfg.model.params.cross_attention_every = 1
    cfg.model.params.garment_scale_routes = ['coarse','middle','detail']
    cfg.model.params.garment_refiner_width = 64
    cfg.model.params.garment_refiner_heads = 4
    cfg.autoencoder.params.ckpt_path = args.vae_checkpoint
    cfg.trainer.params.correspondence_warmup_steps = 0
    # A two-step smoke test must actually update the zero-initialized output.
    # Production uses a 1000-step LR warmup; its first optimizer LR is zero.
    cfg.trainer.params.lr_scheduler_cfg = None
    cfg.trainer.params.garment_dropout_prob = 0
    cfg.trainer.params.correspondence_nll_radius = {'coarse':.16,'middle':.12,'detail':.1}
    module = instantiate_from_config(OmegaConf.to_container(cfg.trainer, resolve=True)).cpu()
    module.flow.zero_time_probability = 0
    module.flow.high_time_probability = 0
    module.flow.t_sampler = lambda shape, device, dtype: torch.full(shape,.5,device=device,dtype=dtype)
    dataset = VTONValidationDataset(args.data_root, image_size=(args.height,args.width),
                                    preview_sample_id='00055_00.jpg', garment_parse_labels=[5,6,7])
    data = default_collate([dataset[0]])
    optimizer = module.configure_optimizers()['optimizer']
    module.train()
    # FLOAT32 CPU avoids slow BF16 emulation; mixed precision is covered by pytest.
    for step in range(2):
        loss, metrics = module(data)
        assert torch.isfinite(loss)
        assert metrics['decoded_samples'] == 1
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
        assert module.model.garment_refiner.output.weight.grad.abs().sum() > 0
        assert module.model.garment_refiner.query.weight.grad.abs().sum() > 0
        if step:
            assert module.model.garment_refiner.value.weight.grad.abs().sum() > 0
        assert all(p.grad is None and not p.requires_grad for p in module.first_stage.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print({'step':step, **{key:float(metrics[key].detach()) for key in (
            'flow_loss','decoded_rgb_loss','decoded_edge_loss','fine_correspondence_loss',
            'fine_target_mass','fine_value_loss','fine_top1_accuracy','fine_rgb_loss'
        )}}, flush=True)
    # Check actual SD decoder numerical parity and latent-input gradient propagation.
    latent = module.encode(data['image']).detach().requires_grad_(True)
    with torch.no_grad():
        reference = module.first_stage.decode(latent)
    decoded = module._decode_with_grad(latent)
    torch.testing.assert_close(decoded, reference, rtol=0, atol=0)
    decoded.mean().backward()
    assert latent.grad.abs().sum() > 0
    print(f'PASS: {args.height}x{args.width} real paired images, frozen SD-VAE and DINO, two optimizer steps, decoded gradient/parity and direct fine correspondence/value supervision. XL GPU peak memory and image quality are not tested.', flush=True)


if __name__ == '__main__':
    main()
