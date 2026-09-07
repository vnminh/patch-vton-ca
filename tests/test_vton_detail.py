"""Regression checks for equal-grid refinement, decoded gradients and masked ATV."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from patch_flow.attention_smoothing import attention_centers, key_coordinates, masked_center_tv
from patch_flow.models.pf_transformer_vton import GarmentLatentRefiner, VTONPatchForcingDiT
from patch_flow.trainer_vton import LatentVTONPatchForcingTrainer
from test_vton_supervision import trainer, batch


def model(refiner=False, match=False):
    return VTONPatchForcingDiT(
        input_size=8, in_channels=4, hidden_size=32, depth=3, num_heads=4,
        num_classes=10, cross_attention_every=1, garment_middle_channels=8,
        garment_detail_channels=8, garment_scale_routes=['coarse', 'middle', 'detail'],
        garment_latent_refiner=refiner, garment_match_query_grid=match,
        garment_refiner_width=32, garment_refiner_heads=4, gradient_checkpointing=True,
    )


def inputs():
    return dict(x=torch.randn(2,4,8,6), t=torch.full((2,12), .5),
                person_agnostic=torch.randn(2,4,8,6), person_mask=torch.ones(2,1,8,6),
                garment=torch.randn(2,4,8,6), garment_middle=torch.randn(2,8,16,12),
                garment_detail=torch.randn(2,8,32,24), garment_mask=torch.ones(2,1,64,48))


def test_equal_query_key_grids_and_fine_output_neutral_initialization():
    old, new = model(), model(refiner=True)
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert all(name.startswith('garment_refiner.') for name in missing.missing_keys)
    data = inputs()
    old.eval()
    new.eval()
    with torch.no_grad():
        torch.testing.assert_close(old(**data), new(**data), rtol=0, atol=0)
        new.garment_match_query_grid = True
        velocity, maps = new(**data, return_garment_attention=True, return_garment_centers=True)
    assert velocity.shape == (2,4,8,6)
    for entry in maps[:-1]:
        assert entry['grid'] == entry['query_grid'] == (4,3)
        assert entry['weights'].shape[-2:] == (12,12)
    assert maps[-1]['query_grid'] == (8,6)
    assert maps[-1]['centers'].shape == (2,48,2)
    data['garment_detail'] = torch.randn(2,8,24,32)
    with pytest.raises(ValueError, match='identical resolution'):
        new(**data)


def test_refiner_gating_fine_phase_and_coordinate_reduction():
    refiner = GarmentLatentRefiner(32, width=32, heads=4)
    nn.init.normal_(refiner.output.weight, std=.1)
    edit = torch.ones(2,1,8,6)
    edit[0,:,:,3:] = 0
    garment = torch.ones(2,1,64,48)
    garment[1] = 0
    args = (torch.randn(2,12,32), torch.randn(2,12,32), torch.randn(2,4,8,6),
            torch.randn(2,4,8,6), torch.randn(2,48,32), torch.randn(1,48,32), edit, garment)
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []
    def capture(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return original(q, k, v, **kwargs)
    with patch('torch.nn.functional.scaled_dot_product_attention', side_effect=capture):
        output, centers = refiner(*args, return_centers=True)
    assert torch.isfinite(output).all() and torch.isfinite(centers).all()
    assert output[0,:,:,:3].abs().sum() > 0
    assert not output[0,:,:,3:].any() and not output[1].any()
    assert not torch.allclose(output[0,:,::2,:2], output[0,:,1::2,:2])
    q, k, v, kwargs = calls[0]
    logits = q @ k.transpose(-2,-1) / q.shape[-1] ** .5
    logits = logits.masked_fill(~kwargs['attn_mask'], float('-inf'))
    expected = (logits.softmax(-1) @ key_coordinates((8,6), q.device)).mean(1)
    torch.testing.assert_close(centers, expected, atol=1e-6, rtol=1e-5)
    output.square().mean().backward()
    for name in ('query_expand','query','key','value','output'):
        assert getattr(refiner,name).weight.grad.abs().sum() > 0


def test_masked_tv_smoothness_boundaries_empty_masks_and_gradients():
    coords = key_coordinates((3,4), 'cpu').unsqueeze(0)
    mask = torch.ones(1,1,3,4)
    smooth = masked_center_tv(coords, (3,4), mask)
    shuffled = coords[:, [0,11,1,10,2,9,3,8,4,7,5,6]]
    assert masked_center_tv(shuffled, (3,4), mask) > smooth
    mask[:,:,:,2:] = 0
    constant = torch.full_like(coords, .7)
    constant[:,[2,3,6,7,10,11]] = -100
    assert masked_center_tv(constant,(3,4),mask) == 0
    logits = torch.randn(1,2,12,12,requires_grad=True)
    centers = attention_centers(logits.softmax(-1),(3,4))
    loss = masked_center_tv(centers,(3,4),mask)
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert not logits.grad[:,:, [2,3,6,7,10,11]].any()
    assert masked_center_tv(centers,(3,4),torch.zeros_like(mask)) == 0
    assert masked_center_tv(torch.zeros(1,1,2),(1,1),torch.ones(1,1,1,1)) == 0


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale, self.shift = .5, .1
        self.post_quant_conv = nn.Conv2d(4,4,1)
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=4,mode='nearest'), nn.Conv2d(4,3,1))

    @torch.no_grad()
    def decode(self, latent):
        return self.decoder(self.post_quant_conv(latent/self.scale+self.shift))


def test_decoded_loss_frozen_decoder_input_gradients_and_pixel_time_pair_masking():
    module = trainer()
    module.first_stage = TinyDecoder().requires_grad_(False)
    module.decoded_rgb_weight, module.decoded_edge_weight = .2, .5
    data = batch()
    data['person_garment_mask'][:,:,:8] = 1
    target = torch.zeros(3,3,16,16)
    latent = torch.randn(3,4,4,4,requires_grad=True)
    times = torch.tensor([[.5,.99,.5,.5],[.5,.5,.5,.5],[.5,.5,.5,.5]])
    loss, metrics = module._decoded_garment_loss(latent,data,{'target_image':target},times,torch.tensor([1.,0.,1.]))
    assert metrics['decoded_samples'] == 1 and loss > 0
    loss.backward()
    assert latent.grad[0,:,:2,:2].abs().sum() > 0
    assert not latent.grad[1:].any()
    assert not latent.grad[0,:,2:].any() and not latent.grad[0,:,:,2:].any()
    assert all(p.grad is None for p in module.first_stage.parameters())
    torch.testing.assert_close(module._decode_with_grad(latent),module.first_stage.decode(latent))
    for times, keep in [(torch.zeros_like(times),None),(torch.ones_like(times),None),
                         (torch.full_like(times,.5),torch.zeros(3))]:
        with patch.object(module,'_decode_with_grad',side_effect=AssertionError('empty mask decoded')):
            loss, metrics = module._decoded_garment_loss(latent,data,{'target_image':target},times,keep)
        assert loss == 0 and metrics['decoded_samples'] == 0


def test_tv_ignores_empty_dropped_unpaired_garments():
    module, data = trainer(), batch()
    data['person_garment_mask'][:] = 1
    entries = [{'scale':'refiner','query_grid':(4,4),'centers':torch.randn(3,16,2,requires_grad=True)}]
    value, _ = module._attention_tv_loss(entries,data,torch.tensor([1.,0.,1.]))
    value.backward()
    assert entries[0]['centers'].grad[0].abs().sum() > 0
    assert not entries[0]['centers'].grad[1:].any()
    data['garment_mask'].zero_()
    assert module._attention_tv_loss(entries,data,None)[0] == 0


def test_strict_legacy_warmstart_only_allows_wholly_new_refiner():
    old, new = trainer(), trainer()
    old.model, new.model = model(), model(refiner=True)
    with pytest.raises(RuntimeError,match='Missing key'):
        new.load_state_dict(old.state_dict())
    new.allow_new_garment_refiner = True
    with pytest.warns(UserWarning,match='fresh optimizer'):
        new.load_state_dict(old.state_dict())
    state = new.state_dict()
    del state['model.garment_refiner.output.weight']
    with pytest.raises(RuntimeError,match='Missing key'):
        new.load_state_dict(state)
    new.load_state_dict(new.state_dict())
    old_opt = old.configure_optimizers()['optimizer']
    new_opt = new.configure_optimizers()['optimizer']
    with pytest.raises(ValueError):
        new_opt.load_state_dict(old_opt.state_dict())


def test_detail_config_keeps_both_images_512x384_and_global_batch_32():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail'])
    assert cfg.model.params.garment_match_query_grid and cfg.model.params.garment_latent_refiner
    assert list(cfg.data.params.train.params.image_size) == [512,384]
    assert cfg.data.params.batch_size * cfg.train_params.accumulate_grad_batches == 32
    assert cfg.trainer.params.decoded_rgb_weight > 0 and cfg.trainer.params.decoded_edge_weight > 0
    assert cfg.trainer.params.attention_tv_weight == .01
