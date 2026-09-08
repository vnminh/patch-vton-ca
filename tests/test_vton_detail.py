"""Regression checks for equal-grid refinement and direct fine supervision."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

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
        velocity, maps = new(**data, return_garment_attention=True, return_refiner_supervision=True)
    assert velocity.shape == (2,4,8,6)
    for entry in maps[:-1]:
        assert entry['grid'] == entry['query_grid'] == (4,3)
        assert entry['weights'].shape[-2:] == (12,12)
    assert maps[-1]['query_grid'] == (8,6)
    assert maps[-1]['query'].shape == (2,4,48,8)
    assert maps[-1]['key'].shape == (2,4,48,8)
    assert maps[-1]['output'].shape == (2,48,32)
    data['garment_detail'] = torch.randn(2,8,24,32)
    with pytest.raises(ValueError, match='identical resolution'):
        new(**data)


def test_refiner_is_garment_transport_only_and_preserves_fine_phase():
    refiner = GarmentLatentRefiner(32, width=32, heads=4)
    nn.init.normal_(refiner.output.weight, std=.1)
    edit = torch.ones(2,1,8,6)
    edit[0,:,:,3:] = 0
    garment = torch.ones(2,1,64,48)
    garment[1] = 0
    args = (torch.randn(2,12,32), torch.randn(2,4,8,6), torch.randn(2,48,32),
            torch.randn(1,48,32), edit, garment)
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []
    def capture(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return original(q, k, v, **kwargs)
    with patch('torch.nn.functional.scaled_dot_product_attention', side_effect=capture):
        output, supervision = refiner(*args, return_supervision=True)
    assert torch.isfinite(output).all()
    assert output[0,:,:,:3].abs().sum() > 0
    assert not output[0,:,:,3:].any() and not output[1].any()
    assert not torch.allclose(output[0,:,::2,:2], output[0,:,1::2,:2])
    assert len(calls) == 1
    assert supervision['query'].shape == supervision['key'].shape == (2,4,48,8)
    handle = refiner.attention_out.register_forward_hook(lambda module, inputs, value: torch.zeros_like(value))
    without_transport = refiner(*args)
    handle.remove()
    assert not without_transport.any(), 'person query must not bypass garment transport into output'
    output.square().mean().backward()
    for name in ('query_expand','query','key','value','output'):
        assert getattr(refiner,name).weight.grad.abs().sum() > 0

def test_direct_fine_correspondence_rejects_fixed_key_collapse_and_has_qk_gradients():
    module = trainer()
    module.fine_correspondence_radius = 0
    query = torch.tensor([[[[8.,0.],[8.,0.],[8.,0.],[8.,0.]]]], requires_grad=True)
    key = torch.tensor([[[[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]]], requires_grad=True)
    target = torch.tensor([[[.125,.5],[.375,.5],[.625,.5],[.875,.5]]])
    weight = torch.ones(1,4)
    valid = torch.ones(1,4,dtype=torch.bool)
    numerator, mass, correct, effective = module._fine_correspondence_chunk(
        query,key,target,weight,valid,1,4
    )
    loss = numerator / effective
    assert loss > 5 and mass / effective < .3 and correct / effective == .25
    loss.backward()
    assert query.grad.abs().sum() > 0 and key.grad.abs().sum() > 0


@pytest.mark.parametrize('empty', [False, True])
def test_fine_correspondence_empty_positive_and_fractional_accuracy(empty):
    module = trainer()
    module.fine_correspondence_radius = 0
    query = torch.randn(1,2,2,4,requires_grad=True)
    key = torch.randn(1,2,4,4,requires_grad=True)
    valid = torch.tensor([[not empty,False,False,False]])
    target = torch.tensor([[[.125,.5],[.875,.5]]])
    result = module._fine_correspondence_chunk(query,key,target,torch.tensor([[.8,0.]]),valid,1,4)
    assert all(torch.isfinite(x) for x in result)
    assert result[2] <= result[3] * 2 + 1e-6
    result[0].backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()
    if empty:
        assert all(x == 0 for x in result)


def test_fine_teacher_never_blends_rejected_or_disconnected_matches():
    module = trainer()
    data = batch()
    data['person_garment_mask'].fill_(1)
    encoded = {'target':torch.zeros(3,4,4,4)}
    uv = torch.tensor([[[.1,.1],[.9,.9],[.2,.2],[.8,.8]]]).expand(3,-1,-1)
    teacher = torch.tensor([[1.,0.,1.,1.]]).expand(3,-1)
    target, weight = module._fine_targets(uv,teacher,data,encoded,None)
    target = target.reshape(3,4,4,2)
    torch.testing.assert_close(target[0,:2,:2],uv[0,0].expand(2,2,2))
    assert not weight.reshape(3,4,4)[0,:2,2:].any()
    assert not weight[2].any()  # unpaired


def test_qk_normalization_bounds_scores_and_matches_inference_transport():
    refiner = GarmentLatentRefiner(32,width=32,heads=4,qk_norm=True,cosine_scale=10.)
    # Deliberately mimic runaway learned positional scale.
    with torch.no_grad():
        refiner.position.weight.mul_(1000)
    args = (torch.randn(1,12,32),torch.randn(1,4,8,6),torch.randn(1,48,32),
            torch.randn(1,48,32),torch.ones(1,1,8,6),torch.ones(1,1,64,48))
    _, entry = refiner(*args,return_supervision=True)
    q,k = entry['query'],entry['key']
    scores = q @ k.transpose(-1,-2) / q.shape[-1]**.5
    assert scores.abs().max() <= 10.00001
    v = refiner.value(args[2]).reshape(1,48,4,8).transpose(1,2)
    expected = (scores.softmax(-1) @ v).transpose(1,2).reshape(1,48,32)
    torch.testing.assert_close(entry['output'],refiner.attention_out(expected),rtol=1e-5,atol=1e-6)
    entry['output'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in refiner.parameters() if p.grad is not None)
    # No learned tensors or optimizer groups change when normalization is enabled.
    raw = GarmentLatentRefiner(32,width=32,heads=4)
    raw.load_state_dict(refiner.state_dict(),strict=True)


def test_fine_position_changes_routing_but_never_value_input():
    refiner = GarmentLatentRefiner(32,width=32,heads=4,qk_norm=True)
    args = [torch.randn(1,12,32),torch.randn(1,4,8,6),torch.randn(1,48,32),
            torch.randn(1,48,32),torch.ones(1,1,8,6),torch.ones(1,1,64,48)]
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []
    def capture(q,k,v,**kwargs):
        calls.append((q.detach().clone(),k.detach().clone(),v.detach().clone()))
        return original(q,k,v,**kwargs)
    with patch('torch.nn.functional.scaled_dot_product_attention',side_effect=capture):
        refiner(*args)
        args[3] = torch.randn_like(args[3])
        refiner(*args)
    assert not torch.allclose(calls[0][0],calls[1][0])
    assert not torch.allclose(calls[0][1],calls[1][1])
    torch.testing.assert_close(calls[0][2],calls[1][2],rtol=0,atol=0)


def test_fixed_rgb_transport_penalizes_wrong_colour_and_has_qk_gradient():
    module = trainer()
    q = torch.tensor([[[[4.,0.],[4.,0.]]]],requires_grad=True)
    k = torch.tensor([[[[1.,0.],[-1.,0.]]]],requires_grad=True)
    colors = torch.tensor([[[1.,0.,0.],[0.,0.,1.]]])
    # Unequal query weights avoid exact cancellation of the two opposing K gradients.
    loss = module._fine_rgb_chunk(q,k,colors,colors,torch.tensor([[.5,1.]]),torch.ones(1,2,dtype=torch.bool))
    assert loss > .6
    loss.backward()
    assert q.grad.abs().sum() > 0 and k.grad.abs().sum() > 0
    zero = module._fine_rgb_chunk(q,k,colors,colors,torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool))
    assert zero == 0


def test_fine_rgb_pooling_excludes_white_background_and_skin():
    module = trainer()
    image = torch.ones(1,3,4,4)
    image[:,:,:,:2] = -1
    mask = torch.zeros(1,1,4,4)
    mask[:,:,:,:2] = 1
    data = {'garment':image, 'garment_mask':mask, 'person_garment_mask':mask, 'agnostic_mask':torch.ones_like(mask)}
    target, garment = module._fine_rgb_targets(data,{'target_image':image},(1,1))
    assert not target.any() and not garment.any()


def test_chunked_checkpoint_gradients_match_full_fine_losses():
    from torch.utils.checkpoint import checkpoint
    module = trainer()
    torch.manual_seed(19)
    q = torch.randn(2,2,7,4,requires_grad=True)
    k = torch.randn(2,2,6,4,requires_grad=True)
    uv = torch.rand(2,7,2)
    weights = torch.rand(2,7)
    valid = torch.tensor([[True,True,False,True,True,True],[True,False,True,True,True,False]])
    rgb_q, rgb_k = torch.rand(2,7,3), torch.rand(2,6,3)
    def objective(query, key, chunked):
        total = query.sum() * 0
        size = 3 if chunked else 7
        for start in range(0,7,size):
            end = start + size
            args = (query[:,:,start:end],key,uv[:,start:end],weights[:,start:end],valid,2,3)
            rgb_args = (query[:,:,start:end],key,rgb_q[:,start:end],rgb_k,weights[:,start:end],valid)
            if chunked:
                total = total + checkpoint(module._fine_correspondence_chunk,*args,use_reentrant=False)[0]
                total = total + checkpoint(module._fine_rgb_chunk,*rgb_args,use_reentrant=False)
            else:
                total = total + module._fine_correspondence_chunk(*args)[0] + module._fine_rgb_chunk(*rgb_args)
        return total
    full = objective(q,k,False)
    full_grads = torch.autograd.grad(full,(q,k))
    chunks = objective(q,k,True)
    chunk_grads = torch.autograd.grad(chunks,(q,k))
    torch.testing.assert_close(full,chunks)
    for a,b in zip(full_grads,chunk_grads):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


def test_stable_experiment_composes_without_changing_resolution():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail-stable'])
    assert cfg.model.params.garment_refiner_qk_norm
    assert cfg.trainer.params.fine_rgb_weight > 0
    assert cfg.trainer.params.fine_correspondence_weight == .05
    assert list(cfg.data.params.train.params.image_size) == [512,384]


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
    module.decoded_max_samples = 0
    data['person_garment_mask'][1,:,:8] = 1
    loss, metrics = module._decoded_garment_loss(
        latent, data, {'target_image':target}, torch.full_like(times,.5), torch.ones(3)
    )
    assert metrics['decoded_samples'] == 2  # third sample is intentionally unpaired
    for times, keep in [(torch.zeros_like(times),None),(torch.ones_like(times),None),
                         (torch.full_like(times,.5),torch.zeros(3))]:
        with patch.object(module,'_decode_with_grad',side_effect=AssertionError('empty mask decoded')):
            loss, metrics = module._decoded_garment_loss(latent,data,{'target_image':target},times,keep)
        assert loss == 0 and metrics['decoded_samples'] == 0


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
    state = new.state_dict()
    state['model.garment_refiner.condition.weight'] = torch.randn(1)
    new.load_state_dict(state)  # obsolete shortcut weights from the collapsed run are allowed
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
    assert cfg.trainer.params.decoded_max_samples == 0
    assert cfg.trainer.params.fine_correspondence_weight > 0
    assert cfg.trainer.params.fine_value_weight > 0
    assert 'attention_tv_weight' not in cfg.trainer.params
    assert not (Path(__file__).resolve().parents[1]/'patch_flow'/'attention_smoothing.py').exists()
