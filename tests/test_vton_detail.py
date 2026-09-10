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


def model(refiner=False, match=False, dense_pose_channels=0, garment_high_frequency_channels=0):
    return VTONPatchForcingDiT(
        input_size=8, in_channels=4, hidden_size=32, depth=3, num_heads=4,
        num_classes=10, cross_attention_every=1, garment_middle_channels=8,
        garment_detail_channels=8, garment_scale_routes=['coarse', 'middle', 'detail'],
        garment_latent_refiner=refiner, garment_match_query_grid=match,
        garment_refiner_width=32, garment_refiner_heads=4, gradient_checkpointing=True,
        dense_pose_channels=dense_pose_channels,
        garment_high_frequency_channels=garment_high_frequency_channels,
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


def test_dense_pose_is_only_zero_initialized_appended_input_channels():
    def wrap(network):
        return LatentVTONPatchForcingTrainer(
            model=network, first_stage=torch.nn.Identity(), ema_rate=0,
            flow={'target':'patch_flow.flow_vton.VTONPatchFlowForcing','params':{'patch_size':2}},
            compute_validation_metrics=False, correspondence_center_weight=0,
            correspondence_nll_weight=0, correspondence_entropy_weight=0,
            correspondence_photometric_weight=0,
        )
    old, new = wrap(model()), wrap(model(dense_pose_channels=4))
    old_weight = old.model.x_embedder.proj.weight.detach().clone()
    with pytest.warns(UserWarning,match='zero-initialized conditioning channel'):
        new.load_state_dict(old.state_dict(),strict=True)
    weight = new.model.x_embedder.proj.weight
    torch.testing.assert_close(weight[:,:9],old_weight,rtol=0,atol=0)
    assert not weight[:,9:].any()
    data = inputs()
    old.eval(); new.eval()
    with torch.no_grad():
        reference = old.model(**data)
        output = new.model(**data,dense_pose=torch.randn(2,4,8,6))
    torch.testing.assert_close(output,reference,rtol=0,atol=0)
    new.train()
    embedded = []
    handle = new.model.x_embedder.register_forward_hook(lambda module,args,output: embedded.append(output))
    new.model(**data,dense_pose=torch.randn(2,4,8,6))
    handle.remove()
    embedded[0].sum().backward()
    assert weight.grad[:,9:].abs().sum() > 0


@pytest.mark.parametrize('ema_rate', [0, .99])
def test_high_frequency_control_preserves_13_inputs_and_warm_starts(ema_rate):
    def wrap(network):
        return LatentVTONPatchForcingTrainer(
            model=network, first_stage=torch.nn.Identity(), ema_rate=ema_rate,
            flow={'target':'patch_flow.flow_vton.VTONPatchFlowForcing','params':{'patch_size':2}},
            compute_validation_metrics=False, correspondence_center_weight=0,
            correspondence_nll_weight=0, correspondence_entropy_weight=0,
            correspondence_photometric_weight=0,
            allow_new_garment_high_frequency=True,
        )
    old = wrap(model(refiner=True, dense_pose_channels=4))
    # Nonzero old output makes prediction parity a meaningful check.
    nn.init.normal_(old.model.final_layer.linear.weight, std=.1)
    new = wrap(model(refiner=True, dense_pose_channels=4, garment_high_frequency_channels=8))
    old_weight = old.model.x_embedder.proj.weight.detach().clone()
    with pytest.warns(UserWarning, match='zero-initialized encoder'):
        new.load_state_dict(old.state_dict(), strict=True)
    weight = new.model.x_embedder.proj.weight
    assert weight.shape[1] == 13
    torch.testing.assert_close(weight, old_weight, rtol=0, atol=0)
    # The zero sits on the encoder, not the head: the head must keep a usable
    # initialisation or nothing upstream of it ever receives gradient.
    assert not new.model.garment_high_frequency_control.encoder.weight.any()
    assert new.model.garment_high_frequency_control.output.weight.any()
    if ema_rate:
        assert not new.ema_model.garment_high_frequency_control.encoder.weight.any()

    data = inputs()
    dense_pose = torch.randn(2, 4, 8, 6)
    high_frequency = torch.randn(2, 8, 32, 24)
    old.eval(); new.eval()
    with torch.no_grad():
        reference = old.model(**data, dense_pose=dense_pose)
        output = new.model(
            **data, dense_pose=dense_pose, garment_high_frequency=high_frequency
        )
    torch.testing.assert_close(output, reference, rtol=0, atol=0)

    # Previously trained 14-channel checkpoints can explicitly discard just HF.
    state = old.state_dict()
    for key in ('model.x_embedder.proj.weight', 'ema_model.x_embedder.proj.weight'):
        if key in state:
            state[key] = torch.cat((state[key], torch.ones_like(state[key][:, :1])), dim=1)
    with pytest.warns(UserWarning):
        new.load_state_dict(state, strict=True)
    torch.testing.assert_close(new.model.x_embedder.proj.weight, old_weight, rtol=0, atol=0)
    new.load_state_dict(new.state_dict(), strict=True)
    partial = new.state_dict()
    del partial['model.garment_high_frequency_control.output.bias']
    with pytest.raises(RuntimeError, match='Missing key'):
        new.load_state_dict(partial, strict=True)


def test_garment_gradient_norms_cover_every_conditioning_branch():
    """train.py is the only caller, so nothing else exercises these attribute paths.

    A stale ``control.encoder[1]`` survived a refactor here and only surfaced as a
    TypeError after 399 real training iterations.
    """
    net = model(refiner=True, dense_pose_channels=4, garment_high_frequency_channels=8)
    module = LatentVTONPatchForcingTrainer(
        model=net, first_stage=torch.nn.Identity(), ema_rate=0,
        flow={'target': 'patch_flow.flow_vton.VTONPatchFlowForcing', 'params': {'patch_size': 2}},
        compute_validation_metrics=False, correspondence_center_weight=0,
        correspondence_nll_weight=0, correspondence_entropy_weight=0,
        correspondence_photometric_weight=0, allow_new_garment_high_frequency=True,
    )
    # A loaded checkpoint has a trained refiner head (L2 0.0286 at step 2000), which is
    # exactly why the zero-initialised `state` receives gradient there. A fresh model's
    # head is zero, so mimic the real case rather than testing a degenerate one.
    nn.init.normal_(net.garment_refiner.output.weight, std=.1)
    data = inputs()
    velocity = module.model(**data, dense_pose=torch.randn(2, 4, 8, 6),
                            garment_high_frequency=torch.randn(2, 8, 32, 24))
    # DiT zero-initialises final_layer.linear, so velocity.square() alone is identically
    # zero and every gradient with it. Regress against a target instead.
    (velocity - torch.randn_like(velocity)).square().mean().backward()
    metrics = module.garment_gradient_norms()
    for key in ('garment_grad/hf/encoder', 'garment_grad/hf/output',
                'garment_grad/refiner/state', 'garment_grad/refiner/query',
                'garment_grad/embedder_detail'):
        assert key in metrics, key
        assert torch.isfinite(metrics[key])
    # The two branches whose gradient path this revision repaired must be alive on the
    # first backward; the HF head's weight is still waiting for a non-zero input.
    assert metrics['garment_grad/hf/encoder'] > 0
    assert metrics['garment_grad/refiner/state'] > 0
    assert metrics['garment_grad/hf/output'] == 0


def test_hf_values_are_centred_across_valid_keys_so_diffuse_attention_gives_zero():
    """A Canny map's VAE features carry a large global mean (measured 0.5591 against
    -0.0102 for garment features). Uniform attention over uncentred values returns that
    mean as a spatially constant latent offset -- a flat colour shift once decoded."""
    net = model(refiner=True, garment_high_frequency_channels=8).eval()
    control = net.garment_high_frequency_control
    nn.init.normal_(control.encoder.weight, std=.1)
    nn.init.constant_(control.encoder.bias, 3.0)          # a large shared component
    hf = torch.randn(2, 8, 32, 24)
    valid = torch.ones(2, 48, dtype=torch.bool)
    valid[:, 24:] = False                                  # only some keys are garment
    query = torch.zeros(2, 4, 48, 8)                       # uniform attention
    key = torch.zeros(2, 4, 48, 8)
    with torch.no_grad():
        residual = control(hf, query, key, valid,
                           torch.ones(2, 1, 8, 6), torch.ones(2, 1, 64, 48))
    # Uniform attention over centred values retrieves the mean of the centred values,
    # which is zero by construction. Any surviving residual is the DC leak.
    assert residual.abs().max() < 1e-4, residual.abs().max()

    # The deviations must still survive: a different edge map must give a different
    # residual once attention is not uniform.
    sharp = torch.randn(2, 4, 48, 8) * 5
    with torch.no_grad():
        a = control(hf, sharp, sharp, valid, torch.ones(2, 1, 8, 6), torch.ones(2, 1, 64, 48))
        b = control(hf.roll(7, -1), sharp, sharp, valid,
                    torch.ones(2, 1, 8, 6), torch.ones(2, 1, 64, 48))
    assert a.abs().sum() > 0 and not torch.allclose(a, b)


@pytest.mark.parametrize('gains', [(1., 1.), (10., .25)])
def test_warm_start_velocity_head_gains_are_explicit_and_weight_only(gains):
    refiner_gain, hf_gain = gains
    net = model(refiner=True, garment_high_frequency_channels=8)
    module = LatentVTONPatchForcingTrainer(
        model=net, first_stage=torch.nn.Identity(), ema_rate=0,
        flow={'target': 'patch_flow.flow_vton.VTONPatchFlowForcing', 'params': {'patch_size': 2}},
        compute_validation_metrics=False, correspondence_center_weight=0,
        correspondence_nll_weight=0, correspondence_entropy_weight=0,
        correspondence_photometric_weight=0, allow_new_garment_high_frequency=True,
        warm_start_refiner_output_gain=refiner_gain,
        warm_start_high_frequency_output_gain=hf_gain,
    )
    state = module.state_dict()
    # A trained-looking pair of heads, as a real checkpoint would carry.
    for key, value in (('model.garment_refiner.output.weight', .000792),
                       ('model.garment_high_frequency_control.output.weight', .031433)):
        state[key] = torch.full_like(state[key], value)
    state['model.garment_refiner.output.bias'] = torch.full_like(
        state['model.garment_refiner.output.bias'], .5)
    module.load_state_dict(state, strict=True)
    torch.testing.assert_close(module.model.garment_refiner.output.weight,
                               torch.full_like(state['model.garment_refiner.output.weight'],
                                               .000792 * refiner_gain))
    torch.testing.assert_close(module.model.garment_high_frequency_control.output.weight,
                               torch.full_like(
                                   state['model.garment_high_frequency_control.output.weight'],
                                   .031433 * hf_gain))
    # Biases are never touched, and a zero head stays zero under any gain.
    torch.testing.assert_close(module.model.garment_refiner.output.bias,
                               state['model.garment_refiner.output.bias'])
    state['model.garment_refiner.output.weight'] = torch.zeros_like(
        state['model.garment_refiner.output.weight'])
    module.load_state_dict(state, strict=True)
    assert not module.model.garment_refiner.output.weight.any()


def test_previous_hf_revision_checkpoint_loads_by_discarding_the_whole_branch():
    """The old pixel encoder is renamed and reshaped, but parts of it collide.

    ``local`` and ``output`` keep matching names and shapes across the two revisions, so
    dropping only the incompatible tensors leaves the branch looking present, suppresses
    the warm-start and turns the genuinely new encoder into a strict missing-key error.
    """
    net = model(refiner=True, dense_pose_channels=4, garment_high_frequency_channels=8)
    module = LatentVTONPatchForcingTrainer(
        model=net, first_stage=torch.nn.Identity(), ema_rate=0,
        flow={'target': 'patch_flow.flow_vton.VTONPatchFlowForcing', 'params': {'patch_size': 2}},
        compute_validation_metrics=False, correspondence_center_weight=0,
        correspondence_nll_weight=0, correspondence_entropy_weight=0,
        correspondence_photometric_weight=0, allow_new_garment_high_frequency=True,
    )
    prefix = 'model.garment_high_frequency_control.'
    legacy = module.state_dict()
    for key in (prefix + 'encoder.weight', prefix + 'encoder.bias'):
        del legacy[key]
    # Shapes and names of the previous pixel-unshuffle encoder.
    legacy[prefix + 'encoder.1.weight'] = torch.randn(32, 64, 1, 1)
    legacy[prefix + 'encoder.1.bias'] = torch.randn(32)
    legacy[prefix + 'encoder.5.weight'] = torch.randn(32, 32, 1, 1)
    # Its bias-carrying local block, which the current revision dropped.
    legacy[prefix + 'local.2.bias'] = torch.randn(32)
    legacy[prefix + 'local.4.bias'] = torch.randn(32)
    # Colliding tensors that must not keep the branch alive.
    nn.init.normal_(module.model.garment_high_frequency_control.output.bias, std=.1)
    legacy[prefix + 'output.bias'] = module.model.garment_high_frequency_control.output.bias.clone()

    with pytest.warns(UserWarning):
        module.load_state_dict(legacy, strict=True)
    control = module.model.garment_high_frequency_control
    assert not control.encoder.weight.any() and not control.encoder.bias.any()
    assert not control.output.bias.any()
    assert control.output.weight.any()

    data = inputs()
    module.model.eval()
    with torch.no_grad():
        residual = []
        handle = control.register_forward_hook(lambda m, a, v: residual.append(v))
        module.model(**data, dense_pose=torch.randn(2, 4, 8, 6),
                     garment_high_frequency=torch.randn(2, 8, 32, 24))
        handle.remove()
    assert not residual[0].any()


def test_hf_control_adds_velocity_only_and_trains_encoder_from_first_step():
    net = model(refiner=True, garment_high_frequency_channels=8).train()
    data = inputs()
    data['edit_mask'] = torch.ones(2, 1, 8, 6)
    data['edit_mask'][0, :, :, 3:] = 0
    hf = torch.randn(2, 8, 32, 24)
    control = net.garment_high_frequency_control
    optimizer = torch.optim.Adam(control.parameters(), lr=.01)
    for step in range(2):
        captured = []
        routing_requires_grad = []
        def capture_control(module, args, value):
            captured.append(value)
            routing_requires_grad.append((args[1].requires_grad, args[2].requires_grad))
        handle = control.register_forward_hook(capture_control)
        velocity, logvar = net(**data, garment_high_frequency=hf, return_uncertainty=True)
        handle.remove()
        # HF copies the detail refiner's attention probabilities, but its loss must not
        # update the shared Q/K routing. RGB/correspondence supervision owns that map.
        assert routing_requires_grad == [(False, False)]
        if step == 0:
            assert not captured[0].any()
        with torch.no_grad():
            baseline, baseline_logvar = net(
                **data, garment_high_frequency=torch.zeros_like(hf), return_uncertainty=True
            )
        torch.testing.assert_close(velocity, baseline + captured[0])
        torch.testing.assert_close(logvar, baseline_logvar, rtol=0, atol=0)
        assert not captured[0][0, :, :, 3:].any()
        (velocity - torch.randn_like(velocity)).square().mean().backward()
        # The point of moving the zero off the head: the encoder is trainable from the
        # very first step instead of waiting for a zero head that never grew. The head's
        # weight has no gradient until the encoder output is non-zero, which costs one
        # step; its bias carries gradient immediately.
        assert control.encoder.weight.grad.abs().sum() > 0
        assert control.output.bias.grad.abs().sum() > 0
        assert (control.output.weight.grad.abs().sum() > 0) == bool(step)
        optimizer.step()
        net.zero_grad(set_to_none=True)
    assert not torch.allclose(control.encoder(hf), control.encoder(hf.roll(1, -1)))
    # Biases cannot leak a residual with dropped cloth or an empty edge map.
    args = []
    handle = control.register_forward_hook(lambda module, inputs, value: args.append(value))
    net(**data, garment_high_frequency=torch.zeros_like(hf))
    data['garment_mask'].zero_()
    net(**data, garment_high_frequency=hf)
    handle.remove()
    assert all(not value.any() for value in args)


def test_refiner_is_garment_transport_only_and_preserves_fine_phase():
    refiner = GarmentLatentRefiner(32, width=32, heads=4)
    nn.init.normal_(refiner.output.weight, std=.1)
    edit = torch.ones(2,1,8,6)
    edit[0,:,:,3:] = 0
    garment = torch.ones(2,1,64,48)
    garment[1] = 0
    args = (torch.randn(2,12,32), torch.randn(2,4,8,6), torch.randn(2,4,8,6),
            torch.randn(2,48,32), torch.randn(1,48,32), edit, garment)
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
    args = (torch.randn(1,12,32),torch.randn(1,4,8,6),torch.randn(1,4,8,6),
            torch.randn(1,48,32),torch.randn(1,48,32),torch.ones(1,1,8,6),torch.ones(1,1,64,48))
    _, entry = refiner(*args,return_supervision=True)
    q,k = entry['query'],entry['key']
    scores = q @ k.transpose(-1,-2) / q.shape[-1]**.5
    assert scores.abs().max() <= 10.00001
    v = refiner.value(args[3]).reshape(1,48,4,8).transpose(1,2)
    expected = (scores.softmax(-1) @ v).transpose(1,2).reshape(1,48,32)
    torch.testing.assert_close(entry['output'],refiner.attention_out(expected),rtol=1e-5,atol=1e-6)
    entry['output'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in refiner.parameters() if p.grad is not None)
    # No learned tensors or optimizer groups change when normalization is enabled.
    raw = GarmentLatentRefiner(32,width=32,heads=4)
    raw.load_state_dict(refiner.state_dict(),strict=True)


def test_fine_position_changes_routing_but_never_value_input():
    refiner = GarmentLatentRefiner(32,width=32,heads=4,qk_norm=True)
    args = [torch.randn(1,12,32),torch.randn(1,4,8,6),torch.randn(1,4,8,6),
            torch.randn(1,48,32),torch.randn(1,48,32),torch.ones(1,1,8,6),torch.ones(1,1,64,48)]
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []
    def capture(q,k,v,**kwargs):
        calls.append((q.detach().clone(),k.detach().clone(),v.detach().clone()))
        return original(q,k,v,**kwargs)
    with patch('torch.nn.functional.scaled_dot_product_attention',side_effect=capture):
        refiner(*args)
        args[4] = torch.randn_like(args[4])
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
    assert cfg.model.params.dense_pose_channels == 4
    assert cfg.model.params.get('garment_high_frequency_channels', 0) == 0
    assert cfg.trainer.params.fine_rgb_weight > 0
    assert cfg.trainer.params.fine_correspondence_weight == .05
    assert list(cfg.data.params.train.params.image_size) == [512,384]
    assert cfg.data.params.train.params.dense_pose_dir == 'image-densepose'
    assert not cfg.data.params.train.params.get('garment_high_frequency', False)


def test_hf_control_experiment_is_separate_and_enabled_for_paired_and_swapped_data():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail-rebalance-hf'])
    assert cfg.model.params.garment_high_frequency_channels == 128
    assert cfg.model.params.garment_latent_refiner
    assert cfg.trainer.params.allow_new_garment_high_frequency
    assert cfg.data.params.train.params.garment_high_frequency
    assert cfg.data.params.validation.params.garment_high_frequency
    assert cfg.name.endswith('hf-control')


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
    # Decoded reconstruction covers the full edit region, including non-garment
    # anatomy. Only the top-right token is excluded by its .99 timestep here.
    assert latent.grad[0,:,2:].abs().sum() > 0
    assert not latent.grad[0,:,:2,2:].any()
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


def test_garment_losses_exclude_arms_while_decoded_reconstruction_includes_them():
    module = trainer()
    data = batch()
    data['person_garment_mask'].zero_()
    data['person_garment_mask'][:, :, :, :8] = 1
    # The right half represents an erased arm: editable, but not garment parsing.
    garment_pixels = module._supervision_pixels(data)
    reconstruction_pixels = module._reconstruction_pixels(data)
    assert garment_pixels[:, :, :, :8].all()
    assert not garment_pixels[:, :, :, 8:].any()
    assert reconstruction_pixels[:, :, :, 8:].all()

    module.first_stage = TinyDecoder().requires_grad_(False)
    module.decoded_rgb_weight, module.decoded_edge_weight = 1., 0.
    module.decoded_max_samples = 0
    latent = torch.randn(3, 4, 4, 4, requires_grad=True)
    target = torch.zeros(3, 3, 16, 16)
    times = torch.full((3, 4), .5)
    loss, metrics = module._decoded_garment_loss(
        latent, data, {'target_image': target}, times, torch.ones(3)
    )
    assert loss > 0 and metrics['decoded_samples'] == 2
    loss.backward()
    # Both halves receive decoded reconstruction gradients, while the third sample
    # remains excluded because batch() marks it unpaired.
    assert latent.grad[:2, :, :, :2].abs().sum() > 0
    assert latent.grad[:2, :, :, 2:].abs().sum() > 0
    assert not latent.grad[2].any()


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
