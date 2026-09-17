"""Regression checks for equal-grid refinement and direct fine supervision."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from patch_flow.models.pf_transformer_vton import (
    GarmentLatentRefiner,
    VTONPatchForcingDiT,
    attention_sampling_grid,
    hard_attention_sampling_grid,
    local_attention_sampling_grid,
    oracle_reachable_fraction,
    sample_attention_heads,
    upsample_displacement_grid,
)
from patch_flow.trainer_vton import (
    LatentVTONPatchForcingTrainer,
    stable_excess_rms_penalty,
)
from test_vton_supervision import trainer, batch


def model(refiner=False, match=False, dense_pose_channels=0, garment_high_frequency_channels=0,
          shared_sampling_grid=False, refiner_global_attention=True,
          velocity_max_backbone_ratio=0., velocity_min_limit=0.,
          detail_activity_floor=1., value_preserve_magnitude=False):
    return VTONPatchForcingDiT(
        input_size=8, in_channels=4, hidden_size=32, depth=3, num_heads=4,
        num_classes=10, cross_attention_every=1, garment_middle_channels=8,
        garment_detail_channels=8, garment_scale_routes=['coarse', 'middle', 'detail'],
        garment_latent_refiner=refiner, garment_match_query_grid=match,
        garment_refiner_width=32, garment_refiner_heads=4, gradient_checkpointing=True,
        dense_pose_channels=dense_pose_channels,
        garment_high_frequency_channels=garment_high_frequency_channels,
        garment_refiner_shared_sampling_grid=shared_sampling_grid,
        garment_refiner_global_attention=refiner_global_attention,
        garment_refiner_velocity_max_backbone_ratio=velocity_max_backbone_ratio,
        garment_refiner_velocity_min_limit=velocity_min_limit,
        garment_refiner_detail_activity_floor=detail_activity_floor,
        garment_value_preserve_magnitude=value_preserve_magnitude,
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
    assert maps[-1]['sampling_grid'].shape == (2,4,48,2)
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


def test_magnitude_value_route_is_zero_init_checkpoint_migration():
    def wrap(network):
        return LatentVTONPatchForcingTrainer(
            model=network, first_stage=torch.nn.Identity(), ema_rate=.99,
            flow={'target':'patch_flow.flow_vton.VTONPatchFlowForcing','params':{'patch_size':2}},
            compute_validation_metrics=False, correspondence_center_weight=0,
            correspondence_nll_weight=0, correspondence_entropy_weight=0,
            correspondence_photometric_weight=0,
        )
    old = wrap(model())
    new = wrap(model(value_preserve_magnitude=True))
    with pytest.warns(UserWarning, match='magnitude-preserving V path'):
        new.load_state_dict(old.state_dict(), strict=True)
    assert all(value.item() == 0 for value in new.model.garment_value_mix.values())
    assert all(value.item() == 0 for value in new.ema_model.garment_value_mix.values())


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
    with pytest.warns(UserWarning, match='zero-init HF-to-RGB'):
        new.load_state_dict(old.state_dict(), strict=True)
    weight = new.model.x_embedder.proj.weight
    assert weight.shape[1] == 13
    torch.testing.assert_close(weight, old_weight, rtol=0, atol=0)
    # The HF extractor is usable immediately; the basis conversion is the exact-zero
    # function-preserving gate.
    assert new.model.garment_high_frequency_control.encoder.weight.any()
    assert new.model.garment_high_frequency_control.feature_out.weight.any()
    assert not new.model.garment_refiner.hf_fusion.weight.any()
    if ema_rate:
        assert new.ema_model.garment_high_frequency_control.encoder.weight.any()
        assert not new.ema_model.garment_refiner.hf_fusion.weight.any()

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
    del partial['model.garment_high_frequency_control.feature_out.weight']
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
    for key in ('garment_grad/hf/encoder', 'garment_grad/hf/feature_out',
                'garment_grad/refiner/state', 'garment_grad/refiner/velocity_condition',
                'garment_grad/refiner/query', 'garment_grad/refiner/hf_fusion',
                'garment_grad/embedder_detail'):
        assert key in metrics, key
        assert torch.isfinite(metrics[key])
    # Fusion opens first while preserving the warm-start prediction exactly. The raw HF
    # extractor receives flow gradient after this zero gate becomes non-zero.
    assert metrics['garment_grad/refiner/hf_fusion'] > 0
    assert metrics['garment_grad/refiner/state'] > 0
    assert metrics['garment_grad/refiner/velocity_condition'] > 0
    assert metrics['garment_grad/hf/encoder'] == 0
    assert metrics['garment_grad/hf/feature_out'] == 0


def test_hf_values_follow_the_shared_coherent_sampling_grid():
    """HF must copy the RGB route instead of falling back to diffuse global A@V."""
    net = model(refiner=True, garment_high_frequency_channels=8).eval()
    control = net.garment_high_frequency_control
    nn.init.normal_(control.encoder.weight, std=.1)
    hf = torch.randn(2, 8, 32, 24)
    valid = torch.ones(2, 48, dtype=torch.bool)
    valid[:, 24:] = False                                  # only some keys are garment
    query = torch.zeros(2, 4, 48, 8)                       # uniform attention
    key = torch.zeros(2, 4, 48, 8)
    identity = LatentVTONPatchForcingTrainer._identity_sampling_grid(
        (8,6), query.device
    ).reshape(1,1,48,2).expand(2,4,-1,-1)
    shifted = identity.roll(1, dims=2)
    with torch.no_grad():
        a = control(hf, query, key, valid, torch.ones(2, 1, 8, 6),
                    torch.ones(2, 1, 64, 48), identity)
        b = control(hf, query, key, valid, torch.ones(2, 1, 8, 6),
                    torch.ones(2, 1, 64, 48), shifted)
    assert a.abs().sum() > 0 and not torch.allclose(a, b)
    torch.testing.assert_close(a.sum((2, 3)), torch.zeros_like(a.sum((2, 3))), atol=1e-4, rtol=0)


@pytest.mark.parametrize('gains', [(1., 1.), (10., .25)])
def test_warm_start_refiner_and_hf_feature_gains_are_explicit_and_weight_only(gains):
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
    # A trained-looking velocity head and HF feature projection.
    for key, value in (('model.garment_refiner.output.weight', .000792),
                       ('model.garment_high_frequency_control.feature_out.weight', .031433)):
        state[key] = torch.full_like(state[key], value)
    module.load_state_dict(state, strict=True)
    torch.testing.assert_close(module.model.garment_refiner.output.weight,
                               torch.full_like(state['model.garment_refiner.output.weight'],
                                               .000792 * refiner_gain))
    torch.testing.assert_close(module.model.garment_high_frequency_control.feature_out.weight,
                               torch.full_like(
                                   state['model.garment_high_frequency_control.feature_out.weight'],
                                   .031433 * hf_gain))
    assert module.model.garment_refiner.output.bias is None
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
    for key in (prefix + 'encoder.weight',):
        del legacy[key]
    # Shapes and names of the previous pixel-unshuffle encoder.
    legacy[prefix + 'encoder.1.weight'] = torch.randn(32, 64, 1, 1)
    legacy[prefix + 'encoder.1.bias'] = torch.randn(32)
    legacy[prefix + 'encoder.5.weight'] = torch.randn(32, 32, 1, 1)
    # Its bias-carrying local block, which the current revision dropped.
    legacy[prefix + 'local.2.bias'] = torch.randn(32)
    legacy[prefix + 'local.4.bias'] = torch.randn(32)
    # Replace the current feature projection with the previous four-channel velocity
    # head. Its incompatible state must reset the whole HF feature branch.
    for key in [name for name in legacy if name.startswith(prefix + 'feature_out.')]:
        del legacy[key]
    legacy[prefix + 'output.weight'] = torch.randn(4, 32, 1, 1)
    legacy[prefix + 'output.bias'] = torch.randn(4)

    with pytest.warns(UserWarning):
        module.load_state_dict(legacy, strict=True)
    control = module.model.garment_high_frequency_control
    assert control.encoder.weight.any()
    assert control.feature_out.weight.any()

    data = inputs()
    module.model.eval()
    with torch.no_grad():
        residual = []
        handle = control.register_forward_hook(lambda m, a, v: residual.append(v))
        module.model(**data, dense_pose=torch.randn(2, 4, 8, 6),
                     garment_high_frequency=torch.randn(2, 8, 32, 24))
        handle.remove()
    assert residual[0].any()
    assert not module.model.garment_refiner.hf_fusion.weight.any()


def test_hf_features_fuse_into_the_single_refiner_velocity():
    net = model(refiner=True, garment_high_frequency_channels=8).train()
    nn.init.normal_(net.garment_refiner.output.weight, std=.05)
    data = inputs()
    data['edit_mask'] = torch.ones(2, 1, 8, 6)
    data['edit_mask'][0, :, :, 3:] = 0
    hf = torch.randn(2, 8, 32, 24)
    control = net.garment_high_frequency_control
    optimizer = torch.optim.Adam(
        [*control.parameters(), *net.garment_refiner.hf_fusion.parameters()], lr=.01
    )
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
        assert captured[0].any()
        with torch.no_grad():
            baseline, baseline_logvar = net(
                **data, garment_high_frequency=torch.zeros_like(hf), return_uncertainty=True
            )
        if step == 0:
            torch.testing.assert_close(velocity, baseline)
        else:
            assert not torch.allclose(velocity, baseline)
        torch.testing.assert_close(logvar, baseline_logvar, rtol=0, atol=0)
        assert captured[0].shape[1] == net.garment_refiner.width
        assert not captured[0][0, :, :, 3:].any()
        (velocity - torch.randn_like(velocity)).square().mean().backward()
        # The zero fusion projection learns first. Once it opens, gradients reach the
        # upstream HF feature extractor without perturbing the warm-start prediction.
        assert net.garment_refiner.hf_fusion.weight.grad.abs().sum() > 0
        assert (control.encoder.weight.grad.abs().sum() > 0) == bool(step)
        assert (control.feature_out.weight.grad.abs().sum() > 0) == bool(step)
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


def test_hf_features_change_only_the_single_fused_refiner_output():
    """HF must augment RGB features before, not velocity after, the refiner."""
    torch.manual_seed(9)
    net = model(refiner=True, garment_high_frequency_channels=8).eval()
    data = inputs()
    data['edit_mask'] = torch.ones(2, 1, 8, 6)
    hf = torch.randn(2, 8, 32, 24)
    with torch.no_grad():
        nn.init.normal_(net.final_layer.linear.weight, std=.05)
        nn.init.normal_(net.garment_refiner.output.weight, std=.05)
        nn.init.normal_(net.garment_refiner.velocity_condition.weight, std=.05)
        nn.init.normal_(net.garment_high_frequency_control.encoder.weight, std=.05)
        nn.init.normal_(net.garment_refiner.hf_fusion.weight, std=.01)

    condition_inputs, hf_outputs = [], []
    condition_handle = net.garment_refiner.velocity_condition.register_forward_pre_hook(
        lambda module, args: condition_inputs.append(args[0].detach().clone())
    )
    hf_handle = net.garment_high_frequency_control.register_forward_hook(
        lambda module, args, value: hf_outputs.append(value.detach().clone())
    )
    output_without_hf, maps_without = net(
        **data, garment_high_frequency=torch.zeros_like(hf),
        return_garment_attention=True, return_refiner_supervision=True,
    )
    output_with_hf, maps_with = net(
        **data, garment_high_frequency=hf,
        return_garment_attention=True, return_refiner_supervision=True,
    )
    condition_handle.remove(); hf_handle.remove()

    assert len(condition_inputs) == len(hf_outputs) == 2
    assert not hf_outputs[0].any() and hf_outputs[1].abs().sum() > 0
    # The state condition stays backbone-only, while HF changes the one refiner output.
    torch.testing.assert_close(condition_inputs[1], condition_inputs[0])
    assert hf_outputs[1].shape[1] == net.garment_refiner.width
    fine_without = [entry for entry in maps_without if entry.get('scale') == 'refiner'][0]
    fine_with = [entry for entry in maps_with if entry.get('scale') == 'refiner'][0]
    assert not torch.allclose(fine_with['fine_velocity'], fine_without['fine_velocity'])
    torch.testing.assert_close(
        output_with_hf - output_without_hf,
        fine_with['fine_velocity'] - fine_without['fine_velocity'],
    )
    support = fine_with['fine_velocity_support']
    torch.testing.assert_close(
        fine_with['fine_velocity'].sum((2, 3)),
        torch.zeros_like(fine_with['fine_velocity'].sum((2, 3))),
        atol=1e-5, rtol=0,
    )


def test_fine_velocity_gate_removes_dc_and_hard_limits_authority():
    refiner = GarmentLatentRefiner(
        32, width=32, heads=4, max_velocity_ratio=.1, min_velocity_limit=.05,
    )
    nn.init.normal_(refiner.output.weight, mean=.1, std=.1)
    # Make the learned gate spatially/channel dependent rather than testing only its
    # identity initialization.
    nn.init.normal_(refiner.fine_gate.weight, std=.1)
    nn.init.normal_(refiner.fine_gate.bias, std=.1)
    features = torch.randn(2, 32, 8, 6)
    backbone = torch.randn(2, 4, 8, 6)
    clean = torch.randn_like(backbone)
    edit = torch.ones(2, 1, 8, 6)
    edit[0, :, :, 4:] = 0
    active = torch.ones(2, dtype=torch.bool)
    activity = torch.zeros(2, 1, 8, 6)
    activity[:, :, 2:6, 1:4] = .75
    velocity, raw, learned_gate, activity_gate, effective_gate, norm_gate, removed_dc, support = (
        refiner.refine_with_gate(features, backbone, clean, edit, active, activity)
    )
    torch.testing.assert_close(
        velocity.sum((2, 3)), torch.zeros_like(velocity.sum((2, 3))),
        atol=1e-5, rtol=0,
    )
    assert not velocity[0, :, :, 4:].any()
    denominator = support.sum((1, 2, 3)) * velocity.shape[1]
    fine_rms = (velocity.float().square().sum((1, 2, 3)) / denominator).sqrt()
    backbone_rms = (
        (backbone.float().square() * support).sum((1, 2, 3)) / denominator
    ).sqrt()
    limit = torch.maximum(backbone_rms * .1, backbone_rms.new_full((2,), .05))
    assert torch.all(fine_rms <= limit + 1e-5)
    assert raw.shape == velocity.shape and removed_dc.shape == (2, 4, 1, 1)
    assert 0 < learned_gate < 2 and 0 <= norm_gate <= 1
    assert 0 < activity_gate < 1 and 0 < effective_gate < 2


def test_hf_activity_gate_is_detached_spatial_and_keeps_configured_floor():
    refiner = GarmentLatentRefiner(
        32, width=32, heads=4, detail_activity_floor=.25,
    )
    hf = torch.zeros(1, 32, 8, 6, requires_grad=True)
    hf.data[:, :, 3:5, 2:4] = 2
    edit = torch.ones(1, 1, 8, 6)
    gate = refiner.detail_activity_gate(hf, edit, torch.ones(1, dtype=torch.bool))
    assert not gate.requires_grad
    assert gate.min() >= .25 and gate.max() <= 1
    assert gate[:, :, 3:5, 2:4].mean() > gate[:, :, :2, :2].mean()


def test_existing_hf_checkpoint_warm_starts_zero_cascade_adapter():
    net = model(refiner=True, garment_high_frequency_channels=8)
    module = LatentVTONPatchForcingTrainer(
        model=net, first_stage=torch.nn.Identity(), ema_rate=0,
        flow={'target': 'patch_flow.flow_vton.VTONPatchFlowForcing', 'params': {'patch_size': 2}},
        compute_validation_metrics=False, correspondence_center_weight=0,
        correspondence_nll_weight=0, correspondence_entropy_weight=0,
        correspondence_photometric_weight=0, allow_new_garment_refiner=True,
        allow_new_garment_high_frequency=True,
    )
    legacy = module.state_dict()
    del legacy['model.garment_refiner.velocity_condition.weight']
    del legacy['model.garment_refiner.velocity_condition.bias']
    nn.init.normal_(module.model.garment_refiner.velocity_condition.weight)
    nn.init.normal_(module.model.garment_refiner.velocity_condition.bias)
    with pytest.warns(UserWarning, match='new garment-refiner adapter'):
        module.load_state_dict(legacy, strict=True)
    assert not module.model.garment_refiner.velocity_condition.weight.any()
    assert not module.model.garment_refiner.velocity_condition.bias.any()


def test_cascade_checkpoint_migrates_dense_query_and_zero_warp_adapters():
    net = model(refiner=True, dense_pose_channels=4, garment_high_frequency_channels=8)
    module = LatentVTONPatchForcingTrainer(
        model=net, first_stage=torch.nn.Identity(), ema_rate=0,
        flow={'target': 'patch_flow.flow_vton.VTONPatchFlowForcing', 'params': {'patch_size': 2}},
        compute_validation_metrics=False, correspondence_center_weight=0,
        correspondence_nll_weight=0, correspondence_entropy_weight=0,
        correspondence_photometric_weight=0, allow_new_garment_refiner=True,
        allow_new_garment_high_frequency=True,
    )
    legacy = module.state_dict()
    state_key = 'model.garment_refiner.state.weight'
    old_state = torch.randn_like(legacy[state_key][:, :8])
    legacy[state_key] = old_state
    for prefix in (
        'model.garment_refiner.warp_mix.',
        'model.garment_refiner.latent_fusion.',
        'model.garment_high_frequency_control.warp_mix.',
    ):
        for key in [name for name in legacy if name.startswith(prefix)]:
            del legacy[key]
    legacy['model.garment_refiner.output.bias'] = torch.randn(4)
    with pytest.warns(UserWarning):
        module.load_state_dict(legacy, strict=True)
    migrated = module.model.garment_refiner.state.weight
    torch.testing.assert_close(migrated[:, :8], old_state)
    assert not migrated[:, 8:].any()
    assert not module.model.garment_refiner.warp_mix.weight.any()
    assert not module.model.garment_refiner.latent_fusion.weight.any()
    assert not module.model.garment_high_frequency_control.warp_mix.weight.any()
    assert module.model.garment_refiner.output.bias is None
    assert not module.model.garment_refiner.fine_gate.weight.any()
    assert not module.model.garment_refiner.fine_gate.bias.any()


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
    module.correspondence_propagation_steps = 4
    module.correspondence_propagated_weight = .25
    data = batch()
    data['person_garment_mask'].fill_(1)
    encoded = {'target':torch.zeros(3,4,4,4)}
    uv = torch.tensor([[[.1,.1],[.9,.9],[.2,.2],[.8,.8]]]).expand(3,-1,-1)
    teacher = torch.tensor([[1.,0.,1.,1.]]).expand(3,-1)
    target, anchor_weight, dense_weight = module._fine_targets(uv,teacher,data,encoded,None)
    target = target.reshape(3,4,4,2)
    # Only the accepted DINO cells are anchors. The rejected top-right cell receives a
    # lower-confidence propagated displacement, never its raw (.9,.9) DINO match.
    assert not anchor_weight.reshape(3,4,4)[0,:2,2:].any()
    assert dense_weight.reshape(3,4,4)[0,:2,2:].any()
    assert target[0,:2,2:].mean() < .8
    assert not anchor_weight[2].any() and not dense_weight[2].any()  # unpaired


def test_identity_coarse_displacement_upsamples_to_identity_fine_flow():
    """Upsampling displacement, not absolute UV, must preserve every fine phase."""
    module = trainer()
    data = batch()
    data = {name: value[:1] for name, value in data.items()}
    data['person_garment_mask'].fill_(1)
    encoded = {'target': torch.zeros(1, 4, 8, 8)}
    coarse, fine = 4, 8
    gy, gx = torch.meshgrid(torch.arange(coarse), torch.arange(coarse), indexing='ij')
    uv = torch.stack(((gx.flatten() + .5) / coarse, (gy.flatten() + .5) / coarse), -1)[None]
    teacher = torch.ones(1, coarse * coarse)
    target, _, _ = module._fine_targets(uv, teacher, data, encoded, None)
    centre_x = (target[0, :, 0] * fine - .5).round().long().reshape(fine, fine)
    centre_y = (target[0, :, 1] * fine - .5).round().long().reshape(fine, fine)
    expected = torch.arange(fine)
    torch.testing.assert_close(centre_y, expected[:, None].expand(fine, fine))
    torch.testing.assert_close(centre_x, expected[None, :].expand(fine, fine))
    # Never leaves the key grid, so the caller's clamp is never load-bearing here.
    assert int(centre_y.min()) == 0 and int(centre_y.max()) == fine - 1


def test_oracle_reachable_fraction_measures_whether_teacher_is_within_local_radius():
    """Guards the diagnostic that tells local_radius-too-small apart from bad scoring."""
    height, width, radius = 8, 6, 1
    cell_x, cell_y = 2.0 / width, 2.0 / height
    base_grid = torch.zeros(1, 1, 4, 2)
    teacher = torch.zeros(1, 1, 4, 2)
    teacher[0, 0, 0, 0] = 0.0  # exactly at the anchor: reachable
    teacher[0, 0, 1, 0] = radius * cell_x  # exactly radius cells away: still reachable
    teacher[0, 0, 2, 0] = (radius + 1) * cell_x  # one cell past the window: not reachable
    teacher[0, 0, 3, 0] = 100 * cell_x  # far away, but masked out below
    mask = torch.tensor([[True, True, True, False]])
    fraction = oracle_reachable_fraction(teacher, base_grid, mask, radius, width, height)
    torch.testing.assert_close(fraction, torch.tensor(2 / 3))

    # A y-axis offset is checked independently of x, using the same cell scale.
    teacher_y = torch.zeros(1, 1, 1, 2)
    teacher_y[0, 0, 0, 1] = (radius + 1) * cell_y
    unreachable = oracle_reachable_fraction(
        teacher_y, torch.zeros(1, 1, 1, 2), torch.tensor([[True]]), radius, width, height
    )
    torch.testing.assert_close(unreachable, torch.tensor(0.0))

    # No masked query at all: a safe zero, never NaN from an empty mean.
    empty_mask = torch.zeros(1, 4, dtype=torch.bool)
    zero = oracle_reachable_fraction(teacher, base_grid, empty_mask, radius, width, height)
    torch.testing.assert_close(zero, torch.tensor(0.0))


def test_route_reports_oracle_within_radius_fraction_only_when_teacher_forced():
    refiner = GarmentLatentRefiner(32, width=32, heads=4)
    args = (
        torch.randn(1, 12, 32), torch.randn(1, 4, 8, 6), torch.randn(1, 4, 8, 6),
        torch.randn(1, 48, 32), torch.randn(1, 48, 32), torch.randn(1, 48, 32),
        torch.ones(1, 1, 64, 48),
    )
    _, entry, _ = refiner.route(*args)
    assert entry["oracle_within_radius_fraction"].item() == 0.0

    target = torch.zeros(1, 48, 2)
    mask = torch.ones(1, 48, dtype=torch.bool)
    _, entry, _ = refiner.route(*args, sampling_grid_target=target, sampling_grid_mask=mask)
    fraction = entry["oracle_within_radius_fraction"]
    assert fraction.shape == ()
    assert 0.0 <= fraction.item() <= 1.0


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
    expected = sample_attention_heads(
        v, entry['sampling_grid'], entry['key_valid'], 8, 6
    ).transpose(1,2).reshape(1,48,32)
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
    # SDPA is now only the optional global-context value path. Hierarchical hard-coarse
    # plus local-residual routing is computed explicitly and transports content-only V.
    assert len(calls) == 2
    assert not torch.allclose(calls[0][0],calls[1][0])
    assert not torch.allclose(calls[0][1],calls[1][1])
    torch.testing.assert_close(calls[0][2],calls[1][2],rtol=0,atol=0)


def test_per_head_sampling_grid_preserves_exact_spatial_values():
    # Strong diagonal attention should recover each cell, not average its neighbours.
    query = torch.eye(4).reshape(1, 1, 4, 4) * 20
    key = query.clone()
    valid = torch.ones(1, 4, dtype=torch.bool)
    values = torch.tensor([[[[1.], [2.], [3.], [4.]]]])
    grid = attention_sampling_grid(query, key, valid, 2, 2)
    sampled = sample_attention_heads(values, grid, valid, 2, 2)
    torch.testing.assert_close(sampled, values, rtol=0, atol=1e-5)


def test_hard_coarse_route_and_local_residual_cannot_average_distant_modes():
    query = torch.eye(4).reshape(1,1,4,4).requires_grad_(True)
    key = query.detach().clone().requires_grad_(True)
    valid = torch.ones(1,4,dtype=torch.bool)
    coarse = hard_attention_sampling_grid(query,key,valid,2,2)
    identity = LatentVTONPatchForcingTrainer._identity_sampling_grid(
        (2,2), query.device
    ).reshape(1,1,4,2)
    torch.testing.assert_close(coarse, identity, rtol=0, atol=1e-6)
    coarse.sum().backward()
    assert query.grad is not None and key.grad is not None

    # A fine query prefers the far-right key globally, but a radius-one residual around
    # the identity base cannot jump there or average it with an unrelated garment part.
    q = torch.tensor([[[[1.,0.]] * 5]])
    k = torch.tensor([[[[-1.,0.],[-1.,0.],[-1.,0.],[-1.,0.],[1.,0.]]]])
    base = LatentVTONPatchForcingTrainer._identity_sampling_grid(
        (1,5), q.device
    ).reshape(1,1,5,2)
    local = local_attention_sampling_grid(q,k,torch.ones(1,5,dtype=torch.bool),base,1,5,1)
    assert local[0,0,0,0] <= -.4 + 1e-6  # no farther than key 1

    upsampled = upsample_displacement_grid(identity, (2,2), (4,4))
    expected = LatentVTONPatchForcingTrainer._identity_sampling_grid(
        (4,4), q.device
    ).reshape(1,1,16,2)
    torch.testing.assert_close(upsampled, expected, rtol=0, atol=1e-6)


def test_fine_velocity_penalty_has_finite_zero_gradient_for_all_cfg_drop():
    mean_square = torch.zeros((), requires_grad=True)
    penalty = stable_excess_rms_penalty(mean_square, mean_square.new_tensor(.1))
    assert penalty == 0
    penalty.backward()
    assert mean_square.grad == 0
    assert torch.isfinite(mean_square.grad)

    zero_limit = torch.zeros((), requires_grad=True)
    stable_excess_rms_penalty(zero_limit, zero_limit.detach()).backward()
    assert zero_limit.grad == 0 and torch.isfinite(zero_limit.grad)

    # Above the limit this is mathematically the original RMS penalty.
    mean_square = torch.tensor(.09, requires_grad=True)
    penalty = stable_excess_rms_penalty(mean_square, mean_square.new_tensor(.1))
    torch.testing.assert_close(penalty, (mean_square.sqrt() - .1).square())
    penalty.backward()
    assert torch.isfinite(mean_square.grad)


def test_consensus_sampling_grid_is_shared_across_value_heads():
    query = torch.randn(1, 4, 4, 8)
    key = torch.randn_like(query)
    valid = torch.ones(1, 4, dtype=torch.bool)
    coarse = hard_attention_sampling_grid(
        query, key, valid, 2, 2, shared_heads=True
    )
    for head in range(1, 4):
        torch.testing.assert_close(coarse[:, head], coarse[:, 0])
    local = local_attention_sampling_grid(
        query, key, valid, coarse, 2, 2, 1, shared_heads=True
    )
    for head in range(1, 4):
        torch.testing.assert_close(local[:, head], local[:, 0])


def test_fine_dense_pose_and_warp_adapters_are_zero_initialized():
    refiner = GarmentLatentRefiner(32, width=32, heads=4, dense_pose_channels=4)
    assert refiner.state.weight.shape[1] == 12
    assert not refiner.state.weight.any()
    assert not refiner.warp_mix.weight.any()
    args = (
        torch.randn(1, 12, 32), torch.randn(1, 4, 8, 6),
        torch.randn(1, 4, 8, 6), torch.randn(1, 48, 32),
        torch.randn(1, 48, 32), torch.ones(1, 1, 8, 6),
        torch.ones(1, 1, 64, 48),
    )
    with pytest.raises(ValueError, match='requires DensePose'):
        refiner(*args)
    _, entry = refiner(*args, dense_pose=torch.randn(1, 4, 8, 6), return_supervision=True)
    assert entry['sampling_grid'].shape == (1, 4, 48, 2)


def test_fine_warp_helpers_identity_and_affine_bending():
    module = trainer()
    identity = module._identity_sampling_grid((2, 3), torch.device('cpu'))
    sampling = identity.reshape(1, 1, 6, 2).requires_grad_(True)
    source = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    sampled = module._sample_fine_tokens(sampling, source, (2, 3))
    torch.testing.assert_close(sampled[:, 0], source, rtol=0, atol=1e-6)
    sampled.sum().backward()
    assert sampling.grad is not None and torch.isfinite(sampling.grad).all()
    weight = torch.ones(1, 6)
    assert module._fine_warp_bending_loss(sampling.detach(), weight, (2, 3)) == 0


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


def test_rgb_hf_experiment_uses_two_baseline_subtracted_vae_streams():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail-rgb-hf'])
    assert cfg.model.params.garment_high_frequency_channels == 256
    assert cfg.trainer.params.hf_detail_loss_weight > 0
    assert cfg.data.params.train.params.garment_high_frequency_mode == 'rgb_dog_gradient'
    assert cfg.data.params.validation.params.garment_high_frequency_mode == 'rgb_dog_gradient'
    assert cfg.train_params.val_check_interval == 250
    assert cfg.name.endswith('detail-rgb-hf')


def test_semantic_hf_experiment_enables_pretrained_encoder_and_source_transport():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail-semantic-hf'])
    assert cfg.trainer.params.learnable_hf_condition_encoder
    assert 0 < cfg.trainer.params.hf_condition_encoder_lr_multiplier < 1
    assert cfg.trainer.params.hf_source_consistency_weight > 0
    assert cfg.trainer.params.hf_source_consistency_scale == 2
    assert cfg.name.endswith('detail-semantic-hf')


def test_logo_hf_experiment_uses_sparse_decoded_supervision_and_dense_teacher():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail-logo-hf'])
    assert not cfg.model.params.garment_high_frequency_global_attention
    assert not cfg.model.params.garment_refiner_global_attention
    assert cfg.model.params.garment_refiner_shared_sampling_grid
    assert cfg.model.params.garment_refiner_velocity_max_backbone_ratio == .3
    assert cfg.model.params.garment_refiner_velocity_min_limit == .05
    # Not 0.25 any more: this gate multiplies the residual *including* the colour
    # correction the branch now owns, so a hard floor there quartered every colour fix
    # in the flat interiors that need it most. It must still separate stroke
    # neighbourhoods from flat cloth, and must not be fully open.
    assert .5 <= cfg.model.params.garment_refiner_detail_activity_floor < 1
    # The backbone reaches the garment only through softmax cross-attention, which
    # returns a blend of garment tokens. This branch warps instead of averaging and is
    # the only one that can correct that blend's colour, so it must keep its DC.
    assert cfg.model.params.garment_refiner_remove_velocity_dc is False
    assert cfg.model.params.garment_refiner_highpass_kernel == 0
    assert cfg.model.params.garment_value_preserve_magnitude
    assert cfg.model.params.garment_value_minimum_mix == 1.0
    assert cfg.trainer.params.ema_rate == 0
    assert cfg.trainer.params.hf_detail_loss_weight == 0
    assert cfg.trainer.params.hf_source_sparse_weight > 0
    assert cfg.trainer.params.hf_decoded_rgb_weight > 0
    assert cfg.trainer.params.hf_decoded_contrast_weight > 0
    assert cfg.trainer.params.hf_decoded_chroma_weight > 0
    assert cfg.trainer.params.hf_decoded_edge_weight > 0
    assert list(cfg.trainer.params.correspondence_teacher_input_size) == [768, 576]
    assert list(cfg.trainer.params.correspondence_garment_grid) == [48, 36]
    assert cfg.data.params.batch_size == 4
    assert cfg.train_params.accumulate_grad_batches == 8
    assert cfg.data.params.batch_size * cfg.train_params.accumulate_grad_batches == 32
    assert cfg.trainer.params.hf_decoded_rgb_weight > cfg.trainer.params.hf_decoded_edge_weight
    assert cfg.trainer.params.hf_decoded_chroma_weight > cfg.trainer.params.hf_decoded_edge_weight
    assert cfg.model.params.garment_refiner_cosine_scale == 10
    assert cfg.trainer.params.garment_refiner_lr_multiplier == 1.0
    assert cfg.trainer.params.garment_high_frequency_lr_multiplier == 1.0
    assert cfg.trainer.params.garment_value_mix_lr_multiplier == 1
    assert cfg.trainer.params.garment_latent_fusion_lr_multiplier == 1
    assert cfg.trainer.params.garment_support_lr_multiplier == 1
    assert cfg.trainer.params.hf_condition_encoder_lr_multiplier == .25
    assert cfg.trainer.params.adapter_lr_multiplier == .25
    assert cfg.trainer.params.decoded_garment_rgb_weight > 0
    assert cfg.trainer.params.decoded_garment_low_frequency_weight > 0
    assert cfg.trainer.params.decoded_garment_mean_weight > 0
    assert cfg.trainer.params.decoded_min_time == 0
    assert cfg.trainer.params.decoded_clean_time_floor == .2
    assert cfg.trainer.params.decoded_max_samples == 2
    assert cfg.trainer.params.decoded_edge_weight == .5
    assert cfg.trainer.params.hf_decoded_edge_weight == .5
    assert cfg.trainer.params.fine_support_weight == .25
    assert cfg.trainer.params.fine_teacher_forcing_start == .75
    assert cfg.trainer.params.correspondence_entropy_weight == .05
    assert cfg.trainer.params.fine_teacher_forcing_steps == 8000
    assert cfg.trainer.params.fine_velocity_regularization_weight == .25
    assert cfg.trainer.params.fine_velocity_max_backbone_ratio == .3
    assert cfg.trainer.params.fine_velocity_min_limit == .05
    assert list(cfg.trainer.params.correspondence_scales) == ['coarse', 'detail']
    assert cfg.trainer.params.correspondence_warmup_steps == 250
    assert cfg.trainer.params.sample_kwargs.cfg_scale == 1.0
    assert cfg.lr_scheduler.params.num_warmup_steps == 500
    assert cfg.train_params.val_check_interval == 250
    assert cfg.checkpoint_params.every_n_train_steps == 500
    assert cfg.name.endswith('detail-logo-hf')


def test_full_resolution_semantic_hf_experiment_preserves_effective_batch():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-1024x768-detail-semantic-hf'])
    assert list(cfg.data.params.train.params.image_size) == [1024, 768]
    assert list(cfg.data.params.validation.params.image_size) == [1024, 768]
    assert cfg.data.params.batch_size == 1
    assert cfg.train_params.accumulate_grad_batches == 32
    assert cfg.data.params.batch_size * cfg.train_params.accumulate_grad_batches == 32
    assert cfg.model.params.garment_refiner_attention_chunk_size == 128
    assert cfg.trainer.params.hf_source_consistency_scale == 1
    assert list(cfg.trainer.params.decoded_supervision_image_size) == [512, 384]
    assert cfg.data.params.validation.params.test_samples == 4
    assert cfg.data.params.validation.params.train_samples == 2
    assert cfg.name.endswith('1024x768-detail-semantic-hf')


def test_cascade_experiment_does_not_repeat_previous_head_rebalance():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail-cascade'])
    assert cfg.model.params.garment_high_frequency_channels == 128
    assert cfg.model.params.garment_latent_refiner
    assert cfg.trainer.params.allow_new_garment_refiner
    assert cfg.trainer.params.allow_new_garment_high_frequency
    assert cfg.trainer.params.warm_start_refiner_output_gain == 1.0
    assert cfg.trainer.params.warm_start_high_frequency_output_gain == 1.0
    assert cfg.name.endswith('detail-cascade')


def test_warp_experiment_enables_coherent_transport_losses():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg = compose(config_name='config',overrides=['experiment=viton-pft-xl-512x384-detail-warp'])
    assert cfg.model.params.dense_pose_channels == 4
    assert cfg.trainer.params.fine_correspondence_weight == .05
    assert cfg.trainer.params.fine_warp_coordinate_weight > 0
    assert cfg.trainer.params.fine_warp_smoothness_weight > 0
    assert cfg.trainer.params.fine_warp_mask_weight > 0
    assert cfg.name.endswith('detail-warp')


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
    module.decoded_garment_rgb_weight = .3
    module.decoded_garment_low_frequency_weight = .4
    module.decoded_garment_mean_weight = .6
    data = batch()
    data['person_garment_mask'][:,:,:8] = 1
    target = torch.zeros(3,3,16,16)
    latent = torch.randn(3,4,4,4,requires_grad=True)
    times = torch.tensor([[.5,.99,.5,.5],[.5,.5,.5,.5],[.5,.5,.5,.5]])
    loss, metrics = module._decoded_garment_loss(latent,data,{'target_image':target},times,torch.tensor([1.,0.,1.]))
    assert metrics['decoded_samples'] == 1 and loss > 0
    assert metrics['decoded_garment_rgb_loss'] > 0
    assert metrics['decoded_garment_low_frequency_loss'] > 0
    assert metrics['decoded_garment_mean_loss'] > 0
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


def _refiner(remove_dc=True, position_gain=False):
    return GarmentLatentRefiner(
        32, channels=4, width=32, heads=4, qk_norm=True,
        max_velocity_ratio=.3, min_velocity_limit=.05,
        remove_velocity_dc=remove_dc, learnable_position_gain=position_gain,
    )


def _residual_args(refiner, constant=.4):
    features = torch.zeros(1, 32, 8, 8)
    velocity = torch.full((1, 4, 8, 8), 1.)
    edit = torch.ones(1, 1, 8, 8)
    active = torch.ones(1, dtype=torch.bool)
    # Drive a purely constant residual so DC is the entire signal.
    nn.init.zeros_(refiner.output.weight)
    refiner.output.weight.data[:, 0] = constant
    features = features.clone()
    features[:, 0] = 1.
    return features, velocity, velocity.clone(), edit, active


def test_refiner_keeps_its_colour_correction_when_dc_removal_is_disabled():
    """The mean the branch writes is a garment colour correction, not an artefact.

    The backbone reaches the garment through softmax cross-attention, which returns a
    convex combination of garment tokens; this branch warps instead, so it is the only
    one that can correct the resulting blend. Deleting its DC deleted that correction --
    measured as fine_velocity_dc_fraction 0.50-0.79, i.e. up to 56% of everything it
    produced.
    """
    removing, keeping = _refiner(remove_dc=True), _refiner(remove_dc=False)
    keeping.load_state_dict(removing.state_dict())
    out_removed = removing.refine_with_gate(*_residual_args(removing))[0]
    out_kept = keeping.refine_with_gate(*_residual_args(keeping))[0]
    # The per-channel spatial mean IS the colour correction. Removal zeroes it exactly;
    # retention passes it through at a magnitude comparable to the residual itself.
    dc_removed = out_removed.mean((2, 3)).abs().max()
    dc_kept = out_kept.mean((2, 3)).abs().max()
    assert dc_removed < 1e-6
    assert dc_kept > .5 * out_kept.abs().mean()
    # Both still carry structure, so this is not simply a larger output.
    assert out_removed.abs().mean() > 0
    # Retention is not a licence to exceed the hard authority cap (0.3 x backbone RMS,
    # and the backbone velocity here is exactly 1.0).
    assert out_kept.square().mean().sqrt() <= .3 + 1e-4


def test_disabled_dc_removal_still_confines_the_residual_to_the_edit_support():
    refiner = _refiner(remove_dc=False)
    features, velocity, clean, edit, active = _residual_args(refiner)
    edit = edit.clone()
    edit[:, :, :, 4:] = 0.
    out = refiner.refine_with_gate(features, velocity, clean, edit, active)[0]
    assert out[:, :, :, 4:].abs().max() == 0
    assert out[:, :, :, :4].abs().mean() > 0


def test_position_gain_is_opt_in_and_identity_at_one():
    """Off by default: adding a parameter costs the optimizer state on the next restart."""
    plain, gained = _refiner(position_gain=False), _refiner(position_gain=True)
    assert not hasattr(plain, 'position_gain')
    assert gained.position_gain.item() == 1.
    assert 'position_gain' not in dict(plain.named_parameters())
    gained.load_state_dict(plain.state_dict(), strict=False)
    torch.manual_seed(0)
    args = (torch.randn(1, 16, 32), torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8),
            torch.randn(1, 64, 32), torch.randn(1, 64, 32), torch.randn(1, 64, 32), None)
    plain.eval(); gained.eval()
    with torch.no_grad():
        reference = plain.route(*args)[1]['sampling_grid']
        output = gained.route(*args)[1]['sampling_grid']
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_position_gain_shrinks_the_positional_share_of_the_key():
    """Hard-normalised, position is permanently half the key energy: an identity prior."""
    refiner = _refiner(position_gain=True)
    torch.manual_seed(0)
    args = (torch.randn(1, 16, 32), torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8),
            torch.randn(1, 64, 32), torch.randn(1, 64, 32), torch.randn(1, 64, 32), None)
    refiner.eval()
    with torch.no_grad():
        full = refiner.route(*args)[1]['key'].clone()
        refiner.position_gain.fill_(0.)
        none = refiner.route(*args)[1]['key'].clone()
    # With the gain at zero the key is content only, so it must differ materially.
    assert (full - none).norm() / full.norm() > .1
    with torch.no_grad():
        refiner.position_gain.fill_(3.)          # clamped to 2 in the forward pass
        clamped = refiner.route(*args)[1]['key'].clone()
        refiner.position_gain.fill_(2.)
        at_limit = refiner.route(*args)[1]['key'].clone()
    torch.testing.assert_close(clamped, at_limit, rtol=0, atol=0)


def _local_inputs(radius=2, height=8, width=8, heads=1, dim=16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(1, heads, height * width, dim)
    k = torch.randn(1, heads, height * width, dim)
    valid = torch.ones(1, height * width, dtype=torch.bool)
    base = upsample_displacement_grid(
        torch.zeros(1, heads, (height // 2) * (width // 2), 2),
        (height // 2, width // 2), (height, width),
    )
    return q, k, valid, base, height, width, radius


def test_local_stage_temperature_controls_whether_it_refines_or_blurs():
    """Its forward pass IS the softmax, unlike the argmax coarse stage above it.

    A flat distribution over the (2r+1)^2 candidates makes sum_c p_c * candidate_c
    return the centroid of the window -- the coarse anchor it was supposed to refine.
    Measured on the trained model at the shared scale of 10: 20.95 of 25 candidates.
    """
    q, k, valid, base, h, w, r = _local_inputs()
    grids, effs = {}, {}
    for ratio in (0.05, 1.0, 20.0):
        grid, eff = local_attention_sampling_grid(
            q, k, valid, base, h, w, r, temperature_ratio=ratio, return_sharpness=True)
        grids[ratio], effs[ratio] = grid, eff.item()
    count = (2 * r + 1) ** 2
    # Flat -> the whole window is averaged and the coordinate barely leaves the anchor.
    assert effs[0.05] > 0.95 * count
    assert (grids[0.05] - base).abs().max() < (grids[20.0] - base).abs().max()
    # Sharp -> a handful of adjacent candidates, i.e. sub-cell interpolation.
    assert effs[20.0] < 0.25 * count
    assert effs[0.05] > effs[1.0] > effs[20.0]


def test_local_temperature_defaults_to_the_shared_scale_and_stays_bounded():
    shared = GarmentLatentRefiner(32, channels=4, width=32, heads=4,
                                  qk_norm=True, cosine_scale=10.)
    assert shared.local_cosine_scale == shared.cosine_scale == 10.
    split = GarmentLatentRefiner(32, channels=4, width=32, heads=4,
                                 qk_norm=True, cosine_scale=10., local_cosine_scale=50.)
    assert (split.cosine_scale, split.local_cosine_scale) == (10., 50.)
    # The coarse scale keeps its own tighter bound; the local one may exceed it because
    # it is a coordinate regressor, not a straight-through gradient path.
    with pytest.raises(ValueError, match='cosine scale'):
        GarmentLatentRefiner(32, channels=4, width=32, heads=4, cosine_scale=50.)
    with pytest.raises(ValueError, match='local cosine scale'):
        GarmentLatentRefiner(32, channels=4, width=32, heads=4, local_cosine_scale=500.)
    with pytest.raises(ValueError, match='local cosine scale'):
        GarmentLatentRefiner(32, channels=4, width=32, heads=4, local_cosine_scale=0.)


def test_a_sharper_local_stage_does_not_change_the_coarse_anchor():
    """The two stages are decoupled: only the sub-cell regressor sees the new scale."""
    base_kwargs = dict(backbone_dim=32, channels=4, width=32, heads=4, qk_norm=True,
                       cosine_scale=10., shared_sampling_grid=True)
    flat = GarmentLatentRefiner(**base_kwargs)
    sharp = GarmentLatentRefiner(**base_kwargs, local_cosine_scale=50.)
    sharp.load_state_dict(flat.state_dict())
    torch.manual_seed(0)
    args = (torch.randn(1, 16, 32), torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8),
            torch.randn(1, 64, 32), torch.randn(1, 64, 32), torch.randn(1, 64, 32), None)
    flat.eval(); sharp.eval()
    with torch.no_grad():
        a, b = flat.route(*args)[1], sharp.route(*args)[1]
    torch.testing.assert_close(a['coarse_sampling_grid'], b['coarse_sampling_grid'],
                               rtol=0, atol=0)
    assert not torch.equal(a['sampling_grid'], b['sampling_grid'])
    assert b['local_effective_candidates'] < a['local_effective_candidates']
