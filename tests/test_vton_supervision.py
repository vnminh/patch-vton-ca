from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import json
import pytest

import numpy as np
import torch
from PIL import Image

from patch_flow.correspondence import correspondence_targets, CorrespondenceAttentionLoss
from patch_flow.models.pf_transformer_vton import VTONPatchForcingDiT
from patch_flow.trainer_vton import LatentVTONPatchForcingTrainer
from patch_flow.vton_data import VTONHDDataset, VTONValidationDataset


def trainer():
    model = VTONPatchForcingDiT(input_size=4, in_channels=4, hidden_size=32,
                               depth=2, num_heads=4, num_classes=10, compile=False,
                               cross_attention_every=1)
    return LatentVTONPatchForcingTrainer(
        model=model, first_stage=torch.nn.Identity(),
        flow={"target": "patch_flow.flow_vton.VTONPatchFlowForcing", "params": {"patch_size": 2}},
        ema_rate=0, compute_validation_metrics=False,
        correspondence_center_weight=0, correspondence_nll_weight=0,
        correspondence_entropy_weight=0, correspondence_photometric_weight=0,
        garment_supervision_only=True, garment_token_min_coverage=0.5,
        detail_pure_noise_only=False, compute_garment_validation_metrics=True,
    )


def batch():
    return {"agnostic_mask": torch.ones(3, 1, 16, 16),
            "person_garment_mask": torch.zeros(3, 1, 16, 16),
            "garment_mask": torch.ones(3, 1, 16, 16),
            "has_ground_truth": torch.tensor([True, True, False])}


def test_cycle_tolerance_accepts_near_returns_but_rejects_far_ones():
    person = torch.tensor([1., 0.]).view(1, 2, 1, 1).expand(1, 2, 1, 3)
    garment = person[:, :, :, :1]
    _, weight, _ = correspondence_targets(person, garment, (1, 1), cycle_tolerance=1.0)
    torch.testing.assert_close(weight, torch.tensor([[1., 1., 0.]]))
    _, weight, _ = correspondence_targets(person, garment, (1, 1), cycle_tolerance=0.0)
    torch.testing.assert_close(weight, torch.tensor([[1., 0., 0.]]))
    valid = torch.tensor([[False, True, True]])
    _, weight, _ = correspondence_targets(person, garment, (1, 1), person_valid=valid, cycle_tolerance=1.)
    torch.testing.assert_close(weight, torch.tensor([[0., 1., 1.]]))


def test_empty_dropped_unpaired_and_skin_queries_are_not_supervised():
    module = trainer()
    data = batch()
    data['person_garment_mask'][0, :, :8, :8] = 1
    data['person_garment_mask'][2] = 1
    encoded = {'target': torch.zeros(3, 4, 4, 4)}
    edits = torch.ones(3, 4, dtype=torch.bool)
    weight = module._supervision_weights(data, encoded, edits, None)
    torch.testing.assert_close(weight, torch.tensor([[1.,0,0,0], [0.,0,0,0], [0.,0,0,0]]))
    assert not module._supervision_weights(data, encoded, edits, torch.zeros(3)).any()
    data['garment_mask'].zero_()
    assert not module._supervision_weights(data, encoded, edits, None).any()


def test_rgb_targets_exclude_non_garment_pixels_at_boundaries():
    module = trainer()
    data = batch()
    data['person_garment_mask'][0, :, :8, :4] = 1
    image = torch.full((3, 3, 16, 16), -1.)
    image[0, :, :8, :4] = 0.6
    data['garment'] = image
    encoded = {'target': torch.zeros(3, 4, 4, 4), 'target_image': image}
    appearance, weight = module._appearance_targets(data, encoded, torch.ones(3,4,dtype=torch.bool), None)
    torch.testing.assert_close(appearance['query'][0,0], torch.full((3,), .6))
    assert weight[0,0] == .5
    assert weight.count_nonzero() == 1


def test_reference_rgb_pooling_excludes_white_background():
    loss = CorrespondenceAttentionLoss(center_weight=0, nll_weight=0, entropy_weight=0, photometric_weight=1)
    garment = torch.ones(1,3,2,2)
    garment[:,:,:,0] = -.5
    mask = torch.zeros(1,1,2,2)
    mask[:,:,:,0] = 1
    appearance = {'garment': garment, 'garment_mask': mask, 'query': torch.full((1,1,3), -.5)}
    value, _ = loss([{'scale':'coarse','grid':(1,1),'block':1,'weights':torch.ones(1,1,1)}],
                    appearance=appearance, appearance_weight=torch.ones(1,1))
    assert value == 0


def test_detail_mask_includes_refinement_and_excludes_skin_and_dropped_pairs():
    module = trainer()
    data = batch()
    data['person_garment_mask'][:, :, :8] = 1
    encoded = {'target': torch.zeros(3,4,4,4)}
    masks = module.flow.prepare_masks(data['agnostic_mask'], (4,4), torch.float32)
    times = torch.tensor([[.5,.99,.5,.5], [0.,0.,0.,0.], [.5,.5,.5,.5]])
    mask = module._detail_supervision_mask(data, encoded, times, masks, None)
    assert mask[0,:,:2,:2].all()
    assert not mask[0,:,:,2:].any()
    assert not mask[0,:,2:,:].any()
    assert mask[1,:,:2,:].all()
    assert not mask[2].any()
    assert not module._detail_supervision_mask(data, encoded, times, masks, torch.zeros(3)).any()


def test_validation_metrics_ignore_swaps_empty_masks_and_non_garment_errors():
    module = trainer()
    data = batch()
    data['person_garment_mask'][0,:,:8] = 1
    data['person_garment_mask'][2] = 1
    data['validation_group'] = ['test_paired','test_paired','test_unpaired']
    target = torch.zeros(3,3,16,16)
    generated = torch.ones_like(target)
    generated[0,:,:8] = 0
    module._record_garment_validation(data, generated, target)
    torch.testing.assert_close(module._garment_validation_totals['test_paired'], torch.tensor([0.,0.,1.]))
    assert 'test_unpaired' not in module._garment_validation_totals


def make_dataset(root):
    for split in ('train','test'):
        for folder in ('image','cloth','agnostic-mask','cloth-mask','image-parse-v3','image-densepose'):
            (root / split / folder).mkdir(parents=True)
        (root / f'{split}_pairs.txt').write_text('a.jpg a.jpg\nb.jpg b.jpg\n')
        for name in ('a','b'):
            for folder in ('image','cloth'):
                Image.new('RGB',(32,32),(64,64,64)).save(root/split/folder/f'{name}.jpg')
            Image.new('RGB',(32,32),(32 if name == 'a' else 224,64,96)).save(
                root/split/'image-densepose'/f'{name}.jpg'
            )
            for folder in ('agnostic-mask','cloth-mask'):
                Image.new('L',(32,32),255).save(root/split/folder/f'{name}.png')
            labels = np.zeros((32,32),dtype=np.uint8)
            labels[8:24,4:12] = 5
            parse = Image.fromarray(labels).convert('P')
            parse.putpalette([0] * 768)  # L conversion would erase clothing completely.
            parse.save(root/split/'image-parse-v3'/f'{name}.png')


def test_palette_labels_follow_person_flip_and_shift_independently_of_garment():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        make_dataset(root)
        dataset = VTONHDDataset(root, image_size=32, random_flip=True, garment_parse_labels=[5])
        with patch('patch_flow.vton_data.random.random', return_value=0.), patch.object(
            dataset, '_sample_shift_scale', side_effect=[([2,0],1.), ([-2,0],1.)]
        ):
            sample = dataset[0]
        expected = torch.zeros(1,32,32)
        expected[:,8:24,22:30] = 1
        torch.testing.assert_close(sample['person_garment_mask'], expected)


def test_fixed_validation_has_paired_train_test_and_swaps_with_shared_noise_seed():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        make_dataset(root)
        dataset = VTONValidationDataset(root, image_size=32, test_samples=2, train_samples=1,
                                        preview_sample_id='b.jpg')
        assert len(dataset) == 5
        paired, swapped = dataset[0], dataset[1]
        assert paired['person_name'] == 'b.jpg'
        assert paired['has_ground_truth'] and not swapped['has_ground_truth']
        assert paired['validation_seed'] == swapped['validation_seed']
        assert paired['person_name'] == swapped['person_name']
        assert dataset[4]['validation_group'] == 'train_paired'


def test_dense_pose_follows_person_and_is_shared_across_garment_swap():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        make_dataset(root)
        dataset = VTONValidationDataset(root, image_size=32, dense_pose_dir='image-densepose',
                                        test_samples=2, train_samples=0)
        paired, swapped = dataset[0], dataset[1]
        torch.testing.assert_close(paired['dense_pose'], swapped['dense_pose'])
        assert paired['dense_pose'].shape == (3,32,32)
        assert paired['person_name'] == swapped['person_name']


def test_fix_experiment_composes_and_keeps_optimizer_shapes():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'), version_base=None):
        cfg = compose(config_name='config', overrides=['experiment=viton-pft-xl-512x384-garment-fix'])
    assert cfg.trainer.params.garment_supervision_only
    assert cfg.trainer.params.correspondence_cycle_tolerance == 1.5
    assert not cfg.trainer.params.detail_pure_noise_only
    assert cfg.data.params.validation.target.endswith('VTONValidationDataset')
    assert cfg.train_params.limit_val_batches == 20
    assert cfg.data.params.batch_size * cfg.train_params.accumulate_grad_batches == 32
    # These options do not create new trainable parameters or optimizer groups.
    original = trainer()
    restored = trainer()
    restored.load_state_dict(original.state_dict(), strict=True)
    old_opt = original.configure_optimizers()['optimizer']
    new_opt = restored.configure_optimizers()['optimizer']
    new_opt.load_state_dict(old_opt.state_dict())


def test_anchor_flow_experiment_uses_sparse_teacher_and_validates_every_50_steps():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'), version_base=None):
        cfg = compose(
            config_name='config',
            overrides=['experiment=viton-pft-xl-512x384-detail-anchor-flow'],
        )
    assert cfg.model.params.garment_refiner_local_radius == 2
    assert cfg.trainer.params.correspondence_min_margin > 0
    assert cfg.trainer.params.correspondence_local_consistency_tolerance == 3
    assert cfg.trainer.params.correspondence_propagation_steps == 16
    assert cfg.trainer.params.correspondence_propagated_weight < 1
    assert cfg.train_params.val_check_interval == 50


@pytest.mark.parametrize('fine_detail', [False, True])
def test_multiscale_bfloat16_backward_and_validation_with_vae_pyramid(fine_detail):
    from jutils.nn.kl_autoencoder import AutoencoderKL
    from test_vton import FakeDinoBackbone

    vae = AutoencoderKL(ddconfig={
        'attn_type':'vanilla', 'double_z':True, 'z_channels':4, 'resolution':64,
        'in_channels':3, 'out_ch':3, 'ch':32, 'ch_mult':[1,2,2,2],
        'num_res_blocks':1, 'attn_resolutions':[], 'dropout':0.,
    })
    model = VTONPatchForcingDiT(
        input_size=8, in_channels=4, hidden_size=32, depth=3, num_heads=4,
        num_classes=10, compile=False, cross_attention_every=1,
        garment_middle_channels=64, garment_detail_channels=32,
        garment_scale_routes=['coarse','middle','detail'], gradient_checkpointing=True,
        garment_match_query_grid=fine_detail, garment_latent_refiner=fine_detail,
        garment_refiner_width=32, garment_refiner_heads=4,
        garment_refiner_qk_norm=fine_detail,
        garment_high_frequency_channels=64 if fine_detail else 0,
    )
    with patch('transformers.AutoModel.from_pretrained', return_value=FakeDinoBackbone()):
        module = LatentVTONPatchForcingTrainer(
            model=model, first_stage=vae, ema_rate=0,
            flow={'target':'patch_flow.flow_vton.VTONPatchFlowForcing','params':{'patch_size':2}},
            compute_validation_metrics=False, compute_garment_validation_metrics=True,
            garment_supervision_only=True, garment_token_min_coverage=.8,
            correspondence_teacher_input_size=None, correspondence_cycle_tolerance=1.5,
            correspondence_min_similarity=0., correspondence_nll_radius=.4,
            correspondence_value_weight=.1, correspondence_entropy_weight=0.,
            detail_loss_weight=.5, detail_pure_noise_only=False, garment_dropout_prob=0.,
            hf_detail_loss_weight=.5 if fine_detail else 0.,
            decoded_rgb_weight=.2 if fine_detail else 0., decoded_edge_weight=.5 if fine_detail else 0.,
            fine_correspondence_weight=.2 if fine_detail else 0.,
            fine_value_weight=.25 if fine_detail else 0.,
            fine_rgb_weight=.1 if fine_detail else 0.,
            fine_warp_coordinate_weight=.1 if fine_detail else 0.,
            fine_warp_smoothness_weight=.02 if fine_detail else 0.,
            fine_warp_mask_weight=.05 if fine_detail else 0.,
            fine_correspondence_radius=1,
            sample_kwargs={'num_steps':2,'cfg_scale':1.,'adaptive':False,'progress':False},
        )
    module.flow.t_sampler = lambda shape, device, dtype: torch.full(shape, .5, device=device, dtype=dtype)
    person = torch.rand(2,3,64,48)*2-1
    edit = torch.ones(2,1,64,48)
    worn = edit.clone()
    worn[:,:,:16] = 0
    data = {'image':person, 'person':person, 'person_agnostic':person*(1-edit),
            'agnostic_mask':edit, 'person_garment_mask':worn, 'garment':person.clone(),
            'garment_mask':edit, 'has_ground_truth':torch.ones(2,dtype=torch.bool),
            'validation_seed':torch.tensor([0,0]), 'validation_group':['test_paired','test_unpaired']}
    if fine_detail:
        data['garment_high_frequency'] = torch.rand(2, 6, 64, 48)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss, metrics = module(data)
    assert torch.isfinite(loss)
    assert metrics['garment_supervision_fraction'] < 1
    assert metrics['detail_active_fraction'] == 1
    if fine_detail:
        assert metrics['fine_rgb_loss'] > 0
        assert metrics['decoded_rgb_loss'] > 0
        assert metrics['decoded_samples'] == 1
        assert metrics['fine_correspondence_loss'] > 0
        assert metrics['fine_value_loss'] > 0
        assert metrics['fine_warp_coordinate_loss'] >= 0
        assert metrics['fine_warp_smoothness_loss'] >= 0
        assert metrics['fine_warp_mask_loss'] >= 0
        assert metrics['fine_supervised_fraction'] > 0
        assert metrics['hf_detail_loss'] > 0
        assert metrics['hf_detail_active_fraction'] == 1
    loss.backward()
    if fine_detail:
        assert model.garment_refiner.output.weight.grad.abs().sum() > 0
        # Zero init is on the encoder, so it is the module that must show gradient
        # on the very first step; the head's weight follows once its input is non-zero.
        assert model.garment_high_frequency_control.encoder.weight.grad.abs().sum() > 0
        assert model.garment_refiner.query.weight.grad.abs().sum() > 0
        assert all(p.grad is None and not p.requires_grad for p in vae.parameters())
    for scale in ('coarse','middle','detail'):
        assert any(block.garment_cross_attention.in_proj_weight.grad.abs().sum() > 0
                   for block in model.blocks if block.garment_scale == scale)
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    assert all(p.grad is None for p in vae.parameters())
    # Exercise the real validation generation/decoding path and retained previews.
    module.eval()
    data['has_ground_truth'][1] = False
    with torch.autocast('cpu',dtype=torch.bfloat16):
        module.validation_step(data,0)
    assert len(module.val_images['tryon']) == 2
    assert len(module._validation_rows) == 2
    assert module._garment_validation_totals['test_paired'][2] == 1
    assert 'test_unpaired' not in module._garment_validation_totals
    with TemporaryDirectory() as folder, patch.object(
        type(module), 'logger', property(lambda self: SimpleNamespace(log_dir=folder))
    ), patch('patch_flow.trainer_vton.log_images'), patch.object(module, 'log') as log:
        module.on_validation_epoch_end()
        assert (Path(folder)/'previews/latest.png').is_file()
        metadata = json.loads((Path(folder)/'previews/step000000.json').read_text())
        assert len(metadata['rows']) == 2
        assert metadata['rows'][1]['has_ground_truth'] is False
        assert any(call.args[0] == 'val/test_paired/garment_rgb_mae' for call in log.call_args_list)
    assert module.val_images is None
    assert not module._garment_validation_totals
