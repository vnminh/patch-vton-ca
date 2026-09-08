import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from patch_flow.correspondence import (
    CorrespondenceAttentionLoss,
    DinoCorrespondenceTeacher,
    correspondence_targets,
    grid_coordinates,
    mask_to_token_valid,
    neighbourhood_mass,
    stable_attention_entropy,
)
from patch_flow.flow_vton import VTONPatchFlowForcing
from patch_flow.models.pf_transformer import PatchForcingDiT
from patch_flow.models.pf_transformer_vton import VTONPatchForcingDiT
from patch_flow.trainer_vton import LatentVTONPatchForcingTrainer
from patch_flow.vton_utils import compose_vton, prepare_vton_masks
from patch_flow.vton_data import VTONHDDataset
from train import repeat_dataloader


class FakeDinoOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeDinoBackbone(torch.nn.Module):
    """DINOv3 stand-in: patch tokens preceded by a CLS token and ``registers`` registers."""

    def __init__(self, hidden=8, patch_size=16, registers=4):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size)
        self.registers = registers
        self.config = SimpleNamespace(patch_size=patch_size, hidden_size=hidden, num_register_tokens=registers)

    def forward(self, pixel_values=None):
        tokens = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        prefix = tokens.new_zeros((tokens.shape[0], 1 + self.registers, tokens.shape[-1]))
        return FakeDinoOutput(torch.cat((prefix, tokens), dim=1))


def fake_teacher(**kwargs):
    backbone = kwargs.pop("backbone", None) or FakeDinoBackbone()
    with patch("transformers.AutoModel.from_pretrained", return_value=backbone):
        return DinoCorrespondenceTeacher(**kwargs)


class ConstantVelocityModel(torch.nn.Module):
    def forward(self, x, t, return_uncertainty=False, **kwargs):
        velocity = torch.ones_like(x)
        if return_uncertainty:
            return velocity, torch.zeros_like(x[:, :1])
        return velocity


class RecordingVelocityModel(ConstantVelocityModel):
    def forward(self, x, t, person_agnostic=None, person_mask=None, edit_mask=None,
                garment_high_frequency=None, **kwargs):
        self.person_condition = person_agnostic.detach().clone()
        self.person_mask = person_mask.detach().clone()
        self.edit_mask = edit_mask.detach().clone()
        self.garment_high_frequency = (
            None if garment_high_frequency is None
            else garment_high_frequency.detach().clone()
        )
        return super().forward(x, t, **kwargs)


class VTONTests(unittest.TestCase):
    def test_correspondence_teacher_is_frozen_and_graph_free(self):
        teacher = fake_teacher()
        self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))
        # Stays in eval even when the trainer switches to train mode, so the target for a
        # given pair is deterministic across steps.
        teacher.train()
        self.assertFalse(teacher.model.training)
        features = teacher.features(torch.randn(1, 3, 512, 384, requires_grad=True))
        self.assertFalse(features.requires_grad)

    def test_teacher_strips_cls_and_register_tokens(self):
        teacher = fake_teacher(backbone=FakeDinoBackbone(hidden=8, patch_size=16, registers=4))
        features = teacher.features(torch.randn(2, 3, 512, 384))
        # 512x384 at patch 16 is exactly the 32x24 PFT token grid; the 5 prefix tokens
        # carry no position and must not be treated as patches.
        self.assertEqual(tuple(features.shape), (2, 8, 32, 24))

    def test_teacher_keeps_the_dataset_aspect_ratio_by_default(self):
        teacher = fake_teacher()
        self.assertIsNone(teacher.input_size)
        self.assertEqual(teacher.resolve_input_size(torch.zeros(1, 3, 256, 256)), (256, 256))
        self.assertEqual(teacher.resolve_input_size(torch.zeros(1, 3, 512, 384)), (512, 384))
        # Rounded to the patch grid, never shrunk to zero.
        self.assertEqual(teacher.resolve_input_size(torch.zeros(1, 3, 250, 200)), (256, 192))

    def test_correspondence_target_is_the_best_matching_garment_position(self):
        # Person token k is an exact copy of garment token (k + 1); the target must be the
        # normalised centre of that garment cell.
        garment_grid = (2, 3)
        garment = torch.randn(1, 5, *garment_grid)
        flat = garment.flatten(2)
        person = torch.stack([flat[0, :, (index + 1) % 6] for index in range(6)], dim=-1)
        person = person[None].reshape(1, 5, 2, 3)
        target, weight, similarity = correspondence_targets(person, garment, garment_grid=garment_grid)
        coordinates = grid_coordinates(garment_grid)
        expected = torch.stack([coordinates[(index + 1) % 6] for index in range(6)])
        torch.testing.assert_close(target[0], expected)
        torch.testing.assert_close(similarity[0], torch.ones(6), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all(weight == 1))

    def test_correspondence_ignores_masked_out_garment_tokens(self):
        garment_grid = (1, 4)
        garment = torch.zeros(1, 2, 1, 4)
        garment[0, :, 0, 0] = torch.tensor([1.0, 0.0])
        garment[0, :, 0, 3] = torch.tensor([0.0, 1.0])
        person = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
        valid = torch.tensor([[False, False, False, True]])
        target, _, _ = correspondence_targets(person, garment, garment_grid=garment_grid, garment_valid=valid)
        # The perfect match at column 0 is masked away, so the only usable key wins.
        torch.testing.assert_close(target[0, 0], grid_coordinates(garment_grid)[3])

    def test_low_confidence_and_dropped_matches_get_zero_weight(self):
        garment = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
        person = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).T.reshape(1, 2, 1, 2)
        _, weight, similarity = correspondence_targets(
            person, garment, garment_grid=(1, 1), min_similarity=0.5
        )
        # Orthogonal features (similarity 0) are below the gate; the identical one is not.
        torch.testing.assert_close(similarity[0], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weight[0], torch.tensor([0.0, 1.0]))

    def test_mutual_check_rejects_one_sided_matches(self):
        # Two person tokens both match garment token 0 best; only the closer one survives
        # cycle consistency.
        garment = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
        person = torch.tensor([[1.0, 0.0], [0.7, 0.7]]).T.reshape(1, 2, 1, 2)
        _, weight, _ = correspondence_targets(person, garment, garment_grid=(1, 1), mutual=True)
        torch.testing.assert_close(weight[0], torch.tensor([1.0, 0.0]))

    def test_mask_to_token_valid_falls_back_when_a_sample_is_empty(self):
        mask = torch.zeros(2, 1, 8, 8)
        mask[0, :, :4, :4] = 1
        valid = mask_to_token_valid(mask, (2, 2))
        torch.testing.assert_close(valid[0], torch.tensor([True, False, False, False]))
        # An all-empty garment mask would make every similarity -inf; fall back to all keys.
        self.assertTrue(torch.all(valid[1]))

    def test_center_of_mass_loss_is_zero_for_perfect_attention(self):
        grid = (4, 4)
        coordinates = grid_coordinates(grid)
        attention = torch.zeros(1, 3, 16)
        attention[0, 0, 5] = 1
        attention[0, 1, 9] = 1
        attention[0, 2, 2] = 1
        target = coordinates[[5, 9, 2]][None]
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=1.0, entropy_weight=0.0, nll_weight=0.0, photometric_weight=0.0
        )
        loss, metrics = loss_fn(
            [{"block": 1, "scale": "coarse", "weights": attention, "grid": grid, "key_padding": None}],
            target,
            torch.ones(1, 3),
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertAlmostEqual(metrics["correspondence_center_loss"].item(), 0.0, places=6)

    def test_entropy_term_separates_sharp_from_diffuse_attention(self):
        """The barycentre alone cannot: a symmetric blur has the same centre of mass as a
        spike at that centre, which is exactly the degenerate solution to rule out."""
        grid = (1, 4)
        coordinates = grid_coordinates(grid)
        sharp = torch.tensor([[[0.0, 0.5, 0.5, 0.0]]])
        diffuse = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
        target = ((coordinates[1] + coordinates[2]) / 2)[None, None]
        center_only = CorrespondenceAttentionLoss(
            center_weight=1.0, entropy_weight=0.0, nll_weight=0.0, photometric_weight=0.0
        )
        with_entropy = CorrespondenceAttentionLoss(
            center_weight=1.0, entropy_weight=1.0, nll_weight=0.0, photometric_weight=0.0
        )
        weight = torch.ones(1, 1)

        def evaluate(loss_fn, attention):
            maps = [{"block": 1, "scale": "coarse", "weights": attention, "grid": grid, "key_padding": None}]
            return loss_fn(maps, target, weight)[0].item()

        self.assertAlmostEqual(evaluate(center_only, sharp), evaluate(center_only, diffuse), places=6)
        self.assertLess(evaluate(with_entropy, sharp), evaluate(with_entropy, diffuse))
        # A uniform distribution over every key is exactly one normalised nat.
        self.assertAlmostEqual(evaluate(with_entropy, diffuse) - evaluate(center_only, diffuse), 1.0, places=5)

    def test_bf16_entropy_backward_is_finite_when_softmax_underflows_to_zero(self):
        logits = torch.tensor(
            [[[[0.0, -100.0, -100.0, -100.0]]]], dtype=torch.bfloat16,
            requires_grad=True,
        )
        attention = logits.softmax(-1)
        self.assertTrue((attention == 0).any())
        entropy = stable_attention_entropy(attention, eps=1e-8).sum()
        entropy.backward()
        self.assertTrue(torch.isfinite(entropy))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_correspondence_entropy_backward_is_finite_with_exact_zero_heads(self):
        attention = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.bfloat16,
            requires_grad=True,
        )
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=0.0,
            entropy_weight=1.0,
            nll_weight=0.0,
            photometric_weight=0.0,
        )
        loss, _ = loss_fn(
            self._map(attention, (1, 2)),
            appearance_weight=torch.ones(1, 1),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(attention.grad).all())

    def test_unsupervised_tokens_do_not_contribute(self):
        grid = (1, 4)
        attention = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])
        target = grid_coordinates(grid)[[0, 0]][None]
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=1.0, entropy_weight=0.0, nll_weight=0.0, photometric_weight=0.0
        )
        maps = [{"block": 1, "scale": "coarse", "weights": attention, "grid": grid, "key_padding": None}]
        # Token 1 is badly placed, but a zero weight (dropped garment, or a match below the
        # confidence gate) must remove it from both numerator and denominator.
        gated, _ = loss_fn(maps, target, torch.tensor([[1.0, 0.0]]))
        self.assertAlmostEqual(gated.item(), 0.0, places=6)
        self.assertGreater(loss_fn(maps, target, torch.ones(1, 2))[0].item(), 0.0)

    def test_warmup_ramp_advances_without_a_lightning_trainer(self):
        """train.py drives this module with accelerate, not a Lightning Trainer. Depending
        on the Lightning version ``global_step`` is then either missing (it raises until
        train.py assigns it after the first optimizer step) or a read-only property pinned
        at 0. Either way the ramp has to come from the module's own counter."""
        trainer = object.__new__(LatentVTONPatchForcingTrainer)
        # What the real base hook touches: no Lightning Trainer, no scheduler, no EMA.
        trainer.__dict__.update(
            {"_trainer": None, "lr_scheduler_cfg": None, "ema_model": None, "_optimizer_steps": 0}
        )
        trainer.correspondence_warmup_steps = 4

        for missing_global_step in (True, False):
            with self.subTest(missing_global_step=missing_global_step):
                if missing_global_step:
                    # Reading raises, exactly as nn.Module.__getattr__ does.
                    patcher = patch.object(
                        LatentVTONPatchForcingTrainer,
                        "global_step",
                        property(lambda self: (_ for _ in ()).throw(AttributeError("global_step"))),
                    )
                else:
                    patcher = patch.object(
                        LatentVTONPatchForcingTrainer, "global_step", property(lambda self: 0)
                    )
                with patcher:
                    trainer._optimizer_steps = 0
                    self.assertEqual(trainer._correspondence_ramp(), 0.0)
                    for _ in range(2):
                        LatentVTONPatchForcingTrainer.on_train_batch_end(trainer, None, None, 0)
                    self.assertEqual(trainer._correspondence_ramp(), 0.5)
                    for _ in range(10):
                        LatentVTONPatchForcingTrainer.on_train_batch_end(trainer, None, None, 0)
                    # Clamped, never past 1.
                    self.assertEqual(trainer._correspondence_ramp(), 1.0)

        # A synthetic Accelerate global step may jump to a loaded checkpoint step, but
        # without a real Lightning Trainer the restart-local counter must still win.
        with patch.object(LatentVTONPatchForcingTrainer, "global_step", property(lambda self: 3)):
            trainer._optimizer_steps = 0
            self.assertEqual(trainer._correspondence_ramp(), 0.0)
            # A genuinely attached Lightning Trainer uses its own advancing counter.
            trainer._trainer = object()
            self.assertEqual(trainer._correspondence_ramp(), 0.75)
            trainer._trainer = None

        # Zero warmup short-circuits before any counter is consulted.
        trainer.correspondence_warmup_steps = 0
        self.assertEqual(trainer._correspondence_ramp(), 1.0)

    def test_pure_noise_detection_ignores_non_editable_timestep_ones(self):
        edit = torch.tensor([[True, True, False], [True, True, False], [False, False, False]])
        times = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.2, 1.0], [1.0, 1.0, 1.0]])
        pure = LatentVTONPatchForcingTrainer._pure_noise_samples(times, edit)
        torch.testing.assert_close(pure, torch.tensor([True, False, False]))

    def test_detail_importance_emphasises_sparse_edges_up_to_six_times(self):
        target = torch.zeros(1, 4, 4, 5)
        target[:, :, :, 3:] = 2.0
        mask = torch.ones(1, 1, 4, 5)
        importance = LatentVTONPatchForcingTrainer._detail_importance(
            target, mask, edge_weight=5.0
        )
        self.assertEqual(tuple(importance.shape), (1, 1, 4, 5))
        self.assertAlmostEqual(importance.min().item(), 1.0, places=6)
        self.assertAlmostEqual(importance.max().item(), 6.0, places=6)

    def test_value_target_ema_stays_frozen_and_tracks_the_student(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        target.load_state_dict(source.state_dict())
        target.requires_grad_(False)
        before = target.weight.detach().clone()
        with torch.no_grad():
            source.weight.add_(2.0)
        LatentVTONPatchForcingTrainer._ema_module(target, source, decay=0.75)
        torch.testing.assert_close(target.weight, before + 0.5)
        self.assertFalse(any(parameter.requires_grad for parameter in target.parameters()))

    def test_legacy_checkpoint_initialises_missing_value_targets_from_loaded_student(self):
        def make_trainer():
            return LatentVTONPatchForcingTrainer(
                model=self._multiscale_model(),
                first_stage=torch.nn.Identity(),
                flow={
                    "target": "patch_flow.flow_vton.VTONPatchFlowForcing",
                    "params": {"patch_size": 2},
                },
                ema_rate=0.0,
                compute_validation_metrics=False,
                correspondence_center_weight=0.0,
                correspondence_nll_weight=0.0,
                correspondence_entropy_weight=0.0,
                correspondence_photometric_weight=0.0,
                correspondence_value_weight=0.1,
                correspondence_scales=["coarse", "middle", "detail"],
            )

        old = make_trainer()
        legacy = {
            key: value for key, value in old.state_dict().items()
            if not key.startswith(("value_target_embedders.", "value_target_norms."))
        }
        restored = make_trainer()
        restored.load_state_dict(legacy, strict=True)
        for scale, target_embedder in restored.value_target_embedders.items():
            source = restored._value_source_embedder(scale)
            for target_parameter, source_parameter in zip(
                target_embedder.parameters(), source.parameters()
            ):
                torch.testing.assert_close(target_parameter, source_parameter)

    @staticmethod
    def _map(attention, grid, padding=None, scale="coarse", block=1):
        return [{"block": block, "scale": scale, "weights": attention,
                 "grid": grid, "key_padding": padding}]

    def test_neighbourhood_mass_matches_a_dense_distance_computation(self):
        grid = (6, 5)
        torch.manual_seed(0)
        attention = torch.rand(2, 4, 30)
        attention = attention / attention.sum(-1, keepdim=True)
        target = torch.rand(2, 4, 2)
        coordinates = grid_coordinates(grid)
        for radius in (0.05, 0.15, 0.4):
            with self.subTest(radius=radius):
                dense = (coordinates[None, None] - target[:, :, None]).norm(dim=-1) <= radius
                torch.testing.assert_close(
                    neighbourhood_mass(attention, target, grid, radius),
                    (attention * dense).sum(-1), rtol=1e-5, atol=1e-6,
                )

    def test_neighbourhood_mass_never_double_counts_at_the_border(self):
        """Out-of-grid offsets must be dropped, not clamped onto an in-grid key."""
        grid = (4, 4)
        attention = torch.zeros(1, 1, 16)
        attention[0, 0, 0] = 1.0
        target = grid_coordinates(grid)[0].view(1, 1, 2)
        self.assertAlmostEqual(float(neighbourhood_mass(attention, target, grid, 0.05)), 1.0, places=6)

    def test_nll_separates_on_target_from_bimodal_straddling(self):
        """The measured failure mode.

        On the real 32x24 run the coarse blocks reached a barycentre 1.7 tokens from the
        target with only 3.8 effective keys -- and put their argmax 4.5 tokens away, with
        5% of mass near the target. Barycentre and entropy both rate that as good; only a
        term on the target mass can reject it.
        """
        grid = (1, 9)
        coordinates = grid_coordinates(grid)
        target = coordinates[4].view(1, 1, 2)
        on_target = torch.zeros(1, 1, 9); on_target[0, 0, 4] = 1.0
        straddling = torch.zeros(1, 1, 9)                # identical barycentre, no mass on it
        straddling[0, 0, 1] = 0.5
        straddling[0, 0, 7] = 0.5
        w = torch.ones(1, 1)
        only = lambda **kw: CorrespondenceAttentionLoss(
            **{"center_weight": 0.0, "entropy_weight": 0.0, "nll_weight": 0.0,
               "photometric_weight": 0.0, **kw})

        gap = lambda fn: abs(fn(self._map(straddling, grid), target, w)[0].item()
                             - fn(self._map(on_target, grid), target, w)[0].item())

        # Barycentre: completely blind, by construction.
        self.assertAlmostEqual(gap(only(center_weight=1.0)), 0.0, places=6)
        # Entropy: both maps are sharp, so at the shipped weight it barely reacts.
        entropy_gap = gap(only(entropy_weight=0.05))
        # Target mass: rejects the straddling map outright.
        nll_gap = gap(only(nll_weight=1.0, nll_radius=0.05))
        self.assertAlmostEqual(
            only(nll_weight=1.0, nll_radius=0.05)(self._map(on_target, grid), target, w)[0].item(),
            0.0, places=4)
        self.assertGreater(nll_gap, 5.0)
        self.assertGreater(nll_gap, 50 * entropy_gap)

    def test_target_mass_metric_reports_the_measured_quantity(self):
        grid = (1, 9)
        target = grid_coordinates(grid)[4].view(1, 1, 2)
        attention = torch.zeros(1, 1, 9)
        attention[0, 0, 4] = 0.3
        attention[0, 0, 0] = 0.7
        loss_fn = CorrespondenceAttentionLoss(nll_radius=0.05, photometric_weight=0.0)
        _, metrics = loss_fn(self._map(attention, grid), target, torch.ones(1, 1))
        self.assertAlmostEqual(metrics["correspondence_target_mass"].item(), 0.3, places=5)

    def test_nll_and_entropy_supervise_each_attention_head(self):
        """A correct head must not hide another head that routes the logo elsewhere."""
        grid = (1, 2)
        target = grid_coordinates(grid)[0].view(1, 1, 2)
        per_head = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        averaged = per_head.mean(dim=1)
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=0.0,
            entropy_weight=0.01,
            nll_weight=1.0,
            nll_radius=0.1,
            photometric_weight=0.0,
        )
        strict, strict_metrics = loss_fn(self._map(per_head, grid), target, torch.ones(1, 1))
        old_loophole, _ = loss_fn(self._map(averaged, grid), target, torch.ones(1, 1))
        self.assertGreater(strict.item(), 5 * old_loophole.item())
        self.assertAlmostEqual(strict_metrics["correspondence_target_mass"].item(), 0.5, places=6)

    def test_nll_radius_can_be_configured_per_garment_scale(self):
        grid = (1, 5)
        target = grid_coordinates(grid)[2].view(1, 1, 2)
        attention = torch.zeros(1, 1, 5)
        attention[0, 0, 3] = 1.0
        maps = [
            {"block": 1, "scale": "coarse", "weights": attention, "grid": grid, "key_padding": None},
            {"block": 2, "scale": "detail", "weights": attention, "grid": grid, "key_padding": None},
        ]
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=0.0,
            entropy_weight=0.0,
            nll_weight=1.0,
            nll_radius={"coarse": 0.21, "detail": 0.05},
            photometric_weight=0.0,
        )
        _, metrics = loss_fn(maps, target, torch.ones(1, 1))
        self.assertAlmostEqual(metrics["correspondence/coarse/target_mass"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["correspondence/detail/target_mass"].item(), 0.0, places=6)

    def test_per_scale_correspondence_metrics_average_their_blocks(self):
        grid = (1, 2)
        target = grid_coordinates(grid)[0].view(1, 1, 2)
        first = torch.tensor([[[1.0, 0.0]]])
        second = torch.tensor([[[0.5, 0.5]]])
        detail = torch.tensor([[[0.0, 1.0]]])
        maps = [
            {"block": 1, "scale": "middle", "weights": first, "grid": grid, "key_padding": None},
            {"block": 2, "scale": "middle", "weights": second, "grid": grid, "key_padding": None},
            {"block": 3, "scale": "detail", "weights": detail, "grid": grid, "key_padding": None},
        ]
        loss_fn = CorrespondenceAttentionLoss(nll_radius=0.1, photometric_weight=0.0)
        _, metrics = loss_fn(maps, target, torch.ones(1, 1))
        self.assertAlmostEqual(metrics["correspondence/middle/target_mass"].item(), 0.75, places=6)
        self.assertAlmostEqual(metrics["correspondence/detail/target_mass"].item(), 0.0, places=6)
        self.assertIn("correspondence/middle/nll", metrics)
        self.assertIn("correspondence/middle/entropy", metrics)

    def test_value_transport_loss_matches_actual_cross_attention_output(self):
        grid = (1, 2)
        attention = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        value_target = torch.randn(1, 2, 8)
        only_value = CorrespondenceAttentionLoss(
            center_weight=0.0,
            entropy_weight=0.0,
            nll_weight=0.0,
            photometric_weight=0.0,
            value_weight=1.0,
        )

        def evaluate(output):
            maps = [{
                "block": 1,
                "scale": "middle",
                "weights": attention,
                "output": output,
                "grid": grid,
                "key_padding": None,
            }]
            return only_value(
                maps,
                appearance_weight=torch.ones(1, 2),
                value_targets={"middle": value_target},
            )

        correct, metrics = evaluate(value_target.clone())
        opposite, _ = evaluate(-value_target)
        self.assertAlmostEqual(correct.item(), 0.0, places=6)
        self.assertGreater(opposite.item(), 1.0)
        self.assertAlmostEqual(metrics["correspondence_value"].item(), 0.0, places=6)
        self.assertAlmostEqual(metrics["correspondence_value_cosine"].item(), 0.0, places=6)
        self.assertAlmostEqual(metrics["correspondence_value_huber"].item(), 0.0, places=6)
        self.assertIn("correspondence/middle/value", metrics)

    def test_value_huber_rejects_the_cosine_scale_shortcut(self):
        grid = (1, 1)
        target = torch.ones(1, 1, 4)
        maps = [{
            "block": 1,
            "scale": "coarse",
            "weights": torch.ones(1, 1, 1),
            "output": target * 0.01,
            "grid": grid,
            "key_padding": None,
        }]
        cosine_only = CorrespondenceAttentionLoss(
            center_weight=0.0, entropy_weight=0.0, nll_weight=0.0,
            photometric_weight=0.0, value_weight=1.0, value_cosine_mix=1.0,
        )
        mixed = CorrespondenceAttentionLoss(
            center_weight=0.0, entropy_weight=0.0, nll_weight=0.0,
            photometric_weight=0.0, value_weight=1.0, value_cosine_mix=0.5,
        )
        kwargs = dict(appearance_weight=torch.ones(1, 1), value_targets={"coarse": target})
        self.assertAlmostEqual(cosine_only(maps, **kwargs)[0].item(), 0.0, places=6)
        mixed_loss, metrics = mixed(maps, **kwargs)
        self.assertGreater(mixed_loss.item(), 0.1)
        self.assertGreater(metrics["correspondence_value_huber"].item(), 0.1)

    def test_photometric_loss_penalises_retrieving_the_garment_mean(self):
        """Exactly the observed defect: every person token retrieving the same average
        colour rather than the colour it needs."""
        garment = torch.zeros(1, 3, 1, 2)
        garment[0, :, 0, 0] = torch.tensor([-0.60, -0.71, -0.55])      # navy
        garment[0, :, 0, 1] = torch.tensor([0.51, 0.40, 0.65])         # lavender
        query = torch.stack([garment[0, :, 0, 0], garment[0, :, 0, 1]])[None]
        appearance = {"garment": garment, "query": query}
        grid = (1, 2)
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=0.0, entropy_weight=0.0, nll_weight=0.0, photometric_weight=1.0)
        w = torch.ones(1, 2)
        got = {}
        for name, a in (("correct", torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
                        ("mean", torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])),
                        ("swapped", torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))):
            loss, metrics = loss_fn(self._map(a, grid), appearance=appearance, appearance_weight=w)
            got[name] = (loss.item(), metrics["correspondence_appearance_error"].item())
        self.assertAlmostEqual(got["correct"][0], 0.0, places=6)
        self.assertGreater(got["mean"][0], 0.4)
        self.assertGreater(got["swapped"][0], got["mean"][0])
        # Reported in RGB L2 units so it reads against the measured 0.817 mean-colour
        # baseline and the 0.119 oracle straight off tensorboard.
        self.assertAlmostEqual(got["mean"][1], got["mean"][0] ** 0.5, places=5)

    def test_photometric_term_needs_no_teacher(self):
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=0.0, nll_weight=0.0, entropy_weight=0.0, photometric_weight=1.0)
        self.assertFalse(loss_fn.needs_target)
        self.assertTrue(loss_fn.enabled)
        grid = (1, 2)
        garment = torch.zeros(1, 3, 1, 2); garment[0, :, 0, 1] = 1.0
        appearance = {"garment": garment, "query": torch.ones(1, 1, 3)}
        loss, metrics = loss_fn(self._map(torch.tensor([[[0.5, 0.5]]]), grid), None, None,
                                appearance=appearance, appearance_weight=torch.ones(1, 1))
        self.assertGreater(loss.item(), 0)
        self.assertNotIn("correspondence/block_01_coarse/nll", metrics)

    def test_dropped_samples_are_excluded_from_every_term(self):
        grid = (1, 4)
        target = grid_coordinates(grid)[0].expand(1, 2, 2).contiguous()
        attention = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])
        appearance = {"garment": torch.randn(1, 3, 1, 4), "query": torch.randn(1, 2, 3)}
        loss_fn = CorrespondenceAttentionLoss(nll_radius=0.05)
        gate = torch.tensor([[1.0, 0.0]])
        gated = loss_fn(self._map(attention, grid), target, gate,
                        appearance=appearance, appearance_weight=gate)[0]
        every = loss_fn(self._map(attention, grid), target, torch.ones(1, 2),
                        appearance=appearance, appearance_weight=torch.ones(1, 2))[0]
        self.assertLess(gated.item(), every.item())

    def test_shipped_init_std_lands_in_the_working_band(self):
        """Guards the fix for the deadlock, at the value the config actually ships.

        At std 1e-3 the garment output projection started at Frobenius norm 1.15 on a
        1152x1152 matrix -- 29x below standard init -- and measured on a real run it did
        not grow: 0.7 per 1000 steps where a coherent Adam trajectory moves ~0.1 per step.
        The branch stayed a ~2% contributor to the residual stream, so the retrieved
        garment could tint the shirt but never restructure it. The garment residual is
        ungated (x = x + cross) while self-attention is gated by adaLN, so parity with the
        pretrained projections is stronger than it looks; 0.2-0.5 of standard init is the
        intended range.
        """
        from omegaconf import OmegaConf

        hidden = 1152
        standard = (2 / (hidden + hidden)) ** 0.5 * hidden          # xavier, ~33.9
        shipped = float(
            OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/model/vton-pft-xl.yaml")
            .params.garment_attention_output_init_std
        )
        torch.manual_seed(0)
        model = VTONPatchForcingDiT(
            input_size=32, patch_size=2, in_channels=4, hidden_size=hidden, depth=2,
            num_heads=16, num_classes=1000, predict_uncertainty=True,
            garment_middle_channels=256, garment_detail_channels=128,
            garment_scale_routes=["coarse", "detail"], cross_attention_every=1,
            garment_attention_output_init_std=shipped, compile=False,
        )
        for block in model.blocks:
            ratio = float(block.garment_cross_attention.out_proj.weight.detach().norm()) / standard
            self.assertGreater(ratio, 0.2, f"init std {shipped} re-enters the muted-branch deadlock")
            self.assertLess(ratio, 0.6, f"init std {shipped} would overpower the ungated residual")
        # The value it replaced is far outside that band -- this is what changed.
        self.assertLess(1e-3 * hidden / standard, 0.05)

    def test_garment_token_norm_equalises_key_scale_across_branches(self):
        """Raw VAE activations differ in scale by branch (measured 25 / 143 / 68 per
        token at 512x384). The norm must remove that."""
        for enabled in (False, True):
            model = self._multiscale_model(garment_token_norm=enabled).eval()
            position = model._position_embedding(8, 6, torch.float32, "cpu")
            with torch.no_grad():
                tokens, _, _ = model._garment_branches(
                    torch.randn(1, 4, 8, 6),
                    torch.randn(1, 16, 16, 12) * 30.0,
                    torch.randn(1, 8, 32, 24) * 3.0,
                    torch.zeros(1, 12, 64), position, 8, 6,
                )
            norms = {k: float(v[0].norm(dim=-1).mean()) for k, v in tokens.items()}
            spread = max(norms.values()) / min(norms.values())
            with self.subTest(garment_token_norm=enabled):
                if enabled:
                    self.assertLess(spread, 1.5, f"branch key scales still diverge: {norms}")
                else:
                    self.assertGreater(spread, 3.0, f"expected raw scale divergence: {norms}")

    def test_garment_token_norm_is_applied_before_the_positional_embedding(self):
        model = self._multiscale_model(garment_token_norm=True).eval()
        self.assertEqual(set(model.garment_token_norms), {"coarse", "middle", "detail"})
        position = model._position_embedding(8, 6, torch.float32, "cpu")
        with torch.no_grad():
            tokens, values, _ = model._garment_branches(
                torch.randn(1, 4, 8, 6) * 100.0, torch.randn(1, 16, 16, 12),
                torch.randn(1, 8, 32, 24), torch.zeros(1, 12, 64), position, 8, 6,
            )
        # Removing the position must leave a unit-variance LayerNorm output behind.
        content = tokens["coarse"] - position
        self.assertAlmostEqual(float(content.var(-1, unbiased=False).mean()), 1.0, places=3)
        self.assertAlmostEqual(float(content.mean()), 0.0, places=4)
        torch.testing.assert_close(content, values["coarse"])

    def test_entropy_is_normalised_by_the_usable_key_count(self):
        """Branches with different key counts must contribute comparably, so a uniform
        distribution scores 1.0 whether it spans 4 keys or 2."""
        grid = (1, 4)
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=0.0, entropy_weight=1.0, nll_weight=0.0, photometric_weight=0.0
        )
        target = grid_coordinates(grid)[[0]][None]
        padding = torch.tensor([[False, False, True, True]])
        uniform_four = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
        uniform_two = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
        wide = loss_fn(
            [{"block": 1, "scale": "coarse", "weights": uniform_four, "grid": grid, "key_padding": None}],
            target,
            torch.ones(1, 1),
        )[0]
        narrow = loss_fn(
            [{"block": 1, "scale": "coarse", "weights": uniform_two, "grid": grid, "key_padding": padding}],
            target,
            torch.ones(1, 1),
        )[0]
        self.assertAlmostEqual(wide.item(), 1.0, places=5)
        self.assertAlmostEqual(narrow.item(), 1.0, places=5)

    def test_preview_sample_is_moved_to_front(self):
        with TemporaryDirectory() as directory:
            pair_list = Path(directory) / "test_pairs.txt"
            pair_list.write_text("first.jpg first.jpg\ndetail.jpg detail.jpg\n", encoding="utf-8")
            dataset = VTONHDDataset(
                directory,
                split="test",
                pair_list=str(pair_list),
                preview_sample_id="detail.jpg",
            )
            self.assertEqual(dataset.pairs[0], ("detail.jpg", "detail.jpg"))

    def test_finite_dataloader_repeats_until_step_limit(self):
        batches = repeat_dataloader([0, 1, 2, 3], max_epochs=-1)
        self.assertEqual([next(batches) for _ in range(10)], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    def test_zero_initialized_model_matches_pft(self):
        torch.manual_seed(7)
        kwargs = dict(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=2,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            compile=False,
        )
        base = PatchForcingDiT(**kwargs).eval()
        vton = VTONPatchForcingDiT(**kwargs, cross_attention_every=1).eval()
        vton.load_pretrained_state_dict(base.state_dict())
        x = torch.randn(2, 4, 8, 8)
        t = torch.rand(2, 16)
        y = torch.randint(0, 10, (2,))
        zeros = torch.zeros_like(x)
        mask = torch.zeros(2, 1, 8, 8)
        with torch.no_grad():
            expected = base(x, t, y)
            actual = vton(
                x,
                t,
                y,
                person_agnostic=zeros,
                person_mask=mask,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_new_conditioning_paths_receive_gradients(self):
        kwargs = dict(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            compile=False,
        )
        model = VTONPatchForcingDiT(
            **kwargs,
            cross_attention_every=1,
            gradient_checkpointing=True,
        ).train()
        with torch.no_grad():
            model.final_layer.linear.weight.normal_(std=0.01)
        x = torch.randn(1, 4, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        output = model(
            x,
            torch.rand(1, 16),
            torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn_like(x),
            person_mask=mask,
            garment=torch.randn(1, 4, 8, 8),
            garment_mask=mask,
        )
        output.square().mean().backward()
        self.assertGreater(model.x_embedder.proj.weight.grad[:, 4:].abs().sum().item(), 0)
        cross = model.blocks[0].garment_cross_attention.out_proj.weight.grad
        self.assertGreater(cross.abs().sum().item(), 0)

    def test_model_supports_rectangular_latents(self):
        model = VTONPatchForcingDiT(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            cross_attention_every=1,
            compile=False,
        ).eval()
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        with torch.no_grad():
            velocity, logvar = model(
                latent,
                torch.rand(1, 12),
                torch.zeros(1, dtype=torch.long),
                person_agnostic=torch.randn_like(latent),
                person_mask=mask,
                edit_mask=mask,
                garment=torch.randn(1, 4, 8, 6),
                garment_mask=mask,
                return_uncertainty=True,
            )
        self.assertEqual(velocity.shape, latent.shape)
        self.assertEqual(logvar.shape, (1, 1, 8, 6))

    @staticmethod
    def _multiscale_model(**overrides):
        kwargs = dict(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=3,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            garment_middle_channels=16,
            garment_detail_channels=8,
            garment_scale_routes=["coarse", "middle", "detail"],
            cross_attention_every=1,
            compile=False,
        )
        kwargs.update(overrides)
        return VTONPatchForcingDiT(**kwargs)

    def test_every_garment_branch_receives_gradients(self):
        model = self._multiscale_model().train()
        with torch.no_grad():
            model.final_layer.linear.weight.normal_(std=0.01)
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        output = model(
            latent,
            torch.rand(1, 12),
            torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn_like(latent),
            person_mask=mask,
            edit_mask=mask,
            garment=torch.randn(1, 4, 8, 6),
            garment_middle=torch.randn(1, 16, 16, 12),
            garment_detail=torch.randn(1, 8, 32, 24),
            garment_mask=mask,
        )
        output.square().mean().backward()
        for name, parameter in (
            ("coarse", model.garment_embedder.proj.weight),
            ("middle", model.garment_middle_embedder.weight),
            ("detail", model.garment_detail_embedder.weight),
        ):
            self.assertIsNotNone(parameter.grad, f"{name} branch received no gradient")
            self.assertGreater(parameter.grad.abs().sum().item(), 0, f"{name} branch gradient is zero")

    def test_garment_gradient_metrics_cover_attention_and_embedders(self):
        model = self._multiscale_model().train()
        with torch.no_grad():
            model.final_layer.linear.weight.normal_(std=0.01)
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        output = model(
            latent,
            torch.rand(1, 12),
            torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn_like(latent),
            person_mask=mask,
            edit_mask=mask,
            garment=torch.randn(1, 4, 8, 6),
            garment_middle=torch.randn(1, 16, 16, 12),
            garment_detail=torch.randn(1, 8, 32, 24),
            garment_mask=mask,
        )
        output.square().mean().backward()

        trainer = object.__new__(LatentVTONPatchForcingTrainer)
        trainer.__dict__["model"] = model
        metrics = LatentVTONPatchForcingTrainer.garment_gradient_norms(trainer)
        for block_index, scale in enumerate(("coarse", "middle", "detail"), start=1):
            prefix = f"garment_grad/block_{block_index:02d}_{scale}"
            for projection in ("out_proj", "q", "k", "v"):
                self.assertIn(f"{prefix}/{projection}", metrics)
                self.assertGreater(metrics[f"{prefix}/{projection}"].item(), 0)
        for scale in ("coarse", "middle", "detail"):
            self.assertGreater(metrics[f"garment_grad/embedder_{scale}"].item(), 0)

    def test_garment_attention_uses_small_nonzero_output_initialization(self):
        model = self._multiscale_model(garment_attention_output_init_std=1e-3)
        for block in model.blocks:
            weight = block.garment_cross_attention.out_proj.weight
            self.assertGreater(weight.abs().sum().item(), 0)
            self.assertLess(weight.std().item(), 2e-3)

    def test_detail_branch_keeps_its_finer_token_grid(self):
        model = self._multiscale_model().eval()
        latent = torch.randn(1, 4, 8, 6)
        position = model._position_embedding(8, 6, latent.dtype, latent.device)
        x = torch.zeros(1, 12, 64)
        tokens, values, grids = model._garment_branches(
            torch.randn(1, 4, 8, 6),
            torch.randn(1, 16, 16, 12),
            torch.randn(1, 8, 32, 24),
            x,
            position,
            8,
            6,
        )
        # coarse/middle align with the person token grid; detail is 4x finer in area.
        self.assertEqual(grids["coarse"], (4, 3))
        self.assertEqual(grids["middle"], (4, 3))
        self.assertEqual(grids["detail"], (8, 6))
        self.assertEqual(tokens["detail"].shape[1], 48)
        self.assertEqual(tokens["coarse"].shape[1], 12)
        self.assertEqual(values["detail"].shape, tokens["detail"].shape)

    def test_garment_values_exclude_position_while_keys_include_it(self):
        model = self._multiscale_model(garment_token_norm=True).eval()
        latent = torch.randn(1, 4, 8, 6)
        position = model._position_embedding(8, 6, latent.dtype, latent.device)
        keys, values, grids = model._garment_branches(
            latent,
            torch.randn(1, 16, 16, 12),
            torch.randn(1, 8, 32, 24),
            torch.zeros(1, 12, 64),
            position,
            8,
            6,
        )
        for scale in ("coarse", "middle", "detail"):
            expected_position = model._grid_position_embedding(
                *grids[scale], values[scale].dtype, values[scale].device
            )
            torch.testing.assert_close(keys[scale], values[scale] + expected_position)

    def test_512x384_detail_fix_config_is_interleaved_and_content_weighted(self):
        from omegaconf import OmegaConf

        config = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "configs/experiment/viton-pft-xl-512x384.yaml"
        )
        routes = list(config.model.params.garment_scale_routes)
        self.assertEqual(len(routes), 14)
        self.assertEqual(routes[:3], ["coarse", "middle", "detail"])
        self.assertEqual(routes.count("coarse"), 4)
        self.assertEqual(routes.count("middle"), 4)
        self.assertEqual(routes.count("detail"), 6)
        self.assertEqual(config.trainer.params.correspondence_nll_weight, 0.1)
        self.assertEqual(config.trainer.params.correspondence_center_weight, 0.05)
        self.assertEqual(config.trainer.params.correspondence_entropy_weight, 0.0)
        self.assertEqual(config.trainer.params.correspondence_photometric_weight, 0.1)
        self.assertEqual(config.trainer.params.correspondence_value_weight, 0.1)
        self.assertEqual(config.trainer.params.detail_loss_weight, 0.5)
        self.assertEqual(config.trainer.params.detail_edge_weight, 5.0)
        self.assertTrue(config.trainer.params.detail_pure_noise_only)

    def test_routes_must_reference_enabled_branches(self):
        with self.assertRaises(ValueError):
            self._multiscale_model(garment_scale_routes=["coarse", "middle", "nope"])
        with self.assertRaises(ValueError):
            self._multiscale_model(
                garment_middle_channels=None,
                garment_detail_channels=None,
                garment_scale_routes=["coarse", "middle", "detail"],
            )

    def test_unit_embed_gain_keeps_garment_attention_non_uniform(self):
        """The 0.1 gain flattened the cross-attention logits to a uniform average over
        every garment token, so only the mean garment feature reached the backbone -- and
        a uniform map is precisely what the correspondence loss now has to move."""
        torch.manual_seed(0)

        # 32x24 garment tokens: the real 512x384 configuration.
        grid_height, grid_width = 32, 24
        keys_count = grid_height * grid_width

        def peak_attention(gain):
            torch.manual_seed(0)
            model = VTONPatchForcingDiT(
                input_size=8,
                patch_size=2,
                in_channels=4,
                hidden_size=64,
                depth=1,
                num_heads=4,
                num_classes=10,
                predict_uncertainty=True,
                use_vae_garment=False,
                garment_middle_channels=16,
                garment_detail_channels=8,
                garment_scale_routes=["middle"],
                garment_embed_gain=gain,
                cross_attention_every=1,
                compile=False,
            ).eval()
            block = model.blocks[0]
            features = torch.randn(1, 16, grid_height * 4, grid_width * 4)
            keys = model.garment_middle_embedder(features).flatten(2).transpose(1, 2)
            queries = block.garment_norm(torch.randn(1, 16, 64))
            weight, bias = block.garment_cross_attention.in_proj_weight, block.garment_cross_attention.in_proj_bias
            q = queries @ weight[:64].T + bias[:64]
            k = keys @ weight[64:128].T + bias[64:128]
            logits = (q @ k.transpose(-1, -2)) / (64 / 4) ** 0.5
            return logits.softmax(-1).max().item()

        uniform = 1.0 / keys_count
        attenuated = peak_attention(0.1)
        unit = peak_attention(1.0)
        self.assertLess(attenuated, uniform * 2)
        self.assertGreater(unit, uniform * 5)
        self.assertGreater(unit / attenuated, 3)

    def test_model_returns_one_attention_map_per_supervised_block(self):
        model = self._multiscale_model().eval()
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        kwargs = dict(
            person_agnostic=torch.randn_like(latent),
            person_mask=mask,
            edit_mask=mask,
            garment=torch.randn(1, 4, 8, 6),
            garment_middle=torch.randn(1, 16, 16, 12),
            garment_detail=torch.randn(1, 8, 32, 24),
            garment_mask=mask,
        )
        with torch.no_grad():
            velocity, maps = model(
                latent,
                torch.rand(1, 12),
                torch.zeros(1, dtype=torch.long),
                return_garment_attention=True,
                **kwargs,
            )
        self.assertEqual([entry["scale"] for entry in maps], ["coarse", "middle", "detail"])
        for entry in maps:
            height, width = entry["grid"]
            # Queries are always the person token grid; keys follow the branch's own grid.
            self.assertEqual(tuple(entry["weights"].shape), (1, 4, 12, height * width))
            self.assertEqual(tuple(entry["output"].shape), (1, 12, model.hidden_size))
            torch.testing.assert_close(
                entry["weights"].sum(-1), torch.ones(1, 4, 12), atol=1e-5, rtol=1e-5
            )

        with torch.no_grad():
            _, restricted = model(
                latent,
                torch.rand(1, 12),
                torch.zeros(1, dtype=torch.long),
                return_garment_attention=True,
                garment_attention_scales=["coarse", "middle"],
                **kwargs,
            )
        # The detail branch's map is (B, queries, 4x keys); skipping it is how the 512x384
        # configuration keeps correspondence supervision affordable.
        self.assertEqual([entry["scale"] for entry in restricted], ["coarse", "middle"])

    def test_checkpointed_blocks_return_identical_attention_gradients(self):
        """The 512x384 config sets gradient_checkpointing AND correspondence_scales, so the
        attention map is an extra output of a checkpointed block. Verify the recompute in
        backward reproduces it exactly, and that unsupervised scales stay gradient-free."""
        def run(gradient_checkpointing):
            torch.manual_seed(0)
            model = self._multiscale_model(gradient_checkpointing=gradient_checkpointing).train()
            torch.manual_seed(1)
            mask = torch.ones(1, 1, 8, 6)
            latent = torch.randn(1, 4, 8, 6)
            _, _, maps = model(
                latent,
                torch.full((1, 12), 0.5),
                torch.zeros(1, dtype=torch.long),
                person_agnostic=torch.randn_like(latent),
                person_mask=mask,
                edit_mask=mask,
                garment=torch.randn(1, 4, 8, 6),
                garment_middle=torch.randn(1, 16, 16, 12),
                garment_detail=torch.randn(1, 8, 32, 24),
                garment_mask=mask,
                return_uncertainty=True,
                return_garment_attention=True,
                garment_attention_scales=["coarse", "middle"],
            )
            self.assertEqual([entry["scale"] for entry in maps], ["coarse", "middle"])
            target = grid_coordinates((4, 3))[-1].expand(1, 12, 2).contiguous()
            loss, _ = CorrespondenceAttentionLoss(
                center_weight=1.0, entropy_weight=0.05, nll_weight=0.0, photometric_weight=0.0
            )(maps, target, torch.ones(1, 12))
            loss.backward()
            grads = [
                block.garment_cross_attention.in_proj_weight.grad for block in model.blocks
            ]
            return loss.detach(), grads

        plain_loss, plain_grads = run(False)
        checkpointed_loss, checkpointed_grads = run(True)
        torch.testing.assert_close(checkpointed_loss, plain_loss, rtol=0, atol=0)
        for scale, plain, checkpointed in zip(
            ("coarse", "middle", "detail"), plain_grads, checkpointed_grads
        ):
            if scale == "detail":
                # Not in correspondence_scales, so this loss must never reach it.
                self.assertIsNone(plain, "detail block received correspondence gradient")
                self.assertIsNone(checkpointed, "detail block received correspondence gradient")
                continue
            self.assertIsNotNone(plain)
            self.assertGreater(plain.abs().sum().item(), 0)
            torch.testing.assert_close(checkpointed, plain, rtol=0, atol=0)

    def test_value_transport_loss_reaches_v_and_output_projection(self):
        torch.manual_seed(0)
        model = self._multiscale_model(depth=1, garment_scale_routes=["coarse"]).train()
        mask = torch.ones(1, 1, 8, 6)
        _, maps = model(
            x=torch.randn(1, 4, 8, 6),
            t=torch.rand(1, 12),
            y=torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn(1, 4, 8, 6),
            person_mask=mask,
            edit_mask=mask,
            garment=torch.randn(1, 4, 8, 6),
            garment_mask=mask,
            return_garment_attention=True,
        )
        target = torch.randn_like(maps[0]["output"])
        loss = CorrespondenceAttentionLoss(
            center_weight=0.0,
            entropy_weight=0.0,
            nll_weight=0.0,
            photometric_weight=0.0,
            value_weight=1.0,
        )(
            maps,
            appearance_weight=torch.ones(1, 12),
            value_targets={"coarse": target},
        )[0]
        loss.backward()

        attention = model.blocks[0].garment_cross_attention
        hidden = attention.embed_dim
        value_gradient = attention.in_proj_weight.grad[2 * hidden :]
        self.assertGreater(value_gradient.abs().sum().item(), 0)
        self.assertGreater(attention.out_proj.weight.grad.abs().sum().item(), 0)

    def test_correspondence_loss_moves_attention_towards_the_matched_token(self):
        """End to end: the loss reaches the cross-attention parameters and sharpens the
        map onto the requested garment cell."""
        torch.manual_seed(0)
        model = self._multiscale_model(depth=1, garment_scale_routes=["coarse"]).train()
        inputs = dict(
            x=torch.randn(1, 4, 8, 6),
            t=torch.rand(1, 12),
            y=torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn(1, 4, 8, 6),
            person_mask=torch.ones(1, 1, 8, 6),
            edit_mask=torch.ones(1, 1, 8, 6),
            garment=torch.randn(1, 4, 8, 6),
            garment_mask=torch.ones(1, 1, 8, 6),
        )
        grid = (4, 3)
        matched = grid[0] * grid[1] - 1
        # Ask every person token to attend to the bottom-right garment token.
        target = grid_coordinates(grid)[matched].expand(1, 12, 2).contiguous()
        loss_fn = CorrespondenceAttentionLoss(
            center_weight=1.0, entropy_weight=0.05, nll_weight=0.0, photometric_weight=0.0
        )
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)

        def step():
            _, maps = model(return_garment_attention=True, **inputs)
            return loss_fn(maps, target, torch.ones(1, 12))[0], maps[0]["weights"]

        first_loss, first_attention = step()
        for _ in range(60):
            optimizer.zero_grad()
            loss, _ = step()
            loss.backward()
            optimizer.step()
        final_loss, final_attention = step()
        self.assertLess(final_loss.item(), first_loss.item())
        self.assertGreater(
            final_attention[0, :, :, matched].mean().item(),
            first_attention[0, :, :, matched].mean().item(),
        )

    def test_edit_mask_is_not_grown_past_the_token_grid(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        raw_mask = torch.zeros(1, 1, 8, 8)
        raw_mask[:, :, 2:6, 2:6] = 1
        masks = flow.prepare_masks(raw_mask, (8, 8), torch.float32)
        # Token-grid rounding is the only permitted growth; a dilated envelope would
        # force the model to re-synthesise identity evidence it could otherwise copy.
        expected_tokens = torch.nn.functional.adaptive_max_pool2d(raw_mask, (4, 4)) > 0
        torch.testing.assert_close(masks.token, expected_tokens.flatten(2).squeeze(1))
        self.assertEqual(masks.latent.sum().item(), raw_mask.sum().item())

    def test_generate_conditions_on_the_single_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        raw_mask = torch.zeros(1, 1, 8, 8)
        raw_mask[:, :, 2:6, 2:6] = 1
        masks = flow.prepare_masks(raw_mask, (8, 8), torch.float32)
        person_context = torch.ones(1, 4, 8, 8) * (1 - masks.latent)
        model = RecordingVelocityModel()
        flow.generate(
            model,
            torch.randn_like(person_context),
            person_context,
            raw_mask,
            garment=torch.zeros_like(person_context),
            person_condition=person_context,
            person_condition_mask=masks.condition,
            num_steps=1,
        )
        torch.testing.assert_close(model.person_mask, masks.condition)
        torch.testing.assert_close(model.edit_mask, masks.condition)

    def test_cfg_treats_high_frequency_map_as_target_garment_condition(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        person_context = torch.zeros(1, 4, 8, 8)
        raw_mask = torch.ones(1, 1, 8, 8)
        high_frequency = torch.ones(1, 1, 64, 64)
        model = RecordingVelocityModel()
        flow.generate(
            model,
            torch.randn_like(person_context),
            person_context,
            raw_mask,
            garment=torch.ones_like(person_context),
            garment_high_frequency=high_frequency,
            cfg_scale=1.5,
            num_steps=1,
        )
        unconditional, conditional = model.garment_high_frequency.chunk(2)
        self.assertFalse(unconditional.any())
        torch.testing.assert_close(conditional, high_frequency)

    def test_sampler_preserves_latent_outside_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        agnostic = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(agnostic)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        output = flow.generate(
            ConstantVelocityModel(),
            noise,
            agnostic,
            mask,
            garment=torch.zeros_like(agnostic),
            num_steps=2,
        )
        latent_mask = prepare_vton_masks(mask, (8, 8), patch_size=2).latent.bool()
        torch.testing.assert_close(output.masked_select(~latent_mask), agnostic.masked_select(~latent_mask))

    def test_adaptive_sampler_preserves_latent_outside_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        agnostic = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        output = flow.generate(
            ConstantVelocityModel(),
            torch.randn_like(agnostic),
            agnostic,
            mask,
            garment=torch.zeros_like(agnostic),
            num_steps=2,
            adaptive=True,
            inner_steps=2,
        )
        latent_mask = prepare_vton_masks(mask, (8, 8), patch_size=2).latent.bool()
        torch.testing.assert_close(output.masked_select(~latent_mask), agnostic.masked_select(~latent_mask))

    def test_interpolants_use_clean_context_times(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        target = torch.randn(1, 4, 8, 8)
        agnostic = torch.randn_like(target)
        noise = torch.randn_like(target)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        sampled_time = torch.full((1, 16), 0.25)
        xt, _, effective_time, masks = flow.get_interpolants(target, agnostic, mask, x0=noise, t=sampled_time)
        self.assertTrue(torch.all(effective_time.masked_select(~masks.token) == 1))
        torch.testing.assert_close(
            xt.masked_select(~masks.latent.bool()),
            agnostic.masked_select(~masks.latent.bool()),
        )

    def test_zero_time_forcing_uses_pure_noise_in_edit_region(self):
        flow = VTONPatchFlowForcing(patch_size=2, zero_time_probability=1.0)
        target = torch.randn(2, 4, 8, 8)
        context = torch.randn_like(target)
        noise = torch.randn_like(target)
        mask = torch.zeros(2, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        xt, _, timesteps, masks = flow.get_interpolants(target, context, mask, x0=noise)
        self.assertTrue(torch.all(timesteps.masked_select(masks.token) == 0))
        torch.testing.assert_close(xt.masked_select(masks.latent.bool()), noise.masked_select(masks.latent.bool()))

    @staticmethod
    def _ltg_sampler(loc, std):
        return {
            "target": "patch_flow.timestep_schedules.LogitNormalTruncatedGaussian",
            "params": {"loc": loc, "std": std, "scale": 1.0},
        }

    def test_high_time_probability_covers_the_detail_regime(self):
        torch.manual_seed(0)
        # The shipped-before configuration: LTG ceiling sigma(0.7+z) plus 30% zero-time.
        starved = VTONPatchFlowForcing(
            patch_size=2,
            timestep_sampler=self._ltg_sampler(0.7, 0.6),
            zero_time_probability=0.3,
        )
        balanced = VTONPatchFlowForcing(
            patch_size=2,
            timestep_sampler=self._ltg_sampler(0.5, 0.25),
            zero_time_probability=0.1,
            high_time_probability=0.25,
            high_time_spread=0.25,
        )
        starved_times = starved._sample_token_times(2048, 64, "cpu", torch.float32)
        balanced_times = balanced._sample_token_times(2048, 64, "cpu", torch.float32)
        starved_detail = (starved_times > 0.9).float().mean().item()
        balanced_detail = (balanced_times > 0.9).float().mean().item()
        # The LTG ceiling alone leaves t>0.9 almost untrained, which is where logo and
        # printed-text structure is written.
        self.assertLess(starved_detail, 0.01)
        self.assertGreater(balanced_detail, 0.05)
        # Garment forcing must survive the rebalance.
        self.assertGreater((balanced_times == 0).float().mean().item(), 0.05)

    def test_high_times_stay_in_range(self):
        torch.manual_seed(0)
        flow = VTONPatchFlowForcing(patch_size=2, high_time_probability=1.0)
        times = flow._sample_token_times(64, 64, "cpu", torch.float32)
        self.assertGreaterEqual(times.min().item(), 0.0)
        self.assertLessEqual(times.max().item(), 1.0)

    def test_rgb_composite_is_exact_outside_mask(self):
        generated = torch.ones(1, 3, 16, 16)
        person = -torch.ones_like(generated)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1
        output = compose_vton(generated, person, mask, feather_radius=2)
        torch.testing.assert_close(output.masked_select(~mask.bool()), person.masked_select(~mask.bool()))


if __name__ == "__main__":
    unittest.main()
