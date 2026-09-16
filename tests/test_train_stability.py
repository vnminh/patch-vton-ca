import pytest
import torch

from patch_flow.trainer_vton import LatentVTONPatchForcingTrainer
from train import clip_grad_norm_finite


class _AcceleratorStub:
    def __init__(self):
        self.unscaled = False

    def unscale_gradients(self, optimizer):
        self.unscaled = True


def test_finite_gradient_guard_clips_norm_without_changing_direction():
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    module.weight.grad = torch.tensor([[3., 4.]])
    accelerator = _AcceleratorStub()
    norm = clip_grad_norm_finite(accelerator, module, optimizer, max_norm=1., step=7)
    assert accelerator.unscaled
    torch.testing.assert_close(norm, torch.tensor(5.))
    torch.testing.assert_close(module.weight.grad.norm(), torch.tensor(1.))


def test_nonfinite_gradient_guard_cancels_step_and_reports_parameter():
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    original = module.weight.detach().clone()
    module.weight.grad = torch.tensor([[float("nan"), 1.]])
    with pytest.raises(FloatingPointError, match=r"optimizer step 716.*weight"):
        clip_grad_norm_finite(
            _AcceleratorStub(), module, optimizer, max_norm=1., step=716
        )
    assert module.weight.grad is None
    torch.testing.assert_close(module.weight, original)


def _step_ramp_trainer():
    trainer = object.__new__(LatentVTONPatchForcingTrainer)
    trainer.__dict__["_trainer"] = None
    trainer.__dict__["_optimizer_steps"] = 0
    trainer.__dict__["correspondence_warmup_steps"] = 250
    trainer.__dict__["fine_teacher_forcing_start"] = 0.75
    trainer.__dict__["fine_teacher_forcing_steps"] = 2000
    return trainer


def test_a_process_restart_that_does_not_seed_optimizer_steps_replays_full_ramps():
    """Guards the bug train.py's ``module._optimizer_steps = global_step`` fixes.

    ``_training_step_count`` reads the private ``_optimizer_steps`` counter, not
    Lightning's ``self.global_step`` -- this script never calls ``Trainer.fit``, so
    ``self._trainer`` is always None. A restart that forgets to seed this counter
    silently restarts the 250-step correspondence warmup and the 2000-step
    teacher-forcing decay from zero even when resuming step 1350 of a continuous run.
    """
    trainer = _step_ramp_trainer()
    assert trainer._correspondence_ramp() == 0.0
    assert trainer._fine_teacher_forcing_ratio() == pytest.approx(0.75)


def test_seeding_optimizer_steps_on_resume_restores_both_ramps():
    trainer = _step_ramp_trainer()
    trainer._optimizer_steps = 1350
    assert trainer._correspondence_ramp() == 1.0
    assert trainer._fine_teacher_forcing_ratio() == pytest.approx(0.75 * (1 - 1350 / 2000))
    trainer._optimizer_steps = 5000
    assert trainer._fine_teacher_forcing_ratio() == 0.0
