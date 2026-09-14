import pytest
import torch

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
