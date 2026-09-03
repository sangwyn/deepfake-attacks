import numpy as np
import torch

from attacks.frequency_pgd import _filter_gradient, attack


class ToyModel(torch.nn.Module):
    def forward(self, x):
        m = x.mean((1, 2, 3))
        return torch.stack((m, -m), dim=1)


def test_band_filter_is_differentiable_and_budgeted():
    gradient = torch.rand(1, 3, 32, 32, requires_grad=True)
    filtered = _filter_gradient(gradient, "low", 4, 12)
    filtered.square().mean().backward()
    assert gradient.grad is not None
    assert torch.isfinite(gradient.grad).all()
    assert filtered.shape == gradient.shape


def test_frequency_attack_output_and_budget():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    classifiers = {
        "vit_b_16": {"model": ToyModel().eval()},
        "densenet121_dct": {"model": ToyModel().eval()},
    }
    result = attack(image, classifiers, torch.device("cpu"),
                    iterations=2, step_size=1 / 255, band="low")
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.max(np.abs(result.astype(int) - image.astype(int))) <= 8
