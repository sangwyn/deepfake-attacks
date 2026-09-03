import numpy as np
import torch

from direction4.isp_joint_pgd import _allocation, attack


class ToyModel(torch.nn.Module):
    def forward(self, x):
        value = x.mean((1, 2, 3))
        return torch.stack((value, -value), dim=1)


def test_allocation_is_finite_and_bounded():
    image = torch.full((1, 3, 32, 32), 0.5)
    allocation, luminance = _allocation(image, True, 0.25)
    assert allocation.shape == luminance.shape == (1, 1, 32, 32)
    assert torch.isfinite(allocation).all()
    assert float(allocation.min()) >= 0.25
    assert float(allocation.max()) <= 1.0


def test_joint_isp_output_is_deterministic_and_budgeted():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    classifiers = {"vit_b_16": {"model": ToyModel().eval()},
                   "densenet121_dct": {"model": ToyModel().eval()}}
    first = attack(image, classifiers, torch.device("cpu"), iterations=2, seed=3)
    second = attack(image, classifiers, torch.device("cpu"), iterations=2, seed=3)
    assert np.array_equal(first, second)
    assert np.max(np.abs(first.astype(int) - image.astype(int))) <= 8
