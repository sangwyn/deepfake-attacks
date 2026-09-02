import numpy as np
import torch

from attacks.vit_pgd import attack


class ToyViT(torch.nn.Module):
    def forward(self, x):
        m = x.mean((1, 2, 3))
        return torch.stack((m, -m), dim=1)


def test_vit_only_output_and_budget():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    classifiers = {"vit_b_16": {"model": ToyViT().eval()}}
    result = attack(image, classifiers, torch.device("cpu"),
                    epsilon=8 / 255, step_size=1 / 255, iterations=2)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert np.max(np.abs(result.astype(int) - image.astype(int))) <= 8
