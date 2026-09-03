import numpy as np
import torch

from attacks.dct_pgd import attack


class ToyDCT(torch.nn.Module):
    def forward(self, x):
        return torch.cat((x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                          -x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)), dim=1)


def test_dct_only_output_and_budget():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    classifiers = {"densenet121_dct": {"model": ToyDCT().eval()}}
    result = attack(image, classifiers, torch.device("cpu"),
                    epsilon=8 / 255, step_size=1 / 255, iterations=2)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert np.max(np.abs(result.astype(int) - image.astype(int))) <= 8
