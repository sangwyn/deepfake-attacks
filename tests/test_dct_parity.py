import numpy as np
import torch
from scipy.fftpack import dct as scipy_dct

from attacks.dual_pgd import dct_preprocess, torch_dct2


def test_dct_matches_scipy():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1, 1, 128, 128)).astype(np.float32)
    actual = torch_dct2(torch.from_numpy(values)).numpy()[0, 0]
    expected = scipy_dct(scipy_dct(values[0, 0], axis=0, norm="ortho"),
                         axis=1, norm="ortho")
    assert np.max(np.abs(actual - expected)) < 1e-4
    assert np.mean(np.abs(actual - expected)) < 1e-5


def test_dct_preprocess_has_valid_gradient():
    image = torch.rand(1, 3, 64, 80, requires_grad=True)
    output = dct_preprocess(image)
    assert output.shape == (1, 1, 128, 128)
    output.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
