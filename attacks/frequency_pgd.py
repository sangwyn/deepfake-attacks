"""PGD with an explicit frequency-band constraint on the RGB gradient."""

import torch
import torch.nn.functional as F

from ._utils import (
    checked_gradient,
    ensemble_loss,
    image_to_tensor,
    project_linf,
    tensor_to_image,
    validate_steps,
)


_DCT_CACHE = {}


def _dct_matrix(size, device, dtype):
    key = (size, device.type, device.index, dtype)
    if key not in _DCT_CACHE:
        frequency = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
        position = torch.arange(size, device=device, dtype=dtype).unsqueeze(0) + 0.5
        matrix = torch.cos(torch.pi * frequency * position / size)
        matrix[0] *= (1.0 / size) ** 0.5
        matrix[1:] *= (2.0 / size) ** 0.5
        _DCT_CACHE[key] = matrix
    return _DCT_CACHE[key]


def _dct2(image):
    row = _dct_matrix(image.shape[-2], image.device, image.dtype)
    column = _dct_matrix(image.shape[-1], image.device, image.dtype)
    return row @ image @ column.t()


def _idct2(coefficients):
    row = _dct_matrix(coefficients.shape[-2], coefficients.device, coefficients.dtype)
    column = _dct_matrix(
        coefficients.shape[-1], coefficients.device, coefficients.dtype
    )
    return row.t() @ coefficients @ column


def filter_gradient(gradient, band, low_cutoff=0.15, high_cutoff=0.45, max_size=256):
    if band == "full":
        return gradient
    if band not in {"low", "mid", "high"}:
        raise ValueError("band must be full, low, mid, or high")
    if not 0 <= low_cutoff < high_cutoff <= 1:
        raise ValueError("frequency cutoffs must satisfy 0 <= low < high <= 1")
    height, width = gradient.shape[-2:]
    work_height = min(height, max_size)
    work_width = min(width, max_size)
    work = gradient
    if (height, width) != (work_height, work_width):
        work = F.interpolate(
            work,
            size=(work_height, work_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    row = torch.linspace(0, 1, work_height, device=work.device)
    column = torch.linspace(0, 1, work_width, device=work.device)
    radius = (row[:, None] + column[None, :]) / 2
    if band == "low":
        mask = radius <= low_cutoff
    elif band == "mid":
        mask = (radius > low_cutoff) & (radius <= high_cutoff)
    else:
        mask = radius > high_cutoff
    filtered = _idct2(_dct2(work) * mask.to(work.dtype))
    if filtered.shape[-2:] != (height, width):
        filtered = F.interpolate(
            filtered,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return filtered


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    step_size=0.5 / 255,
    iterations=40,
    band="full",
    low_cutoff=0.15,
    high_cutoff=0.45,
    objective="targeted_fake_to_real",
    label=None,
    source_weights=None,
    seed=0,
):
    del seed
    validate_steps(epsilon, step_size, iterations)
    original = image_to_tensor(image, device)
    attacked = original.clone()
    for _ in range(iterations):
        attacked.requires_grad_(True)
        loss = ensemble_loss(attacked, classifiers, objective, label, source_weights)
        gradient = checked_gradient(loss, attacked, "frequency PGD")
        gradient = filter_gradient(gradient, band, low_cutoff, high_cutoff)
        if gradient.abs().sum() == 0:
            raise RuntimeError(f"frequency band {band!r} removed the entire gradient")
        attacked = project_linf(
            attacked - float(step_size) * gradient.sign(), original, epsilon
        ).detach()
    return tensor_to_image(attacked)
