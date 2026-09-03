"""Ensemble PGD regularized toward a fixed camera-like noise prior."""

import torch
import torch.nn.functional as F

from ._utils import (
    checked_gradient,
    ensemble_loss,
    image_to_tensor,
    make_generator,
    project_linf,
    tensor_to_image,
    validate_steps,
)


def _noise_allocation(image, texture_mask, mask_floor):
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    luminance = (image * weights).sum(dim=1, keepdim=True)
    if not texture_mask:
        return torch.ones_like(luminance), luminance
    local_mean = F.avg_pool2d(
        F.pad(luminance, (1, 1, 1, 1), mode="reflect"), 3, stride=1
    )
    detail = (luminance - local_mean).abs()
    scale = torch.quantile(detail.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    allocation = mask_floor + (1.0 - mask_floor) * detail / scale.clamp_min(1e-6)
    return allocation.clamp(mask_floor, 1.0), luminance


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    step_size=0.5 / 255,
    iterations=40,
    isp_prior_weight=0.05,
    isp_prior_scale=1.0,
    isp_texture_mask=True,
    isp_mask_floor=0.25,
    shot_noise=0.02,
    read_noise=0.005,
    objective="targeted_fake_to_real",
    label=None,
    source_weights=None,
    seed=0,
):
    validate_steps(epsilon, step_size, iterations)
    if isp_prior_weight < 0 or isp_prior_scale < 0:
        raise ValueError("ISP prior parameters must be non-negative")
    if not 0 < isp_mask_floor <= 1:
        raise ValueError("isp_mask_floor must be in (0, 1]")
    if shot_noise < 0 or read_noise < 0:
        raise ValueError("noise scales must be non-negative")

    original = image_to_tensor(image, device)
    allocation, luminance = _noise_allocation(
        original, bool(isp_texture_mask), float(isp_mask_floor)
    )
    generator = make_generator(device, seed)
    noise = torch.randn(
        original.shape,
        generator=generator,
        device=original.device,
        dtype=original.dtype,
    )
    camera_std = allocation * torch.sqrt(
        float(shot_noise) ** 2 * luminance + float(read_noise) ** 2
    )
    prior_delta = float(isp_prior_scale) * noise * camera_std

    attacked = original.clone()
    for _ in range(iterations):
        attacked.requires_grad_(True)
        classification = ensemble_loss(
            attacked, classifiers, objective, label, source_weights
        )
        prior = (
            ((attacked - original - prior_delta) / max(float(epsilon), 1e-12))
            .square()
            .mean()
        )
        gradient = checked_gradient(
            classification + float(isp_prior_weight) * prior,
            attacked,
            "ISP-PGD",
        )
        attacked = project_linf(
            attacked - float(step_size) * gradient.sign(), original, epsilon
        ).detach()
    return tensor_to_image(attacked)
