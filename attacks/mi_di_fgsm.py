"""Momentum Diverse-Input FGSM with optional expectation over transforms."""

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


def _random_value(generator, device):
    return float(torch.rand((), generator=generator, device=device))


def _diverse_input(image, probability, min_scale, generator):
    if probability == 0 or _random_value(generator, image.device) >= probability:
        return image
    height, width = image.shape[-2:]
    scale = min_scale + (1.0 - min_scale) * _random_value(generator, image.device)
    resized_height = max(2, round(height * scale))
    resized_width = max(2, round(width * scale))
    resized = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    pad_height = height - resized_height
    pad_width = width - resized_width
    top = int(
        torch.randint(pad_height + 1, (), generator=generator, device=image.device)
    )
    left = int(
        torch.randint(pad_width + 1, (), generator=generator, device=image.device)
    )
    return F.pad(
        resized,
        (left, pad_width - left, top, pad_height - top),
        mode="constant",
        value=0.0,
    )


def _eot_view(image, transform):
    height, width = image.shape[-2:]
    if transform == "identity":
        return image
    if transform == "resize":
        small = F.interpolate(
            image,
            size=(max(2, round(height * 0.9)), max(2, round(width * 0.9))),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return F.interpolate(
            small,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    if transform == "crop":
        margin_h = max(1, height // 20)
        margin_w = max(1, width // 20)
        cropped = image[..., margin_h:-margin_h, margin_w:-margin_w]
        return F.interpolate(
            cropped,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    if transform == "jpeg_like":
        quantized = torch.round(image * 63.0) / 63.0
        return image + (quantized - image).detach()
    raise ValueError(f"unsupported EoT transform: {transform}")


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    step_size=None,
    iterations=20,
    momentum=1.0,
    diversity_probability=0.7,
    min_resize_fraction=0.9,
    eot_samples=1,
    eot_transforms=("identity",),
    objective="targeted_fake_to_real",
    label=None,
    source_weights=None,
    seed=0,
):
    if step_size is None:
        step_size = float(epsilon) / iterations
    validate_steps(epsilon, step_size, iterations)
    if momentum < 0:
        raise ValueError("momentum must be non-negative")
    if not 0 <= diversity_probability <= 1:
        raise ValueError("diversity_probability must be in [0, 1]")
    if not 0 < min_resize_fraction <= 1:
        raise ValueError("min_resize_fraction must be in (0, 1]")
    if not isinstance(eot_samples, int) or eot_samples < 1:
        raise ValueError("eot_samples must be a positive integer")
    if not eot_transforms:
        raise ValueError("eot_transforms must not be empty")

    original = image_to_tensor(image, device)
    attacked = original.clone()
    velocity = torch.zeros_like(attacked)
    generator = make_generator(device, seed)
    for iteration in range(iterations):
        attacked.requires_grad_(True)
        loss = attacked.new_zeros(())
        for sample in range(eot_samples):
            view = _diverse_input(
                attacked,
                diversity_probability,
                min_resize_fraction,
                generator,
            )
            transform = eot_transforms[
                (iteration * eot_samples + sample) % len(eot_transforms)
            ]
            loss = loss + ensemble_loss(
                _eot_view(view, transform),
                classifiers,
                objective,
                label,
                source_weights,
            )
        gradient = checked_gradient(loss / eot_samples, attacked, "MI-DI-FGSM")
        normalized = gradient / gradient.abs().mean(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(1e-12)
        velocity = float(momentum) * velocity + normalized
        attacked = project_linf(
            attacked - float(step_size) * velocity.sign(), original, epsilon
        ).detach()
    return tensor_to_image(attacked)
