"""Spectrum-simulation momentum attack (S²I-FGSM)."""

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
from .frequency_pgd import _dct2, _idct2


def _spectrum_transform(image, noise_sigma, amplitude_rho, generator, max_size):
    height, width = image.shape[-2:]
    work_size = (min(height, max_size), min(width, max_size))
    work = image
    if work.shape[-2:] != work_size:
        work = F.interpolate(
            work,
            size=work_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    coefficients = _dct2(work)
    amplitude = torch.rand(
        coefficients.shape,
        generator=generator,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    amplitude = 1.0 - amplitude_rho + 2.0 * amplitude_rho * amplitude
    noise = torch.randn(
        coefficients.shape,
        generator=generator,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    transformed = _idct2(coefficients * (amplitude + noise_sigma * noise))
    if transformed.shape[-2:] != (height, width):
        transformed = F.interpolate(
            transformed,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return transformed


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    step_size=None,
    iterations=10,
    momentum=1.0,
    frequency_samples=20,
    spectrum_noise_sigma=16 / 255,
    spectrum_amplitude_rho=0.5,
    transform_size=256,
    objective="targeted_fake_to_real",
    label=None,
    source_weights=None,
    seed=0,
):
    """Attack an ensemble by averaging gradients over DCT perturbations."""
    if step_size is None:
        step_size = float(epsilon) / iterations
    validate_steps(epsilon, step_size, iterations)
    if momentum < 0:
        raise ValueError("momentum must be non-negative")
    if not isinstance(frequency_samples, int) or frequency_samples < 1:
        raise ValueError("frequency_samples must be a positive integer")
    if spectrum_noise_sigma < 0:
        raise ValueError("spectrum_noise_sigma must be non-negative")
    if not 0 <= spectrum_amplitude_rho <= 1:
        raise ValueError("spectrum_amplitude_rho must be in [0, 1]")
    if not isinstance(transform_size, int) or transform_size < 2:
        raise ValueError("transform_size must be an integer of at least 2")

    original = image_to_tensor(image, device)
    attacked = original.clone()
    velocity = torch.zeros_like(attacked)
    generator = make_generator(device, seed)
    for _ in range(iterations):
        attacked.requires_grad_(True)
        loss = attacked.new_zeros(())
        for _ in range(frequency_samples):
            transformed = _spectrum_transform(
                attacked,
                float(spectrum_noise_sigma),
                float(spectrum_amplitude_rho),
                generator,
                transform_size,
            )
            loss = loss + ensemble_loss(
                transformed,
                classifiers,
                objective,
                label,
                source_weights,
            )
        gradient = checked_gradient(loss / frequency_samples, attacked, "S2I-FGSM")
        gradient = gradient / gradient.abs().mean(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(1e-12)
        velocity = float(momentum) * velocity + gradient
        attacked = project_linf(
            attacked - float(step_size) * velocity.sign(), original, epsilon
        ).detach()
    return tensor_to_image(attacked)
