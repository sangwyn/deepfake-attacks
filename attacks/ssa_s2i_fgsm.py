"""Targeted Spectrum Simulation Attack (S^2 I-FGSM).

The spectrum simulation is differentiable and is applied to the RGB pixel
tensor before the shared detector preprocessing.  The attack therefore keeps
the same pixel-space constraint as the other attack modules.
"""

import torch
import torch.nn.functional as F

from attacklab.preprocessing import (
    dct_matrix,
    from_uint8_image,
    preprocess_for,
    project_linf,
    to_uint8_image,
)


ATTACK_CONTRACT = {
    "version": 1,
    "source_model": "densenet121_dct",
    "supported_source_models": ["vit_b_16", "densenet121_dct"],
    "description": "Targeted spectrum simulation S-squared I-FGSM.",
}


def _dct2(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal 2-D DCT independently to each image channel."""
    height, width = x.shape[-2:]
    row = dct_matrix(height, x.dtype, x.device)
    column = dct_matrix(width, x.dtype, x.device)
    return torch.einsum("ij,ncjk,lk->ncil", row, x, column)


def _idct2(coefficients: torch.Tensor) -> torch.Tensor:
    """Invert :func:`_dct2` exactly in floating point."""
    height, width = coefficients.shape[-2:]
    row = dct_matrix(height, coefficients.dtype, coefficients.device)
    column = dct_matrix(width, coefficients.dtype, coefficients.device)
    return torch.einsum("ji,ncjk,kl->ncil", row, coefficients, column)


def _spectrum_transform(
    x: torch.Tensor,
    spectrum_noise_sigma: float,
    spectrum_amplitude_rho: float,
    spectrum_mask: str = "random",
) -> torch.Tensor:
    """Randomly perturb DCT coefficients while preserving the image shape.

    ``identity`` is intentionally exposed for the ablation required by the
    specification.  Randomness uses PyTorch's caller-provided RNG state.
    """
    if spectrum_mask == "identity" or (
        spectrum_noise_sigma == 0 and spectrum_amplitude_rho == 0
    ):
        return x
    if spectrum_mask != "random":
        raise ValueError("spectrum_mask must be 'random' or 'identity'")
    coefficients = _dct2(x)
    amplitude = torch.empty_like(coefficients).uniform_(
        1.0 - spectrum_amplitude_rho, 1.0 + spectrum_amplitude_rho
    )
    noise = torch.randn_like(coefficients) * spectrum_noise_sigma
    return _idct2(coefficients * (amplitude + noise))


def attack(
    image,
    classifiers,
    device,
    epsilon=8 / 255,
    step_size=None,
    iterations=10,
    momentum=1.0,
    frequency_samples=20,
    spectrum_noise_sigma=16 / 255,
    spectrum_amplitude_rho=0.5,
    transform="orthonormal-dct",
    spectrum_mask="random",
    source_model="vit_b_16",
    target_class=0,
):
    if source_model not in ATTACK_CONTRACT["supported_source_models"]:
        raise ValueError(f"ssa_s2i_fgsm does not support source_model={source_model!r}")
    if source_model not in classifiers:
        raise ValueError("configured classifiers do not contain source_model")
    if target_class not in {0, 1} or epsilon <= 0 or iterations < 1:
        raise ValueError("invalid target_class, epsilon, or iterations")
    if step_size is None:
        step_size = epsilon / iterations
    if step_size <= 0 or momentum < 0 or frequency_samples < 1:
        raise ValueError("step_size, momentum, and frequency_samples must be positive")
    if spectrum_noise_sigma < 0 or not 0 <= spectrum_amplitude_rho <= 1:
        raise ValueError("spectrum noise must be non-negative and amplitude rho in [0, 1]")
    if transform != "orthonormal-dct":
        raise ValueError("only orthonormal-dct is supported")

    model = classifiers[source_model]["model"]
    original = from_uint8_image(image, device)
    attacked = original.clone()
    accumulated = torch.zeros_like(original)
    target = torch.tensor([target_class], device=device)
    for _ in range(iterations):
        attacked.requires_grad_()
        loss = torch.zeros((), device=device)
        for _ in range(frequency_samples):
            transformed = _spectrum_transform(
                attacked, spectrum_noise_sigma, spectrum_amplitude_rho, spectrum_mask
            )
            loss = loss + F.cross_entropy(
                model(preprocess_for(source_model, transformed)), target
            )
        gradient = torch.autograd.grad(loss / frequency_samples, attacked)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("SSA encountered a non-finite gradient")
        normalized = gradient / (gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        accumulated = momentum * accumulated + normalized
        attacked = project_linf(
            attacked - step_size * accumulated.sign(), original, epsilon
        ).detach()
    return to_uint8_image(attacked)
