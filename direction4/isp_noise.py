"""Optimization-free Poisson-Gaussian camera-statistics perturbation."""

import numpy as np


def _texture_mask(rgb: np.ndarray, blur_radius: int = 3,
                  floor: float = 0.25) -> np.ndarray:
    """Return a smooth [floor, 1] allocation map from local luminance detail."""
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]).astype(np.float32) / 255.0
    padded = np.pad(gray, blur_radius, mode="reflect")
    size = 2 * blur_radius + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (size, size)
    )
    local_mean = windows.mean(axis=(-2, -1))
    detail = np.abs(gray - local_mean)
    scale = np.percentile(detail, 95)
    normalized = detail / max(float(scale), 1e-6)
    return np.clip(floor + (1.0 - floor) * normalized, floor, 1.0)


def attack(image, classifiers, device, epsilon=8 / 255,
           poisson_sigma=0.02, gaussian_sigma=0.005,
           texture_mask=False, mask_floor=0.25, seed=0):
    """Add bounded signal-dependent and signal-independent sensor noise."""
    del classifiers, device
    if epsilon < 0 or poisson_sigma < 0 or gaussian_sigma < 0:
        raise ValueError("epsilon and noise scales must be non-negative")
    if not 0 < mask_floor <= 1:
        raise ValueError("mask_floor must be in (0, 1]")

    original = image.astype(np.float32) / 255.0
    luminance = (0.299 * original[..., 0] + 0.587 * original[..., 1]
                 + 0.114 * original[..., 2])
    allocation = (_texture_mask(image, floor=mask_floor)
                  if texture_mask else np.ones_like(luminance))
    rng = np.random.default_rng(seed)
    std = allocation[..., None] * np.sqrt(
        poisson_sigma ** 2 * luminance[..., None]
        + gaussian_sigma ** 2
    )
    noise = rng.normal(0.0, 1.0, original.shape).astype(np.float32) * std
    perturbation = np.clip(noise, -epsilon, epsilon)
    attacked = np.clip(original + perturbation, 0.0, 1.0)
    return np.rint(attacked * 255.0).astype(np.uint8)
