"""Shared differentiable preprocessing and the L-infinity projector.

Every attack must take gradients through the *same* preprocessing and use the
same projection, so that a new attack cannot silently disagree with the one it
is compared against. ``attacks/*`` modules import from here instead of
re-deriving resize/crop/normalize inline.

Fidelity to the official evaluator
----------------------------------
The evaluator preprocesses with PIL on uint8 images
(:func:`evaluate.build_spatial_transform`, :func:`evaluate.build_dct_transform`):
``T.Resize`` for the spatial detectors and ``Image.Resampling.LANCZOS`` for the
DCT detector, both applied before ``ToTensor``. Those operations are not
differentiable with respect to the attacked pixels, so the functions here are
tensor-space surrogates evaluated on float images in ``[0, 1]``:

* the spatial path uses antialiased bilinear interpolation instead of PIL's
  resize, and
* the DCT path uses antialiased bicubic interpolation instead of LANCZOS.

The surrogate is close but *not* bit-identical to the evaluation transform.
Gradients are therefore taken through an approximation of the scored function.
This is the same trade-off the original inline I-FGSM implementation made; the
module only makes it explicit and shared. All reported metrics are still
computed by the runner against the real evaluator transforms, so the surrogate
affects only how perturbations are searched, never how they are scored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


# ImageNet statistics used by evaluate.build_spatial_transform.
SPATIAL_MEAN = (0.485, 0.456, 0.406)
SPATIAL_STD = (0.229, 0.224, 0.225)

# ITU-R 601-2 luma coefficients, matching PIL's Image.convert("L").
LUMA_WEIGHTS = (299.0 / 1000.0, 587.0 / 1000.0, 114.0 / 1000.0)

SPATIAL_RESIZE = 256
VIT_CROP = 224
DCT_RESIZE = 256
DCT_CROP = 128

SUPPORTED_SOURCE_MODELS = ("vit_b_16", "densenet121_dct")


def from_uint8_image(image: np.ndarray, device: Any) -> torch.Tensor:
    """Convert an HWC uint8 RGB array into a 1x3xHxW float tensor in [0, 1]."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image array")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return tensor.unsqueeze(0).float().to(device) / 255.0


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """Convert a 1x3xHxW float tensor in [0, 1] back to an HWC uint8 array."""

    if x.dim() != 4 or x.shape[0] != 1 or x.shape[1] != 3:
        raise ValueError("Expected a 1x3xHxW tensor")
    scaled = x[0].permute(1, 2, 0).detach() * 255.0
    return scaled.round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def project_linf(
    adversarial: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    pixel_min: float = 0.0,
    pixel_max: float = 1.0,
) -> torch.Tensor:
    """Clip the perturbation to the L-infinity ball and the valid pixel range."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if adversarial.shape != original.shape:
        raise ValueError("adversarial and original must have the same shape")
    perturbation = torch.clamp(adversarial - original, -epsilon, epsilon)
    return torch.clamp(original + perturbation, pixel_min, pixel_max)


def normalize_spatial(x: torch.Tensor) -> torch.Tensor:
    return TF.normalize(x, mean=list(SPATIAL_MEAN), std=list(SPATIAL_STD))


def differentiable_spatial(x: torch.Tensor, model_name: str) -> torch.Tensor:
    """Differentiable surrogate of evaluate.build_spatial_transform(model_name)."""

    resized = TF.resize(x, [SPATIAL_RESIZE, SPATIAL_RESIZE], antialias=True)
    if model_name == "vit_b_16":
        resized = TF.center_crop(resized, [VIT_CROP, VIT_CROP])
    return normalize_spatial(resized)


def to_luma(x: torch.Tensor) -> torch.Tensor:
    """Differentiable RGB to greyscale using PIL's ITU-R 601-2 coefficients."""

    weights = torch.tensor(LUMA_WEIGHTS, dtype=x.dtype, device=x.device)
    return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def dct_matrix(size: int, dtype: torch.dtype, device: Any) -> torch.Tensor:
    """Orthonormal DCT-II matrix matching scipy.fft.dct(..., norm='ortho')."""

    indices = torch.arange(size, dtype=torch.float64, device=device)
    angles = torch.pi * (2.0 * indices.view(1, -1) + 1.0) * indices.view(-1, 1)
    matrix = torch.cos(angles / (2.0 * size)) * np.sqrt(2.0 / size)
    matrix[0] = matrix[0] / np.sqrt(2.0)
    return matrix.to(dtype)


def differentiable_dct(x: torch.Tensor, log_scale: bool = True) -> torch.Tensor:
    """Differentiable surrogate of evaluate.build_dct_transform(log_scale).

    Greyscale, resize to 256, centre-crop to 128, 2-D orthonormal DCT-II, then
    the optional log magnitude. The result is a 1x1x128x128 tensor scaled like
    the evaluator's, which reads pixels in [0, 255].
    """

    grey = to_luma(x) * 255.0
    if grey.shape[-1] > DCT_RESIZE or grey.shape[-2] > DCT_RESIZE:
        grey = F.interpolate(
            grey,
            size=(DCT_RESIZE, DCT_RESIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    grey = TF.center_crop(grey, [DCT_CROP, DCT_CROP])
    basis = dct_matrix(DCT_CROP, grey.dtype, grey.device)
    # Separable 2-D DCT: transform the rows, then the columns.
    coefficients = basis @ grey @ basis.transpose(0, 1)
    if log_scale:
        coefficients = torch.log(coefficients.abs() + 1e-6)
    return coefficients


def preprocess_for(model_name: str, x: torch.Tensor, log_scale: bool = True) -> torch.Tensor:
    """Dispatch to the differentiable preprocessing of one detector."""

    if model_name == "densenet121_dct":
        return differentiable_dct(x, log_scale=log_scale)
    if model_name not in SUPPORTED_SOURCE_MODELS:
        raise ValueError(
            f"Unsupported source model {model_name!r}; "
            f"expected one of {SUPPORTED_SOURCE_MODELS}"
        )
    return differentiable_spatial(x, model_name)
