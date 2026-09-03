"""
Shared building blocks for attacks: image<->tensor conversion, differentiable
preprocessing that mirrors evaluate.py's (non-differentiable) transforms, and
transform tricks for transfer attacks (input diversity, differentiable JPEG).

Keep attacks in Torch; keep PIL/NumPy/file encoding in evaluate.py.

NOTE: the Torch preprocessing here is a *surrogate* of evaluate.py — resize is
bilinear+antialias (not PIL LANCZOS) and grayscale skips PIL's rounding. This
only affects gradient quality; evaluate.py always re-scores with its own
transforms on the final pixels.
"""

import torch
import torch.nn.functional as F

# ImageNet stats (same as evaluate.build_spatial_transform)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# image <-> tensor
# ---------------------------------------------------------------------------
def to_tensor(image, device):
    """H×W×3 uint8 RGB numpy -> 1×3×H×W float in [0,1] on `device`."""
    t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return t.float().to(device) / 255.0


def to_numpy(x):
    """1×3×H×W float in [0,1] -> H×W×3 uint8 RGB numpy (rounded)."""
    a = x[0].permute(1, 2, 0) * 255.0
    return a.round().to(torch.uint8).cpu().numpy()


# ---------------------------------------------------------------------------
# differentiable preprocessing (mirrors evaluate.py)
# ---------------------------------------------------------------------------
def spatial_preprocess(x, crop):
    """B×3×H×W in [0,1] -> resize 256 -> center-crop `crop` -> ImageNet norm."""
    x = F.interpolate(x, size=(256, 256), mode="bilinear",
                      align_corners=False, antialias=True)
    if crop != 256:
        top = (256 - crop) // 2
        x = x[:, :, top:top + crop, top:top + crop]
    mean = torch.tensor(_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def dct_matrix(n, device, dtype):
    """Orthonormal DCT-II matrix D (n×n), matching scipy dct(norm='ortho')."""
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
    m = torch.arange(n, device=device, dtype=dtype).view(1, n)
    d = torch.cos(torch.pi * (2 * m + 1) * k / (2 * n)) * (2.0 / n) ** 0.5
    d[0] = d[0] * (0.5 ** 0.5)  # row 0 uses sqrt(1/n) instead of sqrt(2/n)
    return d


def dct_preprocess(x, log_scale=True):
    """B×3×H×W in [0,1] -> grayscale -> resize 256 -> crop 128 -> 2-D DCT -> log.

    Mirrors evaluate.build_dct_transform, which runs on a 0-255 grayscale array,
    so we scale to 0-255 before the DCT to match coefficient magnitudes.
    """
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # ITU-R 601-2
    gray = gray.unsqueeze(1)
    gray = F.interpolate(gray, size=(256, 256), mode="bilinear",
                         align_corners=False, antialias=True)
    top = (256 - 128) // 2
    gray = gray[:, :, top:top + 128, top:top + 128] * 255.0
    d = dct_matrix(128, x.device, x.dtype)
    dct = d @ gray @ d.transpose(-1, -2)
    if log_scale:
        dct = torch.log(torch.abs(dct) + 1e-6)
    return dct


def build_preprocess(name, log_scale):
    """Differentiable preprocessing matching a classifier's name."""
    if name.endswith("_dct"):
        return lambda x: dct_preprocess(x, log_scale=log_scale)
    if name == "vit_b_16":
        return lambda x: spatial_preprocess(x, crop=224)
    return lambda x: spatial_preprocess(x, crop=256)  # other spatial models


# ---------------------------------------------------------------------------
# transfer tricks
# ---------------------------------------------------------------------------
def input_diversity(x, prob, pad_ratio):
    """DI-FGSM input diversity: with `prob`, random resize + random pad.

    x: B×C×H×W in [0,1]. Returns a (possibly larger, black-padded) tensor;
    downstream preprocessing resizes back to a fixed size, so the exact output
    size does not matter — only the induced diversity does.
    """
    if pad_ratio <= 0 or torch.rand(1).item() >= prob:
        return x
    _, _, h, w = x.shape
    hi_h = h + int(round(h * pad_ratio))
    hi_w = w + int(round(w * pad_ratio))
    rnd_h = int(torch.randint(h, hi_h + 1, (1,)).item())
    rnd_w = int(torch.randint(w, hi_w + 1, (1,)).item())
    x = F.interpolate(x, size=(rnd_h, rnd_w), mode="bilinear",
                      align_corners=False, antialias=True)
    pad_top = int(torch.randint(0, hi_h - rnd_h + 1, (1,)).item())
    pad_left = int(torch.randint(0, hi_w - rnd_w + 1, (1,)).item())
    return F.pad(x, [pad_left, hi_w - rnd_w - pad_left,
                     pad_top, hi_h - rnd_h - pad_top], value=0.0)


def jpeg_compress(x, quality):
    """Differentiable JPEG for EOT. Requires kornia (optional dependency)."""
    try:
        from kornia.enhance import jpeg_codec_differentiable
    except ImportError as e:
        raise ImportError(
            "jpeg EOT needs kornia — `pip install kornia` — or drop "
            "`jpeg_quality` from attack_params."
        ) from e
    q = torch.tensor([float(quality)], device=x.device, dtype=x.dtype)
    return jpeg_codec_differentiable(x, q).clamp(0.0, 1.0)
