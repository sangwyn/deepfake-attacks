"""Visibility masks and the chromatic subspace, at the canonical 256x256 grid.

Geometry (derived from evaluate.py preprocessing):
  ViT sees the central 224x224 of the 256 grid  (indices 16..239).
  DCT sees the central 128x128                  (indices 64..191), grayscale only.
Masks are 1.0 where perturbation is ALLOWED, 0.0 where it is forbidden.
"""
import torch

CANON = 256
VIT_CROP = 224
DCT_CROP = 128


def _center_band(crop: int, canon: int = CANON) -> tuple[int, int]:
    off = (canon - crop) // 2
    return off, off + crop


def region_masks(canon: int = CANON, device=None) -> dict:
    """Return boolean/0-1 masks over the (canon,canon) spatial grid."""
    y, x = torch.meshgrid(
        torch.arange(canon, device=device),
        torch.arange(canon, device=device),
        indexing="ij",
    )
    v0, v1 = _center_band(VIT_CROP, canon)
    d0, d1 = _center_band(DCT_CROP, canon)
    vit_sees = (y >= v0) & (y < v1) & (x >= v0) & (x < v1)
    dct_sees = (y >= d0) & (y < d1) & (x >= d0) & (x < d1)

    border = (~vit_sees).float()            # neither model sees -> forbid
    annulus = (vit_sees & ~dct_sees).float()  # ViT only
    center = dct_sees.float()               # both (shared battleground)
    return {
        "border": border,       # 1 where perturbation is wasted (forbidden)
        "annulus": annulus,     # 1 where only ViT sees
        "center": center,       # 1 where both see
        "vit_sees": vit_sees,
        "dct_sees": dct_sees,
    }


def allowed_mask(use_border_mask: bool = True, device=None) -> torch.Tensor:
    """1.0 where perturbation is allowed; 0.0 on the blind border."""
    m = region_masks(device=device)
    allowed = torch.ones_like(m["border"])
    if use_border_mask:
        allowed = allowed * (1.0 - m["border"])
    return allowed.view(1, 1, CANON, CANON)


def project_chroma(delta: torch.Tensor) -> torch.Tensor:
    """Project an RGB perturbation onto the DCT-blind chromatic plane
    (0.299 dR + 0.587 dG + 0.114 dB = 0). delta: (N,3,H,W)."""
    w = torch.tensor([0.299, 0.587, 0.114], device=delta.device, dtype=delta.dtype)
    w = w.view(1, 3, 1, 1)
    luma_proj = (delta * w).sum(dim=1, keepdim=True) * w / (w * w).sum()
    return delta - luma_proj
