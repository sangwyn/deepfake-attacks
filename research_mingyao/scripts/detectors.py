"""Differentiable detector wrappers replicating evaluate.py preprocessing.

Each branch takes a FULL-RESOLUTION adversarial image tensor (N,3,H,W) in
[0,1] and returns logits (N,2). Preprocessing order matches the official
evaluator so that clean-image predictions agree:

  ViT : resize 256 -> center crop 224 -> ImageNet normalize -> vit_b_16
  DCT : grayscale -> resize 256 -> center crop 128 -> DCT-II(ortho)
        -> log(|x|+1e-6) -> densenet121_dct (1-channel input)
"""
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models as tv
from torchvision.models import vit_b_16

from .dct_ops import dct_matrix, rgb_to_gray, stable_log_abs, lanczos_matrix

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 2


def create_dct_model() -> nn.Module:
    """DenseNet-121 with 1-channel input; must match evaluate.py exactly."""
    model = tv.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, NUM_CLASSES),
    )
    return model


def create_vit_model() -> nn.Module:
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model


class DCTBranch(nn.Module):
    """Full-res RGB (N,3,H,W) in [0,1] -> DCT-domain logits (N,2)."""

    def __init__(self, model: nn.Module, canon: int = 256, crop: int = 128,
                 log_eps: float = 1e-6, grad_clip: float | None = None):
        super().__init__()
        self.model = model
        self.canon = canon
        self.crop = crop
        self.log_eps = log_eps
        self.grad_clip = grad_clip
        self.offset = (canon - crop) // 2
        self.register_buffer("dct_m", dct_matrix(crop))
        self._lz_cache: dict = {}

    def _lanczos_resize_gray(self, g: torch.Tensor) -> torch.Tensor:
        """g: (N,1,H,W) grayscale -> (N,1,canon,canon) matching PIL LANCZOS."""
        n, _, h, w = g.shape
        key = (h, w)
        if key not in self._lz_cache:
            mh = lanczos_matrix(h, self.canon, device=g.device)
            mw = lanczos_matrix(w, self.canon, device=g.device)
            self._lz_cache[key] = (mh, mw)
        mh, mw = self._lz_cache[key]
        g0 = g[:, 0]                                      # (N,H,W)
        out = mh @ g0 @ mw.t()                            # (N,canon,canon)
        return out.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = rgb_to_gray(x)                                   # (N,1,H,W) in [0,1]
        g = g * 255.0                                        # official path is uint8 0..255
        # Match the evaluator EXACTLY: PIL convert('L') -> LANCZOS resize to 256.
        # (Differentiable Lanczos via fixed resampling matrices; bicubic differed
        # enough to flip boundary adversarial predictions under the official path.)
        g = self._lanczos_resize_gray(g)
        o = self.offset
        c = self.crop
        g = g[:, :, o:o + c, o:o + c]                        # (N,1,crop,crop)
        g0 = g[:, 0]                                         # (N,crop,crop)
        coeff = torch.matmul(torch.matmul(self.dct_m, g0), self.dct_m.t())
        # Forward == evaluator's log(|x|+1e-6); backward is magnitude-capped to
        # avoid gradient explosion on near-zero high-frequency coefficients.
        feat = stable_log_abs(coeff, eps_fwd=self.log_eps, eps_grad=0.01).unsqueeze(1)
        if self.grad_clip is not None:
            feat = torch.clamp(feat, -self.grad_clip, self.grad_clip)
        return self.model(feat)


class ViTBranch(nn.Module):
    """Full-res RGB (N,3,H,W) in [0,1] -> spatial logits (N,2)."""

    def __init__(self, model: nn.Module, canon: int = 256, crop: int = 224):
        super().__init__()
        self.model = model
        self.canon = canon
        self.crop = crop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.resize(x, [self.canon, self.canon], antialias=True)
        x = TF.center_crop(x, [self.crop, self.crop])
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        return self.model(x)


def load_detectors(weights_dir: str | Path, device: torch.device,
                   use_dct: bool = True, use_vit: bool = True) -> dict:
    """Load detector packs. Missing ViT weights are skipped with a warning
    (2026 vit_b_16.pth is not on the machine yet)."""
    weights_dir = Path(weights_dir)
    packs = {}

    if use_dct:
        p = weights_dir / "densenet121_dct.pth"
        m = create_dct_model()
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval().to(device)
        packs["densenet121_dct"] = {"model": m, "branch": DCTBranch(m).to(device)}
        print(f"[detectors] densenet121_dct loaded from {p}")

    if use_vit:
        p = weights_dir / "vit_b_16.pth"
        if not p.exists():
            print(f"[detectors][WARN] vit_b_16 weights not found at {p} "
                  f"-> ViT branch skipped (DCT-side work continues).")
        else:
            m = create_vit_model()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval().to(device)
            packs["vit_b_16"] = {"model": m, "branch": ViTBranch(m).to(device)}
            print(f"[detectors] vit_b_16 loaded from {p}")

    for p in packs.values():
        for pname in p["model"].parameters():
            pname.requires_grad_(False)
    return packs
