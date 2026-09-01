"""
Minimal smoke checks — run from repo root: `python tests/smoke.py`

Catches the expensive mistakes without needing real weights or data:
  1. torch DCT matches scipy's ortho DCT-II (the differentiable preprocessing).
  2. identity attack leaves the image untouched.
  3. attack output has the right shape/dtype/range and respects the eps-ball.

Uses tiny dummy detectors, so no .pth files are needed.
"""

import os
import sys

# make the repo root importable regardless of where this is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("smoke: needs numpy + torch installed")

from attacks import _common, ifgsm, midi_fgsm, template

DEV = torch.device("cpu")


class Dummy(nn.Module):
    """2-logit detector; gradient flows from input so attacks can optimise."""
    def forward(self, x):
        m = x.float().mean(dim=(1, 2, 3))
        return torch.stack([m, -m], dim=1)


def classifiers():
    # names chosen to exercise both preprocessing paths (spatial + *_dct)
    return {"vit_b_16": {"model": Dummy()},
            "densenet121_dct": {"model": Dummy()}}


def check_dct():
    try:
        from scipy.fftpack import dct as sdct
    except ImportError:
        print("  DCT vs scipy: SKIPPED (scipy not installed)")
        return
    x = (np.random.rand(128, 128).astype(np.float32)) * 255
    d = _common.dct_matrix(128, DEV, torch.float32)
    got = (d @ torch.from_numpy(x) @ d.transpose(-1, -2)).numpy()
    ref = sdct(sdct(x, axis=0, norm="ortho"), axis=1, norm="ortho")
    # coefficients reach ~1e4, so compare relative to scale (float32 rounding).
    rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
    assert rel < 1e-4, f"DCT mismatch: rel err {rel}"
    print(f"  DCT vs scipy: rel err {rel:.2e}  OK")


def check_identity():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = template.attack(img, {}, DEV)
    assert np.array_equal(out, img), "identity attack changed the image"
    print("  identity attack: unchanged  OK")


def check_attack(mod, name, **kw):
    img = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    out = mod.attack(img, classifiers(), DEV, iterations=2, **kw)
    assert out.shape == img.shape, f"shape {out.shape}"
    assert out.dtype == np.uint8, f"dtype {out.dtype}"
    assert out.min() >= 0 and out.max() <= 255, "out of [0,255]"
    eps = kw.get("epsilon", 8 / 255)
    linf = int(np.abs(img.astype(np.int16) - out.astype(np.int16)).max())
    assert linf <= round(eps * 255) + 1, f"linf {linf} exceeds eps"
    print(f"  {name}: shape/range/eps (linf={linf})  OK")


if __name__ == "__main__":
    check_dct()
    check_identity()
    check_attack(ifgsm, "ifgsm")
    check_attack(midi_fgsm, "midi_fgsm", eot_samples=2)  # DI + EOT, jpeg off
    print("SMOKE PASS")
