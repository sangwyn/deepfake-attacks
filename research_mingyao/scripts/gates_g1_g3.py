"""Gate experiments G1 (border mask) and G3 (DCT frequency shaping).

All runs: DCT detector only, MI-FGSM (mu=1), 60 steps, eps=8/255, paired on
the SAME fake images so comparisons are paired.

G1  border mask ON vs OFF : expect EQUAL ASR, quality >= (mask can only help).
    Note: the DCT branch already zeroes gradient outside the center-128 crop,
    so the border/annulus perturbation is ~0 regardless; G1 confirms the hard
    mask causes no harm (its real benefit accrues in the joint ViT+DCT attack).

G3  spectrum allocation (canonical-256 luma perturbation, center-128 DCT band
    projection each step): unrestricted vs LOW-pass vs HIGH-pass band.
    Measure final DCT margin, ASR, and perceptual cost (LPIPS/SSIM) on the
    full-resolution saved image; efficiency = margin gain / LPIPS.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import lpips

from . import data as D
from . import attacks as A
from .detectors import load_detectors
from .data import build_manifest, load_image_tensor, to_uint8_hwc, post_save_linf
from .metrics import official_quality, quality_score
from .attacks import targeted_margin_loss
from .dct_ops import dct_matrix

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/team_repo/weights")
DATA = Path("/data2/aiattacks/dataset")


def dct_margin(branch, x, target=0):
    with torch.no_grad():
        z = branch(x)[0]
    return float(z[1 - target] - z[target])  # >0 wrong, <0 fooled


# ---------------- G1 ----------------
def g1(fakes, packs, lp, device, steps):
    branch = packs["densenet121_dct"]["branch"]
    rows = {}
    for masked in (False, True):
        tag = f"mask={'ON' if masked else 'OFF'}"
        asr = 0; ss = []; ll = []
        for p in fakes:
            x_full, orig = load_image_tensor(p, device)
            adv = A.ifgsm(x_full, packs, device, target=0, steps=steps,
                          use=("densenet121_dct",), momentum=1.0,
                          use_border_mask=masked)
            adv_u = to_uint8_hwc(adv)
            s, l = official_quality(orig, adv_u, lp, device)
            pred = branch(adv).argmax(1).item()
            asr += int(pred == 0); ss.append(s); ll.append(l)
        rows[tag] = (asr / len(fakes), float(np.mean(ss)), float(np.mean(ll)))
    print("\n===== G1 : border mask (DCT-only, MI, %d steps) =====" % steps)
    print(f"  {'arm':12s} {'ASR':>7} {'SSIM':>8} {'LPIPS':>8}")
    for tag, (asr, s, l) in rows.items():
        print(f"  {tag:12s} {asr:7.3f} {s:8.4f} {l:8.4f}")
    a, b = rows["mask=OFF"], rows["mask=ON"]
    g1pass = abs(a[0] - b[0]) < 1e-9 and b[1] >= a[1] - 1e-4 and b[2] <= a[2] + 1e-4
    print(f"  G1 verdict: {'PASS (mask does no harm, quality holds)' if g1pass else 'CHECK'}")
    return g1pass


# ---------------- G3 ----------------
def _band_weight(kind, n=128, cutoff=0.30):
    u = torch.arange(n).float() / (n - 1)
    uu, vv = torch.meshgrid(u, u, indexing="ij")
    r = torch.sqrt(uu ** 2 + vv ** 2)
    if kind == "unrestricted":
        return torch.ones(n, n)
    if kind == "lowpass":
        return (r <= cutoff).float()
    if kind == "highpass":
        return (r > cutoff).float()
    raise ValueError(kind)


def _project_band(d_full, Dm, W):
    """Project a full-res RGB delta onto a chosen 128x128 DCT band in MODEL space.
    Downscale to canonical 256, keep luma, multiply center-128 DCT coefficients
    by W, zero DCT-blind outside region, upscale back. (unrestricted skips this.)
    """
    n, _, h, w = d_full.shape
    d256 = F.interpolate(d_full, size=(256, 256), mode="bicubic",
                         align_corners=False, antialias=True)
    gw = torch.tensor([0.299, 0.587, 0.114], device=d_full.device).view(1, 3, 1, 1)
    gray = (d256 * gw).sum(1, keepdim=True)
    c = gray[0, 0, 64:192, 64:192]
    C = torch.matmul(torch.matmul(Dm, c), Dm.t())
    c2 = torch.matmul(torch.matmul(Dm.t(), C * W), Dm)
    g2 = torch.zeros_like(gray)
    g2[0, 0, 64:192, 64:192] = c2
    d256p = g2.expand(1, 3, 256, 256)
    out = F.interpolate(d256p, size=(h, w), mode="bicubic",
                        align_corners=False, antialias=True)
    return out


def g3_attack(x_full, branch, device, kind, steps=60, eps=8/255, alpha=2/255, mu=1.0):
    """Full-res MI-FGSM; shaped arms project the delta to a DCT band each step."""
    n, _, h, w = x_full.shape
    Dm = dct_matrix(128, device=device)
    W = _band_weight(kind).to(device)
    clean_margin = dct_margin(branch, x_full)
    d = torch.zeros_like(x_full)
    acc = torch.zeros_like(d)
    for _ in range(steps):
        d = d.detach().requires_grad_(True)
        loss = targeted_margin_loss(branch((x_full + d).clamp(0, 1)), 0)
        g = torch.autograd.grad(loss, d)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        with torch.no_grad():
            d = d - alpha * acc.sign()
            d = d.clamp(-eps, eps)
            if kind != "unrestricted":
                d = _project_band(d, Dm, W).clamp(-eps, eps)
    with torch.no_grad():
        adv_full = (x_full + d).clamp(0, 1)
    return adv_full, clean_margin


def g3(fakes, packs, lp, device, steps):
    branch = packs["densenet121_dct"]["branch"]
    kinds = ["unrestricted", "lowpass", "highpass"]
    res = {k: {"asr": 0, "ss": [], "ll": [], "mgain": []} for k in kinds}
    for p in fakes:
        x_full, orig = load_image_tensor(p, device)
        for kind in kinds:
            adv_full, clean_m = g3_attack(x_full, branch, device, kind, steps=steps)
            adv_u = to_uint8_hwc(adv_full)
            s, l = official_quality(orig, adv_u, lp, device)
            adv_m = dct_margin(branch, adv_full)
            pred = branch(adv_full).argmax(1).item()
            res[kind]["asr"] += int(pred == 0)
            res[kind]["ss"].append(s); res[kind]["ll"].append(l)
            res[kind]["mgain"].append(clean_m - adv_m)   # margin reduction toward fooling
    print("\n===== G3 : DCT spectrum allocation (full-res MI, %d steps, %d fakes) =====" % (steps, len(fakes)))
    print(f"  {'arm':14s} {'ASR':>7} {'marginGain':>11} {'SSIM':>8} {'LPIPS':>8} {'Q':>7} {'MG/LPIPS':>9}")
    for k in kinds:
        r = res[k]; n = len(fakes)
        asr = r["asr"] / n; mg = float(np.mean(r["mgain"]))
        s = float(np.mean(r["ss"])); l = float(np.mean(r["ll"]))
        q = quality_score(s, l); eff = mg / max(l, 1e-6)
        print(f"  {k:14s} {asr:7.3f} {mg:11.3f} {s:8.4f} {l:8.4f} {q:7.3f} {eff:9.2f}")
    print("  (marginGain = how far margin moved toward fooling; lowpass ~0 means the")
    print("   model ignores smooth/DC perturbation; high-freq drives the log-DCT model.")
    print("   MG/LPIPS = margin gained per unit perceptual cost = efficiency.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=False)
    lp = lpips.LPIPS(net="alex").to(device).eval()
    fakes = [m["path"] for m in build_manifest(DATA) if m["cls"] == "fake"][: args.n]
    print(f"[data] {len(fakes)} fake images, steps={args.steps}")
    g1(fakes, packs, lp, device, args.steps)
    g3(fakes, packs, lp, device, args.steps)


if __name__ == "__main__":
    main()
