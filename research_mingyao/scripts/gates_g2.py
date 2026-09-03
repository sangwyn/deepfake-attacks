"""Gate G2 — ViT-exclusive (cheap) directions: chroma + annulus go/no-go.

Question: can we fool ViT using perturbation that is DCT-blind (chromatic
subspace and/or the 224-vs-128 annulus) at LOWER perceptual cost than full
RGB? Metric per PLAN: MarginGain / QualityCost (not just ASR).

Arms (fake->real, MI-FGSM, eps=8/255), delta projected into a subspace each step:
  A  full-rgb        : unrestricted (reference, most expensive)
  B  full+vitmask    : full RGB but zeroed outside ViT's 224 crop
  C  chroma-only     : chromatic plane (0.299dR+0.587dG+0.114dB=0) within 224
  D  annulus-luma    : luma only in the annulus ring (ViT sees, DCT crops out)
  E  dct-blind       : chroma in all-224 + luma only in annulus (center-128
                       luma = 0, border = 0) -> invisible to DCT by construction
  F  chroma-smooth   : chromatic plane, spatially Gaussian-smoothed (sigma 12px)
  G  full-smooth     : full RGB, spatially Gaussian-smoothed
F/G test whether SPATIAL smoothness (not channel choice) is what makes the
perturbation cheap for LPIPS/SSIM while still fooling ViT.

We report ViT fooling/margin, LPIPS/SSIM/Q, efficiency (marginGain/LPIPS), and
DCT fooling (arms C/D/E should leave DCT unfooled = blindness confirmed).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips

from .detectors import load_detectors
from .data import DATASETS, build_manifest, load_image_tensor, to_uint8_hwc, post_save_linf
from .metrics import official_quality, quality_score
from .attacks import targeted_margin_loss
from .masks import region_masks, CANON, project_chroma

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")


def full_res_mask(kind256: torch.Tensor, h: int, w: int, device):
    m = F.interpolate(kind256.view(1, 1, CANON, CANON), size=(h, w), mode="nearest")
    return m.to(device)


def _gauss_kernel(sigma: float, device):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    return k1d, radius


def blur(delta: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur per-channel (sigma in full-res pixels)."""
    k1d, r = _gauss_kernel(sigma, delta.device)
    pad = r
    kx = k1d.view(1, 1, 1, 2 * r + 1).expand(delta.shape[1], 1, 1, 2 * r + 1)
    ky = k1d.view(1, 1, 2 * r + 1, 1).expand(delta.shape[1], 1, 2 * r + 1, 1)
    out = F.conv2d(F.pad(delta, (pad, pad, 0, 0), mode="reflect"), kx, groups=delta.shape[1])
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), ky, groups=delta.shape[1])
    return out


def project_delta(delta, vitm, annm, arm, eps=8/255):
    """Project delta into the subspace for an arm. masks: (1,1,H,W) full-res."""
    smooth_sigma = 12.0
    if arm == "A_full-rgb":
        return delta
    if arm == "G_full-smooth":
        d = blur(delta, smooth_sigma)
        return d * (eps / d.abs().max().clamp(min=1e-8))
    if arm == "B_full-vitmask":
        return delta * vitm
    chroma = project_chroma(delta)
    luma = delta - chroma
    if arm == "C_chroma":
        return chroma * vitm
    if arm == "F_chroma-smooth":
        d = blur(chroma, smooth_sigma) * vitm
        return d * (eps / d.abs().max().clamp(min=1e-8))
    if arm == "D_annulus-luma":
        return luma * annm
    if arm == "E_dct-blind":
        # chroma anywhere in ViT's 224 view; luma only in the annulus ring
        return chroma * vitm + luma * annm
    raise ValueError(arm)


def attack(x_full, vit_branch, device, arm, steps=60, eps=8/255, alpha=2/255, mu=1.0, target=0):
    n, _, h, w = x_full.shape
    rm = region_masks(device=device)
    vitm = full_res_mask(rm["annulus"] + rm["center"], h, w, device)   # 224 region
    annm = full_res_mask(rm["annulus"], h, w, device)                  # ring
    clean_margin = margin_of(vit_branch, x_full, target)
    d = torch.zeros_like(x_full)
    acc = torch.zeros_like(d)
    for _ in range(steps):
        d = d.detach().requires_grad_(True)
        loss = targeted_margin_loss(vit_branch((x_full + d).clamp(0, 1)), target)
        g = torch.autograd.grad(loss, d)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        with torch.no_grad():
            d = d - alpha * acc.sign()
            d = project_delta(d, vitm, annm, arm, eps)
            d = d.clamp(-eps, eps)
    adv = (x_full + d).clamp(0, 1)
    return adv.detach(), clean_margin


def margin_of(branch, x, target=0):
    with torch.no_grad():
        z = branch(x)[0]
    return float(z[1 - target] - z[target])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--dataset", default="celebA")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  dataset={args.dataset}")
    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=True)
    vit_b = packs["vit_b_16"]["branch"]
    dct_b = packs["densenet121_dct"]["branch"]
    lp = lpips.LPIPS(net="alex").to(device).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    fakes = [m for m in build_manifest(DATASETS[args.dataset]) if m["cls"] == "fake"][: args.n]
    arms = ["A_full-rgb", "B_full-vitmask", "C_chroma", "F_chroma-smooth",
            "G_full-smooth", "D_annulus-luma", "E_dct-blind"]
    res = {a: {"vit_flip": 0, "dct_flip": 0, "mg": [], "ss": [], "ll": [], "linf": 0.0} for a in arms}

    for i, m in enumerate(fakes):
        x_full, orig = load_image_tensor(m["path"], device)
        for arm in arms:
            adv, clean_m = attack(x_full, vit_b, device, arm, steps=args.steps)
            adv_u = to_uint8_hwc(adv)
            s, l = official_quality(orig, adv_u, lp, device)
            adv_m = margin_of(vit_b, adv)
            pv = int(adv_m < 0)
            pd = int(margin_of(dct_b, adv) < 0)
            r = res[arm]
            r["vit_flip"] += pv; r["dct_flip"] += pd
            r["mg"].append(clean_m - adv_m); r["ss"].append(s); r["ll"].append(l)
            r["linf"] = max(r["linf"], post_save_linf(orig, adv_u))
        if (i + 1) % 4 == 0:
            print(f"  ... {i+1}/{len(fakes)}")

    print(f"\n===== G2 : ViT cheap-direction go/no-go (MI {args.steps} steps, {len(fakes)} fakes) =====")
    print(f"  {'arm':16s} {'ViT-flip':>9} {'DCT-flip':>9} {'marginGain':>11} {'SSIM':>7} {'LPIPS':>7} {'Q':>6} {'MG/LPIPS':>9}")
    for arm in arms:
        r = res[arm]; n = len(fakes)
        mg = np.mean(r["mg"]); s = np.mean(r["ss"]); l = np.mean(r["ll"])
        eff = mg / max(l, 1e-6)
        print(f"  {arm:16s} {r['vit_flip']:>6}/{n} {r['dct_flip']:>6}/{n} {mg:11.3f} "
              f"{s:7.4f} {l:7.4f} {quality_score(s,l):6.3f} {eff:9.2f}")
    print("\n  Interpretation:")
    print("  - C/D/E should have DCT-flip ~0 (perturbation truly DCT-blind).")
    print("  - If C/E achieve meaningful ViT marginGain at much lower LPIPS than A/B,")
    print("    chroma/annulus are efficient ViT-only directions -> keep in ViDA.")
    print("  - If their marginGain ~0, they cannot move ViT alone -> demote chroma to")
    print("    a regularizer (PLAN fallback).")


if __name__ == "__main__":
    main()
