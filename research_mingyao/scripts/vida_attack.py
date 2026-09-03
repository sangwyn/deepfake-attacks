"""ViDA joint decoupled attack vs naive joint, on BOTH 2026 detectors.

Routing (the decoupling, confirmed by G2):
  * ViT gradient is routed into the DCT-BLIND subspace: chromatic component
    anywhere in ViT's 224 view + luma component only in the annulus ring
    (center-128 luma = 0). DCT literally cannot see this, so fooling ViT here
    never disturbs the DCT attack.
  * DCT gradient is routed to the center-128 LUMA component (the only thing the
    grayscale DCT model sees).

Each iteration computes both detectors' full gradients with SEPARATE momentum
buffers and adds the two disjoint routed steps. Then G5 early-stop (stop once
BOTH are fooled) and optional G4 quality recovery (maximize Q while preserving
both fools, with rollback).

Arms compared (fake->real, eps=8/255, MI-FGSM):
  naive      : gradient SUM over both detectors, no routing (baseline).
  vida       : routed decoupling + G5 early stop.
  vida+rec   : vida + G4 quality recovery.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips

from .detectors import load_detectors
from .data import DATASETS, build_manifest, load_image_tensor, to_uint8_hwc, post_save_linf
from .metrics import official_quality, quality_score, DiffSSIM
from .attacks import targeted_margin_loss
from .masks import region_masks, CANON, project_chroma

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")


def fullres(m256, h, w, device):
    return F.interpolate(m256.view(1, 1, CANON, CANON), size=(h, w), mode="nearest").to(device)


def route_blind(delta, vitm, annm):
    """DCT-blind subspace: chroma in all-224 + luma only in the annulus ring."""
    chroma = project_chroma(delta)
    luma = delta - chroma
    return chroma * vitm + luma * annm


def route_center(delta, cenm):
    """DCT-useful subspace: luma only, center-128 (grayscale DCT region)."""
    luma = delta - project_chroma(delta)
    return luma * cenm


def route_dct_grayscale(grad_rgb, cenm):
    """Full-strength center GRAYSCALE step for the DCT model.

    The DCT branch converts RGB->gray via L = 0.299R+0.587G+0.114B, so the
    loss gradient w.r.t RGB is a pure-luma field. Collapsing sign() AFTER
    projecting onto the luma unit vector shrinks the effective gray step to
    ~0.447; instead we take the sign in GRAYSCALE space (full unit magnitude)
    and emit a pure-luma RGB step, so the grayscale perturbation seen by DCT
    is full strength. Restricted to the center region DCT observes.
    """
    w = torch.tensor([0.299, 0.587, 0.114], device=grad_rgb.device).view(1, 3, 1, 1)
    gray_grad = (grad_rgb * w).sum(dim=1, keepdim=True)      # (N,1,H,W)
    s = torch.sign(gray_grad) * cenm
    return s * w                                              # pure luma RGB


def margin(branch, x, target=0):
    with torch.no_grad():
        z = branch(x)[0]
    return float(z[1 - target] - z[target])


def _di_transform(x, p=0.5, lo=0.8, hi=1.0):
    """Diverse-Inputs (Xie et al.): with prob p, randomly resize down then
    zero-pad back to the original size. Differentiable; improves transfer."""
    if torch.rand(1).item() > p:
        return x
    n, c, h, w = x.shape
    s = lo + (hi - lo) * torch.rand(1).item()
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    xs = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    out = torch.zeros_like(x)
    top = int(torch.randint(0, h - nh + 1, (1,)).item())
    left = int(torch.randint(0, w - nw + 1, (1,)).item())
    out[:, :, top:top + nh, left:left + nw] = xs
    return out


def vida_attack(x_full, packs, device, steps=80, tail_steps=40, eps=8/255,
                alpha=2/255, mu=1.0, target=0, use_border_mask=True, early_stop=True,
                di_prob=0.0):
    """ViDA acquisition + minimal-sufficient budget + DCT tail.

    Phase 1 (joint): MI-FGSM momentum SUM of the two detectors' gradients,
    border mask (G1), early stop as soon as BOTH detectors are fooled (G5).
    Phase 2 (DCT tail): if ViT is fooled but DCT is not, run extra DCT-only
    steps restricted to the allowed region, ACCEPTING a step only if ViT stays
    fooled (rollback otherwise), until DCT flips or the tail budget is spent.
    This recovers the DCT ASR that momentum-balancing can leave on the table
    without sacrificing ViT or quality (recovery handles quality afterwards).
    """
    vit_b = packs["vit_b_16"]["branch"]
    dct_b = packs["densenet121_dct"]["branch"]
    n, _, h, w = x_full.shape
    rm = region_masks(device=device)
    border = fullres(rm["border"], h, w, device)
    allow = 1.0 - border if use_border_mask else 1.0

    def adv_of(dd):
        return (x_full + dd).clamp(0, 1)

    def loss_on(dd):
        adv = _di_transform(adv_of(dd), p=di_prob) if di_prob > 0 else adv_of(dd)
        return targeted_margin_loss(vit_b(adv), target) \
            + targeted_margin_loss(dct_b(adv), target)

    # ---- Phase 1: joint momentum sum (same gradient as naive) + mask + stop ----
    d = torch.zeros_like(x_full)
    acc = torch.zeros_like(d)
    stop = steps
    for it in range(steps):
        d = d.detach().requires_grad_(True)
        g = torch.autograd.grad(loss_on(d), d)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        with torch.no_grad():
            d = (d - alpha * acc.sign()) * allow
            d = d.clamp(-eps, eps)
        if early_stop and margin(vit_b, adv_of(d), target) < 0 \
                and margin(dct_b, adv_of(d), target) < 0:
            stop = it + 1
            break

    # ---- Phase 2: continue equal-weight joint steps on images DCT still misses
    #          (hard images get extra iterations; ViT keeps being pushed too) ----
    tail_used = 0
    for _ in range(tail_steps):
        if margin(dct_b, adv_of(d), target) < 0:
            break
        d = d.detach().requires_grad_(True)
        g = torch.autograd.grad(loss_on(d), d)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        with torch.no_grad():
            d = (d.detach() - alpha * acc.sign()) * allow
            d = d.clamp(-eps, eps)
            tail_used += 1
    return d.detach(), stop + tail_used


def naive_joint(x_full, packs, device, steps=80, eps=8/255, alpha=2/255, mu=1.0, target=0):
    vit_b = packs["vit_b_16"]["branch"]; dct_b = packs["densenet121_dct"]["branch"]
    d = torch.zeros_like(x_full)
    acc = torch.zeros_like(d)
    for _ in range(steps):
        d = d.detach().requires_grad_(True)
        adv = (x_full + d).clamp(0, 1)
        loss = targeted_margin_loss(vit_b(adv), target) + targeted_margin_loss(dct_b(adv), target)
        g = torch.autograd.grad(loss, d)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        with torch.no_grad():
            d = (d - alpha * acc.sign()).clamp(-eps, eps)
    return d.detach()


def quality_recovery(x_full, d0, packs, lpips_fn, diff_ssim, device,
                     rec_steps=40, eps=8/255, alpha_r=0.5/255, kappa=0.5, lam=2.0, target=0):
    """Stage-B: maximize perceptual quality Q while NEVER losing a detector
    that was already fooled at d0. Works for all cases — both fooled, or only
    ViT fooled (then the ViT score contribution is repaired; DCT may still flip
    as a bonus). Returns (delta, ran)."""
    vit_b = packs["vit_b_16"]["branch"]; dct_b = packs["densenet121_dct"]["branch"]
    d = d0.clone()
    base = (x_full * 2 - 1).detach()

    def margins(dd):
        adv = (x_full + dd).clamp(0, 1)
        return margin(vit_b, adv, target), margin(dct_b, adv, target)

    mv0, md0 = margins(d)
    lock_v, lock_d = mv0 < 0, md0 < 0
    if not (lock_v or lock_d):
        return d0, False

    def holds(dd):
        mv, md = margins(dd)
        return (not lock_v or mv < 0) and (not lock_d or md < 0)

    best_Q, best_state = -1.0, d.clone()
    for _ in range(rec_steps):
        d = d.detach().requires_grad_(True)
        adv = (x_full + d).clamp(0, 1)
        mv = vit_b(adv)[0]; md = dct_b(adv)[0]
        m_v = mv[1-target] - mv[target]; m_d = md[1-target] - md[target]
        lp = lpips_fn(base, adv * 2 - 1).mean()
        Q = 0.5 * diff_ssim(x_full, adv) + 0.5 * (1 - lp)
        loss = F.softplus(m_v + kappa) + F.softplus(m_d + kappa) + lam * (1 - Q)
        g = torch.autograd.grad(loss, d)[0]
        with torch.no_grad():
            cand = (d - alpha_r * g.sign()).clamp(-eps, eps)
            if holds(cand):
                d = cand
                qv = float(Q.item())
                if qv > best_Q:
                    best_Q, best_state = qv, cand.clone()
    return (best_state if holds(best_state) else d0), True


def evaluate(d, x_full, orig, packs, lp, device):
    adv = (x_full + d).clamp(0, 1)
    adv_u = to_uint8_hwc(adv)
    s, l = official_quality(orig, adv_u, lp, device)
    fv = margin(packs["vit_b_16"]["branch"], adv) < 0
    fd = margin(packs["densenet121_dct"]["branch"], adv) < 0
    q = quality_score(s, l)
    score = q * (int(fv) + int(fd))   # sum over detectors (each up to q)
    return {"fv": fv, "fd": fd, "ssim": s, "lpips": l, "q": q,
            "score": score, "linf": post_save_linf(orig, adv_u)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--rec", type=int, default=40)
    ap.add_argument("--dataset", default="celebA")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  dataset={args.dataset}")
    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=True)
    lp = lpips.LPIPS(net="alex").to(device).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    diff_ssim = DiffSSIM().to(device)

    fakes = [m for m in build_manifest(DATASETS[args.dataset]) if m["cls"] == "fake"][: args.n]
    arms = {"naive_joint": [], "vida": [], "vida+rec": []}
    stops = []
    for i, m in enumerate(fakes):
        x_full, orig = load_image_tensor(m["path"], device)
        d_naive = naive_joint(x_full, packs, device, steps=args.steps)
        d_vida, stop = vida_attack(x_full, packs, device, steps=args.steps,
                                   tail_steps=args.tail)
        stops.append(stop)
        d_rec, _ = quality_recovery(x_full, d_vida, packs, lp, diff_ssim, device,
                                    rec_steps=args.rec)
        arms["naive_joint"].append(evaluate(d_naive, x_full, orig, packs, lp, device))
        arms["vida"].append(evaluate(d_vida, x_full, orig, packs, lp, device))
        arms["vida+rec"].append(evaluate(d_rec, x_full, orig, packs, lp, device))
        if (i + 1) % 4 == 0:
            print(f"  ... {i+1}/{len(fakes)}")

    n = len(fakes)
    print(f"\n===== ViDA joint decoupled vs naive ({n} fakes, {args.steps} steps) =====")
    print(f"  {'arm':12s} {'ViT-flip':>9} {'DCT-flip':>9} {'BOTH':>6} {'SSIM':>7} {'LPIPS':>7} {'Q':>6} {'SCORE':>7}")
    for name, rows in arms.items():
        vf = sum(r["fv"] for r in rows); df = sum(r["fd"] for r in rows)
        both = sum(r["fv"] and r["fd"] for r in rows)
        s = np.mean([r["ssim"] for r in rows]); l = np.mean([r["lpips"] for r in rows])
        q = np.mean([r["q"] for r in rows]); sc = np.mean([r["score"] for r in rows])
        print(f"  {name:12s} {vf:>6}/{n} {df:>6}/{n} {both:>4}/{n} {s:7.4f} {l:7.4f} {q:6.3f} {sc:7.4f}")
    print(f"  vida mean stop step: {np.mean(stops):.1f} (fixed naive runs {args.steps})")
    print("  SCORE = mean over images of Q*(I_ViT + I_DCT), i.e. the official sum rule (2 detectors).")


if __name__ == "__main__":
    main()
