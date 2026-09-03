"""Gate experiments G4 (quality recovery) and G5 (adaptive / minimal-sufficient budget).

DCT-only, MI-FGSM, eps=8/255, paired on the SAME fake images. Three arms:

  A  fixed60   : 60 attack steps, no early stop, no recovery (baseline).
  B  early-stop: stop attacking as soon as the DCT detector is fooled
                 (margin < 0) -> minimal-sufficient perturbation  (G5).
  C  +recovery : from B's delta, run Stage-B quality recovery: optimize
                 L = softplus(margin + kappa) + lambda*(1 - Q_proxy),
                 Q_proxy = 0.5*DiffSSIM + 0.5*(1 - LPIPS_alex), all
                 differentiable; rollback any step that loses fooling;
                 keep the best-Q fooled state                      (G4).

G5 verdict: B vs A at (near-)identical fooling -> B must have higher quality.
G4 verdict: C vs B on the fooled set -> recovery must raise Q (SSIM/LPIPS).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips

from . import attacks as Aatk
from .detectors import load_detectors
from .data import build_manifest, load_image_tensor, to_uint8_hwc
from .metrics import official_quality, quality_score, DiffSSIM
from .attacks import targeted_margin_loss

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/team_repo/weights")
DATA = Path("/data2/aiattacks/dataset")


def margin_of(branch, x, target=0):
    with torch.no_grad():
        z = branch(x)[0]
    return float(z[1 - target] - z[target])   # <0 => fooled


def run_arms(x_full, branch, lpips_fn, diff_ssim, device,
             steps=60, rec_steps=30, eps=8/255, alpha=2/255, mu=1.0,
             kappa=0.5, lam=2.0, alpha_r=0.5/255, target=0):
    """Return dicts of final delta for arms A, B, C and the stop step."""
    d = torch.zeros_like(x_full)
    acc = torch.zeros_like(d)
    d_stop = None
    stop_step = None

    for it in range(steps):
        d = d.detach().requires_grad_(True)
        loss = targeted_margin_loss(branch((x_full + d).clamp(0, 1)), target)
        g = torch.autograd.grad(loss, d)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        with torch.no_grad():
            d = (d - alpha * acc.sign()).clamp(-eps, eps)
        m = margin_of(branch, (x_full + d).clamp(0, 1), target)
        if d_stop is None and m < 0:
            d_stop = d.detach().clone()
            stop_step = it + 1
    d_A = d.detach()
    if d_stop is None:
        d_stop = d.detach()           # never fooled within budget
        stop_step = steps

    # ---- Arm C: quality recovery from d_stop ----
    d_C = d_stop.clone()
    best_Q = -1.0
    best_state = d_C.clone()
    base = (x_full * 2 - 1).detach()
    for _ in range(rec_steps):
        d_C = d_C.detach().requires_grad_(True)
        adv = (x_full + d_C).clamp(0, 1)
        z = branch(adv)[0]
        margin = z[1 - target] - z[target]
        adv_lp = adv * 2 - 1
        lp = lpips_fn(base, adv_lp).mean()
        ds = diff_ssim(x_full, adv)
        Q = 0.5 * ds + 0.5 * (1 - lp)
        loss = F.softplus(margin + kappa) + lam * (1 - Q)
        g = torch.autograd.grad(loss, d_C)[0]
        with torch.no_grad():
            cand = (d_C - alpha_r * g.sign()).clamp(-eps, eps)
            m_new = margin_of(branch, (x_full + cand).clamp(0, 1), target)
            if m_new < 0:                       # keep only fooling-preserving steps
                d_C = cand
                if m_new < 0:
                    qv = float(Q.item())
                    if qv > best_Q:
                        best_Q = qv
                        best_state = cand.clone()
            # else: rollback (discard cand), retry next iteration
    if margin_of(branch, (x_full + best_state).clamp(0, 1), target) < 0:
        d_C = best_state
    return d_A, d_stop, d_C, stop_step


def eval_arm(delta, x_full, orig, branch, lpips_fn, device, target=0):
    adv = (x_full + delta).clamp(0, 1)
    adv_u = to_uint8_hwc(adv)
    s, l = official_quality(orig, adv_u, lpips_fn, device)
    m = margin_of(branch, adv, target)
    fooled = m < 0
    return {"fooled": fooled, "ssim": s, "lpips": l, "q": quality_score(s, l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--rec", type=int, default=30)
    ap.add_argument("--lam", type=float, default=2.0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=False)
    branch = packs["densenet121_dct"]["branch"]
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    diff_ssim = DiffSSIM().to(device)

    fakes = [m["path"] for m in build_manifest(DATA) if m["cls"] == "fake"][: args.n]
    print(f"[data] {len(fakes)} fakes, attack steps={args.steps}, recovery steps={args.rec}")

    arms = {"A_fixed60": [], "B_earlystop": [], "C_recovery": []}
    stop_steps = []
    t0 = time.time()
    for i, p in enumerate(fakes):
        x_full, orig = load_image_tensor(p, device)
        dA, dB, dC, st = run_arms(x_full, branch, lpips_fn, diff_ssim, device,
                                  steps=args.steps, rec_steps=args.rec, lam=args.lam)
        arms["A_fixed60"].append(eval_arm(dA, x_full, orig, branch, lpips_fn, device))
        arms["B_earlystop"].append(eval_arm(dB, x_full, orig, branch, lpips_fn, device))
        arms["C_recovery"].append(eval_arm(dC, x_full, orig, branch, lpips_fn, device))
        stop_steps.append(st)
        if (i + 1) % 4 == 0:
            print(f"  ... {i+1}/{len(fakes)}  ({time.time()-t0:.0f}s)")

    def summarize(rows):
        fooled = [r for r in rows if r["fooled"]]
        n = len(rows)
        return {
            "asr": len(fooled) / n,
            "ssim_all": np.mean([r["ssim"] for r in rows]),
            "lpips_all": np.mean([r["lpips"] for r in rows]),
            "q_all": np.mean([r["q"] for r in rows]),
            "ssim_fool": np.mean([r["ssim"] for r in fooled]) if fooled else float("nan"),
            "lpips_fool": np.mean([r["lpips"] for r in fooled]) if fooled else float("nan"),
            "q_fool": np.mean([r["q"] for r in fooled]) if fooled else float("nan"),
        }

    print("\n===== G4/G5 results (DCT-only, MI, %d fakes) =====" % len(fakes))
    print(f"  {'arm':14s} {'ASR':>6} {'SSIM(all)':>10} {'LPIPS(all)':>11} {'Q(all)':>8} "
          f"{'SSIM(fool)':>11} {'LPIPS(fool)':>12} {'Q(fool)':>8}")
    for name, rows in arms.items():
        s = summarize(rows)
        print(f"  {name:14s} {s['asr']:6.3f} {s['ssim_all']:10.4f} {s['lpips_all']:11.4f} "
              f"{s['q_all']:8.4f} {s['ssim_fool']:11.4f} {s['lpips_fool']:12.4f} {s['q_fool']:8.4f}")
    print(f"\n  mean early-stop step: {np.mean(stop_steps):.1f} "
          f"(median {np.median(stop_steps):.0f}, min {min(stop_steps)}, max {max(stop_steps)})")

    # paired verdicts on images fooled by all three arms
    common = [i for i in range(len(fakes))
              if arms["A_fixed60"][i]["fooled"] and arms["B_earlystop"][i]["fooled"]
              and arms["C_recovery"][i]["fooled"]]
    if common:
        qA = np.mean([arms["A_fixed60"][i]["q"] for i in common])
        qB = np.mean([arms["B_earlystop"][i]["q"] for i in common])
        qC = np.mean([arms["C_recovery"][i]["q"] for i in common])
        lA = np.mean([arms["A_fixed60"][i]["lpips"] for i in common])
        lB = np.mean([arms["B_earlystop"][i]["lpips"] for i in common])
        lC = np.mean([arms["C_recovery"][i]["lpips"] for i in common])
        print(f"\n  Paired on {len(common)} common-fooled images:")
        print(f"    G5  early-stop Q {qA:.4f}(A) -> {qB:.4f}(B)  "
              f"LPIPS {lA:.4f} -> {lB:.4f}  {'PASS' if qB >= qA else 'CHECK'}")
        print(f"    G4  +recovery  Q {qB:.4f}(B) -> {qC:.4f}(C)  "
              f"LPIPS {lB:.4f} -> {lC:.4f}  {'PASS' if qC >= qB else 'CHECK'}")


if __name__ == "__main__":
    main()
