"""Diagnose the margin bottleneck of the hardest fakes (001539, 001556).

For each image we report:
  1. clean-margins: detector logit gaps and softmax Fake-confidence on the
     untouched image (intrinsic difficulty);
  2. gradient conflict: cosine / sign-agreement between the two detectors'
     acquisition gradients, overall and per region (centre-128 shared vs
     annulus ViT-only) and per subspace (luma vs chroma);
  3. per-detector minimal flip scale after a joint acquisition: smallest
     perturbation scale t that fools ViT-only / DCT-only / both (gt gate);
  4. single-detector cost lower bound: MI-FGSM against only one detector +
     line search -> minimal Q (SSIM/LPIPS) each detector costs on its own;
  5. final v3.3 output: logit margins, eps saturation, energy by region.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TEAM = "/data2/aiattacks/Mingyao-Duan/team_repo"
_HERE = str(Path(__file__).resolve().parent)
# Ensure team_repo takes precedence over this script's dir (which contains a
# shadowing attacks.py research module).
sys.path = [TEAM] + [p for p in sys.path if p not in ("", _HERE)]

import evaluate as ev          # noqa: E402
import attacks.vida as vida    # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 8.0 / 255.0
TARGET = 0
IMAGES = [
    ("001539", "hard-worst"),
    ("001556", "hard-2nd"),
    ("001553", "rescue-fixed"),
    ("001549", "easy"),
]
FAKE_DIR = Path("/home/aiattacks/dataset/celebA/TEST/TEST_FAKE")


def build():
    weights = Path(TEAM) / "weights"
    clf = {}
    for name in ["vit_b_16", "densenet121_dct"]:
        model = ev.load_model(name, weights / f"{name}.pth", DEVICE)
        tf = ev.build_dct_transform(True) if name.endswith("_dct") \
            else ev.build_spatial_transform(name)
        clf[name] = {"model": model, "transform": tf}
    branches = {
        "vit_b_16": vida.ViTBranch(clf["vit_b_16"]["model"]).to(DEVICE).eval(),
        "densenet121_dct": vida.DCTBranch(clf["densenet121_dct"]["model"],
                                          DEVICE).to(DEVICE).eval(),
    }
    for b in branches.values():
        for p in b.model.parameters():
            p.requires_grad_(False)
    lp = vida._get_qmodels(DEVICE)[0]
    return clf, branches, lp


def masks(x):
    m256 = torch.ones(1, 1, 256, 256, device=DEVICE)
    m256[:, :, :16, :] = 0; m256[:, :, -16:, :] = 0
    m256[:, :, :, :16] = 0; m256[:, :, :, -16:] = 0
    allow = F.interpolate(m256, size=x.shape[-2:], mode="nearest")
    c256 = torch.zeros(1, 1, 256, 256, device=DEVICE)
    c256[:, :, 64:192, 64:192] = 1.0
    center = F.interpolate(c256, size=x.shape[-2:], mode="nearest")
    return allow, center


def load_img(stem):
    p = FAKE_DIR / f"{stem}_1024x1024.png"
    img = np.array(Image.open(p).convert("RGB"))
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    return img, x


def softmax_conf(clf, pil, name):
    with torch.no_grad():
        t = clf[name]["transform"](pil).unsqueeze(0).to(DEVICE)
        p = F.softmax(clf[name]["model"](t), dim=1)[0]
    return float(p[0]), float(p[1])   # P(Real), P(Fake)


def grad_of(branch, x, d, allow):
    d = d.detach().requires_grad_(True)
    adv = (x + d).clamp(0, 1)
    g = torch.autograd.grad(vida._margin_loss(branch(adv), TARGET), d)[0]
    return (g * allow).detach()


def grad_conflict(gv, gd, center, allow):
    ann = allow * (1 - center)
    cen = allow * center
    out = {}
    for rname, m in [("visible", allow), ("center128", cen), ("annulus", ann)]:
        a, b = gv * m, gd * m
        cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
        agr = float((a.sign() == b.sign()).float()[m.expand_as(a) > 0].mean())
        out[rname] = (cos, agr)
    # luma / chroma split of gradients
    w = torch.tensor([0.299, 0.587, 0.114], device=DEVICE).view(1, 3, 1, 1)
    for sname, proj in [("luma", lambda g: (g * w).sum(1, keepdim=True) * w),
                        ("chroma", lambda g: g - (g * w).sum(1, keepdim=True) * w)]:
        a, b = proj(gv) * allow, proj(gd) * allow
        out[sname] = (float((a * b).sum() / (a.norm() * b.norm() + 1e-12)), None)
    return out


def acquire(x, branches, allow, step=2.0 / 255.0, steps=120, both=True,
            only=None, mu=1.0):
    d = torch.zeros_like(x); acc = torch.zeros_like(x)
    use = [only] if only else list(branches)
    for it in range(steps):
        d = d.detach().requires_grad_(True)
        adv = (x + d).clamp(0, 1)
        loss = sum(vida._margin_loss(branches[n](adv), TARGET) for n in use)
        g = torch.autograd.grad(loss, d)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = mu * acc + g
        with torch.no_grad():
            d = (d - step * acc.sign()) * allow
            d = d.clamp(-EPS, EPS)
        fooled = all(vida._margin(branches[n], (x + d).clamp(0, 1), TARGET) < 0
                     for n in use)
        if fooled:
            break
    return d.detach(), it + 1


def min_scale(x, d, clf, gate_names, iters=14):
    """Smallest t in [0,1] s.t. every detector in gate_names is gt-fooled."""
    if vida._gt_bad(vida._uint8_hwc(x, d), clf, gate_names, DEVICE, TARGET):
        return None  # even t=1 fails
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if vida._gt_bad(vida._uint8_hwc(x, d * mid), clf, gate_names,
                        DEVICE, TARGET):
            lo = mid
        else:
            hi = mid
    return hi


def official_q(clean_np, adv_u, lp):
    s = ev.compute_ssim_rgb(clean_np, adv_u)
    with torch.no_grad():
        l = float(lp(ev.np_to_lpips_tensor(clean_np, DEVICE),
                     ev.np_to_lpips_tensor(adv_u, DEVICE)).item())
    return s, l, 0.5 * s + 0.5 * (1 - l)


def delta_stats(x, d, allow, center):
    adv = (x + d).clamp(0, 1)
    dd = (adv - x).detach()
    vis = allow.expand_as(dd) > 0
    cen = (allow * center).expand_as(dd) > 0
    ann = (allow * (1 - center)).expand_as(dd) > 0
    w = torch.tensor([0.299, 0.587, 0.114], device=DEVICE).view(1, 3, 1, 1)
    luma = (dd * w).sum(1, keepdim=True)          # (1,1,H,W)
    chroma = dd - luma * w
    sat = float((dd[vis].abs() > 7.5 / 255.0).float().mean())
    cen1 = cen[:, :1]; ann1 = ann[:, :1]
    return {
        "linf": float(dd.abs().max()),
        "mean|d| vis": float(dd[vis].abs().mean()),
        "frac at eps cap": sat,
        "L2 center / annulus (luma)": float(
            luma[cen1].norm() / (luma[ann1].norm() + 1e-9)),
        "energy share center": float((dd * center).pow(2).sum() / dd.pow(2).sum()),
        "|chroma|/|luma|": float(chroma.norm() / (luma.norm() + 1e-9)),
    }


def main():
    clf, branches, lp = build()
    names = list(branches)
    for stem, tag in IMAGES:
        print(f"\n{'='*72}\n{stem}  [{tag}]\n{'='*72}")
        clean_np, x = load_img(stem)
        pil = Image.fromarray(clean_np)
        allow, center = masks(x)

        # 1. clean margins / confidence
        mv = vida._margin(branches["vit_b_16"], x, TARGET)
        md = vida._margin(branches["densenet121_dct"], x, TARGET)
        pv = softmax_conf(clf, pil, "vit_b_16")
        pd = softmax_conf(clf, pil, "densenet121_dct")
        print(f"[clean] float margin (fake-real logit): ViT={mv:+.2f} DCT={md:+.2f}")
        print(f"[clean] softmax P(Real)/P(Fake): ViT=({pv[0]:.3f},{pv[1]:.3f}) "
              f"DCT=({pd[0]:.3f},{pd[1]:.3f})")

        # 2. gradient conflict at d=0
        gv = grad_of(branches["vit_b_16"], x, torch.zeros_like(x), allow)
        gd = grad_of(branches["densenet121_dct"], x, torch.zeros_like(x), allow)
        conf = grad_conflict(gv, gd, center, allow)
        print("[grad conflict at d=0]  cosine / sign-agreement:")
        for k, (c, a) in conf.items():
            print(f"    {k:14s} cos={c:+.3f}" + (f"  sign-agr={a:.3f}" if a is not None else ""))

        # 3. joint acquisition (2/255, like pass 1) + per-detector flip scale
        d_j, used = acquire(x, branches, allow, step=2.0 / 255.0, steps=120)
        t_vit = min_scale(x, d_j, clf, ["vit_b_16"])
        t_dct = min_scale(x, d_j, clf, ["densenet121_dct"])
        t_both = min_scale(x, d_j, clf, names)
        print(f"[joint 2/255 acq] stopped at step {used}; "
              f"min flip scale -> ViT-only={t_vit}, DCT-only={t_dct}, both={t_both}")

        # fine-step acquisition (1/255, like rescue)
        d_f, usedf = acquire(x, branches, allow, step=1.0 / 255.0, steps=240)
        tf_both = min_scale(x, d_f, clf, names)
        tf_vit = min_scale(x, d_f, clf, ["vit_b_16"])
        tf_dct = min_scale(x, d_f, clf, ["densenet121_dct"])
        print(f"[joint 1/255 acq] stopped at step {usedf}; "
              f"min flip scale -> ViT-only={tf_vit}, DCT-only={tf_dct}, both={tf_both}")

        # 4. single-detector cost lower bound (fine step)
        singles = {}
        for n in names:
            d_s, us = acquire(x, branches, allow, step=1.0 / 255.0,
                              steps=240, only=n)
            t = min_scale(x, d_s, clf, [n])
            if t is None:
                print(f"[single {n:15s}] NOT fooled in 240 steps")
                continue
            singles[n] = d_s * t
            u = vida._uint8_hwc(x, d_s * t)
            s, l, q = official_q(clean_np, u, lp)
            print(f"[single {n:15s}] step {us}, scale t={t:.3f} -> "
                  f"SSIM={s:.4f} LPIPS={l:.4f} Q={q:.4f}")
            st_s = delta_stats(x, d_s * t, allow, center)
            print(f"           perturb: " + "  ".join(f"{k}={v:.3f}" for k, v in st_s.items()))

        # ViT gradient energy by region (where does ViT ask for perturbation?)
        gv2 = gv * gv
        tot = gv2.sum() + 1e-9
        print(f"[ViT grad energy] center128 share={float((gv2 * center).sum()/tot):.3f} "
              f"annulus share={float((gv2 * allow * (1-center)).sum()/tot):.3f}")
        w = torch.tensor([0.299, 0.587, 0.114], device=DEVICE).view(1, 3, 1, 1)
        gl = (gv * w).sum(1, keepdim=True) * w
        print(f"[ViT grad energy] luma share={float((gl*gl).sum()/tot):.3f} "
              f"chroma share={float(((gv-gl)**2).sum()/tot):.3f}")

        # 5. final v3.3 output margins + perturbation structure
        adv_u = vida.attack(clean_np, clf, DEVICE)
        s, l, q = official_q(clean_np, adv_u, lp)
        xf = torch.from_numpy(adv_u).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        df = (xf - x)
        mv_f = vida._margin(branches["vit_b_16"], xf, TARGET)
        md_f = vida._margin(branches["densenet121_dct"], xf, TARGET)
        st = delta_stats(x, df, allow, center)
        print(f"[final v3.3] SSIM={s:.4f} LPIPS={l:.4f} Q={q:.4f} "
              f"margins ViT={mv_f:+.2f} DCT={md_f:+.2f}")
        print("[final perturbation] " + "  ".join(f"{k}={v:.3f}" for k, v in st.items()))


if __name__ == "__main__":
    main()
