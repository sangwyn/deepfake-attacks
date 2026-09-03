"""DCT-side smoke / correctness test for the ViDA harness.

Checks:
  C0  differentiable DCT-II matches scipy orthonormal DCT.
  C1  differentiable DCT branch reproduces the OFFICIAL PIL/scipy path
      (logits + argmax) on clean images.
  C2  identity attack: SSIM=1, LPIPS~0, prediction unchanged, Linf=0.
  C3  targeted I-FGSM on the DCT detector (fake->real): ASR / SSIM / LPIPS /
      post-save Linf.
  C4  real->fake sanity (target=1).

Run:  .venv/bin/python -m my_attack.smoke_dct --n 8 --steps 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fftpack import dct as scipy_dct

import lpips

from . import data as D
from . import attacks as A
from .detectors import load_detectors, create_dct_model
from .dct_ops import dct_matrix
from .metrics import official_quality, quality_score

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/team_repo/weights")
DATA = Path("/data2/aiattacks/dataset")


# --------------------------------------------------------------------------
# Official (non-differentiable) reference DCT input, copied from evaluate.py
# --------------------------------------------------------------------------
def official_dct_input(pil_img: Image.Image, device) -> torch.Tensor:
    img = pil_img.convert("L")
    if max(img.size) > 256:
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
    w, h = img.size
    left = (w - 128) // 2
    top = (h - 128) // 2
    img = img.crop((left, top, left + 128, top + 128))
    arr = np.array(img, dtype=np.float32)
    coeff = scipy_dct(scipy_dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
    coeff = np.log(np.abs(coeff) + 1e-6)
    return torch.from_numpy(coeff).view(1, 1, 128, 128).to(device)


def c0_dct_matches_scipy():
    n = 128
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, n)).astype(np.float32)
    ref = scipy_dct(scipy_dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")
    Dm = dct_matrix(n)
    xt = torch.from_numpy(x)
    mine = (Dm @ xt @ Dm.t()).numpy()
    err = np.abs(ref - mine).max()
    print(f"[C0] DCT-II max abs diff vs scipy: {err:.3e}  -> {'PASS' if err < 1e-3 else 'FAIL'}")
    return err < 1e-3


@torch.no_grad()
def predict(model, branch_or_tensor, x_full=None):
    if x_full is not None:
        logits = branch_or_tensor(x_full)
    else:
        logits = model(branch_or_tensor)
    pred = logits.argmax(1).item()
    return pred, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--alpha", type=float, default=2 / 255)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=False)
    dct_pack = packs["densenet121_dct"]
    model = dct_pack["model"]
    branch = dct_pack["branch"]
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    ok = True
    ok &= c0_dct_matches_scipy()

    manifest = D.build_manifest(DATA)
    fakes = [m for m in manifest if m["cls"] == "fake"][: args.n]
    reals = [m for m in manifest if m["cls"] == "real"][: args.n]
    print(f"[data] {len(fakes)} fake + {len(reals)} real images")

    # ---- C1: differentiable branch vs official path on CLEAN images ----
    print("\n[C1] differentiable DCT branch vs official PIL/scipy path")
    agree = 0
    max_logit_diff = 0.0
    for m in fakes[: min(8, len(fakes))] + reals[: min(4, len(reals))]:
        pil = Image.open(m["path"]).convert("RGB")
        ref_in = official_dct_input(pil, device)
        ref_logits = model(ref_in)
        ref_pred = ref_logits.argmax(1).item()

        x_full, _ = D.load_image_tensor(m["path"], device)
        diff_logits = branch(x_full)
        diff_pred = diff_logits.argmax(1).item()

        d = (ref_logits - diff_logits).abs().max().item()
        max_logit_diff = max(max_logit_diff, d)
        agree += int(ref_pred == diff_pred)
        print(f"    {Path(m['path']).name:28s} official={ref_pred} diff={diff_pred} "
              f"logit|Δ|max={d:.4f} {'ok' if ref_pred == diff_pred else 'MISMATCH'}")
    total = min(8, len(fakes)) + min(4, len(reals))
    print(f"[C1] argmax agreement {agree}/{total}, max logit diff {max_logit_diff:.4f}")
    c1 = agree == total and max_logit_diff < 1.0
    print(f"[C1] -> {'PASS' if c1 else 'CHECK'}")
    ok &= c1

    # ---- C2: identity ----
    print("\n[C2] identity attack")
    m = fakes[0]
    x_full, orig = D.load_image_tensor(m["path"], device)
    adv = A.identity(x_full, packs, device)
    adv_u = D.to_uint8_hwc(adv)
    s, l = official_quality(orig, adv_u, lpips_fn, device)
    linf = D.post_save_linf(orig, adv_u)
    pred_clean, _ = predict(model, branch, x_full)
    pred_adv, _ = predict(model, branch, adv)
    print(f"    SSIM={s:.4f} LPIPS={l:.5f} Linf={linf:.4f} "
          f"pred clean={pred_clean} adv={pred_adv}")
    c2 = s > 0.999 and linf == 0.0 and pred_clean == pred_adv
    print(f"[C2] -> {'PASS' if c2 else 'FAIL'}")
    ok &= c2

    # ---- C3: targeted I-FGSM fake->real on DCT ----
    print(f"\n[C3] I-FGSM DCT-only  fake->real  steps={args.steps} eps={args.eps:.4f}")
    fooled = 0
    rows = []
    for m in fakes:
        x_full, orig = D.load_image_tensor(m["path"], device)
        clean_pred, _ = predict(model, branch, x_full)
        adv = A.ifgsm(x_full, packs, device, target=0, eps=args.eps,
                      alpha=args.alpha, steps=args.steps,
                      use=("densenet121_dct",))
        adv_u = D.to_uint8_hwc(adv)
        s, l = official_quality(orig, adv_u, lpips_fn, device)
        linf = D.post_save_linf(orig, adv_u)
        adv_pred, _ = predict(model, branch, adv)
        fooled += int(adv_pred == 0)
        rows.append((Path(m["path"]).name, clean_pred, adv_pred, s, l, linf))
    print(f"    {'image':28s} {'clean':>5} {'adv':>4} {'SSIM':>7} {'LPIPS':>7} {'Linf':>6}")
    for name, cp, ap_, s, l, linf in rows:
        print(f"    {name:28s} {cp:>5} {ap_:>4} {s:>7.4f} {l:>7.4f} {linf:>6.4f}")
    asr = fooled / max(1, len(fakes))
    mean_s = np.mean([r[3] for r in rows])
    mean_l = np.mean([r[4] for r in rows])
    max_linf = max(r[5] for r in rows)
    print(f"[C3] ASR(fake->real)={asr:.3f} ({fooled}/{len(fakes)})  "
          f"mean SSIM={mean_s:.4f} mean LPIPS={mean_l:.4f} max Linf={max_linf:.4f}")
    c3 = max_linf <= args.eps + 1.5 / 255
    print(f"[C3] budget check (Linf<=eps+rounding) -> {'PASS' if c3 else 'FAIL'}")
    ok &= c3

    # ---- C4: real->fake sanity ----
    print(f"\n[C4] I-FGSM DCT-only  real->fake  steps={args.steps}")
    fooled_r = 0
    for m in reals[: min(4, len(reals))]:
        x_full, orig = D.load_image_tensor(m["path"], device)
        clean_pred, _ = predict(model, branch, x_full)
        adv = A.ifgsm(x_full, packs, device, target=1, eps=args.eps,
                      alpha=args.alpha, steps=args.steps,
                      use=("densenet121_dct",))
        adv_pred, _ = predict(model, branch, adv)
        adv_u = D.to_uint8_hwc(adv)
        s, l = official_quality(orig, adv_u, lpips_fn, device)
        fooled_r += int(adv_pred == 1)
        print(f"    {Path(m['path']).name:28s} clean={clean_pred} adv={adv_pred} "
              f"SSIM={s:.4f} LPIPS={l:.4f}")
    print(f"[C4] real->fake flip rate {fooled_r}/{min(4, len(reals))} "
          f"(sanity; clean real pred should be 0)")

    print("\n" + "=" * 60)
    print("OVERALL:", "ALL PASS" if ok else "SOME CHECKS NEED ATTENTION")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
