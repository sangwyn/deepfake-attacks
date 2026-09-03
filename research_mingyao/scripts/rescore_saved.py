"""Independent re-scoring of saved adversarial images.

This script does NOT import or use any attack code. It only:
  1. loads original images and the saved adversarial PNGs,
  2. runs the official detector factory + official transforms,
  3. computes SSIM (skimage) and LPIPS-Alex exactly like evaluate.py,
  4. reports per-image and aggregate results.

Usage:
  python my_attack/rescore_saved.py \
      --orig /home/aiattacks/dataset/celebA/TEST \
      --adv  my_attack/verify_outputs/vida_v34_images \
      --weights team_repo/weights
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEAM = "/data2/aiattacks/Mingyao-Duan/team_repo"
sys.path.insert(0, TEAM)

import evaluate as ev          # official evaluator module ONLY (no attacks/)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--adv", required=True)
    ap.add_argument("--weights", default=str(Path(TEAM) / "weights"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    weights = Path(args.weights)
    clf = {}
    for name in ["vit_b_16", "densenet121_dct"]:
        model = ev.load_model(name, weights / f"{name}.pth", device)
        tf = ev.build_dct_transform(True) if name.endswith("_dct") \
            else ev.build_spatial_transform(name)
        clf[name] = {"model": model, "transform": tf}
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    adv_root = Path(args.adv)
    adv_paths = sorted(p for p in adv_root.rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"found {len(adv_paths)} saved adversarial images under {adv_root}")

    total = 0.0
    rows = []
    n_fooled = {"vit_b_16": 0, "densenet121_dct": 0}
    for ap_path in adv_paths:
        rel = ap_path.relative_to(adv_root)
        orig_path = Path(args.orig) / rel
        if not orig_path.exists():
            # saved images are .png; try matching stem under original root
            cand = list(Path(args.orig).rglob(ap_path.stem + ".*"))
            if not cand:
                print(f"  [skip] no original for {rel}")
                continue
            orig_path = cand[0]
        img_o = ev.pil_to_np_rgb(orig_path)
        img_a = ev.pil_to_np_rgb(ap_path)
        ssim_v = ev.compute_ssim_rgb(img_o, img_a)
        with torch.no_grad():
            lpips_v = float(lpips_fn(ev.np_to_lpips_tensor(img_o, device),
                                     ev.np_to_lpips_tensor(img_a, device)).item())
        q = 0.5 * ssim_v + 0.5 * (1.0 - lpips_v)
        pair = 0.0
        preds = {}
        for name, pack in clf.items():
            t = pack["transform"](Image.fromarray(img_a)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(pack["model"](t).argmax(1).item())
            fooled = int(pred == ev.CLASS_IDX_REAL)
            preds[name] = "Real" if fooled else "Fake"
            n_fooled[name] += fooled
            pair += q * fooled
        total += pair
        rows.append((str(rel), ssim_v, lpips_v, q, preds["vit_b_16"],
                     preds["densenet121_dct"], pair))

    print(f"\n{'image':40s} {'SSIM':>7} {'LPIPS':>7} {'Q':>6} {'ViT':>5} {'DCT':>5} {'score':>7}")
    for r in rows:
        print(f"{r[0]:40s} {r[1]:7.4f} {r[2]:7.4f} {r[3]:6.3f} {r[4]:>5} {r[5]:>5} {r[6]:7.4f}")
    n = len(rows)
    print(f"\nimages rescored : {n}")
    print(f"ViT  Real rate  : {n_fooled['vit_b_16']}/{n}")
    print(f"DCT  Real rate  : {n_fooled['densenet121_dct']}/{n}")
    print(f"mean SSIM       : {np.mean([r[1] for r in rows]):.4f}")
    print(f"mean LPIPS      : {np.mean([r[2] for r in rows]):.4f}")
    print(f"TOTAL SCORE (sum, max 2 per image): {total:.4f} / {2*n}")


if __name__ == "__main__":
    main()
