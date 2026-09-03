"""Verify the 2026 ViT weights + celebA test set, and sanity-check white-box
attacks against BOTH detectors.

1. Clean accuracy of vit_b_16 and densenet121_dct on celebA/TEST
   (fakes -> pred 1, reals -> pred 0).
2. Targeted fake->real MI-FGSM against ViT only.
3. Targeted fake->real MI-FGSM against the JOINT ensemble (ViT + DCT).
"""
from pathlib import Path
import numpy as np
import torch
import lpips

from .detectors import load_detectors
from .data import DATASETS, build_manifest, load_image_tensor, to_uint8_hwc, post_save_linf
from .metrics import official_quality, quality_score
from . import attacks as A

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=True)
    lp = lpips.LPIPS(net="alex").to(device).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    root = DATASETS["celebA"]
    man = build_manifest(root)
    fakes = [m for m in man if m["cls"] == "fake"]
    reals = [m for m in man if m["cls"] == "real"]
    print(f"[data] {root}\n  {len(fakes)} fakes, {len(reals)} reals")

    # ---- 1. clean accuracy ----
    print("\n[1] Clean predictions (expect fakes->1, reals->0):")
    for name in packs:
        branch = packs[name]["branch"]
        cf = sum(int(branch(load_image_tensor(m["path"], device)[0]).argmax(1).item() == 1)
                 for m in fakes)
        cr = sum(int(branch(load_image_tensor(m["path"], device)[0]).argmax(1).item() == 0)
                 for m in reals)
        print(f"  {name:18s} fake-correct {cf}/{len(fakes)}  real-correct {cr}/{len(reals)}")

    # ---- 2/3. attacks on a subset ----
    n = 16
    steps = 60
    sub = fakes[:n]
    for tag, use in (("ViT-only", ("vit_b_16",)),
                     ("DCT-only", ("densenet121_dct",)),
                     ("JOINT ViT+DCT", ("vit_b_16", "densenet121_dct"))):
        vit_flip = dct_flip = 0
        ss, ll, linf_max = [], [], 0.0
        for m in sub:
            x_full, orig = load_image_tensor(m["path"], device)
            adv = A.ifgsm(x_full, packs, device, target=0, steps=steps,
                          use=use, momentum=1.0, use_border_mask=True)
            adv_u = to_uint8_hwc(adv)
            s, l = official_quality(orig, adv_u, lp, device)
            ss.append(s); ll.append(l); linf_max = max(linf_max, post_save_linf(orig, adv_u))
            with torch.no_grad():
                pv = packs["vit_b_16"]["branch"](adv).argmax(1).item()
                pd = packs["densenet121_dct"]["branch"](adv).argmax(1).item()
            vit_flip += int(pv == 0); dct_flip += int(pd == 0)
        s_m, l_m = np.mean(ss), np.mean(ll)
        print(f"\n[{tag}] attack source={use}")
        print(f"  ViT fooled {vit_flip}/{n}   DCT fooled {dct_flip}/{n}   "
              f"SSIM={s_m:.4f} LPIPS={l_m:.4f} Q={quality_score(s_m,l_m):.4f} "
              f"maxLinf={linf_max:.4f}")


if __name__ == "__main__":
    main()
