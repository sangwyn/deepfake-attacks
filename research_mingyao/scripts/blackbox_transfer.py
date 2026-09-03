"""Black-box transfer evaluation of ViDA adversarial images.

ViDA is a WHITE-BOX attack crafted against the two graded detectors
(vit_b_16 + densenet121_dct). Oleg added two extra detectors — npr (ResNet-50)
and aide (ConvNeXt-XXL) — purely as TRANSFER (black-box) targets: they are
never used to compute gradients. For each fake image we

  1. generate the adversarial image with the white-box ViDA attack (vit+dct),
  2. run ALL FOUR detectors on both the CLEAN and the ADVERSARIAL image,
  3. report clean accuracy (should say Fake) and fool rate (says Real) per
     detector.

The first two detectors are white-box targets (expect ~100%); npr and aide
measure black-box transfer.

Run:  python -u -m my_attack.blackbox_transfer --n 40
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from .detectors import load_detectors as load_wb
from .data import DATASETS, build_manifest, load_image_tensor
from . import vida_attack as VA

REF = Path("/data2/aiattacks/Mingyao-Duan/my_attack/refcode")
WBW = Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")
BW = Path("/data2/aiattacks/Mingyao-Duan/my_attack/third_weights")


def load_black_box(device):
    sys.path.insert(0, str(REF))
    import detectors as Odet
    arch = REF / "deepfakes_code"
    out = {}
    for name in ["npr", "aide"]:
        out[name] = Odet.load_detector(name, BW / f"{name}.pth", device,
                                       architecture_root=arch)
    return out


def pred_whitebox(packs, x, name):
    with torch.no_grad():
        return int(packs[name]["branch"](x).argmax(1).item())  # 0 = Real


def pred_blackbox(bbdet, x):
    with torch.no_grad():
        return int(bbdet(x).argmax(1).item())                  # 0 = Real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--tail", type=int, default=60)
    ap.add_argument("--rec", type=int, default=40)
    ap.add_argument("--no-rec", action="store_true", help="skip quality recovery")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    wb = load_wb(WBW, device, use_dct=True, use_vit=True)
    print("[load] white-box vit_b_16 + densenet121_dct ready")
    bb = load_black_box(device)
    print("[load] black-box npr + aide ready")

    fakes = [m for m in build_manifest(DATASETS["celebA"]) if m["cls"] == "fake"][: args.n]
    dets = ["vit_b_16", "densenet121_dct", "npr", "aide"]
    clean_real = {d: 0 for d in dets}   # times adapter predicts Real on CLEAN fakes
    adv_real = {d: 0 for d in dets}     # times adapter predicts Real on ADVERSARIAL

    def all_preds(x):
        p = {}
        for name in ("vit_b_16", "densenet121_dct"):
            p[name] = pred_whitebox(wb, x, name)
        for name in ("npr", "aide"):
            p[name] = pred_blackbox(bb[name], x)
        return p  # raw predicted class (0 = Real in the adapter's own orientation)

    for i, m in enumerate(fakes):
        x, _ = load_image_tensor(m["path"], device)
        pc = all_preds(x)
        for d in dets:
            clean_real[d] += int(pc[d] == 0)   # adapter calls clean fake "Real"?

        # white-box ViDA attack
        d, _ = VA.vida_attack(x, wb, device, steps=args.steps, tail_steps=args.tail)
        adv = (x + d).clamp(0, 1)
        if not args.no_rec:
            import lpips
            if not hasattr(main, "_lp"):
                main._lp = lpips.LPIPS(net="alex").to(device).eval()
                for p in main._lp.parameters():
                    p.requires_grad_(False)
                from .metrics import DiffSSIM
                main._ds = DiffSSIM().to(device)
            d2, _ = VA.quality_recovery(x, d, wb, main._lp, main._ds, device,
                                        rec_steps=args.rec)
            adv = (x + d2).clamp(0, 1)

        pa = all_preds(adv)
        for dd in dets:
            adv_real[dd] += int(pa[dd] == 0)

        if (i + 1) % 4 == 0:
            print(f"  ... {i+1}/{len(fakes)}")

    n = len(fakes)
    # Calibrate orientation: clean fakes are known Fake. If a detector's adapter
    # calls most of them "Real" (argmax 0), its orientation is inverted (this
    # happens for the AIDE checkpoint under the shared adapter's column swap).
    print(f"\n===== Transfer results ({n} fakes; ViDA white-box vs vit+dct) =====")
    role = {"vit_b_16": "white-box", "densenet121_dct": "white-box",
            "npr": "BLACK-BOX", "aide": "BLACK-BOX"}
    print(f"  {'detector':16s} {'role':10s} {'clean(Fake acc)':>16s} {'adv fooled->Real':>17s}")
    for det in dets:
        inverted = clean_real[det] > 0.75 * n          # adapter inverted for this detector
        # clean fakes correctly labeled Fake (invert orientation if needed)
        clean_fake_acc = clean_real[det] if inverted else (n - clean_real[det])
        fooled = (n - adv_real[det]) if inverted else adv_real[det]
        flag = " [orientation auto-corrected]" if inverted else ""
        print(f"  {det:16s} {role[det]:10s} {clean_fake_acc:>13}/{n}  "
              f"{fooled:>14}/{n} ({fooled/n*100:5.1f}%){flag}")
    print("  Note: white-box rows should be ~100%; npr/aide rows are the transfer rate.")


if __name__ == "__main__":
    main()
