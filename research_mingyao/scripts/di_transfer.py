"""Does Diverse-Inputs (DI) improve BLACK-BOX transfer of ViDA?

Generates fake->real adversarial images two ways (both white-box vs vit+dct):
  * MI  : ViDA momentum attack (di_prob=0)
  * DI  : ViDA with input diversity (di_prob=0.5)
then scores fool rate on all FOUR detectors: vit/dct (white-box) and
npr/aide (black-box transfer), with clean-label orientation calibration.

Run:  python -u -m my_attack.di_transfer --n 40
"""
import argparse
import sys
from pathlib import Path

import torch

from .detectors import load_detectors as load_wb
from .data import DATASETS, build_manifest, load_image_tensor
from . import vida_attack as VA

WBW = Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")
BW = Path("/data2/aiattacks/Mingyao-Duan/my_attack/third_weights")
REF = Path("/data2/aiattacks/Mingyao-Duan/my_attack/refcode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--tail", type=int, default=60)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    wb = load_wb(WBW, device, use_dct=True, use_vit=True)
    sys.path.insert(0, str(REF))
    import detectors as Odet
    arch = REF / "deepfakes_code"
    bb = {n: Odet.load_detector(n, BW / f"{n}.pth", device, architecture_root=arch)
          for n in ["npr", "aide"]}
    print("[load] 4 detectors ready")

    dets = ["vit_b_16", "densenet121_dct", "npr", "aide"]
    fakes = [m for m in build_manifest(DATASETS["celebA"]) if m["cls"] == "fake"][: args.n]
    n = len(fakes)

    def preds(x):
        p = {}
        with torch.no_grad():
            for k in ("vit_b_16", "densenet121_dct"):
                p[k] = int(wb[k]["branch"](x).argmax(1).item())
            for k in ("npr", "aide"):
                p[k] = int(bb[k](x).argmax(1).item())
        return p

    # clean label orientation calibration (clean fakes should be Fake)
    clean_real = {d: 0 for d in dets}
    for m in fakes:
        x, _ = load_image_tensor(m["path"], device)
        pc = preds(x)
        for d in dets:
            clean_real[d] += int(pc[d] == 0)

    results = {"MI": {d: 0 for d in dets}, "DI": {d: 0 for d in dets}}
    for i, m in enumerate(fakes):
        x, _ = load_image_tensor(m["path"], device)
        for tag, dip in (("MI", 0.0), ("DI", 0.5)):
            d, _ = VA.vida_attack(x, wb, device, steps=args.steps,
                                  tail_steps=args.tail, di_prob=dip)
            pa = preds((x + d).clamp(0, 1))
            for dd in dets:
                results[tag][dd] += int(pa[dd] == 0)
        if (i + 1) % 4 == 0:
            print(f"  ... {i+1}/{n}")

    role = {"vit_b_16": "WB", "densenet121_dct": "WB", "npr": "black-box", "aide": "black-box"}
    print(f"\n===== DI vs MI transfer ({n} fakes) fool->Real rate =====")
    print("  detector           role        MI(%)    DI(%)")
    for det in dets:
        inverted = clean_real[det] > 0.75 * n
        mi = (n - results["MI"][det]) if inverted else results["MI"][det]
        di = (n - results["DI"][det]) if inverted else results["DI"][det]
        print(f"  {det:16s} {role[det]:10s} {mi/n*100:7.1f} {di/n*100:7.1f}")
    print("  (WB rows expected ~100%; npr/aide rows measure transfer; DI should help there)")


if __name__ == "__main__":
    main()
