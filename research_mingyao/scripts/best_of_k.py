"""Best-of-k multi-start experiment for ViDA v3.4-final-no-stage0.

Runs the FULL attack pipeline k independent times per image (independent
random seeds -> independent acquisition/rescue trajectories) and records the
per-run Q, best/mean/std Q, detector fooling and runtime. The selection rule
is argmax(number_of_detectors_fooled, Q).

Images: 001539 (hard), 001556 (hard), 001553 (rescue-effective), 001549 (easy).
Usage: python my_attack/best_of_k.py --k 3 [--gpu 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEAM = "/data2/aiattacks/Mingyao-Duan/team_repo"
HERE = str(Path(__file__).resolve().parent)
sys.path = [TEAM] + [p for p in sys.path if p not in ("", HERE)]

import evaluate as ev          # noqa: E402
import attacks.vida as vida    # noqa: E402

IMAGES = [
    ("001539", "hard"),
    ("001556", "hard"),
    ("001553", "rescue-effective"),
    ("001549", "easy"),
]
FAKE_DIR = Path("/home/aiattacks/dataset/celebA/TEST/TEST_FAKE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gpu", default="2")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    weights = Path(TEAM) / "weights"
    clf = {}
    for name in ["vit_b_16", "densenet121_dct"]:
        model = ev.load_model(name, weights / f"{name}.pth", device)
        tf = ev.build_dct_transform(True) if name.endswith("_dct") \
            else ev.build_spatial_transform(name)
        clf[name] = {"model": model, "transform": tf}
    lp = vida._get_qmodels(device)[0]

    report = {"k": args.k, "images": {}}
    for stem, tag in IMAGES:
        img = np.array(Image.open(FAKE_DIR / f"{stem}_1024x1024.png").convert("RGB"))
        runs = []
        for run in range(args.k):
            torch.manual_seed(10000 + hash((stem, run)) % 50000)
            t0 = time.time()
            adv = vida.attack(img, clf, device)
            dt = time.time() - t0
            s = ev.compute_ssim_rgb(img, adv)
            with torch.no_grad():
                l = float(lp(ev.np_to_lpips_tensor(img, device),
                             ev.np_to_lpips_tensor(adv, device)).item())
            bad = vida._gt_bad(adv, clf, ["vit_b_16", "densenet121_dct"], device, 0)
            q = 0.5 * s + 0.5 * (1.0 - l)
            runs.append({"run": run, "ssim": round(s, 4), "lpips": round(l, 4),
                         "q": round(q, 4), "n_fooled": 2 - len(bad),
                         "bad_detectors": bad, "time_s": round(dt, 1)})
            print(f"  {stem} run{run}: Q={q:.4f} SSIM={s:.4f} LPIPS={l:.4f} "
                  f"fooled={2-len(bad)}/2 t={dt:.0f}s", flush=True)
        qs = np.array([r["q"] for r in runs])
        fooled = np.array([r["n_fooled"] for r in runs])
        # best = argmax(n_fooled, q)
        best_idx = int(np.lexsort((-qs, -fooled))[0])
        summary = {
            "tag": tag,
            "runs": runs,
            "best_run": best_idx,
            "best_q": float(qs.max()),
            "mean_q": float(qs.mean()),
            "std_q": float(qs.std()),
            "min_q": float(qs.min()),
            "all_fooled_every_run": bool((fooled == 2).all()),
            "total_time_s": round(sum(r["time_s"] for r in runs), 1),
        }
        report["images"][stem] = summary
        print(f"{stem} [{tag}]: best Q={qs.max():.4f} mean={qs.mean():.4f} "
              f"std={qs.std():.4f} min={qs.min():.4f} | single-run expected(mean) "
              f"-> best-of-{args.k} gain: {qs.max()-qs.mean():+.4f}", flush=True)

    out = Path(HERE) / f"best_of_k_results_k{args.k}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
