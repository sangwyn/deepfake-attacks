"""Save before/after adversarial samples for visual inspection.

For each selected image: run the final ViDA v3.4 attack, then save
  {stem}_orig.png     original
  {stem}_adv.png      adversarial (uint8, exactly what the evaluator scores)
  {stem}_compare.png  panel: original | adversarial | amplified perturbation
Perturbation panel: gray = no change; colours scaled x10 for visibility.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEAM = "/data2/aiattacks/Mingyao-Duan/team_repo"
HERE = str(Path(__file__).resolve().parent)
sys.path = [TEAM] + [p for p in sys.path if p not in ("", HERE)]

import evaluate as ev          # noqa: E402
import attacks.vida as vida    # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT = Path("/data2/aiattacks/Mingyao-Duan/my_attack/samples")
OUT.mkdir(exist_ok=True)
FAKE = Path("/home/aiattacks/dataset/celebA/TEST/TEST_FAKE")
REAL = Path("/home/aiattacks/dataset/celebA/TEST/TEST_REAL")

PICKS = [
    (FAKE / "001539_1024x1024.png", "fake-hard"),
    (FAKE / "001556_1024x1024.png", "fake-hard"),
    (FAKE / "001549_1024x1024.png", "fake-easy"),
    (REAL / "001501_1024x1024.jpg", "real"),
]


def main():
    weights = Path(TEAM) / "weights"
    clf = {}
    for name in ["vit_b_16", "densenet121_dct"]:
        model = ev.load_model(name, weights / f"{name}.pth", DEVICE)
        tf = ev.build_dct_transform(True) if name.endswith("_dct") \
            else ev.build_spatial_transform(name)
        clf[name] = {"model": model, "transform": tf}
    lp = vida._get_qmodels(DEVICE)[0]

    for p, tag in PICKS:
        orig = np.array(Image.open(p).convert("RGB"))
        adv = vida.attack(orig, clf, DEVICE)
        s = ev.compute_ssim_rgb(orig, adv)
        with torch.no_grad():
            l = float(lp(ev.np_to_lpips_tensor(orig, DEVICE),
                         ev.np_to_lpips_tensor(adv, DEVICE)).item())
        bad = vida._gt_bad(adv, clf, ["vit_b_16", "densenet121_dct"], DEVICE, 0)
        linf = np.abs(adv.astype(int) - orig.astype(int)).max()
        q = 0.5 * s + 0.5 * (1 - l)
        stem = p.stem.split("_")[0]

        Image.fromarray(orig).save(OUT / f"{stem}_{tag}_orig.png")
        Image.fromarray(adv).save(OUT / f"{stem}_{tag}_adv.png")

        # amplified perturbation: delta in [-1,1]-ish -> gray 128, x10
        delta = adv.astype(np.float32) - orig.astype(np.float32)
        amp = np.clip(128 + delta * 10, 0, 255).astype(np.uint8)
        h, w, _ = orig.shape
        panel = np.full((h, 3 * w + 20, 3), 255, dtype=np.uint8)
        panel[:, :w] = orig
        panel[:, w + 10:2 * w + 10] = adv
        panel[:, 2 * w + 20:] = amp
        Image.fromarray(panel).save(OUT / f"{stem}_{tag}_COMPARE.png")

        print(f"{stem} [{tag}]: Q={q:.4f} SSIM={s:.4f} LPIPS={l:.4f} "
              f"fooled={'both' if not bad else bad} L_inf={linf}/255", flush=True)

    print(f"\nsamples -> {OUT}")


if __name__ == "__main__":
    main()
