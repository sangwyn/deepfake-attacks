"""Check run-to-run variance of vida.attack on the two hardest fakes."""
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

DEVICE = torch.device("cuda:0")
weights = Path(TEAM) / "weights"
clf = {}
for name in ["vit_b_16", "densenet121_dct"]:
    model = ev.load_model(name, weights / f"{name}.pth", DEVICE)
    tf = ev.build_dct_transform(True) if name.endswith("_dct") \
        else ev.build_spatial_transform(name)
    clf[name] = {"model": model, "transform": tf}
lp = vida._get_qmodels(DEVICE)[0]

branches = {
    "vit": vida.ViTBranch(clf["vit_b_16"]["model"]).to(DEVICE).eval(),
    "dct": vida.DCTBranch(clf["densenet121_dct"]["model"], DEVICE).to(DEVICE).eval(),
}
for b in branches.values():
    for p in b.model.parameters():
        p.requires_grad_(False)

import time
fake_dir = Path("/home/aiattacks/dataset/celebA/TEST/TEST_FAKE")
for stem in ["001539", "001556"]:
    img = np.array(Image.open(fake_dir / f"{stem}_1024x1024.png").convert("RGB"))
    print(f"\n=== {stem} ===", flush=True)
    for run in range(2):
        t0 = time.time()
        adv = vida.attack(img, clf, DEVICE)
        dt = time.time() - t0
        s = ev.compute_ssim_rgb(img, adv)
        with torch.no_grad():
            l = float(lp(ev.np_to_lpips_tensor(img, DEVICE),
                         ev.np_to_lpips_tensor(adv, DEVICE)).item())
        bad = vida._gt_bad(adv, clf, ["vit_b_16", "densenet121_dct"], DEVICE, 0)
        xf = torch.from_numpy(adv).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        x0 = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        mv = vida._margin(branches["vit"], xf, 0)
        md = vida._margin(branches["dct"], xf, 0)
        linf = float((xf - x0).abs().max())
        print(f"  run{run}: SSIM={s:.4f} LPIPS={l:.4f} Q={0.5*s+0.5*(1-l):.4f} "
              f"gt_bad={bad} margins ViT={mv:+.2f} DCT={md:+.2f} "
              f"linf={linf:.4f} time={dt:.0f}s", flush=True)
