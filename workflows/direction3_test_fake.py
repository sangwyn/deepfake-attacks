"""Evaluate Direction 3 universal-plus-residual on every TEST_FAKE image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.dual_pgd import _vit_preprocess, dct_preprocess  # noqa: E402
from evaluate import compute_ssim_rgb, load_model, np_to_lpips_tensor  # noqa: E402


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["input_root"])
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    assert len(paths) == 100, f"expected 100 TEST_FAKE images, found {len(paths)}"
    pack = torch.load(cfg["universal_checkpoint"], map_location=device)
    universal = pack["delta"].to(device)
    vit = load_model("vit_b_16", Path(cfg["models_dir"]) / "vit_b_16.pth", device).eval()
    dct = load_model("densenet121_dct", Path(cfg["models_dir"]) / "densenet121_dct.pth", device).eval()
    for model in (vit, dct):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        original = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        size = original.shape[0]
        batch = torch.from_numpy(original).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
        u = F.interpolate(universal, size=(size, size), mode="bilinear", align_corners=False)
        residual = torch.zeros_like(batch)
        started = time.perf_counter()
        for _ in range(int(cfg["iterations"])):
            residual.requires_grad_(True)
            attacked = (batch + u + residual).clamp(0, 1)
            target = torch.zeros(1, dtype=torch.long, device=device)
            loss = float(cfg["vit_weight"]) * F.cross_entropy(vit(_vit_preprocess(attacked)), target)
            loss += float(cfg["dct_weight"]) * F.cross_entropy(dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")), target)
            gradient = torch.autograd.grad(loss, residual)[0]
            residual = (residual - float(cfg["step_size"]) * gradient.sign()).clamp(
                -float(cfg["residual_epsilon"]), float(cfg["residual_epsilon"])).detach()
        attacked = (batch + u + residual).clamp(0, 1)
        elapsed = time.perf_counter() - started
        output = (attacked[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        Image.fromarray(output).save(output_root / path.name)
        with torch.no_grad():
            vit_pred = int(vit(_vit_preprocess(attacked)).argmax(1).item())
            dct_pred = int(dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")).argmax(1).item())
            lp = float(lpips_fn(np_to_lpips_tensor(original, device), np_to_lpips_tensor(output, device)).item())
        rows.append({"image": path.name, "vit_real": vit_pred == 0, "dct_real": dct_pred == 0,
                     "ssim": float(compute_ssim_rgb(original, output)), "lpips": lp,
                     "seconds": elapsed})
    result = {"images": len(rows), "method": "universal_plus_residual", "rows": rows,
              "vit_real_rate": float(np.mean([r["vit_real"] for r in rows])),
              "dct_real_rate": float(np.mean([r["dct_real"] for r in rows])),
              "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
              "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
              "seconds": float(sum(r["seconds"] for r in rows))}
    Path(cfg["output_json"]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(yaml.safe_load(Path(args.config).read_text()))
