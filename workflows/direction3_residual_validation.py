"""Evaluate universal noise followed by a small per-image joint residual."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import lpips

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import CLASS_IDX_REAL, compute_ssim_rgb, load_model, np_to_lpips_tensor, pil_to_np_rgb
from attacks.dual_pgd import _vit_preprocess, dct_preprocess


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(cfg["split_json"]).read_text())
    paths = split[cfg.get("split", "validation")]
    pack = torch.load(cfg["universal_checkpoint"], map_location=device)
    universal = pack["delta"].to(device)
    eps = float(cfg.get("epsilon", 8 / 255))
    residual_eps = float(cfg.get("residual_epsilon", 2 / 255))
    step = float(cfg.get("step_size", 0.5 / 255))
    iterations = int(cfg.get("iterations", 10))
    batch_size = int(cfg.get("batch_size", 2))
    vit_weight = float(cfg.get("vit_weight", 0.5))
    dct_weight = float(cfg.get("dct_weight", 0.5))
    models = Path(cfg["models_dir"])
    vit = load_model("vit_b_16", models / "vit_b_16.pth", device)
    dct = load_model("densenet121_dct", models / "densenet121_dct.pth", device)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    vit_hits = dct_hits = 0
    ssims, lpips_values = [], []
    groups = {}
    for name in paths:
        groups.setdefault(Image.open(name).size, []).append(name)
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            names = group[start:start + batch_size]
            size = Image.open(names[0]).size[0]
            images = [torch.from_numpy(np.asarray(Image.open(name).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 for name in names]
            original = torch.stack(images).to(device)
            universal_local = F.interpolate(universal, size=(size, size), mode="bilinear", align_corners=False).expand_as(original)
            residual = torch.zeros_like(original)
            base = original.clone()
            for _ in range(iterations):
                residual.requires_grad_(True)
                attacked = (base + universal_local + residual).clamp(0, 1)
                target = torch.zeros(len(names), dtype=torch.long, device=device)
                loss = vit_weight * F.cross_entropy(vit(_vit_preprocess(attacked)), target) + dct_weight * F.cross_entropy(dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")), target)
                grad = torch.autograd.grad(loss, residual)[0]
                residual = (residual - step * grad.sign()).clamp(-residual_eps, residual_eps).detach()
            attacked = (base + universal_local + residual).clamp(0, 1)
            with torch.no_grad():
                vit_pred = vit(_vit_preprocess(attacked)).argmax(1)
                dct_pred = dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")).argmax(1)
            vit_hits += int((vit_pred == CLASS_IDX_REAL).sum())
            dct_hits += int((dct_pred == CLASS_IDX_REAL).sum())
            for i, name in enumerate(names):
                height, width = attacked.shape[-2:]
                orig = np.asarray(Image.open(name).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)).copy()
                adv = (attacked[i].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
                ssims.append(compute_ssim_rgb(orig, adv))
                lpips_values.append(lpips_fn(np_to_lpips_tensor(orig, device), np_to_lpips_tensor(adv, device)).item())
    n = len(paths)
    result = {"images": n, "universal_checkpoint": cfg["universal_checkpoint"], "residual_epsilon": residual_eps, "iterations": iterations, "vit_success": vit_hits / n, "dct_success": dct_hits / n, "mean_ssim": float(np.mean(ssims)), "mean_lpips": float(np.mean(lpips_values))}
    Path(cfg["output_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output_json"]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(yaml.safe_load(Path(args.config).read_text()))
