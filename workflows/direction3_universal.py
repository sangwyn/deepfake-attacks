"""Train and evaluate a reusable additive perturbation on held-out images."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Allow the workflow to import the evaluator when launched by absolute path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import (
    CLASS_IDX_REAL,
    compute_ssim_rgb,
    get_device,
    load_model,
    np_to_lpips_tensor,
    pil_to_np_rgb,
)
from attacks.dual_pgd import _vit_preprocess, dct_preprocess
import lpips


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def collect_images(root: Path, exclude: Path | None) -> list[Path]:
    excluded = {p.name for p in exclude.rglob("*") if p.is_file() and p.suffix.lower() in EXTS} if exclude else set()
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS and p.name not in excluded)


def make_split(paths: list[Path], seed: int, tuning: int, validation: int, out: Path) -> dict:
    order = list(range(len(paths)))
    random.Random(seed).shuffle(order)
    split = {
        "seed": seed,
        "source_root": str(paths[0].parents[0]) if paths else "",
        "tuning": [str(paths[i]) for i in order[:tuning]],
        "validation": [str(paths[i]) for i in order[tuning:tuning + validation]],
        "held_out": [str(paths[i]) for i in order[tuning + validation:]],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2))
    return split


def load_split(cfg: dict, paths: list[Path]) -> dict:
    split_path = Path(cfg["split_json"])
    if split_path.exists():
        return json.loads(split_path.read_text())
    return make_split(paths, int(cfg.get("seed", 0)), int(cfg.get("tuning_images", 100)), int(cfg.get("validation_images", 100)), split_path)


def to_batch(paths: list[str], size: int, device: torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(images).to(device)


def apply_delta(batch: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if delta.shape[-2:] != batch.shape[-2:]:
        delta = F.interpolate(delta, size=batch.shape[-2:], mode="bilinear", align_corners=False)
    return (batch + delta).clamp(0, 1)


def resized_metric_pair(path: str, attacked: torch.Tensor, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Put the original and attacked image on the same canvas for metrics."""
    height, width = attacked.shape[-2:]
    original = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    original_np = np.asarray(original).copy()
    attacked_np = (attacked[index].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    return original_np, attacked_np


def loss_for(batch: torch.Tensor, delta: torch.Tensor, vit, dct, target: int, vit_weight: float, dct_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attacked = apply_delta(batch, delta)
    target_tensor = torch.full((batch.shape[0],), target, dtype=torch.long, device=batch.device)
    vit_loss = F.cross_entropy(vit(_vit_preprocess(attacked)), target_tensor)
    dct_loss = F.cross_entropy(dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")), target_tensor)
    return vit_weight * vit_loss + dct_weight * dct_loss, vit_loss, dct_loss


def evaluate_split(paths: list[str], delta: torch.Tensor, vit, dct, lpips_fn, device: torch.device, batch_size: int) -> dict:
    vit_hits = dct_hits = 0
    ssims, lpips_values = [], []
    groups = {}
    for path in paths:
        groups.setdefault(Image.open(path).size, []).append(path)
    with torch.no_grad():
        for image_size, group in groups.items():
            for start in range(0, len(group), batch_size):
                names = group[start:start + batch_size]
                batch = to_batch(names, image_size[0], device)
                attacked = apply_delta(batch, delta)
                vit_pred = vit(_vit_preprocess(attacked)).argmax(1)
                dct_pred = dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")).argmax(1)
                vit_hits += int((vit_pred == CLASS_IDX_REAL).sum())
                dct_hits += int((dct_pred == CLASS_IDX_REAL).sum())
                for i, name in enumerate(names):
                    original, adv = resized_metric_pair(name, attacked, i)
                    ssims.append(compute_ssim_rgb(original, adv))
                    lpips_values.append(lpips_fn(np_to_lpips_tensor(original, device), np_to_lpips_tensor(adv, device)).item())
    n = len(paths)
    return {"images": n, "vit_success": vit_hits / n, "dct_success": dct_hits / n, "mean_ssim": float(np.mean(ssims)), "mean_lpips": float(np.mean(lpips_values))}


def main(cfg: dict) -> None:
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device(cfg.get("device", "cuda"))
    root = Path(cfg["original_root"])
    paths = collect_images(root, Path(cfg["exclude_root"]) if cfg.get("exclude_root") else None)
    split = load_split(cfg, paths)
    tuning, validation, held_out = split["tuning"], split["validation"], split["held_out"]
    models_dir = Path(cfg["models_dir"])
    vit = load_model("vit_b_16", models_dir / "vit_b_16.pth", device)
    dct = load_model("densenet121_dct", models_dir / "densenet121_dct.pth", device)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    size = int(cfg.get("universal_size", 256))
    epsilon = float(cfg.get("epsilon", 8 / 255))
    step_size = float(cfg.get("step_size", 1 / 255))
    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("epochs", 10))
    vit_weight = float(cfg.get("vit_weight", 0.5))
    dct_weight = float(cfg.get("dct_weight", 0.5))
    if vit_weight < 0 or dct_weight < 0 or vit_weight + dct_weight <= 0:
        raise ValueError("vit_weight and dct_weight must be non-negative and not both zero")
    delta = torch.zeros((1, 3, size, size), device=device, requires_grad=True)
    best_delta = delta.detach().clone()
    best_score = float("-inf")
    history = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(tuning)
        losses = []
        for start in range(0, len(tuning), batch_size):
            batch = to_batch(tuning[start:start + batch_size], size, device)
            loss, vit_loss, dct_loss = loss_for(batch, delta, vit, dct, CLASS_IDX_REAL, vit_weight, dct_weight)
            gradient = torch.autograd.grad(loss, delta)[0]
            delta = (delta - step_size * gradient.sign()).clamp(-epsilon, epsilon).detach().requires_grad_(True)
            losses.append((float(loss), float(vit_loss), float(dct_loss)))
        val = evaluate_split(validation, delta.detach(), vit, dct, lpips_fn, device, batch_size)
        record = {"epoch": epoch + 1, "loss": float(np.mean([x[0] for x in losses])), "vit_loss": float(np.mean([x[1] for x in losses])), "dct_loss": float(np.mean([x[2] for x in losses])), "validation": val}
        history.append(record)
        score = val["vit_success"] + val["dct_success"] - 0.5 * val["mean_lpips"]
        if score > best_score:
            best_score = score
            best_delta = delta.detach().clone()
        print(json.dumps(record), flush=True)
    out = Path(cfg["output_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta": best_delta.cpu(), "epsilon": epsilon, "seed": seed}, Path(cfg["checkpoint"]))
    result = {"config": cfg, "split": {k: len(v) for k, v in split.items() if isinstance(v, list)}, "history": history, "validation": evaluate_split(validation, best_delta, vit, dct, lpips_fn, device, batch_size), "held_out": evaluate_split(held_out, best_delta, vit, dct, lpips_fn, device, batch_size), "delta_linf": float(best_delta.abs().max())}
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(out), "checkpoint": str(cfg["checkpoint"]), "validation": result["validation"], "held_out": result["held_out"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(yaml.safe_load(Path(args.config).read_text()))
