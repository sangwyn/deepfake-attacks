"""Run a strict held-out ablation for reusable and image-specific perturbations."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.dual_pgd import _vit_preprocess, dct_preprocess
from evaluate import CLASS_IDX_REAL, compute_ssim_rgb, load_model, np_to_lpips_tensor


def load_batch(names: list[str], size: int, device: torch.device) -> torch.Tensor:
    images = []
    for name in names:
        image = Image.open(name).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255)
    return torch.stack(images).to(device)


def metric_pair(name: str, attacked: torch.Tensor, index: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = attacked.shape[-2:]
    original = np.asarray(Image.open(name).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)).copy()
    adv = (attacked[index].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    return original, adv


def evaluate_case(cfg: dict, paths: list[str], universal: torch.Tensor | None, mode: str,
                  residual_eps: float, vit, dct, lpips_fn, device: torch.device) -> dict:
    batch_size = int(cfg.get("batch_size", 2))
    iterations = int(cfg.get("iterations", 10)) if mode != "universal_only" else 0
    step = float(cfg.get("step_size", 0.5 / 255))
    vit_weight = float(cfg.get("vit_weight", 0.5))
    dct_weight = float(cfg.get("dct_weight", 0.5))
    epsilon = float(cfg.get("epsilon", 8 / 255))
    clean_vit = clean_dct = adv_vit = adv_dct = 0
    vit_eligible = dct_eligible = joint_eligible = 0
    vit_asr = dct_asr = joint_asr = 0
    ssims, lpips_values = [], []
    elapsed = 0.0
    groups: dict[tuple[int, int], list[str]] = {}
    for name in paths:
        groups.setdefault(Image.open(name).size, []).append(name)

    for image_size, group in groups.items():
        size = image_size[0]
        for start in range(0, len(group), batch_size):
            names = group[start:start + batch_size]
            batch = load_batch(names, size, device)
            with torch.no_grad():
                clean_vit_pred = vit(_vit_preprocess(batch)).argmax(1)
                clean_dct_pred = dct(dct_preprocess(batch, log_scale=True, resize_mode="bicubic")).argmax(1)
            clean_vit += int((clean_vit_pred == CLASS_IDX_REAL).sum())
            clean_dct += int((clean_dct_pred == CLASS_IDX_REAL).sum())
            if universal is None:
                universal_local = torch.zeros_like(batch)
            else:
                universal_local = F.interpolate(universal, size=(size, size), mode="bilinear", align_corners=False).expand_as(batch)
            residual = torch.zeros_like(batch)
            started = time.perf_counter()
            for _ in range(iterations):
                residual.requires_grad_(True)
                base_delta = universal_local if mode == "universal_plus_residual" else torch.zeros_like(batch)
                attacked = (batch + base_delta + residual).clamp(0, 1)
                target = torch.full((len(names),), CLASS_IDX_REAL, dtype=torch.long, device=device)
                loss = vit_weight * F.cross_entropy(vit(_vit_preprocess(attacked)), target)
                loss = loss + dct_weight * F.cross_entropy(dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")), target)
                grad = torch.autograd.grad(loss, residual)[0]
                residual = (residual - step * grad.sign()).clamp(-residual_eps, residual_eps).detach()
            base_delta = universal_local if mode != "residual_only" else torch.zeros_like(batch)
            attacked = (batch + base_delta + residual).clamp(0, 1)
            elapsed += time.perf_counter() - started
            with torch.no_grad():
                adv_vit_pred = vit(_vit_preprocess(attacked)).argmax(1)
                adv_dct_pred = dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")).argmax(1)
            adv_vit += int((adv_vit_pred == CLASS_IDX_REAL).sum())
            adv_dct += int((adv_dct_pred == CLASS_IDX_REAL).sum())
            vit_mask = clean_vit_pred != CLASS_IDX_REAL
            dct_mask = clean_dct_pred != CLASS_IDX_REAL
            joint_mask = vit_mask & dct_mask
            vit_eligible += int(vit_mask.sum()); dct_eligible += int(dct_mask.sum()); joint_eligible += int(joint_mask.sum())
            vit_asr += int((vit_mask & (adv_vit_pred == CLASS_IDX_REAL)).sum())
            dct_asr += int((dct_mask & (adv_dct_pred == CLASS_IDX_REAL)).sum())
            joint_asr += int((joint_mask & (adv_vit_pred == CLASS_IDX_REAL) & (adv_dct_pred == CLASS_IDX_REAL)).sum())
            originals, adversarials = [], []
            for index, name in enumerate(names):
                original, adv = metric_pair(name, attacked, index)
                ssims.append(compute_ssim_rgb(original, adv))
                originals.append(np_to_lpips_tensor(original, device))
                adversarials.append(np_to_lpips_tensor(adv, device))
            lpips_values.extend(lpips_fn(torch.cat(originals), torch.cat(adversarials)).flatten().cpu().tolist())
    n = len(paths)
    return {
        "mode": mode, "residual_epsilon": residual_eps, "iterations": iterations, "images": n,
        "clean_vit_real": clean_vit / n, "clean_dct_real": clean_dct / n,
        "raw_vit_target_rate": adv_vit / n, "raw_dct_target_rate": adv_dct / n,
        "vit_clean_corrected_asr": vit_asr / vit_eligible if vit_eligible else None,
        "dct_clean_corrected_asr": dct_asr / dct_eligible if dct_eligible else None,
        "joint_clean_corrected_asr": joint_asr / joint_eligible if joint_eligible else None,
        "vit_eligible": vit_eligible, "dct_eligible": dct_eligible, "joint_eligible": joint_eligible,
        "mean_ssim": float(np.mean(ssims)), "mean_lpips": float(np.mean(lpips_values)),
        "attack_seconds": elapsed, "seconds_per_image": elapsed / n,
    }


def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(cfg["split_json"]).read_text())
    paths = split[cfg.get("eval_split", "held_out")]
    if cfg.get("max_images"):
        paths = paths[:int(cfg["max_images"])]
    pack = torch.load(cfg["universal_checkpoint"], map_location=device) if cfg.get("universal_checkpoint") else None
    universal = pack["delta"].to(device) if pack else None
    models = Path(cfg["models_dir"])
    vit = load_model("vit_b_16", models / "vit_b_16.pth", device)
    dct = load_model("densenet121_dct", models / "densenet121_dct.pth", device)
    for model in (vit, dct):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    results = []
    for mode in cfg.get("modes", ["universal_only", "residual_only", "universal_plus_residual"]):
        epsilons = [0.0] if mode == "universal_only" else [float(x) for x in cfg.get("residual_epsilons", [2 / 255])]
        for residual_eps in epsilons:
            result = evaluate_case(cfg, paths, universal, mode, residual_eps, vit, dct, lpips_fn, device)
            results.append(result)
            print(json.dumps(result), flush=True)
            partial = Path(cfg["output_json"]).with_suffix(".partial.json")
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps({"config": cfg, "eval_split": cfg.get("eval_split", "held_out"), "results": results}, indent=2))
    output = Path(cfg["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": cfg, "eval_split": cfg.get("eval_split", "held_out"), "results": results}, indent=2))
    print(json.dumps({"output": str(output), "cases": len(results)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(yaml.safe_load(Path(args.config).read_text()))
