"""Measure RGB-domain residual statistics in the local real/fake dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def image_paths(root: Path) -> dict[str, list[Path]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return {
        label: sorted(p for p in (root / label).rglob("*")
                      if p.is_file() and p.suffix.lower() in exts)
        for label in ("real", "fake")
    }


def collect(paths: list[Path], bins: int) -> dict:
    values = [[] for _ in range(bins)]
    residuals = []
    channel_covariances = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            rgb = np.asarray(image, dtype=np.float32) / 255.0
            smooth = np.asarray(image.filter(ImageFilter.GaussianBlur(1.0)),
                                dtype=np.float32) / 255.0
        residual = rgb - smooth
        luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                     + 0.114 * rgb[..., 2])
        indices = np.minimum((luminance * bins).astype(np.int32), bins - 1)
        for index in range(bins):
            selected = residual[indices == index]
            if selected.size:
                values[index].append(selected)
        flat = residual.reshape(-1, 3)
        channel_covariances.append(np.cov(flat, rowvar=False))
        residuals.append(flat)

    variance_by_bin = []
    for selected in values:
        data = np.concatenate(selected, axis=0) if selected else np.empty((0, 3))
        variance_by_bin.append({
            "count": int(len(data)),
            "mean": data.mean(axis=0).tolist() if len(data) else [0.0] * 3,
            "variance": data.var(axis=0).tolist() if len(data) else [0.0] * 3,
            "std_luma": float(np.std(data @ np.array([0.299, 0.587, 0.114])))
            if len(data) else 0.0,
        })
    covariance = np.mean(channel_covariances, axis=0)
    flat_all = np.concatenate(residuals, axis=0)
    return {
        "images": len(paths),
        "pixels": int(len(flat_all)),
        "variance_by_luminance_bin": variance_by_bin,
        "mean_channel_covariance": covariance.tolist(),
        "mean_abs_residual": float(np.abs(flat_all).mean()),
        "p95_abs_residual": float(np.percentile(np.abs(flat_all), 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Directory containing real/ and fake/ subdirectories")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bins <= 1:
        raise ValueError("bins must be greater than one")

    paths = image_paths(Path(args.root))
    if not paths["real"] or not paths["fake"]:
        raise RuntimeError("root must contain non-empty real/ and fake/ directories")
    result = {
        "root": str(Path(args.root).resolve()),
        "bins": args.bins,
        "classes": {label: collect(class_paths, args.bins)
                    for label, class_paths in paths.items()},
    }
    real = result["classes"]["real"]
    fake = result["classes"]["fake"]
    real_variance = np.array([x["std_luma"] for x in
                              real["variance_by_luminance_bin"]])
    fake_variance = np.array([x["std_luma"] for x in
                              fake["variance_by_luminance_bin"]])
    result["comparison"] = {
        "std_luma_ratio_fake_over_real": np.divide(
            fake_variance, real_variance,
            out=np.zeros_like(fake_variance), where=real_variance > 1e-12
        ).tolist(),
        "mean_abs_residual_difference_fake_minus_real": (
            fake["mean_abs_residual"] - real["mean_abs_residual"]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
