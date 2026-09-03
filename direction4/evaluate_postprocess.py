"""Evaluate saved Direction 4 images after common image post-processing."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import (  # noqa: E402
    CLASS_IDX_REAL, build_dct_transform, build_spatial_transform,
    get_device, load_model,
)


def transforms(size: tuple[int, int]):
    width, height = size

    def jpeg(image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()

    def resize(image):
        small = image.resize((max(1, int(width * 0.75)),
                              max(1, int(height * 0.75))), Image.Resampling.BICUBIC)
        return small.resize((width, height), Image.Resampling.BICUBIC)

    def blur(image):
        return image.filter(ImageFilter.GaussianBlur(radius=0.5))

    return {"clean": lambda image: image, "jpeg75": jpeg,
            "resize75": resize, "blur05": blur}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--attacked-root", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    models_dir = Path(args.models_dir)
    packs = {}
    for name in ("vit_b_16", "densenet121_dct"):
        packs[name] = {
            "model": load_model(name, models_dir / f"{name}.pth", device),
            "transform": (build_dct_transform(True) if name.endswith("_dct")
                          else build_spatial_transform(name)),
        }

    original_root = Path(args.original_root)
    attacked_root = Path(args.attacked_root)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    attacked_paths = sorted(p for p in attacked_root.rglob("*")
                            if p.is_file() and p.suffix.lower() in image_exts)
    if not attacked_paths:
        raise RuntimeError(f"No attacked images under {attacked_root}")

    kinds = transforms((1, 1))
    counts = {name: {kind: 0 for kind in kinds} for name in packs}
    total = 0
    for attacked_path in attacked_paths:
        relative = attacked_path.relative_to(attacked_root)
        original_path = original_root / relative
        if not original_path.exists():
            raise FileNotFoundError(f"Missing original pair: {original_path}")
        image = Image.open(attacked_path).convert("RGB")
        for kind, transform_image in transforms(image.size).items():
            processed = transform_image(image)
            for name, pack in packs.items():
                tensor = pack["transform"](processed).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = pack["model"](tensor).argmax(1).item()
                counts[name][kind] += int(prediction == CLASS_IDX_REAL)
        total += 1

    result = {"images": total, "attacked_root": str(attacked_root),
              "target": "Real", "real_rate": {
                  name: {kind: value / total for kind, value in values.items()}
                  for name, values in counts.items()
              }}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
