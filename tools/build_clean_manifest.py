"""Build a reproducible clean-prediction manifest for the four detectors."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import (
    IMAGE_EXTS,
    _AIDELogitModel,
    _NPRLogitModel,
    build_aide_transform,
    build_dct_transform,
    build_spatial_transform,
    get_device,
    load_model,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    device = get_device(args.device)
    names = ["vit_b_16", "densenet121_dct", "npr", "aide"]
    packs = {}

    for name in names:
        path = models_dir / f"{name}.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = load_model(name, path, device)
        if name == "aide":
            transform = build_aide_transform(raw, device)
            model = _AIDELogitModel(raw).eval()
        elif name == "npr":
            transform = build_spatial_transform(name)
            model = _NPRLogitModel(raw).eval()
        else:
            transform = (build_dct_transform(True)
                         if name.endswith("_dct")
                         else build_spatial_transform(name))
            model = raw
        packs[name] = {"model": model, "transform": transform}

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    records = []
    for path in files:
        parent = path.parent.name
        if parent not in {"TEST_REAL", "TEST_FAKE"}:
            raise ValueError(f"unexpected label directory: {path}")
        label = 0 if parent == "TEST_REAL" else 1
        image = Image.open(path).convert("RGB")
        item = {
            "id": str(path.relative_to(root)),
            "path": str(path),
            "label": label,
            "class": "real" if label == 0 else "fake",
            "sha256": sha256(path),
            "size": list(image.size),
            "predictions": {},
        }
        for name, pack in packs.items():
            tensor = pack["transform"](image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = pack["model"](tensor)
            if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid {name} output for {path}")
            item["predictions"][name] = {
                "logits": [float(v) for v in logits[0].detach().cpu()],
                "prediction": int(logits.argmax(1).item()),
            }
        records.append(item)

    counts = {"real": sum(x["label"] == 0 for x in records),
              "fake": sum(x["label"] == 1 for x in records)}
    manifest = {
        "schema_version": 1,
        "root": str(root),
        "images": len(records),
        "class_counts": counts,
        "detectors": names,
        "label_mapping": {"0": "Real", "1": "Fake"},
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "images": len(records),
                      "class_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
