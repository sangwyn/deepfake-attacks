"""Evaluate the four frozen detectors on a directory-labeled image set."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

import evaluate


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
DETECTORS = ("vit_b_16", "densenet121_dct", "npr", "aide")


def build_detector(name, weights_dir, device):
    raw = evaluate.load_model(name, weights_dir / f"{name}.pth", device)
    if name == "aide":
        transform = evaluate.build_aide_transform(raw, device)
        model = evaluate._AIDELogitModel(raw).eval().to(device)
    elif name == "npr":
        transform = evaluate.build_spatial_transform(name)
        model = evaluate._NPRLogitModel(raw).eval().to(device)
    else:
        transform = (evaluate.build_dct_transform(True)
                     if name.endswith("_dct")
                     else evaluate.build_spatial_transform(name))
        model = raw
    return model, transform


def predict(model, transform, path, device):
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return int(model(tensor).argmax(1).item())


def evaluate_dataset(root, weights_dir, device):
    samples = []
    directory_names = (("real", "fake") if (root / "real").is_dir()
                       else ("real_100", "fake_100"))
    for directory_name, label in zip(directory_names, (0, 1)):
        directory = root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"missing labeled directory: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append((path, label))
    if not samples:
        raise RuntimeError(f"no supported images under {root}")

    detectors = {
        name: build_detector(name, weights_dir, device) for name in DETECTORS
    }
    predictions = {name: [] for name in DETECTORS}
    for path, label in samples:
        for name, (model, transform) in detectors.items():
            predictions[name].append((label, predict(model, transform, path, device)))

    result = {"root": str(root), "images": len(samples), "detectors": {}}
    for name, rows in predictions.items():
        real = sum(label == 0 for label, _ in rows)
        fake = sum(label == 1 for label, _ in rows)
        correct = sum(label == prediction for label, prediction in rows)
        result["detectors"][name] = {
            "n": len(rows),
            "accuracy": correct / len(rows),
            "balanced_accuracy": 0.5 * (
                sum(label == 0 and prediction == 0 for label, prediction in rows) / real
                + sum(label == 1 and prediction == 1 for label, prediction in rows) / fake
            ),
            "real_recall": sum(label == 0 and prediction == 0 for label, prediction in rows) / real,
            "fake_recall": sum(label == 1 and prediction == 1 for label, prediction in rows) / fake,
            "real_as_real": sum(label == 0 and prediction == 0 for label, prediction in rows),
            "real_as_fake": sum(label == 0 and prediction == 1 for label, prediction in rows),
            "fake_as_real": sum(label == 1 and prediction == 0 for label, prediction in rows),
            "fake_as_fake": sum(label == 1 and prediction == 1 for label, prediction in rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    report = [evaluate_dataset(root, args.weights_dir, device) for root in args.root]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
