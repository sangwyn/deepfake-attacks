"""
Baseline health-check script (direction-independent)
=====================================================
Only measures how the detectors classify CLEAN fake images; no attack involved.

NOTE (2026): this script was written against the AADD-2025 4-detector setup
(resnet50, densenet121, vit_b_16, densenet121_dct). AADD 2026 evaluates only
2 detectors (vit_b_16, densenet121_dct). Trim CLASSIFIERS below if you only
have the 2026 weights; missing weight files will otherwise raise FileNotFoundError.

For each fake image, each detector outputs pred (0=real, 1=fake):
  - pred==1 (fake)  -> detector is correct (it recognizes the forgery)
  - pred==0 (real)  -> detector is already fooled by the clean image (misclassified without any attack)

Reports, per detector:
  - clean_detect_rate : fraction classified as fake (higher = stronger detector / larger attack room)
  - already_real_rate : fraction of clean images already classified as real (the "free" starting point for the attack)
  Broken down by hq / lq / ALL.

Reuses the verified model-loading and transform logic from evaluate.py to stay consistent with official evaluation.

Usage:
  .venv/bin/python baseline_check.py --data-root aadd-2025/test_set_deepfake/test/fake \
      --models-dir aadd-2025/.models --limit 0
  (--limit N runs only the first N images for a quick smoke test; 0 = all)
"""
import argparse, warnings, json, time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, densenet121, vit_b_16
from torchvision import models as tv_models
from scipy.fftpack import dct
from tqdm import tqdm

CLASS_IDX_REAL = 0
CLASSES = 2
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
CLASSIFIERS = ['resnet50', 'densenet121', 'vit_b_16', 'densenet121_dct']


# ---- The model/transform logic below is identical to evaluate.py ----
def create_densenet121_dct():
    model = tv_models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Sequential(nn.Dropout(0.2),
                                      nn.Linear(model.classifier.in_features, CLASSES))
    return model


def dct2(np_img):
    return dct(dct(np_img, axis=0, norm='ortho'), axis=1, norm='ortho')


def build_dct_transform(log_scale=True):
    def _t(pil_img):
        img = pil_img.convert('L')
        if max(img.size) > 256:
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
        w, h = img.size
        left, top = (w - 128) // 2, (h - 128) // 2
        img = img.crop((left, top, left + 128, top + 128))
        np_img = np.array(img, dtype=np.float32)
        d = dct2(np_img)
        if log_scale:
            d = np.log(np.abs(d) + 1e-6)
        return torch.from_numpy(d).unsqueeze(0)
    return _t


def build_spatial_transform(model_name):
    if model_name == 'vit_b_16':
        return T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224)), T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return T.Compose([T.Resize((256, 256)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_model(name, weight_path, device):
    if name == 'resnet50':
        model = resnet50(); model.fc = nn.Linear(model.fc.in_features, CLASSES)
    elif name == 'densenet121':
        model = densenet121(); model.classifier = nn.Linear(model.classifier.in_features, CLASSES)
    elif name == 'vit_b_16':
        model = vit_b_16(); model.heads.head = nn.Linear(model.heads.head.in_features, CLASSES)
    elif name == 'densenet121_dct':
        model = create_densenet121_dct()
    else:
        raise ValueError(name)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    return model.eval().to(device)


def subset_of(path, data_root):
    """Return whether the image belongs to hq or lq (first segment of the relative path)."""
    rel = path.relative_to(data_root)
    return rel.parts[0] if len(rel.parts) > 1 else 'root'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0=all, N=only first N images (smoke test)')
    ap.add_argument('--save-json', default='baseline_report.json')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    data_root = Path(args.data_root)
    models_dir = Path(args.models_dir)

    # Load the detectors
    packs = {}
    for name in CLASSIFIERS:
        w = models_dir / f"{name}.pth"
        if not w.exists():
            raise FileNotFoundError(w)
        transform = build_dct_transform() if name.endswith('_dct') else build_spatial_transform(name)
        packs[name] = {'model': load_model(name, w, device), 'transform': transform}
        print(f"[MODEL] {name} loaded")

    # Collect images
    paths = sorted(p for p in data_root.rglob('*')
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.limit > 0:
        paths = paths[:args.limit]
    print(f"[DATA] {len(paths)} clean fake images\n")

    # Stats container: stats[classifier][subset] = {'fake':n, 'real':n}
    subsets = set()
    stats = {n: {} for n in CLASSIFIERS}

    t0 = time.time()
    for p in tqdm(paths, desc='eval'):
        sub = subset_of(p, data_root)
        subsets.add(sub)
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            warnings.warn(f"open failed {p}: {e}"); continue
        for name, pack in packs.items():
            tensor = pack['transform'](img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = pack['model'](tensor).argmax(1).item()
            for key in (sub, 'ALL'):
                d = stats[name].setdefault(key, {'fake': 0, 'real': 0})
                d['real' if pred == CLASS_IDX_REAL else 'fake'] += 1

    dt = time.time() - t0
    print(f"\n[DONE] {len(paths)} imgs in {dt:.1f}s ({dt/max(len(paths),1)*1000:.1f} ms/img)\n")

    # Report
    report = {'num_images': len(paths), 'subsets': sorted(subsets), 'per_classifier': {}}
    print("=" * 70)
    print(f"{'classifier':<16}{'subset':<8}{'detect_fake%':>14}{'already_real%':>15}")
    print("-" * 70)
    for name in CLASSIFIERS:
        report['per_classifier'][name] = {}
        for key in sorted(stats[name].keys()):
            d = stats[name][key]
            tot = d['fake'] + d['real']
            df = 100.0 * d['fake'] / tot if tot else 0.0
            ar = 100.0 * d['real'] / tot if tot else 0.0
            report['per_classifier'][name][key] = {
                'n': tot, 'detect_fake_pct': round(df, 2), 'already_real_pct': round(ar, 2)}
            print(f"{name:<16}{key:<8}{df:>13.2f}%{ar:>14.2f}%")
        print("-" * 70)

    with open(args.save_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[SAVE] {args.save_json}")
    print("\nReading: high detect_fake% = strong detector (needs attacking); already_real% = the 'free' starting point already fooled without any attack")


if __name__ == '__main__':
    main()
