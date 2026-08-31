"""
evaluate.py — Adversarial Robustness Evaluation
=================================================
Evaluates the robustness of image-classification models against adversarial
versions of a given image set.  For every original / adversarial image pair:

1. Computes visual similarity (SSIM + LPIPS).
2. Runs each classifier on the adversarial image.
3. Flags successful attacks (prediction == "Real" class).
4. Aggregates a weighted score across classifiers and pairs.

Supported classifiers
---------------------
  vit_b_16        : Vision Transformer B/16  (spatial, 224×224)
  densenet121_dct : DenseNet-121 in DCT space (1-channel, 128×128)

YAML configuration (WHAT YOU MUST PROVIDE)
-------------------------------
  original_root  : path to orignal Test 
  attack         : attack module name from the attacks folder
  save_attacked_dir : optional directory for attacked images
  models_dir     : folder containing <model_name>.pth weight files
  classifiers    : list of model names to evaluate
  device         : "auto" | "cpu" | "cuda"
  save_json      : path to write a JSON report

YAML configuration (WHAT YOU MUST LEAVE UNCHANGED) -----> the results will be evaluated based on these settings
-------------------------------
  dct_log_scale  : bool — log-scale the DCT coefficients
  weights        : per-classifier score weight (default 1.0)
  aggregate      : "sum"
  alpha          : weight for SSIM vs (1-LPIPS) in the similarity score


Dependencies
-------------------------------
  Python       == 3.11.11
  numpy        == 1.26.4
  PyYAML       == 6.0.2
  Pillow       == 11.0.0
  scipy        == 1.15.3
  scikit-image == 0.26.0
  tqdm         == 4.67.1
  torch        == 2.3.0+cu118
  torchvision  == 0.18.0+cu118
  lpips        == 0.1.4

GPU acceleration is automatic if CUDA is available and `device: auto`
in the YAML; otherwise CPU is used.

Usage
-----
  1. Install the dependencies (e.g. via pip)
  2. Prepare only these directories in the config:
        - original_root: with the original AADD_2026_Test images
        - attack: attack module name from the attacks folder
        - save_attacked_dir: where to optionally save attacked images
        - models_dir: with the .pth weight files
        - save_json: where you want the results JSON to be written
  3. Run the evaluation:
  python AADD_2026_evaluation.py --config AADD_2026_config.yaml
"""

import argparse
from importlib import import_module
import json
import warnings
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.fftpack import dct as scipy_dct
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models as tv_models
from torchvision.models import vit_b_16

import lpips

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_IDX_REAL = 0
CLASSES        = 2
IMAGE_EXTS     = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
SUPPORTED      = {'vit_b_16', 'densenet121_dct'}


# ============================================================================
# MODEL FACTORIES  — must match train.py exactly
# ============================================================================

def _create_densenet121_dct() -> nn.Module:
    """
    DenseNet-121 with 1-channel input for DCT features.
    """
    model = tv_models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, CLASSES)
    )
    return model


def load_model(name: str, weight_path: Path, device: torch.device) -> nn.Module:
    if name not in SUPPORTED:
        raise ValueError(
            f"Unsupported classifier '{name}'. "
            f"Supported: {sorted(SUPPORTED)}"
        )

    print(f"[MODEL] Loading '{name}' from {weight_path} …")

    if name == 'vit_b_16':
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, CLASSES)

    elif name == 'densenet121_dct':
        model = _create_densenet121_dct()

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"[MODEL] '{name}' ready on {device}\n")
    return model


# ============================================================================
# TRANSFORMS
# ============================================================================

def _dct2(np_img: np.ndarray) -> np.ndarray:
    return scipy_dct(scipy_dct(np_img, axis=0, norm='ortho'), axis=1, norm='ortho')


def build_dct_transform(log_scale: bool = True):
    """
    Grayscale → resize 256 → center-crop 128 → DCT → (optional) log-scale.
    Returns a callable: PIL Image → 1×128×128 tensor.
    """
    def _transform(pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.convert('L')
        if max(img.size) > 256:
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
        w, h  = img.size
        left  = (w - 128) // 2
        top   = (h - 128) // 2
        img   = img.crop((left, top, left + 128, top + 128))
        arr   = np.array(img, dtype=np.float32)
        dct_a = _dct2(arr)
        if log_scale:
            dct_a = np.log(np.abs(dct_a) + 1e-6)
        return torch.from_numpy(dct_a).unsqueeze(0)   # 1×128×128
    return _transform


def build_spatial_transform(model_name: str) -> T.Compose:
    """
    Standard ImageNet-normalised spatial transform.
    vit_b_16 uses 224×224 (CenterCrop); others use 256×256.
    """
    if model_name == 'vit_b_16':
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# SIMILARITY METRICS
# ============================================================================

def pil_to_np_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def compute_ssim_rgb(im1: np.ndarray, im2: np.ndarray) -> float:
    """Mean SSIM over the three RGB channels."""
    return sum(
        ssim(im1[..., c], im2[..., c], data_range=255)
        for c in range(3)
    ) / 3.0


def np_to_lpips_tensor(np_img: np.ndarray,
                        device: torch.device) -> torch.Tensor:
    """HWC uint8 → 1×3×H×W float in [-1, 1] (LPIPS convention)."""
    t = torch.from_numpy(np_img).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"[CONFIG] Loaded from {cfg_path}")
    return cfg


def get_device(choice: str) -> torch.device:
    if choice == 'auto':
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(choice)
    print(f"[DEVICE] {dev}")
    return dev


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate(cfg: dict):
    device    = get_device(cfg.get('device', 'auto'))
    log_scale = bool(cfg.get('dct_log_scale', True))
    alpha     = float(cfg.get('alpha', 0.5))
    attack_fn = import_module(f"attacks.{cfg['attack']}").attack
    save_attacked_dir = cfg['save_attacked_dir']

    # ── LPIPS perceptual similarity ──────────────────────────────────────
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # ── Per-classifier weights from YAML (default 1.0) ───────────────────
    weight_cfg: dict = cfg.get('weights', {})

    # ── Load classifiers ─────────────────────────────────────────────────
    models_dir  = Path(cfg['models_dir'])
    clf_names   = cfg['classifiers']
    classifiers = {}

    for name in clf_names:
        w_path = models_dir / f"{name}.pth"
        if not w_path.exists():
            raise FileNotFoundError(
                f"Weight file for '{name}' not found: {w_path}"
            )
        transform = (build_dct_transform(log_scale)
                     if name.endswith('_dct')
                     else build_spatial_transform(name))

        classifiers[name] = {
            'model':      load_model(name, w_path, device),
            'transform':  transform,
            'clf_weight': float(weight_cfg.get(name, 1.0)),
            'indicators': [],
            'ssim_vals':  [],
            'lpips_vals': [],
        }

    print(f"[SETUP] {len(classifiers)} classifier(s) loaded\n")
    for n, p in classifiers.items():
        print(f"  {n:<22s}  clf_weight={p['clf_weight']:.2f}")
    print()

    # ── Collect image pairs ───────────────────────────────────────────────
    original_root = Path(cfg['original_root'])

    orig_paths = [p for p in original_root.rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not orig_paths:
        raise RuntimeError(f"No images found under: {original_root}")
    print(f"[DATA] {len(orig_paths)} original image(s) found\n")

    running_sum = 0.0
    total_pairs = 0

    for o_path in tqdm(orig_paths, desc="Images"):
        rel = o_path.relative_to(original_root)

        print(f"[IMAGE] {rel}")

        img_o = pil_to_np_rgb(o_path)
        img_a = attack_fn(img_o, classifiers, device)

        if save_attacked_dir:
            save_path = Path(save_attacked_dir) / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img_a).save(save_path)

        # SSIM
        try:
            ssim_val = compute_ssim_rgb(img_o, img_a)
        except Exception as e:
            warnings.warn(f"SSIM failed for {rel}: {e}")
            ssim_val = 0.0
        print(f"    SSIM : {ssim_val:.4f}")

        # LPIPS
        with torch.no_grad():
            lpips_val = lpips_fn(
                np_to_lpips_tensor(img_o, device),
                np_to_lpips_tensor(img_a, device)
            ).item()
        print(f"    LPIPS: {lpips_val:.4f}")

        sim_weight = alpha * ssim_val + (1.0 - alpha) * (1.0 - lpips_val)
        pair_contribution = 0.0

        for name, pack in classifiers.items():
            tensor = pack['transform'](Image.fromarray(img_a))
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = pack['model'](tensor).argmax(1).item()

            indicator = int(pred == CLASS_IDX_REAL)
            pack['indicators'].append(indicator)
            pack['ssim_vals'].append(ssim_val)
            pack['lpips_vals'].append(lpips_val)

            contribution = pack['clf_weight'] * sim_weight * indicator
            pair_contribution += contribution

            print(
                f"    [{name:<22s}]  "
                f"pred={'Real' if pred == 0 else 'Fake'}  "
                f"indicator={indicator}  "
                f"contribution={contribution:.4f}"
            )

        print(f"    Pair total contribution: {pair_contribution:.4f}")
        running_sum += pair_contribution
        total_pairs += 1

    if total_pairs == 0:
        print("[RESULT] No valid image pairs — score = 0")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────
    total_weight = sum(p['clf_weight'] for p in classifiers.values())
    if cfg.get('aggregate', 'mean').lower() == 'mean':
        final_score = running_sum / (total_pairs * total_weight)
    else:
        final_score = running_sum

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[RESULT] SUMMARY")
    print("=" * 55)
    print(f"  Images evaluated        : {total_pairs}")
    print(f"  Classifiers             : {len(classifiers)}")
    print(f"  Aggregate               : {cfg.get('aggregate', 'mean')}")
    print(f"  Alpha (SSIM weight)     : {alpha}")
    print(f"  Final score             : {final_score:.6f}")
    print()
    for name, pack in classifiers.items():
        asr       = np.mean(pack['indicators'])   if pack['indicators']   else 0.0
        m_ssim    = np.mean(pack['ssim_vals'])    if pack['ssim_vals']    else 0.0
        m_lpips   = np.mean(pack['lpips_vals'])   if pack['lpips_vals']   else 0.0
        print(
            f"  [{name:<22s}]  "
            f"attack_success={asr:.4f}  "
            f"mean_ssim={m_ssim:.4f}  "
            f"mean_lpips={m_lpips:.4f}  "
            f"clf_weight={pack['clf_weight']:.2f}"
        )

    # ── JSON report ──────────────────────────────────────────────
    out_json = cfg.get('save_json')
    if out_json:
        report = {
            'final_score':       final_score,
            'aggregate':         cfg.get('aggregate', 'mean'),
            'alpha':             alpha,
            'images_evaluated':  total_pairs,
            'per_classifier': {
                n: {
                    'clf_weight':     p['clf_weight'],
                    'attack_success': float(np.mean(p['indicators'])
                                           if p['indicators'] else 0.0),
                    'mean_ssim':      float(np.mean(p['ssim_vals'])
                                           if p['ssim_vals'] else 0.0),
                    'mean_lpips':     float(np.mean(p['lpips_vals'])
                                           if p['lpips_vals'] else 0.0),
                }
                for n, p in classifiers.items()
            },
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[RESULT] JSON report → {out_json}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness (vit_b_16 + densenet121_dct).")
    parser.add_argument('--config', required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()
    evaluate(load_cfg(args.config))
