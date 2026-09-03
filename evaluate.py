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
  npr             : NPR ResNet-50 (spatial residual, 224×224)
  aide            : AIDE high-pass + ConvNeXt detector (five-view, 256×256)

YAML configuration (WHAT YOU MUST PROVIDE)
-------------------------------
  original_root  : path to orignal Test 
  manifest       : optional immutable JSONL TEST manifest
  attack         : attack module name from the attacks folder
  attack_params  : optional dict of kwargs forwarded to the attack's attack()
  save_attacked_dir : optional directory for attacked images
                      (when set, images are scored after save+reload)
  models_dir     : folder containing <model_name>.pth weight files
  source_classifiers : detector names used to generate attacks
  target_classifiers : detector names used to evaluate attacks
  objective      : targeted_fake_to_real | untargeted
  attack_params  : attack-specific parameters recorded in the config
  device         : "auto" | "cpu" | "cuda"
  metric_device  : optional LPIPS device (default "cpu" to preserve attack GPU memory)
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
  python evaluate.py --config configs/AADD_2026_config.yaml
"""

import argparse
import hashlib
from importlib import import_module
import inspect
import json
import time
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

from detectors import SUPPORTED_DETECTORS, load_detector
from manifests import IMAGE_EXTENSIONS, load_test_manifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_IDX_REAL = 0
CLASSES        = 2
IMAGE_EXTS     = IMAGE_EXTENSIONS
SUPPORTED      = SUPPORTED_DETECTORS


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
    print(f"[MODEL] Loading '{name}' from {weight_path} …")
    model = load_detector(name, weight_path, device).model
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
        # Always create the 256x256 canvas expected by the checkpoint.
        if img.size != (256, 256):
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
    if model_name in {'vit_b_16', 'npr'}:
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


def build_aide_transform(model: nn.Module, device: torch.device):
    """PIL RGB -> the five-branch tensor expected by the AIDE model."""
    from detectors.aide.adapter import AIDEAdapter
    adapter = AIDEAdapter(model, device)

    def _transform(pil_img: Image.Image) -> torch.Tensor:
        image = T.ToTensor()(pil_img).unsqueeze(0).to(device)
        return adapter(image)[0]

    return _transform


class _NPRLogitModel(nn.Module):
    """Expose the scalar NPR score as [Real, Fake] logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        score = self.model(image)
        # The verified NPR checkpoint produces positive scores for Fake.
        return torch.cat((-score, score), dim=1)


class _AIDELogitModel(nn.Module):
    """Map the upstream AIDE output ordering to [Real, Fake]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        logits = self.model(image)
        # The upstream wrapper maps its second-class probability above 0.5
        # to Real, so swap the raw output into the shared [Real, Fake] order.
        return logits[:, [1, 0]]


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


def np_to_detector_tensor(np_img: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    """HWC uint8 → 1×3×H×W float in [0, 1]."""
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).div_(255.0)


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def load_cfg(cfg_path: str) -> dict:
    path = Path(cfg_path).resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f)
    cfg['_config_path'] = str(path)
    cfg['_config_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"[CONFIG] Loaded from {cfg_path}")
    return cfg


def get_device(choice: str) -> torch.device:
    if choice == 'auto':
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(choice)
    print(f"[DEVICE] {dev}")
    return dev


def get_metric_device(choice: str = 'cpu') -> torch.device:
    if choice not in {'cpu', 'cuda', 'cuda:0'}:
        raise ValueError("metric_device must be 'cpu', 'cuda', or 'cuda:0'")
    if choice.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("metric_device requests CUDA, but CUDA is unavailable")
    return torch.device(choice)


def resolve_experiment_models(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    target_names = cfg.get('target_classifiers', cfg.get('classifiers'))
    if not isinstance(target_names, list) or not target_names:
        raise ValueError("Configure target_classifiers as a non-empty list")
    source_names = cfg.get('source_classifiers', target_names)
    if not isinstance(source_names, list) or not source_names:
        raise ValueError("Configure source_classifiers as a non-empty list")
    unknown = (set(source_names) | set(target_names)) - SUPPORTED
    if unknown:
        raise ValueError(
            f"Unsupported classifier(s) {sorted(unknown)}; "
            f"supported: {sorted(SUPPORTED)}"
        )
    active_source_names = [] if cfg.get('reuse_attacked_dir') else source_names
    loaded_names = list(dict.fromkeys([*active_source_names, *target_names]))
    return source_names, target_names, loaded_names


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def predict_np(pack: dict, np_img: np.ndarray, device: torch.device) -> int:
    """Class prediction of one detector on an H×W×3 uint8 RGB image."""
    tensor = pack['transform'](Image.fromarray(np_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        return int(pack['model'](tensor).argmax(1).item())


def evaluate(cfg: dict):
    import lpips

    started_at = time.perf_counter()
    device    = get_device(cfg.get('device', 'auto'))
    metric_device = get_metric_device(cfg.get('metric_device', 'cpu'))
    print(f"[METRICS] LPIPS device: {metric_device}")
    log_scale = bool(cfg.get('dct_log_scale', True))
    alpha     = float(cfg.get('alpha', 0.5))
    save_attacked_dir = cfg['save_attacked_dir']
    reuse_attacked_dir = cfg.get('reuse_attacked_dir')
    if reuse_attacked_dir is not None and not isinstance(reuse_attacked_dir, str):
        raise ValueError("reuse_attacked_dir must be a path string or null")
    if reuse_attacked_dir and save_attacked_dir:
        raise ValueError(
            "A reuse config cannot also declare save_attacked_dir"
        )
    source_names, target_names, clf_names = resolve_experiment_models(cfg)
    attack_fn = (
        None if reuse_attacked_dir
        else import_module(f"attacks.{cfg['attack']}").attack
    )

    # ── LPIPS perceptual similarity ──────────────────────────────────────
    lpips_fn = lpips.LPIPS(net='alex').to(metric_device)
    lpips_fn.eval()

    # ── Per-classifier weights from YAML (default 1.0) ───────────────────
    weight_cfg: dict = cfg.get('weights', {})

    # ── Load classifiers ─────────────────────────────────────────────────
    models_dir = Path(cfg['models_dir'])
    classifiers = {}
    architecture_root = cfg.get('architecture_root')

    for name in clf_names:
        w_path = models_dir / f"{name}.pth"
        if not w_path.exists():
            raise FileNotFoundError(
                f"Weight file for '{name}' not found: {w_path}"
            )
        raw_model = load_model(name, w_path, device)
        if name == 'aide':
            transform = build_aide_transform(raw_model, device)
            model = _AIDELogitModel(raw_model).eval().to(device)
        elif name == 'npr':
            transform = build_spatial_transform(name)
            model = _NPRLogitModel(raw_model).eval().to(device)
        else:
            transform = (build_dct_transform(log_scale)
                         if name.endswith('_dct')
                         else build_spatial_transform(name))
            model = raw_model

        print(f"[MODEL] Loading '{name}' from {w_path} …")
        adapter = load_detector(
            name,
            w_path,
            device,
            log_dct=log_scale,
            architecture_root=(Path(architecture_root)
                               if architecture_root else None),
        )
        classifiers[name] = {
            'model':      adapter.model,
            'adapter':    adapter,
            'transform':  transform,
            'clf_weight': float(weight_cfg.get(name, 1.0)),
            'indicators': [],
            'clean_indicators': [],
            'ssim_vals':  [],
            'lpips_vals': [],
            'eligible':   [],
            'successes':  [],
            'clean_ok':   [],
            'adv_ok':     [],
        }
        print(f"[MODEL] '{name}' ready on {device}\n")

    # Initialize LPIPS only after configuration and checkpoint validation.
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    print(f"[SETUP] {len(classifiers)} classifier(s) loaded\n")
    for n, p in classifiers.items():
        print(f"  {n:<22s}  clf_weight={p['clf_weight']:.2f}")
    print()

    # ── Collect image pairs ───────────────────────────────────────────────
    original_root = Path(cfg['original_root']).resolve()
    manifest_path = cfg.get('manifest')
    if manifest_path:
        samples = load_test_manifest(Path(manifest_path), original_root)
    else:
        samples = [
            {'sample_id': str(path.relative_to(original_root)),
             'path': path, 'label': None}
            for path in original_root.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
    include_labels = cfg.get('include_labels')
    if include_labels is not None:
        include_labels = set(include_labels)
        samples = [sample for sample in samples
                   if sample['label'] in include_labels]
    if not samples:
        raise RuntimeError(f"No images found under: {original_root}")
    print(f"[DATA] {len(samples)} original image(s) found\n")

    running_sum = 0.0
    total_pairs = 0
    sample_records = []
    l2_values = []
    linf_values = []
    objective = cfg.get('objective', 'targeted_fake_to_real')
    attack_params = dict(cfg.get('attack_params', {}))
    attack_classifiers = {
        name: classifiers[name] for name in source_names if name in classifiers
    }

    for sample in tqdm(samples, desc="Images"):
        o_path = sample['path']
        label = sample['label']
        rel = o_path.relative_to(original_root)
        print(f"[IMAGE] {rel}")

        img_o = pil_to_np_rgb(o_path)
        if reuse_attacked_dir:
            reused_path = Path(reuse_attacked_dir) / rel
            if not reused_path.is_file():
                raise FileNotFoundError(
                    f"Reused adversarial image not found: {reused_path}"
                )
            img_a = pil_to_np_rgb(reused_path)
        else:
            call_params = dict(attack_params)
            if cfg['attack'] == 'fgsm':
                call_params.setdefault('objective', objective)
                call_params.setdefault('label', label)
            img_a = attack_fn(img_o, attack_classifiers, device, **call_params)

        if save_attacked_dir:
            save_path = Path(save_attacked_dir) / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img_a).save(save_path)
            img_a = pil_to_np_rgb(save_path)

        delta = (img_a.astype(np.float32) - img_o.astype(np.float32)) / 255.0
        l2_val = float(np.linalg.norm(delta.reshape(-1), ord=2))
        linf_val = float(np.abs(delta).max())
        l2_values.append(l2_val)
        linf_values.append(linf_val)

        # SSIM
        try:
            img_o = pil_to_np_rgb(o_path)
            img_a = attack_fn(img_o, classifiers, device, **attack_params)

        # LPIPS
        with torch.no_grad():
            lpips_val = lpips_fn(
                np_to_lpips_tensor(img_o, metric_device),
                np_to_lpips_tensor(img_a, metric_device)
            ).item()
        print(f"    LPIPS: {lpips_val:.4f}")

        sim_weight = alpha * ssim_val + (1.0 - alpha) * (1.0 - lpips_val)
        pair_contribution = 0.0

        detector_records = {}
        for name in target_names:
            pack = classifiers[name]
            with torch.no_grad():
                clean_tensor = np_to_detector_tensor(img_o, device)
                clean_pred = pack['adapter'](clean_tensor).argmax(1).item()
                del clean_tensor
                adv_tensor = np_to_detector_tensor(img_a, device)
                pred = pack['adapter'](adv_tensor).argmax(1).item()
                del adv_tensor

            indicator = int(pred == CLASS_IDX_REAL)
            clean_ok = label is not None and clean_pred == label
            adv_ok = label is not None and pred == label
            if objective == 'targeted_fake_to_real':
                eligible = clean_ok and label == 1
                success = eligible and pred == CLASS_IDX_REAL
            elif objective == 'untargeted':
                eligible = clean_ok
                success = eligible and pred != label
            else:
                raise ValueError(f"Unsupported objective: {objective}")
            pack['indicators'].append(indicator)
            pack['ssim_vals'].append(ssim_val)
            pack['lpips_vals'].append(lpips_val)
            pack['eligible'].append(int(eligible))
            pack['successes'].append(int(success))
            pack['clean_ok'].append(int(clean_ok))
            pack['adv_ok'].append(int(adv_ok))

            print(f"    SSIM : {ssim_val:.4f}    LPIPS: {lpips_val:.4f}")
            sim_weight = alpha * ssim_val + (1.0 - alpha) * (1.0 - lpips_val)
            pair_contribution = 0.0
            rec = {'path': str(rel), 'ssim': ssim_val, 'lpips': lpips_val,
                   'linf': linf, 'classifiers': {}}

            print(
                f"    [{name:<22s}]  "
                f"pred={'Real' if pred == 0 else 'Fake'}  "
                f"indicator={indicator}  "
                f"contribution={contribution:.4f}"
            )
            detector_records[name] = {
                'clean_prediction': clean_pred,
                'adversarial_prediction': pred,
                'eligible': bool(eligible),
                'success': bool(success),
            }

                contribution = pack['clf_weight'] * sim_weight * indicator
                pair_contribution += contribution
                rec['classifiers'][name] = {
                    'clean':     'Real' if clean_pred == 0 else 'Fake',
                    'attacked':  'Real' if pred == 0 else 'Fake',
                    'indicator': indicator,
                }
                print(
                    f"    [{name:<22s}]  "
                    f"clean={'Real' if clean_pred == 0 else 'Fake'}  "
                    f"pred={'Real' if pred == 0 else 'Fake'}  "
                    f"indicator={indicator}  "
                    f"contribution={contribution:.4f}"
                )

            print(f"    L_inf: {linf}/255   Pair total contribution: {pair_contribution:.4f}")
        except Exception as e:
            warnings.warn(f"[SKIP] {rel}: {e!r}")
            continue

        per_image.append(rec)
        running_sum += pair_contribution
        total_pairs += 1
        sample_records.append({
            'sample_id': sample['sample_id'],
            'relative_path': str(rel),
            'label': label,
            'ssim': ssim_val,
            'lpips': lpips_val,
            'l2': l2_val,
            'linf': linf_val,
            'detectors': detector_records,
        })

    if total_pairs == 0:
        print("[RESULT] No valid image pairs — score = 0")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────
    total_weight = sum(classifiers[name]['clf_weight'] for name in target_names)
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
    for name in target_names:
        pack = classifiers[name]
        asr       = np.mean(pack['indicators'])   if pack['indicators']   else 0.0
        m_ssim    = np.mean(pack['ssim_vals'])    if pack['ssim_vals']    else 0.0
        m_lpips   = np.mean(pack['lpips_vals'])   if pack['lpips_vals']   else 0.0
        denominator = sum(pack['eligible'])
        conditional_asr = (sum(pack['successes']) / denominator
                           if denominator else 0.0)
        print(
            f"  [{name:<22s}]  "
            f"attack_success={asr:.4f}  "
            f"conditional_asr={conditional_asr:.4f}  "
            f"denominator={denominator}  "
            f"mean_ssim={m_ssim:.4f}  "
            f"mean_lpips={m_lpips:.4f}  "
            f"clf_weight={pack['clf_weight']:.2f}"
        )

    # ── JSON report ──────────────────────────────────────────────
    out_json = cfg.get('save_json')
    if out_json:
        report = {
            'config_path':       cfg.get('_config_path'),
            'config_sha256':     cfg.get('_config_sha256'),
            'final_score':       final_score,
            'aggregate':         cfg.get('aggregate', 'mean'),
            'alpha':             alpha,
            'metric_device':     str(metric_device),
            'objective':         objective,
            'source_classifiers': list(source_names),
            'target_classifiers': list(target_names),
            'reuse_attacked_dir': reuse_attacked_dir,
            'images_evaluated':  total_pairs,
            'runtime_seconds':   time.perf_counter() - started_at,
            'mean_l2':           float(np.mean(l2_values)),
            'max_linf':          float(np.max(linf_values)),
            'per_classifier': {
                n: {
                    'clf_weight':     p['clf_weight'],
                    'attack_success': float(np.mean(p['indicators'])
                                           if p['indicators'] else 0.0),
                    'mean_ssim':      float(np.mean(p['ssim_vals'])
                                           if p['ssim_vals'] else 0.0),
                    'mean_lpips':     float(np.mean(p['lpips_vals'])
                                           if p['lpips_vals'] else 0.0),
                    'clean_accuracy': float(np.mean(p['clean_ok'])),
                    'adversarial_accuracy': float(np.mean(p['adv_ok'])),
                    'conditional_asr': (
                        float(sum(p['successes']) / sum(p['eligible']))
                        if sum(p['eligible']) else 0.0
                    ),
                    'conditional_asr_denominator': sum(p['eligible']),
                }
                for n, p in ((name, classifiers[name])
                             for name in target_names)
            },
            'samples': sample_records,
        }
        if manifest_records is not None:
            report['protocol'] = {
                'manifest': str(Path(manifest_path).resolve()),
                'eligible_source': eligible_source,
                'eligible_target': eligible_target,
                'direction': direction,
                'eligible_count': total_pairs,
                'conditional_asr': float(np.mean(protocol_target_indicators)
                                         if protocol_target_indicators else 0.0),
                # The run intentionally attacks only the eligible subset, so
                # an all-manifest attacked rate is unavailable here.
                'all_sample_target_rate': None,
                'evaluated_subset_target_rate': float(
                    np.mean(protocol_target_indicators)
                    if protocol_target_indicators else 0.0
                ),
            }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[RESULT] JSON report → {out_json}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness.")
    parser.add_argument('--config', required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()
    evaluate(load_cfg(args.config))
