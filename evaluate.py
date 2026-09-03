"""Canonical adversarial-attack runner for the four submitted detectors."""

import argparse
import hashlib
import json
import platform
import random
import time
from importlib import import_module, metadata
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

from attacks import AVAILABLE_ATTACKS
from detectors import SUPPORTED_DETECTORS, load_detector
from manifests import discover_test_samples, load_test_manifest


CLASS_IDX_REAL = 0
CLASS_IDX_FAKE = 1
SUPPORTED = SUPPORTED_DETECTORS
PATH_KEYS = (
    "original_root",
    "manifest",
    "models_dir",
    "save_attacked_dir",
    "reuse_attacked_dir",
    "save_json",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cfg(cfg_path):
    """Load YAML and resolve every filesystem path relative to that YAML file."""
    path = Path(cfg_path).resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    base = path.parent
    for key in PATH_KEYS:
        value = cfg.get(key)
        if value is not None:
            candidate = Path(value)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (base / candidate).resolve()
            )
            cfg[key] = str(resolved)
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = _file_sha256(path)
    return cfg


def get_device(choice):
    if choice == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def get_metric_device(choice="cpu"):
    return get_device(choice)


def configure_reproducibility(seed):
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_versions():
    versions = {"python": platform.python_version()}
    for distribution in (
        "numpy",
        "PyYAML",
        "Pillow",
        "scipy",
        "scikit-image",
        "tqdm",
        "torch",
        "torchvision",
        "lpips",
        "open-clip-torch",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def resolve_experiment_models(cfg):
    target_names = cfg.get("target_classifiers", cfg.get("classifiers"))
    source_names = cfg.get("source_classifiers", target_names)
    for field, names in (
        ("source_classifiers", source_names),
        ("target_classifiers", target_names),
    ):
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(x, str) for x in names)
        ):
            raise ValueError(f"{field} must be a non-empty list of detector names")
        if len(names) != len(set(names)):
            raise ValueError(f"{field} contains duplicate detector names")
    unknown = (set(source_names) | set(target_names)) - SUPPORTED
    if unknown:
        raise ValueError(
            f"Unsupported detectors {sorted(unknown)}; choose from {sorted(SUPPORTED)}"
        )
    active_sources = [] if cfg.get("reuse_attacked_dir") else source_names
    loaded_names = list(dict.fromkeys([*active_sources, *target_names]))
    return source_names, target_names, loaded_names


def load_attack(name):
    if name not in AVAILABLE_ATTACKS:
        raise ValueError(f"Unknown attack {name!r}; choose from {AVAILABLE_ATTACKS}")
    function = import_module(f"attacks.{name}").attack
    if not callable(function):
        raise TypeError(f"attacks.{name}.attack is not callable")
    return function


def load_model(name, weight_path, device):
    """Compatibility helper returning the frozen raw model."""
    return load_detector(name, weight_path, device).model


def build_spatial_transform(model_name):
    if model_name not in {"vit_b_16", "npr"}:
        raise ValueError("spatial transform is defined for vit_b_16 and npr")
    return T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_dct_transform(log_scale=True):
    """Return the same RGB-to-DCT transform used by the differentiable adapter."""
    from detectors.registry import DetectorAdapter

    adapter = DetectorAdapter("densenet121_dct", torch.nn.Identity(), log_dct=log_scale)

    def transform(image):
        tensor = T.ToTensor()(image.convert("RGB")).unsqueeze(0)
        return adapter._dct_input(tensor)[0]

    return transform


def pil_to_np_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compute_ssim_rgb(original, attacked):
    if original.shape != attacked.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    return float(
        structural_similarity(original, attacked, channel_axis=2, data_range=255)
    )


def np_to_lpips_tensor(image, device):
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
    return tensor.unsqueeze(0).to(device).div_(127.5).sub_(1.0)


def np_to_detector_tensor(image, device):
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).div_(255.0)


def build_lpips_metric(device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("LPIPS is required; install requirements.txt") from exc
    return lpips.LPIPS(net="alex").eval().to(device)


def predict_np(pack, image, device):
    with torch.no_grad():
        logits = pack["adapter"](np_to_detector_tensor(image, device))
    return int(logits.argmax(dim=1).item())


def _collect_samples(cfg):
    root = Path(cfg["original_root"]).resolve()
    manifest = cfg.get("manifest")
    samples = (
        load_test_manifest(Path(manifest), root)
        if manifest
        else discover_test_samples(root)
    )
    include_labels = cfg.get("include_labels")
    if include_labels is not None:
        if not isinstance(include_labels, list) or not set(include_labels) <= {0, 1}:
            raise ValueError("include_labels must be a list containing only 0 and/or 1")
        allowed = set(include_labels)
        samples = [sample for sample in samples if sample["label"] in allowed]
    sample_limit = cfg.get("sample_limit")
    if sample_limit is not None:
        if not isinstance(sample_limit, int) or sample_limit < 1:
            raise ValueError("sample_limit must be a positive integer or null")
        samples = samples[:sample_limit]
    if not samples:
        raise RuntimeError("No samples remain after applying the configured filters")
    output_paths = [
        sample["path"].relative_to(root).with_suffix(".png") for sample in samples
    ]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Sample names collide after conversion to lossless PNG output")
    return root, samples


def _validate_attack_output(attacked, original):
    if not isinstance(attacked, np.ndarray):
        raise TypeError("attack() must return a numpy array")
    if attacked.dtype != np.uint8:
        raise TypeError("attack() must return uint8 pixels")
    if attacked.shape != original.shape:
        raise ValueError(
            f"attack() changed image shape from {original.shape} to {attacked.shape}"
        )


def _adversarial_relative_path(relative_path):
    return relative_path.with_suffix(".png")


def _success(objective, label, prediction):
    if objective == "targeted_fake_to_real":
        return prediction == CLASS_IDX_REAL
    return prediction != label


def _validate_configuration(cfg, samples, source_names, target_names):
    objective = cfg.get("objective", "targeted_fake_to_real")
    if objective not in {"targeted_fake_to_real", "untargeted"}:
        raise ValueError("objective must be targeted_fake_to_real or untargeted")
    if objective == "targeted_fake_to_real" and any(
        sample["label"] != CLASS_IDX_FAKE for sample in samples
    ):
        raise ValueError("targeted_fake_to_real may evaluate TEST_FAKE samples only")
    alpha = float(cfg.get("alpha", 0.5))
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    aggregate = cfg.get("aggregate", "sum")
    if aggregate not in {"sum", "mean"}:
        raise ValueError("aggregate must be sum or mean")
    attack_params = cfg.get("attack_params", {})
    if not isinstance(attack_params, dict):
        raise ValueError("attack_params must be a mapping")
    if cfg.get("reuse_attacked_dir") and cfg.get("save_attacked_dir"):
        raise ValueError(
            "reuse_attacked_dir and save_attacked_dir are mutually exclusive"
        )
    if not cfg.get("reuse_attacked_dir") and not source_names:
        raise ValueError("attack generation requires at least one source detector")
    weights = cfg.get("weights", {})
    if not isinstance(weights, dict):
        raise ValueError("weights must be a mapping")
    values = [float(weights.get(name, 1.0)) for name in target_names]
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError(
            "target detector weights must be non-negative and not all zero"
        )
    return objective, alpha, aggregate, attack_params, weights


def evaluate(cfg):
    """Generate adversarial images, score every target, and return a JSON-ready report."""
    started = time.perf_counter()
    required = ("original_root", "models_dir")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Missing required configuration fields: {missing}")

    source_names, target_names, loaded_names = resolve_experiment_models(cfg)
    dataset_root, samples = _collect_samples(cfg)
    objective, alpha, aggregate, attack_params, weight_cfg = _validate_configuration(
        cfg, samples, source_names, target_names
    )
    reuse_root = (
        Path(cfg["reuse_attacked_dir"]) if cfg.get("reuse_attacked_dir") else None
    )
    save_root = Path(cfg["save_attacked_dir"]) if cfg.get("save_attacked_dir") else None
    attack_function = (
        None if reuse_root is not None else load_attack(cfg.get("attack", "identity"))
    )
    if reuse_root is not None and not reuse_root.is_dir():
        raise FileNotFoundError(f"Reused adversarial directory not found: {reuse_root}")

    models_root = Path(cfg["models_dir"])
    checkpoints = {name: models_root / f"{name}.pth" for name in loaded_names}
    absent = [str(path) for path in checkpoints.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing detector checkpoints: {absent}")
    checkpoint_records = {
        name: {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in checkpoints.items()
    }

    seed = cfg.get("seed", 0)
    configure_reproducibility(seed)
    device = get_device(cfg.get("device", "auto"))
    metric_device = get_metric_device(cfg.get("metric_device", "cpu"))
    classifiers = {}
    for name in loaded_names:
        adapter = load_detector(
            name,
            checkpoints[name],
            device,
            log_dct=bool(cfg.get("dct_log_scale", True)),
        )
        classifiers[name] = {
            "adapter": adapter,
            "weight": float(weight_cfg.get(name, 1.0)),
            "clean_correct": [],
            "adversarial_correct": [],
            "eligible": [],
            "success": [],
            "raw_success": [],
        }

    metric = build_lpips_metric(metric_device)
    source_classifiers = {
        name: classifiers[name] for name in source_names if name in classifiers
    }
    sample_records = []
    score_sum = 0.0
    l2_values = []
    linf_values = []

    iterator = tqdm(samples, desc="Images", disable=not cfg.get("progress", True))
    for sample in iterator:
        original_path = sample["path"]
        relative_path = original_path.relative_to(dataset_root)
        adversarial_relative = _adversarial_relative_path(relative_path)
        original = pil_to_np_rgb(original_path)

        if reuse_root is not None:
            attacked_path = reuse_root / adversarial_relative
            if not attacked_path.is_file():
                raise FileNotFoundError(
                    f"Reused adversarial image not found: {attacked_path}"
                )
            attacked = pil_to_np_rgb(attacked_path)
        else:
            parameters = dict(attack_params)
            parameters.setdefault("objective", objective)
            parameters.setdefault("label", sample["label"])
            parameters.setdefault("seed", seed)
            attacked = attack_function(
                original, source_classifiers, device, **parameters
            )
            _validate_attack_output(attacked, original)
            if save_root is not None:
                attacked_path = save_root / adversarial_relative
                attacked_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(attacked).save(attacked_path, format="PNG")
                attacked = pil_to_np_rgb(attacked_path)
        _validate_attack_output(attacked, original)

        difference = (attacked.astype(np.float32) - original.astype(np.float32)) / 255.0
        l2_value = float(np.linalg.norm(difference.reshape(-1)))
        linf_value = float(np.abs(difference).max())
        epsilon = attack_params.get("epsilon") if reuse_root is None else None
        if epsilon is not None:
            allowed = np.ceil(float(epsilon) * 255.0) / 255.0
            if linf_value > allowed + 1e-12:
                raise RuntimeError(
                    f"Attack exceeded epsilon for {relative_path}: {linf_value} > {allowed}"
                )
        l2_values.append(l2_value)
        linf_values.append(linf_value)

        ssim_value = compute_ssim_rgb(original, attacked)
        with torch.no_grad():
            lpips_value = float(
                metric(
                    np_to_lpips_tensor(original, metric_device),
                    np_to_lpips_tensor(attacked, metric_device),
                ).item()
            )
        similarity = alpha * ssim_value + (1.0 - alpha) * (1.0 - lpips_value)
        detector_records = {}
        for name in target_names:
            pack = classifiers[name]
            clean_prediction = predict_np(pack, original, device)
            adversarial_prediction = predict_np(pack, attacked, device)
            clean_correct = clean_prediction == sample["label"]
            adversarial_correct = adversarial_prediction == sample["label"]
            eligible = clean_correct
            raw_success = _success(objective, sample["label"], adversarial_prediction)
            success = eligible and raw_success
            pack["clean_correct"].append(int(clean_correct))
            pack["adversarial_correct"].append(int(adversarial_correct))
            pack["eligible"].append(int(eligible))
            pack["success"].append(int(success))
            pack["raw_success"].append(int(raw_success))
            score_sum += pack["weight"] * similarity * int(raw_success)
            detector_records[name] = {
                "clean_prediction": clean_prediction,
                "adversarial_prediction": adversarial_prediction,
                "eligible": bool(eligible),
                "success": bool(success),
            }

        sample_records.append(
            {
                "sample_id": sample["sample_id"],
                "relative_path": relative_path.as_posix(),
                "adversarial_relative_path": adversarial_relative.as_posix(),
                "label": sample["label"],
                "ssim": ssim_value,
                "lpips": lpips_value,
                "l2": l2_value,
                "linf": linf_value,
                "detectors": detector_records,
            }
        )

    total_target_weight = sum(classifiers[name]["weight"] for name in target_names)
    final_score = (
        score_sum
        if aggregate == "sum"
        else score_sum / (len(samples) * total_target_weight)
    )
    per_classifier = {}
    for name in target_names:
        pack = classifiers[name]
        denominator = sum(pack["eligible"])
        per_classifier[name] = {
            "clf_weight": pack["weight"],
            "attack_success": float(np.mean(pack["raw_success"])),
            "clean_accuracy": float(np.mean(pack["clean_correct"])),
            "adversarial_accuracy": float(np.mean(pack["adversarial_correct"])),
            "conditional_asr": float(sum(pack["success"]) / denominator)
            if denominator
            else None,
            "conditional_asr_denominator": denominator,
        }

    report = {
        "config_path": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
        "manifest": cfg.get("manifest"),
        "objective": objective,
        "attack": None if reuse_root is not None else cfg.get("attack", "identity"),
        "attack_params": attack_params,
        "seed": seed,
        "determinism": {
            "seeded_python_numpy_torch": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "environment": environment_versions(),
        "source_classifiers": list(source_names),
        "target_classifiers": list(target_names),
        "checkpoints": checkpoint_records,
        "images_evaluated": len(samples),
        "aggregate": aggregate,
        "alpha": alpha,
        "final_score": final_score,
        "mean_ssim": float(np.mean([sample["ssim"] for sample in sample_records])),
        "mean_lpips": float(np.mean([sample["lpips"] for sample in sample_records])),
        "mean_l2": float(np.mean(l2_values)),
        "max_linf": float(np.max(linf_values)),
        "per_classifier": per_classifier,
        "samples": sample_records,
        "runtime_seconds": time.perf_counter() - started,
    }
    output_path = cfg.get("save_json")
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    arguments = parser.parse_args()
    report = evaluate(load_cfg(arguments.config))
    print(f"Evaluated {report['images_evaluated']} images")
    print(f"Final score: {report['final_score']:.6f}")


if __name__ == "__main__":
    main()
