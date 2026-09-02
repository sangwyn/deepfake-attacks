"""Reproducible evaluator wrapper with post-save constraint auditing."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import verify_run
from .attack_api import invoke_attack, load_attack_module
from .config import PROJECT_ROOT, load_experiment_config, load_server_config
from .io import (
    ContractError,
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    sha256_file,
    utc_now,
)
from .manifest import load_manifest
from .preflight import run_preflight
from .provenance import collect_provenance


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for row in rows
    )


def _set_determinism(seed: int, np: Any, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only, not a hard error: the differentiable resize every attack needs
    # has no deterministic CUDA backward, so warn_only=False aborts the first
    # sample of any GPU run. Seeds, cuDNN determinism, and the cuBLAS workspace
    # remain enforced, and byte-for-byte reproducibility is still proven
    # empirically by the per-sample output hashes the verifier re-checks.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import lpips
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ContractError(
            "Scientific runtime dependencies are missing; install requirements.lock"
        ) from exc
    return np, torch, Image, lpips


def _prediction(model_pack: dict[str, Any], image: Any, device: Any, torch: Any) -> int:
    tensor = model_pack["transform"](image).unsqueeze(0).to(device)
    with torch.no_grad():
        return int(model_pack["model"](tensor).argmax(1).item())


def _require_tracked_inputs(paths: list[Path]) -> None:
    for path in paths:
        try:
            relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise ContractError(f"Versioned input is outside project root: {path}") from exc
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=20,
        )
        if result.returncode != 0:
            raise ContractError(f"Scientific input is not committed to Git: {relative}")


def run_experiment(config_path: Path, run_dir: Path) -> dict[str, Any]:
    """Execute one immutable experiment attempt and verify all produced evidence."""

    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    config = load_experiment_config(config_path)
    server_path = (PROJECT_ROOT / config["server_config"]).resolve()
    server = load_server_config(server_path)

    backbone_path = Path(server["assets"]["lpips"]["alexnet_backbone_path"])
    try:
        torch_home = backbone_path.parents[2]
    except IndexError as exc:
        raise ContractError("Configured LPIPS backbone path has no Torch cache root") from exc
    os.environ["TORCH_HOME"] = str(torch_home)
    # cuBLAS reads this when its handle is first created, so it must be set
    # before anything imports torch.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    preflight = run_preflight(server_path, deep=True)
    if preflight["status"] != "pass":
        failed = [
            item["name"] for item in preflight["checks"] if item["status"] == "fail"
        ]
        raise ContractError("Deep preflight failed: " + ", ".join(failed))

    run_dir.mkdir(parents=True, exist_ok=True)
    protected_names = {
        "resolved_config.yaml",
        "summary.json",
        "per_sample_metrics.jsonl",
        "verification.json",
    }
    existing = sorted(name for name in protected_names if (run_dir / name).exists())
    if existing:
        raise ContractError(
            "Attempt directory is immutable; existing artifacts: " + ", ".join(existing)
        )
    atomic_write_json(run_dir / "preflight.json", preflight)

    manifest_path = (PROJECT_ROOT / config["dataset"]["manifest"]).resolve()
    environment_path = (PROJECT_ROOT / server["environment"]["lock_file"]).resolve()
    _require_tracked_inputs(
        [config_path, server_path, manifest_path, environment_path]
    )
    records = load_manifest(manifest_path)
    sample_limit = config["dataset"].get("sample_limit")
    if sample_limit is not None:
        records = records[: int(sample_limit)]
    source_label = int(config["dataset"]["source_label"])
    wrong_labels = [record["sample_id"] for record in records if record["label"] != source_label]
    if wrong_labels:
        raise ContractError(
            f"Manifest selection contains {len(wrong_labels)} rows outside source_label"
        )

    shutil.copyfile(config_path, run_dir / "resolved_config.yaml")
    shutil.copyfile(server_path, run_dir / "resolved_server_config.yaml")
    shutil.copyfile(manifest_path, run_dir / "manifest.snapshot.jsonl")

    np, torch, Image, lpips_module = _load_runtime_dependencies()
    _set_determinism(int(config["seed"]), np, torch)
    try:
        import evaluate as legacy
    except Exception as exc:
        raise ContractError(f"Cannot import legacy model/evaluation functions: {exc}") from exc

    device = legacy.get_device("auto")
    weights_root = Path(server["assets"]["weights_root"])
    weight_paths = {
        name: weights_root / server["assets"]["models"][name]["filename"]
        for name in config["models"]
    }
    classifiers: dict[str, dict[str, Any]] = {}
    for name in config["models"]:
        transform = (
            legacy.build_dct_transform(True)
            if name.endswith("_dct")
            else legacy.build_spatial_transform(name)
        )
        classifiers[name] = {
            "model": legacy.load_model(name, weight_paths[name], device),
            "transform": transform,
        }

    _, attack_function = load_attack_module(
        config["attack"]["module"], config["attack"]["source_model"]
    )
    lpips_fn = lpips_module.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    relative_attempt = run_dir.relative_to(PROJECT_ROOT)
    heavy_key = canonical_json_sha256(
        {"experiment_id": config["experiment_id"], "run_dir": relative_attempt.as_posix()}
    )[:16]
    heavy_root = (
        Path(server["runtime"]["runs_root"])
        / config["experiment_id"]
        / heavy_key
    )
    if heavy_root.exists():
        raise ContractError(f"Heavy artifact directory already exists: {heavy_root}")
    images_root = heavy_root / "images"
    images_root.mkdir(parents=True, exist_ok=False)

    dataset_root = Path(server["dataset"]["celeb_a_root"])
    source_model = config["attack"]["source_model"]
    target_class = int(config["attack"]["target_class"])
    require_clean_correct = bool(config["dataset"].get("require_clean_correct", True))
    epsilon = float(config["constraint"]["epsilon"])
    tolerance = (1.0 / 255.0) + 1e-12
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    model_clean_correct = {name: 0 for name in config["models"]}
    model_success = {name: 0 for name in config["models"]}
    max_linf = 0.0
    violations = 0

    for record in records:
        input_path = dataset_root / record["relative_path"]
        if not input_path.is_file():
            raise ContractError(f"Manifest input is missing: {input_path}")
        actual_input_sha = sha256_file(input_path)
        if actual_input_sha != record["sha256"]:
            raise ContractError(f"Manifest input hash changed: {input_path}")
        original_pil = Image.open(input_path).convert("RGB")
        original = np.array(original_pil, dtype=np.uint8)
        clean_predictions = {
            name: _prediction(classifiers[name], original_pil, device, torch)
            for name in config["models"]
        }
        for name, prediction in clean_predictions.items():
            model_clean_correct[name] += int(prediction == source_label)
        eligible = (not require_clean_correct) or (
            clean_predictions[source_model] == source_label
        )
        selection_rows.append(
            {
                "schema_version": 1,
                "sample_id": record["sample_id"],
                "relative_path": record["relative_path"],
                "source_label": source_label,
                "clean_predictions": clean_predictions,
                "eligible": eligible,
            }
        )
        if not eligible:
            continue

        attacked = invoke_attack(
            attack_function,
            original,
            classifiers,
            device,
            source_model,
            target_class,
            dict(config["attack"]["parameters"]),
        )
        if not isinstance(attacked, np.ndarray):
            raise ContractError("attack() must return a numpy.ndarray")
        if attacked.dtype != np.uint8 or attacked.shape != original.shape:
            raise ContractError(
                "attack() must return uint8 HxWx3 with the original image shape"
            )
        output_path = images_root / f"{record['sha256']}.png"
        Image.fromarray(attacked, mode="RGB").save(output_path, format="PNG")

        # Audit the bytes that downstream evaluators will actually decode.
        saved = np.array(Image.open(output_path).convert("RGB"), dtype=np.uint8)
        linf = float(
            np.max(np.abs(saved.astype(np.int16) - original.astype(np.int16))) / 255.0
        )
        max_linf = max(max_linf, linf)
        violation = linf > epsilon + tolerance
        violations += int(violation)
        ssim_value = float(legacy.compute_ssim_rgb(original, saved))
        with torch.no_grad():
            lpips_value = float(
                lpips_fn(
                    legacy.np_to_lpips_tensor(original, device),
                    legacy.np_to_lpips_tensor(saved, device),
                ).item()
            )
        attacked_pil = Image.fromarray(saved, mode="RGB")
        adversarial_predictions = {
            name: _prediction(classifiers[name], attacked_pil, device, torch)
            for name in config["models"]
        }
        for name, prediction in adversarial_predictions.items():
            model_success[name] += int(prediction == target_class)
        metric_rows.append(
            {
                "schema_version": 1,
                "sample_id": record["sample_id"],
                "relative_path": record["relative_path"],
                "source_label": source_label,
                "target_class": target_class,
                "clean_predictions": clean_predictions,
                "adversarial_predictions": adversarial_predictions,
                "source_success": adversarial_predictions[source_model]
                == target_class,
                "linf": linf,
                "constraint_violation": violation,
                "ssim": ssim_value,
                "lpips": lpips_value,
                "output": {
                    "path": str(output_path),
                    "sha256": sha256_file(output_path),
                    "format": "png",
                },
            }
        )

    selected_count = len(selection_rows)
    eligible_count = len(metric_rows)
    per_model = {
        name: {
            "clean_accuracy_on_selected": (
                model_clean_correct[name] / selected_count if selected_count else 0.0
            ),
            "targeted_asr_on_source_eligible": (
                model_success[name] / eligible_count if eligible_count else 0.0
            ),
            "successes": model_success[name],
            "denominator": eligible_count,
        }
        for name in config["models"]
    }
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "experiment_id": config["experiment_id"],
        "scope": config["scope"],
        "seed": config["seed"],
        "samples_selected": selected_count,
        "samples_eligible": eligible_count,
        "samples_evaluated": eligible_count,
        "source_model": source_model,
        "target_class": target_class,
        "epsilon": epsilon,
        "constraint_violations": violations,
        "mean_ssim": float(np.mean([row["ssim"] for row in metric_rows]))
        if metric_rows
        else None,
        "mean_lpips": float(np.mean([row["lpips"] for row in metric_rows]))
        if metric_rows
        else None,
        "per_model": per_model,
    }
    norm_audit = {
        "schema_version": 1,
        "norm": "linf",
        "measurement": "post-save PNG decoded uint8 RGB in [0,1]",
        "epsilon": epsilon,
        "quantization_tolerance": tolerance,
        "samples": eligible_count,
        "max_linf": max_linf,
        "violations": violations,
    }
    atomic_write_text(run_dir / "selection.jsonl", _jsonl(selection_rows))
    atomic_write_text(run_dir / "per_sample_metrics.jsonl", _jsonl(metric_rows))
    atomic_write_json(run_dir / "norm_audit.json", norm_audit)
    atomic_write_json(run_dir / "summary.json", summary)
    lpips_calibration = (
        Path(lpips_module.__file__).resolve().parent
        / server["assets"]["lpips"]["calibration_relative_path"]
    )
    provenance_weights = {
        **weight_paths,
        "lpips_alexnet_backbone": backbone_path,
        "lpips_calibration": lpips_calibration,
    }
    provenance = collect_provenance(
        PROJECT_ROOT,
        run_dir / "resolved_config.yaml",
        run_dir / "manifest.snapshot.jsonl",
        provenance_weights,
    )
    atomic_write_json(run_dir / "provenance.json", provenance)
    metadata_files = (
        "resolved_config.yaml",
        "resolved_server_config.yaml",
        "manifest.snapshot.jsonl",
        "selection.jsonl",
        "per_sample_metrics.jsonl",
        "norm_audit.json",
        "summary.json",
        "provenance.json",
        "preflight.json",
    )
    atomic_write_json(
        run_dir / "artifacts.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "metadata_root": str(run_dir),
            "heavy_root": str(heavy_root),
            "files": {
                name: sha256_file(run_dir / name)
                for name in metadata_files
            },
            "generated_images": {
                "directory": str(images_root),
                "count": eligible_count,
            },
        },
    )
    report = verify_run(run_dir, write_report=True)
    if report["outcome"] != "passed":
        raise ContractError("Run verification failed: " + "; ".join(report["errors"]))
    return report
