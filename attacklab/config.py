"""Strict loaders for server and experiment configuration contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import ContractError, canonical_json_sha256, load_yaml, project_relative_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
SUPPORTED_MODELS = {"vit_b_16", "densenet121_dct"}
SUPPORTED_SCOPES = {"smoke", "development", "full", "official"}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a mapping")
    return value


def _require_keys(value: dict[str, Any], name: str, keys: set[str]) -> None:
    missing = keys - value.keys()
    if missing:
        raise ContractError(f"{name} is missing: {', '.join(sorted(missing))}")


def load_server_config(path: Path) -> dict[str, Any]:
    config = _mapping(load_yaml(path), "server config")
    if config.get("schema_version") != 1:
        raise ContractError("Server config must use schema_version: 1")
    _require_keys(
        config,
        "server config",
        {"project_root", "dataset", "assets", "runtime", "environment", "opencode"},
    )
    for field in ("project_root",):
        if not isinstance(config[field], str) or not Path(config[field]).is_absolute():
            raise ContractError(f"{field} must be an absolute path")

    dataset = _mapping(config["dataset"], "dataset")
    _require_keys(dataset, "dataset", {"celeb_a_root", "classes"})
    if not Path(str(dataset["celeb_a_root"])).is_absolute():
        raise ContractError("dataset.celeb_a_root must be absolute")
    classes = _mapping(dataset["classes"], "dataset.classes")
    if set(classes) != {"TRAIN_REAL", "TRAIN_FAKE", "TEST_REAL", "TEST_FAKE"}:
        raise ContractError("dataset.classes must define the four audited CelebA classes")
    labels = [entry.get("label") for entry in classes.values() if isinstance(entry, dict)]
    if sorted(set(labels)) != [0, 1]:
        raise ContractError("dataset class labels must use 0=real and 1=fake")

    assets = _mapping(config["assets"], "assets")
    _require_keys(assets, "assets", {"weights_root", "models"})
    if not Path(str(assets["weights_root"])).is_absolute():
        raise ContractError("assets.weights_root must be absolute")
    models = _mapping(assets["models"], "assets.models")
    if set(models) != SUPPORTED_MODELS:
        raise ContractError(
            "assets.models must define vit_b_16 and densenet121_dct exactly"
        )
    for name, model in models.items():
        model = _mapping(model, f"assets.models.{name}")
        _require_keys(model, f"assets.models.{name}", {"filename", "sha256"})
        digest = model["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"assets.models.{name}.sha256 is invalid")
    lpips = _mapping(assets.get("lpips"), "assets.lpips")
    _require_keys(
        lpips,
        "assets.lpips",
        {
            "alexnet_backbone_path",
            "alexnet_backbone_sha256",
            "calibration_relative_path",
            "calibration_sha256",
        },
    )
    if not Path(str(lpips["alexnet_backbone_path"])).is_absolute():
        raise ContractError("assets.lpips.alexnet_backbone_path must be absolute")
    for key in ("alexnet_backbone_sha256", "calibration_sha256"):
        if not isinstance(lpips[key], str) or not re.fullmatch(r"[0-9a-f]{64}", lpips[key]):
            raise ContractError(f"assets.lpips.{key} is invalid")

    runtime = _mapping(config["runtime"], "runtime")
    _require_keys(runtime, "runtime", {"runs_root", "tracking_root"})
    if not Path(str(runtime["runs_root"])).is_absolute():
        raise ContractError("runtime.runs_root must be absolute")
    if Path(str(runtime["tracking_root"])).is_absolute():
        raise ContractError("runtime.tracking_root must be project-relative")
    return config


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = _mapping(load_yaml(path), "experiment config")
    if config.get("schema_version") != 1:
        raise ContractError("Experiment config must use schema_version: 1")
    _require_keys(
        config,
        "experiment config",
        {
            "experiment_id",
            "scope",
            "seed",
            "server_config",
            "dataset",
            "attack",
            "models",
            "constraint",
            "metrics",
        },
    )
    experiment_id = config["experiment_id"]
    if not isinstance(experiment_id, str) or not ID_RE.fullmatch(experiment_id):
        raise ContractError("experiment_id must be a stable lowercase identifier")
    if config["scope"] not in SUPPORTED_SCOPES:
        raise ContractError(f"Unsupported experiment scope: {config['scope']!r}")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ContractError("seed must be a non-negative integer")
    for field in ("server_config",):
        project_relative_path(PROJECT_ROOT, str(config[field]))

    dataset = _mapping(config["dataset"], "dataset")
    _require_keys(dataset, "dataset", {"manifest", "source_label", "selection"})
    project_relative_path(PROJECT_ROOT, str(dataset["manifest"]))
    if dataset["source_label"] not in {0, 1}:
        raise ContractError("dataset.source_label must be 0 or 1")
    if dataset["selection"] not in {"manifest-order", "all"}:
        raise ContractError("dataset.selection must be manifest-order or all")
    sample_limit = dataset.get("sample_limit")
    if sample_limit is not None and (
        not isinstance(sample_limit, int) or sample_limit < 1
    ):
        raise ContractError("dataset.sample_limit must be null or a positive integer")

    attack = _mapping(config["attack"], "attack")
    _require_keys(
        attack,
        "attack",
        {"name", "module", "source_model", "target_class", "parameters"},
    )
    if attack["source_model"] not in SUPPORTED_MODELS:
        raise ContractError("attack.source_model is unsupported")
    if attack["target_class"] not in {0, 1}:
        raise ContractError("attack.target_class must be 0 or 1")
    if attack["target_class"] == dataset["source_label"]:
        raise ContractError("target_class must differ from dataset.source_label")
    if not isinstance(attack["name"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]*", attack["name"]
    ):
        raise ContractError("attack.name must be a lowercase attack identifier")
    if not isinstance(attack["module"], str) or not re.fullmatch(
        r"attacks\.[A-Za-z0-9_.]+", attack["module"]
    ):
        raise ContractError("attack.module must name a module under attacks")
    _mapping(attack["parameters"], "attack.parameters")

    models = config["models"]
    if (
        not isinstance(models, list)
        or not models
        or len(set(models)) != len(models)
        or not set(models).issubset(SUPPORTED_MODELS)
    ):
        raise ContractError("models must be a unique non-empty supported-model list")
    if attack["source_model"] not in models:
        raise ContractError("attack.source_model must be present in models")

    constraint = _mapping(config["constraint"], "constraint")
    _require_keys(constraint, "constraint", {"norm", "epsilon", "pixel_range"})
    if constraint["norm"] != "linf":
        raise ContractError("Only linf is supported by contract version 1")
    epsilon = constraint["epsilon"]
    if not isinstance(epsilon, (int, float)) or not 0 < float(epsilon) <= 1:
        raise ContractError("constraint.epsilon must be in (0, 1]")
    if constraint["pixel_range"] != [0.0, 1.0]:
        raise ContractError("constraint.pixel_range must be [0.0, 1.0]")
    if constraint.get("output_format", "png") != "png":
        raise ContractError("constraint.output_format must be png")
    attack_epsilon = attack["parameters"].get("epsilon")
    if attack_epsilon is None or not isinstance(attack_epsilon, (int, float)):
        raise ContractError("attack.parameters.epsilon is mandatory")
    if abs(float(attack_epsilon) - float(epsilon)) > 1e-12:
        raise ContractError(
            "attack.parameters.epsilon must equal constraint.epsilon exactly"
        )

    metrics = _mapping(config["metrics"], "metrics")
    _require_keys(metrics, "metrics", {"ssim", "lpips", "targeted_asr"})
    if any(metrics[key] is not True for key in ("ssim", "lpips", "targeted_asr")):
        raise ContractError("SSIM, LPIPS, and targeted ASR are mandatory")

    config["config_sha256"] = canonical_json_sha256(config)
    return config
