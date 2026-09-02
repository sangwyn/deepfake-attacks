"""Fail-closed server, asset, environment, and dataset preflight checks."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_server_config
from .io import ContractError, load_json, project_relative_path, sha256_file, utc_now
from .manifest import IMAGE_SUFFIXES


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _run(argv: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )


def _version_matches(distribution: str, expected: str | set[str]) -> Check:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return Check(f"package:{distribution}", "fail", "not installed")
    accepted = {expected} if isinstance(expected, str) else expected
    status = "pass" if actual in accepted else "fail"
    return Check(f"package:{distribution}", status, f"expected={expected}, actual={actual}")


def run_preflight(config_path: Path, deep: bool = False) -> dict[str, Any]:
    config = load_server_config(config_path)
    checks: list[Check] = []
    configured_root = Path(config["project_root"]).resolve()
    actual_root = PROJECT_ROOT.resolve()
    checks.append(
        Check(
            "project-root-binding",
            "pass" if configured_root == actual_root else "fail",
            f"configured={configured_root}, actual={actual_root}",
        )
    )

    dataset = config["dataset"]
    dataset_root = Path(dataset["celeb_a_root"])
    checks.append(
        Check(
            "dataset-root",
            "pass" if dataset_root.is_dir() else "fail",
            str(dataset_root),
        )
    )
    for class_name, contract in dataset["classes"].items():
        class_root = dataset_root / contract["path"]
        if class_root.is_dir():
            count = sum(
                1
                for path in class_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            expected = int(contract["expected_count"])
            checks.append(
                Check(
                    f"dataset:{class_name}",
                    "pass" if count == expected else "fail",
                    f"expected={expected}, actual={count}, path={class_root}",
                )
            )
        else:
            checks.append(Check(f"dataset:{class_name}", "fail", f"missing {class_root}"))

    weights_root = Path(config["assets"]["weights_root"])
    for model_name, contract in config["assets"]["models"].items():
        path = weights_root / contract["filename"]
        if not path.is_file():
            checks.append(Check(f"weight:{model_name}", "fail", f"missing {path}"))
            continue
        actual = sha256_file(path)
        expected = contract["sha256"]
        checks.append(
            Check(
                f"weight:{model_name}",
                "pass" if actual == expected else "fail",
                f"expected_sha256={expected}, actual_sha256={actual}, path={path}",
            )
        )

    lpips_contract = config["assets"].get("lpips")
    if not isinstance(lpips_contract, dict):
        checks.append(Check("weight:lpips", "fail", "assets.lpips is not configured"))
    else:
        backbone = Path(str(lpips_contract.get("alexnet_backbone_path", "")))
        expected = lpips_contract.get("alexnet_backbone_sha256")
        if not backbone.is_file():
            checks.append(Check("weight:lpips-alexnet", "fail", f"missing {backbone}"))
        else:
            actual = sha256_file(backbone)
            checks.append(
                Check(
                    "weight:lpips-alexnet",
                    "pass" if actual == expected else "fail",
                    f"expected_sha256={expected}, actual_sha256={actual}, path={backbone}",
                )
            )
        calibration_relative = lpips_contract.get("calibration_relative_path")
        calibration_expected = lpips_contract.get("calibration_sha256")
        if deep:
            try:
                import lpips
            except Exception as exc:
                checks.append(
                    Check("weight:lpips-calibration", "fail", f"lpips import failed: {exc}")
                )
            else:
                calibration = Path(lpips.__file__).resolve().parent / str(
                    calibration_relative
                )
                if not calibration.is_file():
                    checks.append(
                        Check(
                            "weight:lpips-calibration",
                            "fail",
                            f"missing {calibration}",
                        )
                    )
                else:
                    actual = sha256_file(calibration)
                    checks.append(
                        Check(
                            "weight:lpips-calibration",
                            "pass" if actual == calibration_expected else "fail",
                            f"expected_sha256={calibration_expected}, actual_sha256={actual}, path={calibration}",
                        )
                    )

    runtime_root = Path(config["runtime"]["runs_root"])
    runtime_parent = runtime_root if runtime_root.exists() else runtime_root.parent
    checks.append(
        Check(
            "runtime-parent",
            "pass" if runtime_parent.is_dir() else "fail",
            f"expected existing parent for {runtime_root}: {runtime_parent}",
        )
    )
    tracking_root = project_relative_path(actual_root, config["runtime"]["tracking_root"])
    checks.append(
        Check(
            "tracking-root",
            "pass" if tracking_root.is_dir() else "fail",
            str(tracking_root),
        )
    )

    environment_lock = project_relative_path(
        actual_root, config["environment"]["lock_file"]
    )
    checks.append(
        Check(
            "environment-lock",
            "pass" if environment_lock.is_file() else "fail",
            str(environment_lock),
        )
    )
    if environment_lock.is_file():
        locked = load_json(environment_lock)
        expected_python = locked.get("python") if isinstance(locked, dict) else None
        actual_python = platform.python_version()
        checks.append(
            Check(
                "python-version",
                "pass" if actual_python == expected_python else "fail",
                f"expected={expected_python}, actual={actual_python}",
            )
        )
        if isinstance(locked, dict):
            for path_key, hash_key, check_name in (
                ("lock_file", "lock_file_sha256", "requirements-lock-hash"),
                ("reference_freeze", "reference_freeze_sha256", "reference-freeze-hash"),
            ):
                raw = locked.get(path_key)
                expected_hash = locked.get(hash_key)
                if not isinstance(raw, str) or not isinstance(expected_hash, str):
                    checks.append(Check(check_name, "fail", "path/hash missing in lock"))
                    continue
                artifact = project_relative_path(actual_root, raw)
                if not artifact.is_file():
                    checks.append(Check(check_name, "fail", f"missing {artifact}"))
                else:
                    actual_hash = sha256_file(artifact)
                    checks.append(
                        Check(
                            check_name,
                            "pass" if actual_hash == expected_hash else "fail",
                            f"expected_sha256={expected_hash}, actual_sha256={actual_hash}",
                        )
                    )

    opencode = config["opencode"]
    binary = Path(opencode["binary"])
    if not binary.is_file():
        checks.append(Check("opencode-binary", "fail", f"missing {binary}"))
    else:
        result = _run([str(binary), "--version"])
        actual_version = result.stdout.strip() or result.stderr.strip()
        checks.append(
            Check(
                "opencode-version",
                "pass"
                if result.returncode == 0 and actual_version == str(opencode["version"])
                else "fail",
                f"expected={opencode['version']}, actual={actual_version}",
            )
        )

    if shutil.which("nvidia-smi") is None:
        checks.append(Check("nvidia-smi", "fail", "not found on PATH"))
    else:
        result = _run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = [line for line in result.stdout.splitlines() if line.strip()]
        checks.append(
            Check(
                "nvidia-smi",
                "pass" if result.returncode == 0 and rows else "fail",
                f"visible_gpus={len(rows)}; query_return_code={result.returncode}",
            )
        )

    if deep:
        pinned = {
            "attrs": "25.3.0",
            "filelock": "3.32.5",
            "fsspec": "2026.7.0",
            "ImageIO": "2.37.4",
            "Jinja2": "3.1.6",
            "numpy": "1.26.4",
            "PyYAML": "6.0.2",
            "Pillow": "11.0.0",
            "lazy-loader": "0.5",
            "MarkupSafe": "3.0.3",
            "mpmath": "1.3.0",
            "networkx": "3.6.1",
            "packaging": "26.3",
            "referencing": "0.36.2",
            "rpds-py": "0.27.0",
            "scipy": "1.15.3",
            "scikit-image": "0.26.0",
            "sympy": "1.14.0",
            "tifffile": "2026.3.3",
            "tqdm": "4.67.1",
            "torch": {"2.3.0", "2.3.0+cu121"},
            "torchvision": {"0.18.0", "0.18.0+cu121"},
            "lpips": "0.1.4",
            "jsonschema": "4.23.0",
            "jsonschema-specifications": "2025.4.1",
            "typing_extensions": "4.16.0",
        }
        checks.extend(_version_matches(name, version) for name, version in pinned.items())
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on server runtime
            checks.append(Check("torch-cuda", "fail", f"torch import failed: {exc}"))
        else:
            available = bool(torch.cuda.is_available())
            runtime_exact = (
                torch.__version__ == "2.3.0+cu121"
                and torch.version.cuda == "12.1"
                and torch.backends.cudnn.version() == 8902
            )
            checks.append(
                Check(
                    "torch-cuda",
                    "pass" if available else "fail",
                    f"available={available}, device_count={torch.cuda.device_count()}",
                )
            )
            checks.append(
                Check(
                    "torch-runtime-version",
                    "pass" if runtime_exact else "fail",
                    "expected=torch 2.3.0+cu121,cuda 12.1,cudnn 8902; "
                    f"actual=torch {torch.__version__},cuda {torch.version.cuda},"
                    f"cudnn {torch.backends.cudnn.version()}",
                )
            )

    status = "pass" if all(check.status != "fail" for check in checks) else "fail"
    return {
        "schema_version": 1,
        "checked_at": utc_now(),
        "status": status,
        "config": str(config_path.resolve()),
        "deep": deep,
        "checks": [asdict(check) for check in checks],
    }


def print_preflight(report: dict[str, Any]) -> None:
    for check in report["checks"]:
        print(f"[{check['status'].upper():4}] {check['name']}: {check['detail']}")
    print(json.dumps({"status": report["status"], "deep": report["deep"]}))
