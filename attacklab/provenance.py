"""Capture reproducibility metadata without copying credentials or full env vars."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import sha256_file, utc_now


PACKAGE_NAMES = (
    "numpy",
    "PyYAML",
    "Pillow",
    "scipy",
    "scikit-image",
    "tqdm",
    "torch",
    "torchvision",
    "lpips",
    "jsonschema",
)


def _git(project_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _tracked_diff_sha256(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "HEAD", "--"],
        cwd=project_root,
        check=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=20,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def collect_provenance(
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    weight_paths: dict[str, Path],
) -> dict[str, Any]:
    status = _git(project_root, "status", "--porcelain=v1", "--untracked-files=no")
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "git": {
            "commit": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "tracked_worktree_clean": status == "",
            "tracked_diff_sha256": _tracked_diff_sha256(project_root),
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "weights": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in sorted(weight_paths.items())
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": _packages(),
        },
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime
        provenance["runtime"]["torch_import_error"] = str(exc)
    else:
        provenance["runtime"]["determinism"] = {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            # warn_only: a nondeterministic kernel warns instead of aborting.
            # See _set_determinism in attacklab/runner.py.
            "deterministic_algorithms_warn_only": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        provenance["runtime"]["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "visible_device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    return provenance
