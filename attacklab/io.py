"""Small, dependency-free helpers shared by the control layer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when a versioned pipeline contract is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read JSON {path}: {exc}") from exc


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ContractError(
            "PyYAML is required; install the pinned requirements.lock environment"
        ) from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"Cannot read YAML {path}: {exc}") from exc


def dump_yaml(path: Path, value: Any) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ContractError(
            "PyYAML is required; install the pinned requirements.lock environment"
        ) from exc
    atomic_write_text(
        path,
        yaml.safe_dump(value, sort_keys=True, allow_unicode=True),
    )


def project_relative_path(project_root: Path, raw_path: str | Path) -> Path:
    """Resolve a project-relative path and reject traversal or absolute input."""

    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ContractError(f"Expected a project-relative path, got {candidate}")
    resolved_root = project_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"Path escapes project root: {candidate}") from exc
    return resolved
