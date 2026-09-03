"""Immutable job specification and queue state definitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .errors import SpecError


SCHEMA_VERSION = 1
CANONICAL_TASK_KIND = "attack-experiment"

JOB_STATES = frozenset(
    {
        "queued",
        "reserving",
        "running",
        "validating",
        "succeeded",
        "failed",
        "retry_wait",
        "cancelled",
        "orphaned",
    }
)
ACTIVE_STATES = frozenset({"reserving", "running", "validating"})
READY_STATES = frozenset({"queued", "retry_wait"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "orphaned"})

INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "task_kind",
        "config_path",
        "run_dir",
        "requested_memory_mb",
        "timeout_seconds",
        "priority",
        "max_attempts",
    }
)


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return the one canonical encoding used for hashes and persistence."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise SpecError(f"{field} must be between {minimum} and {maximum}")
    return value


def _safe_relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{field} must be a non-empty project-relative path")
    if "\\" in value or "\x00" in value:
        raise SpecError(f"{field} contains unsupported characters")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SpecError(f"{field} must be a normalized project-relative path")
    return path


def resolve_inside(project_root: Path, relative: PurePosixPath, *, strict: bool) -> Path:
    root = project_root.resolve(strict=True)
    try:
        resolved = (root / Path(*relative.parts)).resolve(strict=strict)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SpecError(f"Path escapes or cannot be resolved inside project: {relative}") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class JobSpec:
    """Normalized, immutable schema-v1 GPU job specification."""

    schema_version: int
    task_kind: str
    config_path: str
    config_sha256: str
    run_dir: str
    requested_memory_mb: int
    timeout_seconds: int
    priority: int = 0
    max_attempts: int = 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], project_root: Path) -> "JobSpec":
        if not isinstance(raw, Mapping):
            raise SpecError("Job specification must be a JSON object")
        unknown = set(raw) - INPUT_FIELDS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise SpecError(f"Unknown job specification fields: {names}")
        missing = {
            "schema_version",
            "task_kind",
            "config_path",
            "run_dir",
            "requested_memory_mb",
            "timeout_seconds",
        } - set(raw)
        if missing:
            raise SpecError(
                "Missing job specification fields: " + ", ".join(sorted(missing))
            )

        schema_version = _integer(
            raw["schema_version"],
            "schema_version",
            minimum=SCHEMA_VERSION,
            maximum=SCHEMA_VERSION,
        )
        task_kind = raw["task_kind"]
        if task_kind != CANONICAL_TASK_KIND:
            raise SpecError(
                f"task_kind must be exactly {CANONICAL_TASK_KIND!r}; commands are not accepted"
            )

        root = project_root.resolve(strict=True)
        config_relative = _safe_relative_path(raw["config_path"], "config_path")
        config_absolute = resolve_inside(root, config_relative, strict=True)
        if not config_absolute.is_file():
            raise SpecError(f"config_path is not a regular file: {config_relative}")
        if config_absolute.suffix.lower() not in {".yaml", ".yml", ".json"}:
            raise SpecError("config_path must end in .yaml, .yml, or .json")

        run_relative = _safe_relative_path(raw["run_dir"], "run_dir")
        if len(run_relative.parts) < 4 or run_relative.parts[:2] != ("tracking", "runs"):
            raise SpecError(
                "run_dir must be tracking/runs/<campaign>/<task> or a deeper path"
            )
        resolve_inside(root, run_relative, strict=False)

        return cls(
            schema_version=schema_version,
            task_kind=task_kind,
            config_path=config_relative.as_posix(),
            config_sha256=sha256_file(config_absolute),
            run_dir=run_relative.as_posix(),
            requested_memory_mb=_integer(
                raw["requested_memory_mb"],
                "requested_memory_mb",
                minimum=1,
                maximum=1024 * 1024,
            ),
            timeout_seconds=_integer(
                raw["timeout_seconds"],
                "timeout_seconds",
                minimum=10,
                maximum=7 * 24 * 60 * 60,
            ),
            priority=_integer(
                raw.get("priority", 0), "priority", minimum=-100, maximum=100
            ),
            max_attempts=_integer(
                raw.get("max_attempts", 1), "max_attempts", minimum=1, maximum=5
            ),
        )

    @classmethod
    def from_persisted(cls, raw: Mapping[str, Any]) -> "JobSpec":
        """Load an already validated normalized spec without reading its config."""

        expected = {field.name for field in fields(cls)}
        if set(raw) != expected:
            raise SpecError("Persisted job specification has an invalid field set")
        try:
            spec = cls(**raw)
        except TypeError as exc:
            raise SpecError(f"Persisted job specification is invalid: {exc}") from exc
        if spec.schema_version != SCHEMA_VERSION or spec.task_kind != CANONICAL_TASK_KIND:
            raise SpecError("Persisted job specification uses an unsupported schema")
        return spec

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def canonical(self) -> str:
        return canonical_json(self.as_dict())

    @property
    def idempotency_key(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()

    def config_absolute(self, project_root: Path) -> Path:
        return resolve_inside(project_root, PurePosixPath(self.config_path), strict=True)

    def run_absolute(self, project_root: Path) -> Path:
        return resolve_inside(project_root, PurePosixPath(self.run_dir), strict=False)

    def verify_config_unchanged(self, project_root: Path) -> None:
        actual = sha256_file(self.config_absolute(project_root))
        if actual != self.config_sha256:
            raise SpecError(
                f"Frozen config hash changed: expected {self.config_sha256}, got {actual}"
            )
