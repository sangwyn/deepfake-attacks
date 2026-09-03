"""Schema-backed status validation shared by controllers and workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import ContractError, load_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_status(value: dict[str, Any], kind: str) -> None:
    if kind not in {"attack", "review"}:
        raise ContractError(f"Unknown status kind: {kind}")
    schema_path = PROJECT_ROOT / "schemas" / f"{kind}-status.schema.json"
    schema = load_json(schema_path)
    try:
        import jsonschema
    except ImportError as exc:
        raise ContractError(
            "jsonschema is required; install the pinned requirements.lock environment"
        ) from exc
    try:
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(item) for item in exc.absolute_path) or "<root>"
        raise ContractError(f"Invalid {kind} status at {location}: {exc.message}") from exc


def load_and_validate_status(path: Path, kind: str) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ContractError(f"Status must be a JSON object: {path}")
    validate_status(value, kind)
    return value
