"""Attack idea specifications and their agreement with the actual project.

A specification states what an attack is and how it must be judged. It is
authored by hand and reviewed; nothing in the pipeline may edit one. The point
of this module is to fail loudly and early when a specification disagrees with
what the checkout can actually run, instead of discovering it after a GPU has
been reserved.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_server_config
from .io import ContractError, load_json, load_yaml

SPEC_ROOT = PROJECT_ROOT / "specs" / "attacks"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "attack-spec.schema.json"
STEP_SIZE_EXPRESSION = "epsilon / "


def validate_against_schema(spec: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise ContractError(
            "jsonschema is required; install the pinned requirements.lock environment"
        ) from exc
    schema = load_json(SCHEMA_PATH)
    try:
        jsonschema.Draft202012Validator(schema).validate(spec)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(item) for item in exc.absolute_path) or "<root>"
        raise ContractError(
            f"Invalid attack specification at {location}: {exc.message}"
        ) from exc


def load_spec(path: Path) -> dict[str, Any]:
    spec = load_yaml(path)
    if not isinstance(spec, dict):
        raise ContractError(f"Attack specification must be a mapping: {path}")
    validate_against_schema(spec)
    if spec["idea_id"] != path.stem:
        raise ContractError(
            f"idea_id {spec['idea_id']!r} does not match file name {path.stem!r}"
        )
    return spec


def load_all_specs(root: Path | None = None) -> dict[str, dict[str, Any]]:
    directory = root or SPEC_ROOT
    specs = {}
    for path in sorted(directory.glob("*.yaml")):
        spec = load_spec(path)
        specs[spec["idea_id"]] = spec
    if not specs:
        raise ContractError(f"No attack specifications found under {directory}")
    return specs


def resolve_step_size(spec: dict[str, Any], epsilon: float) -> float:
    """Turn the declared step size into the number a config must record."""

    parameters = spec["parameters"]
    declared = parameters.get("step_size")
    iterations = parameters["iterations"]
    if declared is None:
        if iterations == 1:
            return float(epsilon)
        raise ContractError(f"{spec['idea_id']} declares no step_size")
    if isinstance(declared, str):
        # "epsilon / iterations" or "epsilon / <n>": the specification states the
        # ratio, the resolved config must record the resulting number.
        divisor_text = declared[len(STEP_SIZE_EXPRESSION):]
        if divisor_text == "iterations":
            if not isinstance(iterations, int):
                raise ContractError(
                    f"{spec['idea_id']} derives step_size from a "
                    "non-numeric iteration count"
                )
            divisor = iterations
        else:
            divisor = int(divisor_text)
        if divisor < 1:
            raise ContractError(f"{spec['idea_id']} has a non-positive step divisor")
        return float(epsilon) / divisor
    return float(declared)


def readiness(
    spec: dict[str, Any], server: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return (blockers, warnings) for one specification against this checkout."""

    blockers: list[str] = []
    warnings: list[str] = []
    identifier = spec["idea_id"]

    configured = set(server["assets"]["models"])
    requested = set(spec["evaluation"]["target_models"])
    unconfigured = sorted(requested - configured)
    if unconfigured:
        blockers.append(
            f"{identifier}: target models are not configured in the server "
            f"contract: {', '.join(unconfigured)}"
        )

    manifest = PROJECT_ROOT / spec["evaluation"]["manifest"]
    if not manifest.is_file():
        blockers.append(f"{identifier}: manifest does not exist: {manifest}")

    module = spec["implementation"]["module"]
    if importlib.util.find_spec(module) is None:
        warnings.append(f"{identifier}: {module} is not implemented yet")

    if spec["evaluation"]["source_label"] == spec["evaluation"]["target_class"]:
        blockers.append(f"{identifier}: a targeted attack needs target != source label")

    if spec["evaluation"]["first_scope"] == "full":
        warnings.append(
            f"{identifier}: declares first_scope full, which needs explicit "
            "authorization; the campaign still starts at smoke"
        )

    if spec["implementation"]["source_model"] == "leave-one-detector-out-ensemble" and (
        len(configured) < 3
    ):
        warnings.append(
            f"{identifier}: leave-one-detector-out leaves {len(configured) - 1} "
            "source(s) with the detectors configured here; the result is not "
            "held-out transfer evidence"
        )

    gate = spec["retention_gate"]
    if gate["minimum_gain_percentage_points"] <= 0:
        warnings.append(
            f"{identifier}: retention gate accepts any non-negative gain, so it "
            "does not discriminate between candidates"
        )
    return blockers, warnings


def report(server_config: Path, root: Path | None = None) -> dict[str, Any]:
    """Machine-readable readiness of every specification."""

    server = load_server_config(server_config)
    specs = load_all_specs(root)
    blockers: list[str] = []
    warnings: list[str] = []
    for spec in specs.values():
        spec_blockers, spec_warnings = readiness(spec, server)
        blockers.extend(spec_blockers)
        warnings.extend(spec_warnings)
    return {
        "schema_version": 1,
        "specifications": sorted(specs),
        "status": "fail" if blockers else "pass",
        "blockers": blockers,
        "warnings": warnings,
    }
