"""Deterministic validation for completed scientific run artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import load_experiment_config
from .io import ContractError, atomic_write_json, load_json, sha256_file, utc_now


REQUIRED_FILES = (
    "resolved_config.yaml",
    "resolved_server_config.yaml",
    "manifest.snapshot.jsonl",
    "selection.jsonl",
    "per_sample_metrics.jsonl",
    "norm_audit.json",
    "summary.json",
    "provenance.json",
    "artifacts.json",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"Cannot read JSONL {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid JSON at {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"Expected JSON object at {path}:{number}")
        rows.append(row)
    return rows


def _finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ContractError(f"{field} must be finite")
    return converted


def verify_run(run_dir: Path, write_report: bool = True) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    errors: list[str] = []
    if missing:
        errors.append("missing required artifacts: " + ", ".join(missing))
        report = {
            "schema_version": 1,
            "verified_at": utc_now(),
            "outcome": "failed",
            "run_dir": str(run_dir),
            "errors": errors,
        }
        if write_report:
            atomic_write_json(run_dir / "verification.json", report)
        return report

    try:
        config = load_experiment_config(run_dir / "resolved_config.yaml")
        summary = load_json(run_dir / "summary.json")
        provenance = load_json(run_dir / "provenance.json")
        norm_audit = load_json(run_dir / "norm_audit.json")
        artifacts = load_json(run_dir / "artifacts.json")
        selection = _jsonl(run_dir / "selection.jsonl")
        metrics = _jsonl(run_dir / "per_sample_metrics.jsonl")
        if not isinstance(summary, dict) or summary.get("schema_version") != 1:
            raise ContractError("summary.json must be a schema-version-1 object")
        if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
            raise ContractError("provenance.json must be a schema-version-1 object")
        if not isinstance(norm_audit, dict) or norm_audit.get("schema_version") != 1:
            raise ContractError("norm_audit.json must be a schema-version-1 object")
        if not isinstance(artifacts, dict) or artifacts.get("schema_version") != 1:
            raise ContractError("artifacts.json must be a schema-version-1 object")

        selected = int(summary.get("samples_selected", -1))
        eligible = int(summary.get("samples_eligible", -1))
        evaluated = int(summary.get("samples_evaluated", -1))
        if selected != len(selection):
            raise ContractError(
                f"samples_selected={selected} but selection.jsonl has {len(selection)} rows"
            )
        observed_eligible = sum(row.get("eligible") is True for row in selection)
        if eligible != observed_eligible:
            raise ContractError(
                f"samples_eligible={eligible} but selection has {observed_eligible} eligible rows"
            )
        if evaluated != len(metrics) or evaluated != eligible:
            raise ContractError(
                "samples_evaluated, eligible selection rows, and metric rows must agree"
            )
        if evaluated < 1:
            raise ContractError("A passed run requires at least one eligible sample")

        ids = [row.get("sample_id") for row in metrics]
        if not all(isinstance(value, str) and value for value in ids):
            raise ContractError("Every metric row requires sample_id")
        if len(set(ids)) != len(ids):
            raise ContractError("per_sample_metrics.jsonl contains duplicate sample ids")

        epsilon = float(config["constraint"]["epsilon"])
        # PNG uint8 round-trip permits at most one quantization step beyond a
        # non-integer epsilon expressed in [0,1] space.
        tolerance = (1.0 / 255.0) + 1e-12
        violations = 0
        for index, row in enumerate(metrics, start=1):
            linf = _finite_number(row.get("linf"), f"metrics row {index}.linf")
            _finite_number(row.get("ssim"), f"metrics row {index}.ssim")
            _finite_number(row.get("lpips"), f"metrics row {index}.lpips")
            if linf > epsilon + tolerance:
                violations += 1
            clean_predictions = row.get("clean_predictions")
            adversarial_predictions = row.get("adversarial_predictions")
            if not isinstance(clean_predictions, dict) or not isinstance(
                adversarial_predictions, dict
            ):
                raise ContractError(
                    f"metrics row {index} requires clean and adversarial predictions"
                )
            output = row.get("output")
            if not isinstance(output, dict):
                raise ContractError(f"metrics row {index} requires output metadata")
            output_path = Path(str(output.get("path", "")))
            output_sha = output.get("sha256")
            if not output_path.is_file():
                raise ContractError(f"metrics row {index} output is missing: {output_path}")
            if not isinstance(output_sha, str) or sha256_file(output_path) != output_sha:
                raise ContractError(f"metrics row {index} output hash mismatch")

        declared_violations = int(summary.get("constraint_violations", -1))
        audit_violations = int(norm_audit.get("violations", -1))
        if violations != declared_violations or violations != audit_violations:
            raise ContractError("Constraint violation counts disagree")
        if violations:
            raise ContractError(f"Found {violations} post-save L-inf constraint violations")
        if norm_audit.get("epsilon") != config["constraint"]["epsilon"]:
            raise ContractError("norm_audit epsilon differs from resolved config")

        timing = summary.get("timing")
        if not isinstance(timing, dict) or timing.get("schema_version") != 1:
            raise ContractError("summary.timing must be a schema-version-1 object")
        for field in ("started_at", "finished_at", "measurement"):
            value = timing.get(field)
            if not isinstance(value, str) or not value:
                raise ContractError(f"summary.timing.{field} is missing")
        elapsed = _finite_number(timing.get("elapsed_seconds"), "timing.elapsed_seconds")
        if elapsed < 0:
            raise ContractError("timing.elapsed_seconds must not be negative")

        inputs = provenance.get("inputs")
        if not isinstance(inputs, dict):
            raise ContractError("provenance.inputs is missing")
        if inputs.get("config", {}).get("sha256") != sha256_file(
            run_dir / "resolved_config.yaml"
        ):
            raise ContractError("Resolved config hash differs from provenance")
        if inputs.get("manifest", {}).get("sha256") != sha256_file(
            run_dir / "manifest.snapshot.jsonl"
        ):
            raise ContractError("Manifest snapshot hash differs from provenance")
    except (ContractError, OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    report = {
        "schema_version": 1,
        "verified_at": utc_now(),
        "outcome": "passed" if not errors else "failed",
        "run_dir": str(run_dir),
        "errors": errors,
        "verified_files": {
            name: sha256_file(run_dir / name)
            for name in REQUIRED_FILES
            if (run_dir / name).is_file()
        },
    }
    if write_report:
        atomic_write_json(run_dir / "verification.json", report)
    return report
