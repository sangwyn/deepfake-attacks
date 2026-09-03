#!/usr/bin/env python3
"""Run the attack campaign as isolated OpenCode sessions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

# Importable both as "python3 scripts/run_campaign.py" (sys.path[0] is scripts/)
# and as "from scripts import run_campaign". ops.gpuq imports only the standard
# library, so binding the queue at module load cannot pull in torch or CUDA.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.gpuq import GpuQueueError, QueueDatabase  # noqa: E402

DEFAULT_MANIFEST = ROOT / "CAMPAIGN.yaml"
RUNTIME_ROOT = ROOT / ".campaign"
RUNS_ROOT = RUNTIME_ROOT / "runs"
LATEST_FILE = RUNTIME_ROOT / "latest.json"

# Every outcome the status contract recognises. A worker may only produce the
# provisional subset; the controller owns running, passed, and cancelled.
ATTACK_OUTCOMES = {"queued", "running", "passed", "failed", "blocked", "cancelled"}
WORKER_OUTCOMES = {"queued", "failed", "blocked"}
REVIEW_OUTCOMES = {"passed", "failed", "blocked"}
DECISIONS = {"retain", "reject", "baseline", "pending", "not_applicable"}
TERMINAL_STATES = {"passed", "failed", "blocked", "cancelled", "skipped"}
ABORT_STATES = {"failed", "blocked", "cancelled"}
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Mirrors ops.gpuq.models.TERMINAL_STATES; a job in any other state is still
# owned by the scheduler and must be polled again.
GPUQ_TERMINAL = {"succeeded", "failed", "cancelled", "orphaned"}
POLL_INTERVAL_SECONDS = 10.0

HAVE_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None


class CampaignError(RuntimeError):
    """Raised for an invalid campaign or state transition."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"Expected a JSON object in {path}")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError(f"Cannot read campaign manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError("Campaign manifest must contain a mapping")
    if value.get("schema_version") != 1:
        raise CampaignError("CAMPAIGN.yaml must use schema_version: 1")
    profiles = value.get("profiles")
    if (
        not isinstance(profiles, dict)
        or "development" not in profiles
        or "full" not in profiles
    ):
        raise CampaignError("CAMPAIGN.yaml requires development and full profiles")
    validate_development_tasks(profiles["development"].get("tasks"))
    full = profiles["full"]
    if not isinstance(full, dict) or full.get("scope") != "full":
        raise CampaignError("The full profile must declare scope: full")
    max_finalists = full.get("max_finalists")
    if not isinstance(max_finalists, int) or max_finalists < 1:
        raise CampaignError("The full profile requires a positive max_finalists")
    return value


def validate_development_tasks(tasks: Any) -> None:
    if not isinstance(tasks, list) or not tasks:
        raise CampaignError("The development profile requires a non-empty task list")
    seen: set[str] = set()
    for raw in tasks:
        if not isinstance(raw, dict):
            raise CampaignError("Each campaign task must be a mapping")
        task_id = raw.get("id")
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            raise CampaignError(f"Invalid campaign task id: {task_id!r}")
        if task_id in seen:
            raise CampaignError(f"Duplicate campaign task id: {task_id}")
        kind = raw.get("kind", "attack")
        if kind not in {"attack", "review"}:
            raise CampaignError(f"Unsupported task kind for {task_id}: {kind}")
        if not isinstance(raw.get("required"), bool):
            raise CampaignError(f"Task {task_id} must declare required: true|false")
        if kind == "attack":
            if not isinstance(raw.get("attack"), str):
                raise CampaignError(f"Attack task {task_id} has no attack name")
            if raw.get("scope") not in {"smoke", "development"}:
                raise CampaignError(f"Attack task {task_id} has an invalid scope")
            source = raw.get("source")
            if not isinstance(source, str) or not source:
                raise CampaignError(
                    f"Attack task {task_id} must name the gradient source detector"
                )
        for field in ("needs", "after"):
            dependencies = raw.get(field, [])
            if not isinstance(dependencies, list) or not all(
                isinstance(item, str) for item in dependencies
            ):
                raise CampaignError(f"Task {task_id} has invalid {field}")
            unknown = set(dependencies) - seen
            if unknown:
                names = ", ".join(sorted(unknown))
                raise CampaignError(
                    f"Task {task_id} references non-earlier {field}: {names}"
                )
        seen.add(task_id)


def materialize_tasks(raw_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for raw in raw_tasks:
        task = {
            "id": raw["id"],
            "kind": raw.get("kind", "attack"),
            "required": raw["required"],
            "needs": list(raw.get("needs", [])),
            "after": list(raw.get("after", [])),
            "state": "pending",
            "summary": "",
            "attempts": [],
        }
        if task["kind"] == "attack":
            task["attack"] = raw["attack"]
            task["scope"] = raw["scope"]
            task["source"] = raw["source"]
        tasks.append(task)
    return tasks


def campaign_attacks(manifest: dict[str, Any]) -> set[str]:
    return {
        task["attack"]
        for task in manifest["profiles"]["development"]["tasks"]
        if task.get("kind", "attack") == "attack"
    }


def build_full_tasks(
    finalists: list[str], max_finalists: int, allowed_attacks: set[str] | None = None
) -> list[dict[str, Any]]:
    if not finalists:
        raise CampaignError("No finalists were selected by the development campaign")
    if len(finalists) > max_finalists:
        raise CampaignError(
            f"Selected {len(finalists)} finalists; the maximum is {max_finalists}"
        )
    if len(set(finalists)) != len(finalists):
        raise CampaignError("Finalist attack names must be unique")
    if allowed_attacks is not None:
        unknown = set(finalists) - allowed_attacks
        if unknown:
            raise CampaignError(
                "Unknown finalist attack names: " + ", ".join(sorted(unknown))
            )
    tasks: list[dict[str, Any]] = []
    previous: str | None = None
    for attack in finalists:
        if not isinstance(attack, str) or not TASK_ID_RE.fullmatch(attack):
            raise CampaignError(f"Invalid finalist attack name: {attack!r}")
        task_id = f"{attack}-full"
        tasks.append(
            {
                "id": task_id,
                "kind": "attack",
                "required": True,
                "needs": [],
                "after": [previous] if previous else [],
                "state": "pending",
                "summary": "",
                "attempts": [],
                "attack": attack,
                "scope": "full",
                "source": "vit_b_16",
            }
        )
        previous = task_id
    return tasks


def campaign_id(profile: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{profile}-{uuid.uuid4().hex[:8]}"


def create_state(
    profile: str,
    tasks: list[dict[str, Any]],
    manifest_path: Path,
    finalists: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    identifier = campaign_id(profile)
    run_dir = RUNS_ROOT / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_snapshot = run_dir / "CAMPAIGN.yaml"
    shutil.copy2(manifest_path, manifest_snapshot)
    state = {
        "schema_version": 1,
        "campaign_id": identifier,
        "profile": profile,
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "manifest": str(manifest_snapshot.resolve()),
        "manifest_source": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_snapshot),
        "run_dir": str(run_dir.resolve()),
        "finalists": finalists or [],
        "tasks": tasks,
    }
    state_path = run_dir / "state.json"
    save_state(state_path, state)
    atomic_write_json(
        LATEST_FILE,
        {"campaign_id": identifier, "state_file": str(state_path.resolve())},
    )
    return state, state_path


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)


def load_latest_state() -> tuple[dict[str, Any], Path]:
    if not LATEST_FILE.exists():
        raise CampaignError("No campaign exists yet")
    latest = load_json(LATEST_FILE)
    raw_path = latest.get("state_file")
    if not isinstance(raw_path, str):
        raise CampaignError(f"Invalid latest-campaign pointer: {LATEST_FILE}")
    state_path = Path(raw_path)
    return load_json(state_path), state_path


def latest_completed_development() -> tuple[dict[str, Any], Path]:
    candidates: list[tuple[str, dict[str, Any], Path]] = []
    if RUNS_ROOT.exists():
        for state_path in RUNS_ROOT.glob("*/state.json"):
            try:
                state = load_json(state_path)
            except CampaignError:
                continue
            if (
                state.get("profile") == "development"
                and state.get("status") == "completed"
            ):
                candidates.append((str(state.get("created_at", "")), state, state_path))
    if not candidates:
        raise CampaignError("No completed development campaign is available")
    _, state, state_path = max(candidates, key=lambda item: item[0])
    return state, state_path


def latest_state_for_profile(profile: str) -> tuple[dict[str, Any], Path] | None:
    candidates: list[tuple[str, dict[str, Any], Path]] = []
    if RUNS_ROOT.exists():
        for state_path in RUNS_ROOT.glob("*/state.json"):
            try:
                state = load_json(state_path)
            except CampaignError:
                continue
            if state.get("profile") == profile:
                candidates.append((str(state.get("created_at", "")), state, state_path))
    if not candidates:
        return None
    _, state, state_path = max(candidates, key=lambda item: item[0])
    return state, state_path


def selected_finalists(state: dict[str, Any], selector_task: str) -> list[str]:
    for task in state.get("tasks", []):
        if task.get("id") != selector_task:
            continue
        status_path = task.get("status_file")
        if task.get("state") != "passed" or not isinstance(status_path, str):
            raise CampaignError(f"Finalist selector {selector_task} did not pass")
        status = load_json(Path(status_path))
        finalists = status.get("finalists")
        if not isinstance(finalists, list) or not all(
            isinstance(item, str) for item in finalists
        ):
            raise CampaignError(f"Finalist selector {selector_task} has invalid output")
        return finalists
    raise CampaignError(f"Development campaign has no selector task {selector_task}")


def validate_finalists_development(state: dict[str, Any], finalists: list[str]) -> None:
    passed = {
        task.get("attack")
        for task in state.get("tasks", [])
        if task.get("kind") == "attack"
        and task.get("scope") == "development"
        and task.get("state") == "passed"
    }
    unavailable = set(finalists) - passed
    if unavailable:
        raise CampaignError(
            "Finalists lack passed development runs: " + ", ".join(sorted(unavailable))
        )


def validate_preflight(manifest: dict[str, Any], dry_run: bool) -> None:
    preflight = manifest.get("preflight", {})
    if not isinstance(preflight, dict):
        raise CampaignError("preflight must be a mapping")
    missing: list[str] = []
    for raw_path in preflight.get("directories", []):
        path = Path(raw_path)
        if not path.is_dir():
            missing.append(f"directory {path}")
    for raw_path in preflight.get("files", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            missing.append(f"file {path}")
    if missing:
        message = "Preflight paths are missing: " + ", ".join(missing)
        if dry_run:
            print(f"[dry-run warning] {message}")
        else:
            raise CampaignError(message)


def opencode_binary(manifest: dict[str, Any]) -> str:
    """Resolve the OpenCode binary the campaign manifest declares.

    A non-interactive shell does not read the user's profile, so relying on
    PATH alone makes the driver work from an attached terminal and fail from
    ssh or a service. The manifest already records the absolute path.
    """

    control = manifest.get("control")
    declared = control.get("opencode_binary") if isinstance(control, dict) else None
    if isinstance(declared, str) and declared:
        path = Path(declared)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CampaignError(
                f"control.opencode_binary is not an executable file: {path}"
            )
        return str(path)
    found = shutil.which("opencode")
    if found is None:
        raise CampaignError(
            "opencode is not on PATH and CAMPAIGN.yaml declares no opencode_binary"
        )
    return found


def validate_launch_environment(manifest: dict[str, Any]) -> None:
    opencode_binary(manifest)
    validate_preflight(manifest, dry_run=False)


def resolve_artifact(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def require_artifact_list(status: dict[str, Any], field: str) -> None:
    values = status.get(field)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise CampaignError(f"Passed attack status requires a non-empty {field} list")
    missing = [value for value in values if not resolve_artifact(value).exists()]
    if missing:
        raise CampaignError(f"Status references missing {field}: {', '.join(missing)}")


def structural_attack_status(status_path: Path) -> dict[str, Any]:
    """Validate the schema's required shape without a jsonschema dependency.

    This is a strict subset of schemas/attack-status.schema.json so that the
    controller still runs in a dependency-light environment.
    """

    status = load_json(status_path)
    if status.get("schema_version") != 1:
        raise CampaignError("Attack status must use schema_version 1")
    for field in ("task_id", "attack", "scope", "outcome", "decision", "summary"):
        if field not in status:
            raise CampaignError(f"Attack status is missing required field {field}")
    for field in ("task_id", "attack"):
        value = status.get(field)
        if not isinstance(value, str) or not TASK_ID_RE.fullmatch(value):
            raise CampaignError(f"Attack status has an invalid {field}: {value!r}")
    if status.get("scope") not in {"smoke", "development", "full", "official"}:
        raise CampaignError(f"Invalid attack scope: {status.get('scope')!r}")
    if status.get("outcome") not in ATTACK_OUTCOMES:
        raise CampaignError(f"Invalid attack outcome: {status.get('outcome')!r}")
    summary = status.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise CampaignError("Attack status requires a non-empty summary")
    if status.get("outcome") == "queued":
        for field in ("job_id", "job_spec"):
            value = status.get(field)
            if not isinstance(value, str) or not value:
                raise CampaignError(f"A queued attack status requires {field}")
    return status


def load_attack_status(status_path: Path) -> dict[str, Any]:
    """Validate against the shared JSON Schema when jsonschema is available."""

    if not HAVE_JSONSCHEMA:
        return structural_attack_status(status_path)
    from attacklab.io import ContractError
    from attacklab.status import load_and_validate_status

    try:
        return load_and_validate_status(status_path, "attack")
    except ContractError as exc:
        raise CampaignError(f"Invalid attack status: {exc}") from exc
    except OSError as exc:
        raise CampaignError(f"Cannot read attack status {status_path}: {exc}") from exc


def validate_attack_status(status_path: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Validate a worker-written provisional status against the campaign task."""

    status = load_attack_status(status_path)
    if status.get("task_id") != task["id"]:
        raise CampaignError("Attack status task_id does not match the campaign task")
    if status.get("attack") != task["attack"]:
        raise CampaignError("Attack status name does not match the campaign task")
    if status.get("scope") != task["scope"]:
        raise CampaignError("Attack status scope does not match the campaign task")
    outcome = status.get("outcome")
    if outcome not in WORKER_OUTCOMES:
        raise CampaignError(
            f"A worker may not report the controller-owned outcome {outcome!r}"
        )
    decision = status.get("decision")
    if decision not in DECISIONS:
        raise CampaignError(f"Invalid scientific decision: {decision!r}")
    if outcome == "queued":
        for field in ("job_id", "job_spec"):
            value = status.get(field)
            if not isinstance(value, str) or not value:
                raise CampaignError(f"A queued attack status requires {field}")
    return status


def validate_final_attack_status(
    status_path: Path, task: dict[str, Any]
) -> dict[str, Any]:
    """Re-validate the controller-written status before declaring a task passed."""

    status = load_attack_status(status_path)
    if status.get("task_id") != task["id"]:
        raise CampaignError("Final status task_id does not match the campaign task")
    if status.get("outcome") not in ATTACK_OUTCOMES:
        raise CampaignError(f"Invalid final outcome: {status.get('outcome')!r}")
    if status.get("outcome") == "passed":
        require_artifact_list(status, "configs")
        require_artifact_list(status, "results")
        require_artifact_list(status, "evidence")
        report = status.get("verifier_report")
        if not isinstance(report, str) or not resolve_artifact(report).is_file():
            raise CampaignError("A passed status requires an existing verifier report")
    return status


def validate_review_document(
    status: dict[str, Any],
    task: dict[str, Any],
    max_finalists: int,
    allowed_attacks: set[str],
) -> dict[str, Any]:
    if status.get("schema_version") != 1:
        raise CampaignError("Review status must use schema_version 1")
    if status.get("task") != task["id"]:
        raise CampaignError("Review status task id does not match")
    outcome = status.get("outcome")
    if outcome not in REVIEW_OUTCOMES:
        raise CampaignError(f"Invalid review outcome: {outcome!r}")
    summary = status.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise CampaignError("Review status requires a non-empty summary")
    if outcome == "passed":
        finalists = status.get("finalists")
        if not isinstance(finalists, list) or not finalists:
            raise CampaignError("Passed finalist review requires at least one finalist")
        if len(finalists) > max_finalists or len(set(finalists)) != len(finalists):
            raise CampaignError(
                "Finalist review contains too many or duplicate attacks"
            )
        if not all(
            isinstance(item, str) and TASK_ID_RE.fullmatch(item) for item in finalists
        ):
            raise CampaignError("Finalist review contains an invalid attack name")
        unknown = set(finalists) - allowed_attacks
        if unknown:
            raise CampaignError(
                "Finalist review contains unknown attacks: "
                + ", ".join(sorted(unknown))
            )
        require_artifact_list(status, "evidence")
    return status


def validate_review_status(
    status_path: Path,
    task: dict[str, Any],
    max_finalists: int,
    allowed_attacks: set[str],
) -> dict[str, Any]:
    return validate_review_document(
        load_json(status_path), task, max_finalists, allowed_attacks
    )


def persist_review_status_from_log(
    log_path: Path,
    status_path: Path,
    task: dict[str, Any],
    max_finalists: int,
    allowed_attacks: set[str],
) -> dict[str, Any]:
    """Validate the reviewer's sole JSON response, then persist it atomically.

    The reviewer is intentionally read-only, so it cannot create ``status.json``.
    OpenCode writes the model response to stdout, which the controller already
    captures in ``agent.log``. Only a single standalone JSON object is accepted;
    tool output, prose, multiple candidates, and schema-invalid objects fail
    closed instead of being treated as a scientific decision.
    """

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise CampaignError(f"Cannot read reviewer log {log_path}: {exc}") from exc

    candidates: list[dict[str, Any]] = []
    for raw_line in lines:
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)

    if len(candidates) != 1:
        raise CampaignError(
            "Reviewer must return exactly one standalone JSON object; "
            f"found {len(candidates)} in {log_path}"
        )

    status = validate_review_document(
        candidates[0], task, max_finalists, allowed_attacks
    )
    atomic_write_json(status_path, status)
    return status


def gpu_queue_state_dir(manifest: dict[str, Any]) -> Path:
    control = manifest.get("control")
    raw = ".gpuq"
    if isinstance(control, dict):
        candidate = control.get("gpu_queue_state", ".gpuq")
        if not isinstance(candidate, str) or not candidate:
            raise CampaignError("control.gpu_queue_state must be a non-empty string")
        raw = candidate
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def open_queue(manifest: dict[str, Any]) -> QueueDatabase:
    try:
        return QueueDatabase(gpu_queue_state_dir(manifest), ROOT)
    except (GpuQueueError, OSError) as exc:
        raise CampaignError(f"Cannot open the GPU queue: {exc}") from exc


def gpuq_attempt_dir(database: QueueDatabase, record: dict[str, Any]) -> Path | None:
    """Locate the newest attempt directory the scheduler produced for a job."""

    run_dir = Path(database.project_root) / str(record["run_dir"])
    try:
        attempt = int(record.get("attempt_count") or 0)
    except (TypeError, ValueError):
        attempt = 0
    preferred = run_dir / f"attempt-{attempt:04d}"
    if preferred.is_dir():
        return preferred
    if not run_dir.is_dir():
        return None
    candidates = sorted(item for item in run_dir.glob("attempt-*") if item.is_dir())
    return candidates[-1] if candidates else None


def read_verifier_report(attempt_dir: Path | None) -> dict[str, Any] | None:
    if attempt_dir is None:
        return None
    report_path = attempt_dir / "verification.json"
    if not report_path.is_file():
        return None
    try:
        return load_json(report_path)
    except CampaignError:
        return None


def map_terminal(
    record: dict[str, Any], verifier: dict[str, Any] | None
) -> tuple[str, str] | None:
    """Map a terminal queue state to a task state, or None while still active.

    A job only becomes `passed` through the deterministic verifier report; a
    scheduler exit status alone is never sufficient scientific evidence.
    """

    state = str(record.get("state"))
    if state not in GPUQ_TERMINAL:
        return None
    error = record.get("error")
    if state == "cancelled":
        return "cancelled", str(error or "GPU job was cancelled")
    if state in {"failed", "orphaned"}:
        return "failed", str(error or f"GPU job ended as {state}")
    if verifier is None:
        return "failed", "GPU job succeeded but verification.json is missing"
    if verifier.get("outcome") == "passed":
        return "passed", "GPU job succeeded and the deterministic verifier accepted it"
    errors = verifier.get("errors")
    detail = "; ".join(str(item) for item in errors) if isinstance(errors, list) else ""
    return "failed", f"Verifier rejected the run: {detail or 'no reason recorded'}"


def poll_until_terminal(
    database: QueueDatabase, job_id: str, deadline: float | None
) -> dict[str, Any] | None:
    """Read the queue until the job is terminal. Returns None on poll timeout.

    This only ever reads. The scheduler owns a running job, so an interrupt
    propagates without cancelling anything.
    """

    while True:
        try:
            record = database.get(job_id)
        except GpuQueueError as exc:
            raise CampaignError(f"GPU job {job_id} is unknown to the queue: {exc}") from exc
        if str(record.get("state")) in GPUQ_TERMINAL:
            return record
        if deadline is not None and time.monotonic() >= deadline:
            return None
        time.sleep(POLL_INTERVAL_SECONDS)


def finalize_status(
    task: dict[str, Any],
    attempt: dict[str, Any],
    status_path: Path,
    provisional: dict[str, Any],
    task_state: str,
    summary: str,
    record: dict[str, Any],
    attempt_dir: Path | None,
) -> dict[str, Any]:
    """Persist the worker's provisional status and write the controller's own."""

    worker_copy = status_path.with_name("status.worker.json")
    if not worker_copy.exists():
        atomic_write_json(worker_copy, provisional)
    attempt["worker_status_file"] = str(worker_copy.resolve())

    final: dict[str, Any] = {
        "schema_version": 1,
        "task_id": task["id"],
        "attack": task["attack"],
        "scope": task["scope"],
        "outcome": task_state,
        "decision": provisional.get("decision", "pending"),
        "summary": summary,
        "job_id": str(record["id"]),
        "attempt": max(1, int(record.get("attempt_count") or 1)),
        "updated_at": utc_now(),
    }
    job_spec = provisional.get("job_spec")
    if isinstance(job_spec, str) and job_spec:
        final["job_spec"] = job_spec
    created_at = provisional.get("created_at")
    if isinstance(created_at, str) and created_at:
        final["created_at"] = created_at
    if task_state == "passed" and attempt_dir is not None:
        relative = Path(str(record["run_dir"])) / attempt_dir.name
        final["configs"] = [(relative / "resolved_config.yaml").as_posix()]
        final["results"] = [(relative / "summary.json").as_posix()]
        final["evidence"] = [(relative / "norm_audit.json").as_posix()]
        final["verifier_report"] = (relative / "verification.json").as_posix()
    atomic_write_json(status_path, final)
    return final


def reconcile_task(
    database: QueueDatabase,
    state: dict[str, Any],
    state_path: Path,
    task: dict[str, Any],
) -> bool:
    """Idempotently advance one queued task. Returns True when it is terminal."""

    if task["state"] in TERMINAL_STATES:
        return True
    job_id = task.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return False
    try:
        record = database.get(job_id)
    except GpuQueueError:
        task["state"] = "failed"
        task["summary"] = f"GPU job {job_id} is unknown to the queue"
        save_state(state_path, state)
        return True

    attempt_dir = gpuq_attempt_dir(database, record)
    mapped = map_terminal(record, read_verifier_report(attempt_dir))
    if mapped is None:
        return False
    task_state, summary = mapped

    if not task["attempts"]:
        raise CampaignError(f"Task {task['id']} has a job but no recorded attempt")
    attempt = task["attempts"][-1]
    status_path = Path(str(attempt["status_file"]))
    try:
        provisional = load_json(status_path)
    except CampaignError:
        provisional = {"decision": "pending"}

    final = finalize_status(
        task, attempt, status_path, provisional, task_state, summary, record, attempt_dir
    )
    if task_state == "passed":
        try:
            validate_final_attack_status(status_path, task)
        except CampaignError as exc:
            task_state = "failed"
            summary = f"Controller could not confirm a passed status: {exc}"
            final = finalize_status(
                task, attempt, status_path, provisional, task_state, summary, record, None
            )

    task["state"] = task_state
    task["summary"] = summary
    task["status_file"] = str(status_path.resolve())
    task["decision"] = final["decision"]
    attempt["gpuq_state"] = record.get("state")
    attempt["reconciled_at"] = utc_now()
    save_state(state_path, state)
    return True


def reconcile_running_tasks(
    database: QueueDatabase, state: dict[str, Any], state_path: Path
) -> None:
    """Recover every task left mid-flight by an interrupted controller."""

    for task in state.get("tasks", []):
        if task.get("kind", "attack") != "attack":
            continue
        if task.get("state") in TERMINAL_STATES or not task.get("job_id"):
            continue
        reconcile_task(database, state, state_path, task)


def task_command(
    task: dict[str, Any],
    status_path: Path,
    run_dir: Path,
    campaign_name: str,
    model: str | None,
    variant: str | None,
    auto: bool,
    binary: str = "opencode",
) -> list[str]:
    command_name = "attack" if task["kind"] == "attack" else "campaign-review"
    command = [
        binary,
        "run",
        "--command",
        command_name,
        "--dir",
        str(ROOT),
        "--title",
        f"{campaign_name}:{task['id']}",
    ]
    if model:
        command.extend(["--model", model])
    if variant:
        command.extend(["--variant", variant])
    if auto:
        command.append("--auto")
    if task["kind"] == "attack":
        # The task id is passed explicitly so the worker's status contract does
        # not have to rely on the <attack>-<scope> naming convention.
        command.extend(
            [
                task["attack"],
                task["scope"],
                str(status_path),
                task["id"],
                task["source"],
            ]
        )
    else:
        command.extend([str(status_path), str(run_dir)])
    return command


def dependency_state(tasks_by_id: dict[str, dict[str, Any]], task_id: str) -> str:
    try:
        return str(tasks_by_id[task_id]["state"])
    except KeyError as exc:
        raise CampaignError(f"State references unknown dependency {task_id}") from exc


def run_process(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=None,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def await_queued_task(
    database: QueueDatabase,
    state: dict[str, Any],
    state_path: Path,
    task: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    """Wait for one queued job and reconcile it. False means the campaign stops."""

    job_id = str(task["job_id"])
    timeout = getattr(args, "poll_timeout_seconds", None)
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    try:
        record = poll_until_terminal(database, job_id, deadline)
    except KeyboardInterrupt:
        save_state(state_path, state)
        print(
            f"\nInterrupted while polling {task['id']}. GPU job {job_id} is still "
            "owned by the scheduler; run resume to reconcile it."
        )
        raise
    if record is None:
        task["state"] = "failed"
        task["summary"] = (
            f"Controller poll timed out after {timeout}s; GPU job {job_id} is "
            "still owned by the scheduler"
        )
        save_state(state_path, state)
    else:
        reconcile_task(database, state, state_path, task)

    if task["state"] != "passed":
        print(f"Task {task['id']} ended as {task['state']}: {task['summary']}")
        if task["required"]:
            state["status"] = task["state"]
            save_state(state_path, state)
            return False
    return True


def run_campaign(
    state: dict[str, Any],
    state_path: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    validate_launch_environment(manifest)
    binary = opencode_binary(manifest)
    database = open_queue(manifest)
    tasks = state["tasks"]
    tasks_by_id = {task["id"]: task for task in tasks}
    max_finalists = int(manifest["profiles"]["full"]["max_finalists"])
    allowed_attacks = campaign_attacks(manifest)
    state["status"] = "running"
    save_state(state_path, state)

    # Recover anything a previously interrupted controller left mid-flight
    # before deciding what still needs an agent session.
    reconcile_running_tasks(database, state, state_path)
    if args.auto:
        print(
            "WARNING: child OpenCode sessions use --auto; run only in an isolated "
            "environment without sensitive credentials."
        )

    for index, task in enumerate(tasks, start=1):
        current = task["state"]
        if current in {"passed", "skipped"}:
            continue
        if current in ABORT_STATES:
            if task["required"]:
                state["status"] = current
                save_state(state_path, state)
                print(
                    f"Required task {task['id']} is {current}; "
                    f"resume with --retry {task['id']}."
                )
                return 2
            continue
        if current == "running":
            if task.get("job_id"):
                # The queue job outlived the controller and reconciliation above
                # found it still active. Keep waiting instead of re-running the
                # agent and submitting a second experiment.
                print(
                    f"\n[{index}/{len(tasks)}] Resuming {task['id']} "
                    f"on GPU job {task['job_id']}"
                )
                if not await_queued_task(database, state, state_path, task, args):
                    return 2
                continue
            task["state"] = "pending"

        incomplete_after = [
            dependency
            for dependency in task["after"]
            if dependency_state(tasks_by_id, dependency) not in TERMINAL_STATES
        ]
        if incomplete_after:
            raise CampaignError(
                f"Task {task['id']} reached before after-dependencies completed: "
                + ", ".join(incomplete_after)
            )

        unmet_needs = [
            dependency
            for dependency in task["needs"]
            if dependency_state(tasks_by_id, dependency) != "passed"
        ]
        if unmet_needs:
            task["state"] = "skipped"
            task["summary"] = "Unmet successful dependencies: " + ", ".join(unmet_needs)
            save_state(state_path, state)
            if task["required"]:
                state["status"] = "failed"
                save_state(state_path, state)
                print(f"Required task {task['id']} was skipped: {task['summary']}")
                return 2
            print(f"Skipping optional task {task['id']}: {task['summary']}")
            continue

        attempt_number = len(task["attempts"]) + 1
        attempt_dir = Path(state["run_dir"]) / (
            f"{index:02d}-{task['id']}-attempt-{attempt_number}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        status_path = attempt_dir / "status.json"
        log_path = attempt_dir / "agent.log"
        command = task_command(
            task,
            status_path,
            Path(state["run_dir"]),
            state["campaign_id"],
            args.model,
            args.variant,
            args.auto,
            binary,
        )
        atomic_write_json(
            attempt_dir / "command.json",
            {"argv": command, "started_at": utc_now()},
        )
        task["state"] = "running"
        task["attempts"].append(
            {
                "attempt": attempt_number,
                "directory": str(attempt_dir.resolve()),
                "status_file": str(status_path.resolve()),
                "log": str(log_path.resolve()),
                "started_at": utc_now(),
            }
        )
        save_state(state_path, state)

        print(f"\n[{index}/{len(tasks)}] Running {task['id']}")
        print("Command:", " ".join(command))
        process_error: OSError | None = None
        try:
            return_code = run_process(command, log_path)
        except OSError as exc:
            process_error = exc
            return_code = 127
        attempt = task["attempts"][-1]
        attempt["finished_at"] = utc_now()
        attempt["return_code"] = return_code

        if process_error is not None:
            task["state"] = "failed"
            task["summary"] = f"Could not start OpenCode: {process_error}"
        elif return_code != 0:
            task["state"] = "failed"
            task["summary"] = f"OpenCode exited with status {return_code}"
        elif task["kind"] == "attack" and not status_path.is_file():
            task["state"] = "failed"
            task["summary"] = "OpenCode exited without writing the required status JSON"
        else:
            queued = False
            try:
                if task["kind"] == "attack":
                    status = validate_attack_status(status_path, task)
                elif status_path.is_file():
                    status = validate_review_status(
                        status_path, task, max_finalists, allowed_attacks
                    )
                else:
                    status = persist_review_status_from_log(
                        log_path,
                        status_path,
                        task,
                        max_finalists,
                        allowed_attacks,
                    )
            except CampaignError as exc:
                task["state"] = "failed"
                task["summary"] = f"Invalid status JSON: {exc}"
            else:
                task["summary"] = status["summary"]
                task["status_file"] = str(status_path.resolve())
                if "decision" in status:
                    task["decision"] = status["decision"]
                if "finalists" in status:
                    task["finalists"] = status["finalists"]
                if task["kind"] == "attack" and status["outcome"] == "queued":
                    queued = True
                    task["job_id"] = status["job_id"]
                    task["job_spec"] = status["job_spec"]
                    # The controller, not the worker, owns "running".
                    task["state"] = "running"
                    attempt = task["attempts"][-1]
                    attempt["job_id"] = status["job_id"]
                    attempt["job_spec"] = status["job_spec"]
                else:
                    task["state"] = status["outcome"]

            if queued:
                save_state(state_path, state)
                print(
                    f"Task {task['id']} queued as GPU job {task['job_id']}; "
                    "waiting for the scheduler."
                )
                if not await_queued_task(database, state, state_path, task, args):
                    return 2
                continue

        save_state(state_path, state)
        if task["state"] != "passed":
            print(f"Task {task['id']} ended as {task['state']}: {task['summary']}")
            if task["required"]:
                state["status"] = task["state"]
                save_state(state_path, state)
                return 2

    state["status"] = "completed"
    save_state(state_path, state)
    print(f"\nCampaign completed: {state['campaign_id']}")
    print(f"State: {state_path}")
    return 0


def reset_for_retry(state: dict[str, Any], task_id: str) -> None:
    ids = [task["id"] for task in state["tasks"]]
    if task_id not in ids:
        raise CampaignError(f"Unknown retry task: {task_id}")
    start = ids.index(task_id)
    for task in state["tasks"][start:]:
        task["state"] = "pending"
        task["summary"] = ""
        # Drop the queue binding so a retried task is never reconciled against
        # its previous job. The old job stays with the scheduler; the controller
        # never cancels GPU work.
        for field in ("status_file", "decision", "finalists", "job_id", "job_spec"):
            task.pop(field, None)
    state["status"] = "running"


def print_status(state: dict[str, Any], state_path: Path) -> None:
    print(f"Campaign: {state['campaign_id']}")
    print(f"Profile:  {state['profile']}")
    print(f"Status:   {state['status']}")
    print(f"State:    {state_path}")
    print()
    for task in state["tasks"]:
        marker = "required" if task["required"] else "optional"
        decision = f", {task['decision']}" if task.get("decision") else ""
        print(f"{task['id']:<32} {task['state']:<8} ({marker}{decision})")


def print_dry_run(
    profile: str,
    tasks: list[dict[str, Any]],
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    validate_preflight(manifest, dry_run=True)
    fake_run_dir = RUNTIME_ROOT / "runs" / f"DRY-RUN-{profile}"
    print(f"Profile: {profile}")
    for index, task in enumerate(tasks, start=1):
        command = task_command(
            task,
            fake_run_dir / task["id"] / "status.json",
            fake_run_dir,
            f"dry-run-{profile}",
            args.model,
            args.variant,
            args.auto,
        )
        print(f"{index:02d}. {task['id']}: {' '.join(command)}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run or resume the isolated OpenCode attack campaign."
    )
    result.add_argument("action", choices=("development", "full", "resume", "status"))
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument(
        "--model", help="OpenCode provider/model override for child sessions"
    )
    result.add_argument(
        "--variant", help="OpenCode reasoning variant for child sessions"
    )
    result.add_argument(
        "--auto",
        action="store_true",
        help="Pass OpenCode --auto to child sessions; use only in an isolated environment",
    )
    result.add_argument(
        "--poll-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Give up waiting for a queued GPU job after this many seconds. "
            "The job itself is never cancelled; the default waits indefinitely."
        ),
    )
    result.add_argument(
        "--dry-run", action="store_true", help="Print tasks without writing state"
    )
    result.add_argument(
        "--new", action="store_true", help="Start a new development campaign"
    )
    result.add_argument(
        "--retry", metavar="TASK_ID", help="Reset this task and later tasks"
    )
    result.add_argument(
        "--confirm-full",
        action="store_true",
        help="Required for direct full launches",
    )
    result.add_argument(
        "--finalists",
        nargs="+",
        help="Override automatically selected finalists (maximum from CAMPAIGN.yaml)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)

        if args.action == "status":
            state, state_path = load_latest_state()
            state_manifest = state.get("manifest")
            if isinstance(state_manifest, str) and Path(state_manifest).is_file():
                try:
                    database = open_queue(load_manifest(Path(state_manifest)))
                except CampaignError as exc:
                    print(f"warning: queue reconciliation skipped: {exc}", file=sys.stderr)
                else:
                    reconcile_running_tasks(database, state, state_path)
            print_status(state, state_path)
            return 0

        if args.action == "resume":
            state, state_path = load_latest_state()
            state_manifest = state.get("manifest")
            if not isinstance(state_manifest, str):
                raise CampaignError("Campaign state has no frozen manifest snapshot")
            manifest = load_manifest(Path(state_manifest))
            if args.retry:
                if args.dry_run:
                    state = copy.deepcopy(state)
                    reset_for_retry(state, args.retry)
                else:
                    reset_for_retry(state, args.retry)
                    save_state(state_path, state)
            if args.dry_run:
                print_dry_run(state["profile"], state["tasks"], manifest, args)
                return 0
            return run_campaign(state, state_path, manifest, args)

        if args.action == "development":
            tasks = materialize_tasks(manifest["profiles"]["development"]["tasks"])
            if args.dry_run:
                print_dry_run("development", tasks, manifest, args)
                return 0
            latest_development = latest_state_for_profile("development")
            if latest_development is not None and not args.new:
                _, latest_path = latest_development
                raise CampaignError(
                    f"A development campaign already exists at {latest_path}; "
                    "use resume or pass --new explicitly"
                )
            validate_launch_environment(manifest)
            state, state_path = create_state("development", tasks, manifest_path)
            return run_campaign(state, state_path, manifest, args)

        if not args.confirm_full:
            raise CampaignError(
                "Full runs require --confirm-full; /campaign full supplies this authorization"
            )
        full_profile = manifest["profiles"]["full"]
        development: dict[str, Any] | None = None
        if args.finalists and args.dry_run:
            finalists = args.finalists
        else:
            development, _ = latest_completed_development()
            development_manifest = development.get("manifest")
            if not isinstance(development_manifest, str):
                raise CampaignError("Development state has no frozen manifest snapshot")
            manifest_path = Path(development_manifest)
            manifest = load_manifest(manifest_path)
            full_profile = manifest["profiles"]["full"]
            if args.finalists:
                finalists = args.finalists
            else:
                finalists = selected_finalists(
                    development, str(full_profile["selector_task"])
                )
        if development is not None:
            validate_finalists_development(development, finalists)
        tasks = build_full_tasks(
            finalists,
            int(full_profile["max_finalists"]),
            allowed_attacks=campaign_attacks(manifest),
        )
        if args.dry_run:
            print_dry_run("full", tasks, manifest, args)
            return 0
        validate_launch_environment(manifest)
        state, state_path = create_state(
            "full", tasks, manifest_path, finalists=finalists
        )
        return run_campaign(state, state_path, manifest, args)
    except CampaignError as exc:
        print(f"campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
