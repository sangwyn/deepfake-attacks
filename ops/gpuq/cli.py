"""Command-line interface for the conservative GPU queue."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .db import QueueDatabase
from .errors import GpuQueueError, InventoryError
from .inventory import NvidiaSmiInventory
from .locks import scheduler_lock
from .models import JOB_STATES, JobSpec
from .scheduler import CycleReport, GpuScheduler, SchedulerPolicy


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _load_spec(path: Path, project_root: Path) -> JobSpec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuQueueError(f"Cannot read job specification {path}: {exc}") from exc
    return JobSpec.from_mapping(raw, project_root)


def _store(args: argparse.Namespace) -> QueueDatabase:
    project_root = Path(args.project_root).resolve(strict=True)
    state_dir = (
        Path(args.state_dir).resolve(strict=False)
        if args.state_dir
        else project_root / ".gpuq"
    )
    return QueueDatabase(state_dir, project_root)


def _compact(record: dict[str, Any]) -> str:
    gpu = record["assigned_gpu_uuid"] or "-"
    return (
        f"{record['id']}  {record['state']:<11}  gpu={gpu:<20} "
        f"attempt={record['attempt_count']}/{record['max_attempts']} "
        f"run={record['run_dir']}"
    )


def _doctor(database: QueueDatabase, nvidia_smi: str) -> tuple[dict[str, Any], bool]:
    checks: dict[str, Any] = {
        "schema_version": 1,
        "project_root": str(database.project_root),
        "state_dir": str(database.state_dir),
        "database": {"ok": database.path.is_file(), "path": str(database.path)},
    }
    runner_candidates = (
        database.project_root / "src" / "attacklab" / "cli.py",
        database.project_root / "attacklab" / "cli.py",
    )
    runner_path = next((path for path in runner_candidates if path.is_file()), None)
    checks["fixed_runner"] = {
        "ok": runner_path is not None,
        "module": "attacklab.cli",
        "path": str(runner_path) if runner_path else None,
    }
    binary = shutil.which(nvidia_smi) if "/" not in nvidia_smi else nvidia_smi
    checks["nvidia_smi"] = {"ok": bool(binary), "path": binary}
    inventory_ok = False
    if binary:
        try:
            inventory = NvidiaSmiInventory(binary).snapshot()
            checks["inventory"] = {
                "ok": True,
                "gpus": [
                    {
                        "uuid": gpu.uuid,
                        "index": gpu.index,
                        "free_memory_mb": gpu.free_memory_mb,
                        "utilization_percent": gpu.utilization_percent,
                        "compute_pids": list(gpu.compute_pids),
                    }
                    for gpu in inventory
                ],
            }
            inventory_ok = True
        except InventoryError as exc:
            checks["inventory"] = {"ok": False, "error": str(exc)}
    else:
        checks["inventory"] = {"ok": False, "error": "nvidia-smi not found"}

    lock = scheduler_lock(database.locks_dir)
    available = lock.acquire(blocking=False)
    if available:
        lock.release()
    checks["scheduler_lock"] = {
        "ok": True,
        "available": available,
        "note": "busy is healthy when one scheduler daemon is already running",
    }
    ok = bool(
        checks["database"]["ok"]
        and checks["fixed_runner"]["ok"]
        and checks["nvidia_smi"]["ok"]
        and inventory_ok
    )
    checks["ok"] = ok
    return checks, ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ops.gpuq",
        description="Durable GPU queue; jobs never supply shell commands.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="canonical attack-pipeline root (default: current directory)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="runtime DB/lock/log directory (default: PROJECT_ROOT/.gpuq)",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    submit = subparsers.add_parser("submit", help="validate and enqueue a JSON job spec")
    submit.add_argument("job_spec", help="path to schema-v1 JSON job specification")

    listing = subparsers.add_parser("list", help="list jobs")
    listing.add_argument("--state", choices=sorted(JOB_STATES))
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status", help="show one job and its state history")
    status.add_argument("job_id")
    status.add_argument("--json", action="store_true")

    cancel = subparsers.add_parser("cancel", help="cancel queued work or request owned-process cancellation")
    cancel.add_argument("job_id")

    doctor = subparsers.add_parser("doctor", help="read-only queue/GPU readiness checks")
    doctor.add_argument("--nvidia-smi", default="nvidia-smi")
    doctor.add_argument("--json", action="store_true")

    scheduler = subparsers.add_parser(
        "run-scheduler",
        help="preview scheduling by default; --execute is required to launch jobs",
    )
    scheduler.add_argument("--execute", action="store_true")
    scheduler.add_argument("--once", action="store_true")
    scheduler.add_argument("--max-running", type=int, default=1)
    scheduler.add_argument("--poll-seconds", type=float, default=10.0)
    scheduler.add_argument("--idle-samples", type=int, default=3)
    scheduler.add_argument("--headroom-mb", type=int, default=4096)
    scheduler.add_argument("--max-idle-utilization", type=int, default=5)
    scheduler.add_argument("--retry-delay-seconds", type=float, default=30.0)
    scheduler.add_argument("--validation-timeout-seconds", type=float, default=600.0)
    scheduler.add_argument("--nvidia-smi", default="nvidia-smi")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        database = _store(args)
        if args.action == "submit":
            spec = _load_spec(Path(args.job_spec), database.project_root)
            record, created = database.submit(spec)
            _json(
                {
                    "schema_version": 1,
                    "created": created,
                    "job_id": record["id"],
                    "idempotency_key": record["idempotency_key"],
                    "state": record["state"],
                    "job_spec_snapshot": str(
                        database.project_root / record["run_dir"] / "job_spec.json"
                    ),
                }
            )
            return 0

        if args.action == "list":
            records = database.list_jobs(state=args.state, limit=args.limit)
            if args.json:
                _json(records)
            elif records:
                print("\n".join(_compact(record) for record in records))
            else:
                print("No jobs.")
            return 0

        if args.action == "status":
            record = database.get(args.job_id)
            record["events"] = database.events(args.job_id)
            if args.json:
                _json(record)
            else:
                print(_compact(record))
                if record["error"]:
                    print(f"error: {record['error']}")
                print(f"events: {len(record['events'])}")
            return 0

        if args.action == "cancel":
            record = database.request_cancel(args.job_id)
            _json(
                {
                    "job_id": record["id"],
                    "state": record["state"],
                    "cancel_requested": record["cancel_requested"],
                }
            )
            return 0

        if args.action == "doctor":
            result, ok = _doctor(database, args.nvidia_smi)
            if args.json:
                _json(result)
            else:
                for name, value in result.items():
                    if name not in {"schema_version", "ok"}:
                        print(f"{name}: {value}")
                print("ready" if ok else "not ready")
            return 0 if ok else 2

        if args.action == "run-scheduler":
            policy = SchedulerPolicy(
                max_running=args.max_running,
                poll_seconds=args.poll_seconds,
                idle_samples=args.idle_samples,
                headroom_mb=args.headroom_mb,
                max_idle_utilization_percent=args.max_idle_utilization,
                retry_delay_seconds=args.retry_delay_seconds,
                validation_timeout_seconds=args.validation_timeout_seconds,
            )
            scheduler = GpuScheduler(
                database,
                inventory=NvidiaSmiInventory(args.nvidia_smi),
                policy=policy,
                execute=args.execute,
            )
            if not args.execute:
                print(
                    "[gpuq] safe preview only; pass --execute to launch fixed commands",
                    file=sys.stderr,
                )

            def report(cycle: CycleReport) -> None:
                print(json.dumps(cycle.as_dict(), sort_keys=True), flush=True)

            scheduler.serve(once=args.once, report=report)
            return 0

        parser.error(f"Unsupported action: {args.action}")
    except (GpuQueueError, OSError, ValueError) as exc:
        print(f"gpuq: {exc}", file=sys.stderr)
        return 2
    return 2
