"""SQLite-backed durable queue with audited state transitions."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .errors import QueueStateError, SpecError
from .models import ACTIVE_STATES, JOB_STATES, READY_STATES, JobSpec


DATABASE_SCHEMA_VERSION = "1"


def _utc(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


class QueueDatabase:
    """Durable job store pinned to exactly one project root."""

    def __init__(self, state_dir: Path, project_root: Path):
        self.project_root = project_root.resolve(strict=True)
        self.state_dir = state_dir.resolve(strict=False)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.state_dir, 0o700)
        except OSError:
            pass
        self.logs_dir = self.state_dir / "logs"
        self.locks_dir = self.state_dir / "locks"
        self.logs_dir.mkdir(mode=0o700, exist_ok=True)
        self.locks_dir.mkdir(mode=0o700, exist_ok=True)
        self.path = self.state_dir / "queue.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    spec_json TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    run_dir TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (
                        state IN (
                            'queued', 'reserving', 'running', 'validating',
                            'succeeded', 'failed', 'retry_wait', 'cancelled',
                            'orphaned'
                        )
                    ),
                    priority INTEGER NOT NULL,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL,
                    available_ts REAL NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    assigned_gpu_uuid TEXT,
                    lease_owner TEXT,
                    pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (
                        cancel_requested IN (0, 1)
                    ),
                    exit_code INTEGER,
                    error TEXT,
                    log_path TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id),
                    event_ts REAL NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    message TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS jobs_ready
                    ON jobs(state, available_ts, priority, created_ts);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_job_per_gpu
                    ON jobs(assigned_gpu_uuid)
                    WHERE state IN ('reserving', 'running', 'validating');

                CREATE TRIGGER IF NOT EXISTS immutable_job_spec
                BEFORE UPDATE OF
                    idempotency_key, spec_json, spec_sha256, task_kind,
                    run_dir, priority, max_attempts
                ON jobs
                BEGIN
                    SELECT RAISE(ABORT, 'job specification is immutable');
                END;
                """
            )
            # Added after the first deployments, so migrate in place. SQLite has
            # no IF NOT EXISTS for ADD COLUMN; an existing column simply raises.
            for column in ("run_seconds", "validate_seconds"):
                try:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} REAL")
                except sqlite3.OperationalError:
                    pass
            metadata = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            expected_root = str(self.project_root)
            if not metadata:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (
                        ("schema_version", DATABASE_SCHEMA_VERSION),
                        ("project_root", expected_root),
                    ),
                )
            else:
                if metadata.get("schema_version") != DATABASE_SCHEMA_VERSION:
                    raise QueueStateError("Unsupported GPU queue database schema")
                if metadata.get("project_root") != expected_root:
                    raise QueueStateError(
                        "GPU queue is pinned to a different project root: "
                        + str(metadata.get("project_root"))
                    )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["spec"] = json.loads(record.pop("spec_json"))
        record["cancel_requested"] = bool(record["cancel_requested"])
        record["created_at"] = _utc(record.pop("created_ts"))
        record["updated_at"] = _utc(record.pop("updated_ts"))
        record["available_at"] = _utc(record.pop("available_ts"))
        return record

    def _get_row(self, connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise QueueStateError(f"Unknown GPU queue job: {job_id}")
        return row

    def _sync_tracking(self, record: Mapping[str, Any]) -> None:
        run_dir = self.project_root / record["run_dir"]
        status = {
            "schema_version": 1,
            "job_id": record["id"],
            "idempotency_key": record["idempotency_key"],
            "task_kind": record["task_kind"],
            "state": record["state"],
            "attempt_count": record["attempt_count"],
            "max_attempts": record["max_attempts"],
            "assigned_gpu_uuid": record["assigned_gpu_uuid"],
            "cancel_requested": record["cancel_requested"],
            "exit_code": record["exit_code"],
            "error": record["error"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "run_seconds": record.get("run_seconds"),
            "validate_seconds": record.get("validate_seconds"),
        }
        _atomic_json(run_dir / "gpuq_status.json", status)

    def submit(self, spec: JobSpec) -> tuple[dict[str, Any], bool]:
        """Submit once by content hash and snapshot the normalized spec in Git space."""

        identifier = f"gpuq-{uuid.uuid4().hex}"
        now = time.time()
        created_run_dir: Path | None = None
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?",
                (spec.idempotency_key,),
            ).fetchone()
            if existing is not None:
                return self._record(existing), False
            conflict = connection.execute(
                "SELECT id FROM jobs WHERE run_dir = ?", (spec.run_dir,)
            ).fetchone()
            if conflict is not None:
                raise SpecError(
                    f"run_dir is already owned by queue job {conflict['id']}"
                )

            run_dir = spec.run_absolute(self.project_root)
            if run_dir.exists():
                raise SpecError(f"run_dir already exists and will not be overwritten: {spec.run_dir}")
            run_dir.mkdir(parents=True, exist_ok=False)
            created_run_dir = run_dir
            try:
                _atomic_json(run_dir / "job_spec.json", spec.as_dict())
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, idempotency_key, spec_json, spec_sha256,
                        task_kind, run_dir, state, priority, created_ts,
                        updated_ts, available_ts, max_attempts
                    ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        spec.idempotency_key,
                        spec.canonical,
                        spec.idempotency_key,
                        spec.task_kind,
                        spec.run_dir,
                        spec.priority,
                        now,
                        now,
                        now,
                        spec.max_attempts,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO events(job_id, event_ts, from_state, to_state, message)
                    VALUES (?, ?, NULL, 'queued', 'immutable job submitted')
                    """,
                    (identifier, now),
                )
                row = self._get_row(connection, identifier)
            except BaseException:
                (run_dir / "job_spec.json").unlink(missing_ok=True)
                try:
                    run_dir.rmdir()
                except OSError:
                    pass
                raise
        record = self._record(row)
        self._sync_tracking(record)
        return record, True

    def get(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            return self._record(self._get_row(connection, job_id))

    def get_spec(self, job_id: str) -> JobSpec:
        return JobSpec.from_persisted(self.get(job_id)["spec"])

    def list_jobs(
        self, *, state: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if state is not None and state not in JOB_STATES:
            raise QueueStateError(f"Unknown job state: {state}")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise QueueStateError("limit must be between 1 and 10000")
        query = "SELECT * FROM jobs"
        parameters: tuple[Any, ...] = ()
        if state:
            query += " WHERE state = ?"
            parameters = (state,)
        query += " ORDER BY created_ts DESC LIMIT ?"
        parameters += (limit,)
        with self._connect() as connection:
            return [self._record(row) for row in connection.execute(query, parameters)]

    def ready_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state IN ('queued', 'retry_wait')
                  AND available_ts <= ?
                  AND cancel_requested = 0
                ORDER BY priority DESC, created_ts ASC
                LIMIT ?
                """,
                (now, limit),
            )
            return [self._record(row) for row in rows]

    def _transition(
        self,
        job_id: str,
        allowed_from: Sequence[str],
        to_state: str,
        message: str,
        **updates: Any,
    ) -> dict[str, Any]:
        if to_state not in JOB_STATES:
            raise QueueStateError(f"Invalid destination state: {to_state}")
        allowed_columns = {
            "available_ts",
            "attempt_count",
            "assigned_gpu_uuid",
            "lease_owner",
            "pid",
            "cancel_requested",
            "exit_code",
            "error",
            "log_path",
        }
        if set(updates) - allowed_columns:
            raise QueueStateError("Internal transition attempted unsupported updates")
        now = time.time()
        with self._transaction() as connection:
            row = self._get_row(connection, job_id)
            if row["state"] not in allowed_from:
                raise QueueStateError(
                    f"Cannot transition {job_id} from {row['state']} to {to_state}"
                )
            assignments = ["state = ?", "updated_ts = ?"]
            values: list[Any] = [to_state, now]
            for column, value in updates.items():
                assignments.append(f"{column} = ?")
                values.append(value)
            values.append(job_id)
            connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", values
            )
            connection.execute(
                """
                INSERT INTO events(job_id, event_ts, from_state, to_state, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, now, row["state"], to_state, message),
            )
            updated = self._get_row(connection, job_id)
        record = self._record(updated)
        self._sync_tracking(record)
        return record

    def claim(self, job_id: str, gpu_uuid: str, owner: str) -> dict[str, Any]:
        if not gpu_uuid.startswith("GPU-"):
            raise QueueStateError("Only full GPU UUIDs are accepted for scheduling")
        now = time.time()
        with self._transaction() as connection:
            row = self._get_row(connection, job_id)
            if row["state"] not in READY_STATES or row["available_ts"] > now:
                raise QueueStateError(f"Job {job_id} is not ready")
            if row["cancel_requested"]:
                raise QueueStateError(f"Job {job_id} has a pending cancellation")
            connection.execute(
                """
                UPDATE jobs
                SET state = 'reserving', updated_ts = ?, assigned_gpu_uuid = ?,
                    lease_owner = ?, pid = NULL, exit_code = NULL, error = NULL
                WHERE id = ?
                """,
                (now, gpu_uuid, owner, job_id),
            )
            connection.execute(
                """
                INSERT INTO events(job_id, event_ts, from_state, to_state, message)
                VALUES (?, ?, ?, 'reserving', ?)
                """,
                (job_id, now, row["state"], f"reserved cooperative GPU {gpu_uuid}"),
            )
            updated = self._get_row(connection, job_id)
        record = self._record(updated)
        self._sync_tracking(record)
        return record

    def mark_running(self, job_id: str, pid: int, log_path: str) -> dict[str, Any]:
        current = self.get(job_id)
        if current["attempt_count"] >= current["max_attempts"]:
            raise QueueStateError(f"Job {job_id} exhausted its attempts")
        return self._transition(
            job_id,
            ("reserving",),
            "running",
            "owned runner process started",
            attempt_count=current["attempt_count"] + 1,
            pid=pid,
            log_path=log_path,
        )

    def record_duration(self, job_id: str, column: str, seconds: float) -> None:
        """Persist how long an owned process ran, so run time outlives the log."""

        if column not in {"run_seconds", "validate_seconds"}:
            raise QueueStateError(f"Unknown duration column: {column}")
        with self._transaction() as connection:
            connection.execute(
                f"UPDATE jobs SET {column} = ? WHERE id = ?",
                (float(seconds), job_id),
            )

    def mark_validating(self, job_id: str) -> dict[str, Any]:
        return self._transition(
            job_id,
            ("running",),
            "validating",
            "runner exited successfully; deterministic validation started",
            pid=None,
            exit_code=0,
        )

    def set_active_pid(self, job_id: str, pid: int) -> dict[str, Any]:
        now = time.time()
        with self._transaction() as connection:
            row = self._get_row(connection, job_id)
            if row["state"] not in ACTIVE_STATES:
                raise QueueStateError(f"Job {job_id} is not active")
            connection.execute(
                "UPDATE jobs SET pid = ?, updated_ts = ? WHERE id = ?",
                (pid, now, job_id),
            )
            updated = self._get_row(connection, job_id)
        record = self._record(updated)
        self._sync_tracking(record)
        return record

    def mark_succeeded(self, job_id: str) -> dict[str, Any]:
        return self._transition(
            job_id,
            ("validating",),
            "succeeded",
            "validator accepted all required artifacts",
            pid=None,
            assigned_gpu_uuid=None,
            lease_owner=None,
            exit_code=0,
            error=None,
        )

    def mark_failed(self, job_id: str, reason: str, exit_code: int | None = None) -> dict[str, Any]:
        return self._transition(
            job_id,
            tuple(ACTIVE_STATES),
            "failed",
            reason,
            pid=None,
            assigned_gpu_uuid=None,
            lease_owner=None,
            exit_code=exit_code,
            error=reason,
        )

    def retry_or_fail(self, job_id: str, reason: str, delay_seconds: float) -> dict[str, Any]:
        current = self.get(job_id)
        if current["cancel_requested"]:
            return self.mark_cancelled(job_id, "cancelled before retry")
        if current["attempt_count"] >= current["max_attempts"]:
            return self.mark_failed(job_id, reason)
        return self._transition(
            job_id,
            tuple(ACTIVE_STATES),
            "retry_wait",
            reason,
            available_ts=time.time() + max(0.0, delay_seconds),
            pid=None,
            assigned_gpu_uuid=None,
            lease_owner=None,
            error=reason,
        )

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        current = self.get(job_id)
        if current["state"] in {"queued", "retry_wait"}:
            return self._transition(
                job_id,
                (current["state"],),
                "cancelled",
                "cancelled before execution",
                cancel_requested=1,
                assigned_gpu_uuid=None,
                lease_owner=None,
            )
        if current["state"] in ACTIVE_STATES:
            now = time.time()
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_ts = ? WHERE id = ?",
                    (now, job_id),
                )
                updated = self._get_row(connection, job_id)
            record = self._record(updated)
            self._sync_tracking(record)
            return record
        return current

    def mark_cancelled(self, job_id: str, reason: str) -> dict[str, Any]:
        return self._transition(
            job_id,
            tuple(ACTIVE_STATES),
            "cancelled",
            reason,
            cancel_requested=1,
            pid=None,
            assigned_gpu_uuid=None,
            lease_owner=None,
            error=reason,
        )

    def cancel_requested(self, job_id: str) -> bool:
        return bool(self.get(job_id)["cancel_requested"])

    def recover_orphans(self, owner: str) -> list[str]:
        """Mark stale active records orphaned; never signal a recorded PID."""

        recovered: list[str] = []
        with self._connect() as connection:
            ids = [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM jobs WHERE state IN ('reserving','running','validating')"
                )
            ]
        for job_id in ids:
            self._transition(
                job_id,
                tuple(ACTIVE_STATES),
                "orphaned",
                f"scheduler {owner} recovered stale active state without signalling its PID",
                pid=None,
                assigned_gpu_uuid=None,
                lease_owner=None,
                error="scheduler restart found an unowned active job; manual inspection required",
            )
            recovered.append(job_id)
        return recovered

    def events(self, job_id: str) -> list[dict[str, Any]]:
        self.get(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE job_id = ? ORDER BY sequence", (job_id,)
            )
            return [dict(row) for row in rows]
