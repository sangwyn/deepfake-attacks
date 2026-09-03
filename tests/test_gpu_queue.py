"""Tests for the stdlib-only GPU queue; no real GPU is accessed."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from ops.gpuq.commands import commands_for
from ops.gpuq.db import QueueDatabase
from ops.gpuq.errors import InventoryError, QueueStateError, SpecError
from ops.gpuq.inventory import GpuInfo, NvidiaSmiInventory
from ops.gpuq.models import JobSpec
from ops.gpuq.process import ProcessResult
from ops.gpuq.scheduler import GpuScheduler, SchedulerPolicy


def job_mapping(
    *,
    config_path: str = "configs/attack.yaml",
    run_dir: str = "tracking/runs/campaign/task",
    requested_memory_mb: int = 12000,
    max_attempts: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_kind": "attack-experiment",
        "config_path": config_path,
        "run_dir": run_dir,
        "requested_memory_mb": requested_memory_mb,
        "timeout_seconds": 600,
        "priority": 0,
        "max_attempts": max_attempts,
    }


class ProjectFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project"
        (self.root / "configs").mkdir(parents=True)
        self.config = self.root / "configs" / "attack.yaml"
        self.config.write_text("attack: ifgsm\nseed: 0\n", encoding="utf-8")
        self.state_dir = Path(self.temporary.name) / "state"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def spec(self, **changes: object) -> JobSpec:
        raw = job_mapping()
        raw.update(changes)
        return JobSpec.from_mapping(raw, self.root)

    def database(self) -> QueueDatabase:
        return QueueDatabase(self.state_dir, self.root)


class JobSpecTests(ProjectFixture):
    def test_normalizes_and_hashes_the_frozen_config(self) -> None:
        first = self.spec()
        second = self.spec()
        self.assertEqual(first.idempotency_key, second.idempotency_key)
        self.assertEqual(len(first.config_sha256), 64)
        self.assertEqual(first.task_kind, "attack-experiment")

    def test_rejects_command_or_environment_injection_fields(self) -> None:
        for field, value in (
            ("command", "rm -rf /"),
            ("argv", ["bash"]),
            ("env", {"LD_PRELOAD": "bad.so"}),
        ):
            raw = job_mapping()
            raw[field] = value
            with self.subTest(field=field), self.assertRaises(SpecError):
                JobSpec.from_mapping(raw, self.root)

    def test_rejects_unknown_task_kind_and_path_escape(self) -> None:
        with self.assertRaises(SpecError):
            self.spec(task_kind="shell")
        with self.assertRaises(SpecError):
            self.spec(config_path="../secret.yaml")
        with self.assertRaises(SpecError):
            self.spec(run_dir="runs/campaign/task")
        with self.assertRaises(SpecError):
            self.spec(run_dir="tracking/runs/../outside/task")

    def test_rejects_boolean_as_integer(self) -> None:
        with self.assertRaises(SpecError):
            self.spec(requested_memory_mb=True)

    def test_command_registry_is_fixed_and_project_relative(self) -> None:
        commands = commands_for(self.spec(), 2)
        self.assertEqual(commands.run[1:4], ("-m", "attacklab.cli", "run"))
        self.assertEqual(commands.validate[1:4], ("-m", "attacklab.cli", "verify"))
        self.assertIn("configs/attack.yaml", commands.run)
        self.assertEqual(
            commands.attempt_dir, "tracking/runs/campaign/task/attempt-0002"
        )


class QueueDatabaseTests(ProjectFixture):
    def test_submit_is_idempotent_and_writes_git_visible_snapshots(self) -> None:
        database = self.database()
        first, created = database.submit(self.spec())
        second, created_again = database.submit(self.spec())
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first["id"], second["id"])
        run_dir = self.root / "tracking" / "runs" / "campaign" / "task"
        persisted = json.loads((run_dir / "job_spec.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "gpuq_status.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["config_sha256"], self.spec().config_sha256)
        self.assertEqual(status["state"], "queued")

    def test_same_run_dir_cannot_be_reused_by_a_different_spec(self) -> None:
        database = self.database()
        database.submit(self.spec())
        self.config.write_text("attack: pgd\n", encoding="utf-8")
        with self.assertRaises(SpecError):
            database.submit(self.spec())

    def test_database_trigger_keeps_spec_immutable(self) -> None:
        database = self.database()
        record, _ = database.submit(self.spec())
        with sqlite3.connect(database.path) as connection, self.assertRaises(
            sqlite3.IntegrityError
        ):
            connection.execute(
                "UPDATE jobs SET spec_json = '{}' WHERE id = ?", (record["id"],)
            )

    def test_successful_state_machine_and_event_history(self) -> None:
        database = self.database()
        record, _ = database.submit(self.spec())
        database.claim(record["id"], "GPU-aaaaaaaa", "test-owner")
        database.mark_running(record["id"], 12345, "/tmp/owned.log")
        database.mark_validating(record["id"])
        final = database.mark_succeeded(record["id"])
        self.assertEqual(final["state"], "succeeded")
        self.assertIsNone(final["assigned_gpu_uuid"])
        self.assertEqual(
            [event["to_state"] for event in database.events(record["id"])],
            ["queued", "reserving", "running", "validating", "succeeded"],
        )

    def test_cancel_queued_job_without_a_process(self) -> None:
        database = self.database()
        record, _ = database.submit(self.spec())
        cancelled = database.request_cancel(record["id"])
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertTrue(cancelled["cancel_requested"])

    def test_restart_marks_active_job_orphaned_without_signalling_pid(self) -> None:
        database = self.database()
        record, _ = database.submit(self.spec())
        database.claim(record["id"], "GPU-aaaaaaaa", "dead-owner")
        database.mark_running(record["id"], 999999, "/tmp/old.log")
        recovered = database.recover_orphans("new-owner")
        self.assertEqual(recovered, [record["id"]])
        self.assertEqual(database.get(record["id"])["state"], "orphaned")

    def test_invalid_transition_is_rejected(self) -> None:
        database = self.database()
        record, _ = database.submit(self.spec())
        with self.assertRaises(QueueStateError):
            database.mark_succeeded(record["id"])


class InventoryTests(unittest.TestCase):
    def test_inventory_attaches_compute_pids_to_gpu_uuid(self) -> None:
        outputs = iter(
            (
                "GPU-aaa, 0, 46000, 1000, 45000, 2\nGPU-bbb, 1, 46000, 5000, 41000, 8\n",
                "GPU-bbb, 4321, 4000\n",
            )
        )

        def runner(argv: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, next(outputs), "")

        gpus = NvidiaSmiInventory(runner=runner).snapshot()
        self.assertEqual(gpus[0].compute_pids, ())
        self.assertEqual(gpus[1].compute_pids, (4321,))
        self.assertTrue(
            gpus[0].eligible(
                requested_memory_mb=40000,
                headroom_mb=4096,
                max_utilization_percent=5,
            )
        )
        self.assertFalse(
            gpus[1].eligible(
                requested_memory_mb=1000,
                headroom_mb=4096,
                max_utilization_percent=10,
            )
        )

    def test_inventory_fails_closed_on_malformed_output(self) -> None:
        outputs = iter(("not,a,valid,row\n", ""))

        def runner(argv: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, next(outputs), "")

        with self.assertRaises(InventoryError):
            NvidiaSmiInventory(runner=runner).snapshot()


class FakeInventory:
    def __init__(self, gpus: list[GpuInfo]):
        self.gpus = gpus
        self.calls = 0

    def snapshot(self) -> list[GpuInfo]:
        self.calls += 1
        return list(self.gpus)


class SchedulerTests(ProjectFixture):
    @staticmethod
    def gpu(
        uuid: str = "GPU-aaaaaaaa",
        *,
        index: int = 0,
        free_mb: int = 45000,
        utilization: int = 0,
        pids: tuple[int, ...] = (),
    ) -> GpuInfo:
        return GpuInfo(
            uuid=uuid,
            index=index,
            total_memory_mb=46000,
            used_memory_mb=46000 - free_mb,
            free_memory_mb=free_mb,
            utilization_percent=utilization,
            compute_pids=pids,
        )

    def test_default_policy_is_single_job_and_conservative(self) -> None:
        policy = SchedulerPolicy()
        self.assertEqual(policy.max_running, 1)
        self.assertEqual(policy.idle_samples, 3)
        self.assertEqual(policy.headroom_mb, 4096)

    def test_dry_scheduler_never_claims_or_launches(self) -> None:
        database = self.database()
        record, _ = database.submit(self.spec())
        launched: list[object] = []

        def runner(*args: object, **kwargs: object) -> ProcessResult:
            launched.append((args, kwargs))
            return ProcessResult(0, "exited", 0.0)

        scheduler = GpuScheduler(
            database,
            inventory=FakeInventory([self.gpu()]),
            policy=SchedulerPolicy(idle_samples=1),
            execute=False,
            process_runner=runner,
        )
        report = scheduler.run_cycle()
        scheduler.close()
        self.assertFalse(report.execute)
        self.assertEqual(database.get(record["id"])["state"], "queued")
        self.assertEqual(launched, [])

    def test_scheduler_waits_for_idle_window_and_uses_gpu_uuid(self) -> None:
        database = self.database()
        record, _ = database.submit(self.spec())
        inventory = FakeInventory(
            [
                self.gpu("GPU-small", index=0, free_mb=15000),
                self.gpu("GPU-large", index=1, free_mb=45000),
            ]
        )
        calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

        def runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
            environment = dict(kwargs["env"])
            calls.append((tuple(argv), environment))
            kwargs["on_start"](20000 + len(calls))
            return ProcessResult(0, "exited", 0.01)

        scheduler = GpuScheduler(
            database,
            inventory=inventory,
            policy=SchedulerPolicy(idle_samples=3, headroom_mb=4096),
            execute=True,
            process_runner=runner,
        )
        self.assertEqual(scheduler.run_cycle().started, ())
        self.assertEqual(scheduler.run_cycle().started, ())
        started = scheduler.run_cycle().started
        scheduler.wait_for_workers()
        scheduler.close()
        self.assertEqual(started, ((record["id"], "GPU-large"),))
        self.assertEqual(database.get(record["id"])["state"], "succeeded")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["CUDA_VISIBLE_DEVICES"], "GPU-large")
        self.assertEqual(calls[0][1]["GPUQ_ASSIGNED_GPU_UUID"], "GPU-large")
        self.assertEqual(calls[0][0][3], "run")
        self.assertEqual(calls[1][0][3], "verify")

    def test_compute_process_and_headroom_both_fail_closed(self) -> None:
        for name, gpu in (
            ("foreign-process", self.gpu(pids=(1111,))),
            ("insufficient-headroom", self.gpu(free_mb=15000)),
        ):
            with self.subTest(name=name):
                # Use a separate project because run_dir ownership is immutable.
                nested = self.root / name
                (nested / "configs").mkdir(parents=True)
                (nested / "configs" / "attack.yaml").write_text(
                    "attack: ifgsm\n", encoding="utf-8"
                )
                database = QueueDatabase(self.state_dir / name, nested)
                spec = JobSpec.from_mapping(job_mapping(), nested)
                record, _ = database.submit(spec)
                scheduler = GpuScheduler(
                    database,
                    inventory=FakeInventory([gpu]),
                    policy=SchedulerPolicy(idle_samples=1, headroom_mb=4096),
                    execute=True,
                    process_runner=lambda *a, **k: ProcessResult(0, "exited", 0),
                )
                self.assertEqual(scheduler.run_cycle().started, ())
                scheduler.close()
                self.assertEqual(database.get(record["id"])["state"], "queued")




class SharedGpuPolicyTests(unittest.TestCase):
    """Sharing a card is opt-in, and memory headroom still guards both users."""

    @staticmethod
    def _gpu(free_memory_mb: int, compute_pids=(), utilization_percent: int = 80):
        return GpuInfo(
            index=0,
            uuid="GPU-shared-0001",
            total_memory_mb=46068,
            used_memory_mb=46068 - free_memory_mb,
            free_memory_mb=free_memory_mb,
            utilization_percent=utilization_percent,
            compute_pids=tuple(compute_pids),
        )

    def test_busy_card_is_refused_by_default(self) -> None:
        gpu = self._gpu(free_memory_mb=44081, compute_pids=(3429274,))
        self.assertFalse(
            gpu.eligible(
                requested_memory_mb=8192,
                headroom_mb=4096,
                max_utilization_percent=5,
            )
        )

    def test_busy_card_is_accepted_when_sharing_is_allowed(self) -> None:
        gpu = self._gpu(free_memory_mb=44081, compute_pids=(3429274,))
        self.assertTrue(
            gpu.eligible(
                requested_memory_mb=8192,
                headroom_mb=4096,
                max_utilization_percent=5,
                allow_shared=True,
            )
        )

    def test_sharing_still_enforces_memory_headroom(self) -> None:
        gpu = self._gpu(free_memory_mb=153, compute_pids=(2823431,))
        self.assertFalse(
            gpu.eligible(
                requested_memory_mb=8192,
                headroom_mb=4096,
                max_utilization_percent=5,
                allow_shared=True,
            )
        )

    def test_sharing_still_rejects_a_non_uuid_card(self) -> None:
        gpu = GpuInfo(
            index=0,
            uuid="Unknown Error",
            total_memory_mb=46068,
            used_memory_mb=0,
            free_memory_mb=46068,
            utilization_percent=0,
            compute_pids=(),
        )
        self.assertFalse(
            gpu.eligible(
                requested_memory_mb=8192,
                headroom_mb=4096,
                max_utilization_percent=5,
                allow_shared=True,
            )
        )




class DurationPersistenceTests(unittest.TestCase):
    """Queue time and process time must survive the job, not only the log."""

    def _database(self, root: Path) -> tuple[QueueDatabase, str]:
        (root / "configs").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "exp.yaml").write_text("a: 1\n", encoding="utf-8")
        database = QueueDatabase(root / ".gpuq", root)
        spec = JobSpec.from_mapping(
            {
                "schema_version": 1,
                "task_kind": "attack-experiment",
                "config_path": "configs/exp.yaml",
                "run_dir": "tracking/runs/camp/task",
                "requested_memory_mb": 1024,
                "timeout_seconds": 60,
            },
            root,
        )
        record, _ = database.submit(spec)
        return database, record["id"]

    def test_durations_are_recorded_and_exported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id = self._database(root)
            database.record_duration(job_id, "run_seconds", 12.5)
            database.record_duration(job_id, "validate_seconds", 3.25)
            record = database.get(job_id)
            self.assertAlmostEqual(record["run_seconds"], 12.5)
            self.assertAlmostEqual(record["validate_seconds"], 3.25)

            database.claim(job_id, "GPU-dur-0001", "owner")
            status = json.loads(
                (root / "tracking/runs/camp/task/gpuq_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertAlmostEqual(status["run_seconds"], 12.5)
            self.assertAlmostEqual(status["validate_seconds"], 3.25)

    def test_unknown_duration_column_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database, job_id = self._database(root)
            with self.assertRaises(QueueStateError):
                database.record_duration(job_id, "wall_seconds; DROP TABLE jobs", 1.0)


if __name__ == "__main__":
    unittest.main()
