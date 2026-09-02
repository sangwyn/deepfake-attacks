"""Conservative GPU scheduler with fixed commands and owned-process supervision."""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .commands import commands_for
from .db import QueueDatabase
from .errors import GpuQueueError, QueueStateError
from .inventory import GpuInfo, NvidiaSmiInventory
from .locks import AdvisoryLock, gpu_lock, scheduler_lock
from .models import ACTIVE_STATES, JobSpec
from .process import ProcessResult, run_owned_process


ProcessRunner = Callable[..., ProcessResult]


@dataclass(frozen=True, slots=True)
class SchedulerPolicy:
    max_running: int = 1
    poll_seconds: float = 10.0
    idle_samples: int = 3
    headroom_mb: int = 4096
    max_idle_utilization_percent: int = 5
    retry_delay_seconds: float = 30.0
    validation_timeout_seconds: float = 600.0
    # Off by default: waiting for an exclusive card is the conservative policy.
    # On a cluster every card shares other users' processes permanently, so
    # without this the scheduler polls forever and never starts a job.
    allow_shared_gpu: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_running, bool) or not 1 <= self.max_running <= 8:
            raise ValueError("max_running must be between 1 and 8")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if isinstance(self.idle_samples, bool) or not 1 <= self.idle_samples <= 60:
            raise ValueError("idle_samples must be between 1 and 60")
        if self.headroom_mb < 0:
            raise ValueError("headroom_mb cannot be negative")
        if not 0 <= self.max_idle_utilization_percent <= 100:
            raise ValueError("max_idle_utilization_percent must be between 0 and 100")
        if self.retry_delay_seconds < 0 or self.validation_timeout_seconds <= 0:
            raise ValueError("retry and validation timeouts are invalid")


@dataclass(frozen=True, slots=True)
class CycleReport:
    execute: bool
    inventory: tuple[GpuInfo, ...]
    ready_job_ids: tuple[str, ...]
    eligible_gpu_uuids: tuple[str, ...]
    started: tuple[tuple[str, str], ...]
    running_job_ids: tuple[str, ...]
    idle_observations: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["inventory"] = [asdict(gpu) for gpu in self.inventory]
        value["started"] = [
            {"job_id": job_id, "gpu_uuid": gpu_uuid}
            for job_id, gpu_uuid in self.started
        ]
        return value


class GpuScheduler:
    """One scheduler process capable of supervising a small number of jobs."""

    def __init__(
        self,
        database: QueueDatabase,
        *,
        inventory: NvidiaSmiInventory | Any | None = None,
        policy: SchedulerPolicy | None = None,
        execute: bool = False,
        process_runner: ProcessRunner = run_owned_process,
    ):
        self.database = database
        self.inventory = inventory or NvidiaSmiInventory()
        self.policy = policy or SchedulerPolicy()
        self.execute = execute
        self.process_runner = process_runner
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
        self._idle_observations: dict[str, int] = {}
        self._busy_gpus: set[str] = set()
        self._futures: dict[Future[None], tuple[str, str]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=self.policy.max_running, thread_name_prefix="gpuq"
        )
        self._shutdown = threading.Event()
        self._closed = False

    def _observe(self, inventory: Sequence[GpuInfo]) -> None:
        observed = {gpu.uuid for gpu in inventory}
        for stale in set(self._idle_observations) - observed:
            del self._idle_observations[stale]
        for gpu in inventory:
            base_idle = self.policy.allow_shared_gpu or (
                not gpu.compute_pids
                and gpu.utilization_percent
                <= self.policy.max_idle_utilization_percent
            )
            self._idle_observations[gpu.uuid] = (
                self._idle_observations.get(gpu.uuid, 0) + 1 if base_idle else 0
            )

    def _eligible(self, gpu: GpuInfo, requested_memory_mb: int) -> bool:
        return (
            gpu.uuid not in self._busy_gpus
            and self._idle_observations.get(gpu.uuid, 0) >= self.policy.idle_samples
            and gpu.eligible(
                requested_memory_mb=requested_memory_mb,
                headroom_mb=self.policy.headroom_mb,
                max_utilization_percent=self.policy.max_idle_utilization_percent,
                allow_shared=self.policy.allow_shared_gpu,
            )
        )

    def _reap_done(self) -> None:
        for future, (job_id, gpu_uuid) in list(self._futures.items()):
            if not future.done():
                continue
            del self._futures[future]
            self._busy_gpus.discard(gpu_uuid)
            try:
                future.result()
            except Exception as exc:  # defensive: worker also records failures
                try:
                    if self.database.get(job_id)["state"] in ACTIVE_STATES:
                        self.database.mark_failed(
                            job_id, f"scheduler worker crashed: {type(exc).__name__}: {exc}"
                        )
                except GpuQueueError:
                    pass

    def _final_gpu_check(self, gpu_uuid: str, requested_memory_mb: int) -> GpuInfo | None:
        fresh = {gpu.uuid: gpu for gpu in self.inventory.snapshot()}.get(gpu_uuid)
        if fresh is None or not fresh.eligible(
            requested_memory_mb=requested_memory_mb,
            headroom_mb=self.policy.headroom_mb,
            max_utilization_percent=self.policy.max_idle_utilization_percent,
            allow_shared=self.policy.allow_shared_gpu,
        ):
            self._idle_observations[gpu_uuid] = 0
            return None
        return fresh

    def run_cycle(self) -> CycleReport:
        if self._closed:
            raise RuntimeError("Scheduler is closed")
        self._reap_done()
        inventory = tuple(self.inventory.snapshot())
        self._observe(inventory)
        ready = self.database.ready_jobs()
        eligible_uuids = tuple(
            gpu.uuid
            for gpu in inventory
            if gpu.uuid not in self._busy_gpus
            and self._idle_observations.get(gpu.uuid, 0) >= self.policy.idle_samples
            and (self.policy.allow_shared_gpu or not gpu.compute_pids)
        )
        started: list[tuple[str, str]] = []
        capacity = self.policy.max_running - len(self._futures)
        if self.execute and capacity > 0:
            remaining_gpus = list(inventory)
            for record in ready:
                if capacity <= 0:
                    break
                spec = JobSpec.from_persisted(record["spec"])
                candidates = [
                    gpu
                    for gpu in remaining_gpus
                    if self._eligible(gpu, spec.requested_memory_mb)
                ]
                candidates.sort(key=lambda item: (-item.free_memory_mb, item.index))
                selected: GpuInfo | None = None
                lease: AdvisoryLock | None = None
                for candidate in candidates:
                    proposed = gpu_lock(self.database.locks_dir, candidate.uuid)
                    if not proposed.acquire(blocking=False):
                        continue
                    fresh = self._final_gpu_check(
                        candidate.uuid, spec.requested_memory_mb
                    )
                    if fresh is None:
                        proposed.release()
                        continue
                    selected, lease = fresh, proposed
                    break
                if selected is None or lease is None:
                    continue
                try:
                    claimed = self.database.claim(record["id"], selected.uuid, self.owner)
                except (GpuQueueError, Exception):
                    lease.release()
                    continue
                self._busy_gpus.add(selected.uuid)
                remaining_gpus = [gpu for gpu in remaining_gpus if gpu.uuid != selected.uuid]
                future = self._executor.submit(self._execute_job, claimed, selected, lease)
                self._futures[future] = (record["id"], selected.uuid)
                started.append((record["id"], selected.uuid))
                capacity -= 1

        return CycleReport(
            execute=self.execute,
            inventory=inventory,
            ready_job_ids=tuple(record["id"] for record in ready),
            eligible_gpu_uuids=eligible_uuids,
            started=tuple(started),
            running_job_ids=tuple(sorted(job_id for job_id, _gpu in self._futures.values())),
            idle_observations=dict(self._idle_observations),
        )

    def _environment(self, job_id: str, attempt: int, gpu_uuid: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "GPUQ_ASSIGNED_GPU_UUID": gpu_uuid,
                "GPUQ_JOB_ID": job_id,
                "GPUQ_ATTEMPT": str(attempt),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def _execute_job(
        self, record: Mapping[str, Any], gpu: GpuInfo, lease: AdvisoryLock
    ) -> None:
        job_id = str(record["id"])
        try:
            if self.database.cancel_requested(job_id):
                self.database.mark_cancelled(job_id, "cancelled while reserving")
                return
            spec = JobSpec.from_persisted(record["spec"])
            spec.verify_config_unchanged(self.database.project_root)
            attempt = int(record["attempt_count"]) + 1
            commands = commands_for(spec, attempt)
            attempt_absolute = self.database.project_root / commands.attempt_dir
            attempt_absolute.mkdir(parents=True, exist_ok=False)
            log_path = (
                self.database.logs_dir / job_id / f"attempt-{attempt:04d}.log"
            )
            environment = self._environment(job_id, attempt, gpu.uuid)

            run_result = self.process_runner(
                commands.run,
                cwd=self.database.project_root,
                env=environment,
                log_path=log_path,
                timeout_seconds=spec.timeout_seconds,
                cancel_requested=lambda: self.database.cancel_requested(job_id),
                shutdown_requested=self._shutdown.is_set,
                on_start=lambda pid: self.database.mark_running(
                    job_id, pid, str(log_path)
                ),
            )
            if run_result.reason == "cancelled":
                self.database.mark_cancelled(job_id, "owned runner cancelled")
                return
            if run_result.reason == "shutdown":
                self.database.retry_or_fail(
                    job_id,
                    "scheduler shutdown terminated its owned runner; retry is safe",
                    self.policy.retry_delay_seconds,
                )
                return
            if run_result.reason == "timeout":
                self.database.mark_failed(
                    job_id,
                    f"runner exceeded timeout of {spec.timeout_seconds} seconds",
                    run_result.returncode,
                )
                return
            if run_result.returncode != 0:
                self.database.mark_failed(
                    job_id,
                    f"runner exited with code {run_result.returncode}",
                    run_result.returncode,
                )
                return

            self.database.mark_validating(job_id)
            validation_result = self.process_runner(
                commands.validate,
                cwd=self.database.project_root,
                env=environment,
                log_path=log_path,
                timeout_seconds=self.policy.validation_timeout_seconds,
                cancel_requested=lambda: self.database.cancel_requested(job_id),
                shutdown_requested=self._shutdown.is_set,
                on_start=lambda pid: self.database.set_active_pid(job_id, pid),
            )
            if validation_result.reason == "cancelled":
                self.database.mark_cancelled(job_id, "owned validator cancelled")
            elif validation_result.reason == "shutdown":
                self.database.retry_or_fail(
                    job_id,
                    "scheduler shutdown terminated its owned validator; retry is safe",
                    self.policy.retry_delay_seconds,
                )
            elif validation_result.reason == "timeout":
                self.database.mark_failed(
                    job_id,
                    "deterministic validator timed out",
                    validation_result.returncode,
                )
            elif validation_result.returncode != 0:
                self.database.mark_failed(
                    job_id,
                    f"deterministic validator exited with code {validation_result.returncode}",
                    validation_result.returncode,
                )
            else:
                self.database.mark_succeeded(job_id)
        except Exception as exc:
            try:
                if self.database.get(job_id)["state"] in ACTIVE_STATES:
                    self.database.mark_failed(
                        job_id, f"job supervisor error: {type(exc).__name__}: {exc}"
                    )
            except GpuQueueError:
                pass
        finally:
            lease.release()

    def wait_for_workers(self) -> None:
        """Wait for currently dispatched work without requesting cancellation."""

        for future in list(self._futures):
            try:
                future.result()
            except Exception:
                pass
        self._reap_done()

    def request_shutdown(self) -> None:
        """Ask owned children to stop; unrelated and recovered PIDs are untouched."""

        self._shutdown.set()

    def close(self, *, stop_running: bool = False) -> None:
        if self._closed:
            return
        if stop_running:
            self.request_shutdown()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._reap_done()
        self._closed = True

    def serve(
        self,
        *,
        once: bool = False,
        report: Callable[[CycleReport], None] | None = None,
    ) -> None:
        """Hold the singleton lock and run one cycle or a persistent scheduler."""

        lock = scheduler_lock(self.database.locks_dir)
        with lock:
            if self.execute:
                self.database.recover_orphans(self.owner)
            try:
                while True:
                    cycle = self.run_cycle()
                    if report is not None:
                        report(cycle)
                    if once or not self.execute:
                        break
                    self._shutdown.wait(self.policy.poll_seconds)
                    if self._shutdown.is_set():
                        break
                if once and self.execute:
                    self.wait_for_workers()
            except KeyboardInterrupt:
                self.request_shutdown()
                raise
            finally:
                self.close(stop_running=self._shutdown.is_set())
