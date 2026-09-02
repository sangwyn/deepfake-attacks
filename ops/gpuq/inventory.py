"""Strict NVIDIA inventory obtained through read-only ``nvidia-smi`` queries."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .errors import InventoryError


@dataclass(frozen=True, slots=True)
class GpuInfo:
    uuid: str
    index: int
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: int
    compute_pids: tuple[int, ...] = ()

    def eligible(
        self,
        *,
        requested_memory_mb: int,
        headroom_mb: int,
        max_utilization_percent: int,
    ) -> bool:
        return (
            self.uuid.startswith("GPU-")
            and not self.compute_pids
            and self.utilization_percent <= max_utilization_percent
            and self.free_memory_mb >= requested_memory_mb + headroom_mb
        )


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _integer(raw: str, field: str) -> int:
    value = raw.strip()
    if value in {"", "N/A", "[N/A]"}:
        raise InventoryError(f"nvidia-smi returned no numeric {field}")
    try:
        return int(value)
    except ValueError as exc:
        raise InventoryError(f"Invalid {field} from nvidia-smi: {raw!r}") from exc


def _csv_rows(output: str) -> Iterable[list[str]]:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "no running processes found" in line.lower():
            continue
        yield [part.strip() for part in line.split(",")]


class NvidiaSmiInventory:
    """Read GPU and compute-process state without changing driver state."""

    def __init__(
        self, binary: str = "nvidia-smi", runner: CommandRunner | None = None
    ):
        self.binary = binary
        self._runner = runner or _default_runner

    def _query(self, query: str) -> str:
        argv = (
            self.binary,
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        )
        try:
            result = self._runner(argv)
        except (OSError, subprocess.SubprocessError) as exc:
            raise InventoryError(f"Cannot execute {' '.join(argv)}: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise InventoryError(f"nvidia-smi query failed ({result.returncode}): {detail}")
        return result.stdout

    def snapshot(self) -> list[GpuInfo]:
        gpu_output = self._query(
            "gpu=uuid,index,memory.total,memory.used,memory.free,utilization.gpu"
        )
        process_output = self._query("compute-apps=gpu_uuid,pid,used_gpu_memory")

        processes: dict[str, list[int]] = {}
        for row in _csv_rows(process_output):
            if len(row) != 3:
                raise InventoryError(f"Unexpected compute-app row: {row!r}")
            gpu_uuid, raw_pid, _used_memory = row
            if not gpu_uuid.startswith("GPU-"):
                raise InventoryError(f"Unexpected compute-app GPU UUID: {gpu_uuid!r}")
            processes.setdefault(gpu_uuid, []).append(_integer(raw_pid, "process pid"))

        inventory: list[GpuInfo] = []
        seen: set[str] = set()
        for row in _csv_rows(gpu_output):
            if len(row) != 6:
                raise InventoryError(f"Unexpected GPU inventory row: {row!r}")
            gpu_uuid = row[0]
            if not gpu_uuid.startswith("GPU-") or gpu_uuid in seen:
                raise InventoryError(f"Invalid or duplicate GPU UUID: {gpu_uuid!r}")
            seen.add(gpu_uuid)
            inventory.append(
                GpuInfo(
                    uuid=gpu_uuid,
                    index=_integer(row[1], "GPU index"),
                    total_memory_mb=_integer(row[2], "total memory"),
                    used_memory_mb=_integer(row[3], "used memory"),
                    free_memory_mb=_integer(row[4], "free memory"),
                    utilization_percent=_integer(row[5], "GPU utilization"),
                    compute_pids=tuple(sorted(processes.get(gpu_uuid, ()))),
                )
            )
        if not inventory:
            raise InventoryError("nvidia-smi returned no full GPUs")
        unknown_process_gpus = set(processes) - seen
        if unknown_process_gpus:
            raise InventoryError(
                "Compute processes reference unknown GPUs: "
                + ", ".join(sorted(unknown_process_gpus))
            )
        return sorted(inventory, key=lambda gpu: gpu.index)
