"""Supervise only subprocesses created by this scheduler instance."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int | None
    reason: str
    duration_seconds: float


def _signal_owned_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    """Signal the process group whose leader is our own Popen child.

    This function deliberately cannot accept a PID loaded from SQLite. After a
    scheduler crash, old PIDs are marked orphaned and are never signalled.
    """

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def terminate_owned_process(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 10.0
) -> None:
    if process.poll() is not None:
        return
    _signal_owned_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        _signal_owned_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            # We do not broaden the target. The OS remains responsible for a
            # child that cannot be reaped after signalling its owned group.
            pass


def run_owned_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
    cancel_requested: Callable[[], bool],
    shutdown_requested: Callable[[], bool],
    on_start: Callable[[int], None],
    poll_seconds: float = 0.5,
) -> ProcessResult:
    """Run a fixed argv without a shell and supervise timeout/cancellation."""

    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("argv must contain non-empty strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    start = time.monotonic()
    with log_path.open("ab", buffering=0) as log:
        marker = f"\n[gpuq] exec argv={list(argv)!r}\n".encode("utf-8")
        log.write(marker)
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )
        try:
            on_start(process.pid)
        except BaseException:
            terminate_owned_process(process)
            raise
        deadline = start + timeout_seconds
        while True:
            returncode = process.poll()
            if returncode is not None:
                return ProcessResult(
                    returncode=returncode,
                    reason="exited",
                    duration_seconds=time.monotonic() - start,
                )
            if cancel_requested():
                terminate_owned_process(process)
                return ProcessResult(
                    returncode=process.poll(),
                    reason="cancelled",
                    duration_seconds=time.monotonic() - start,
                )
            if shutdown_requested():
                terminate_owned_process(process)
                return ProcessResult(
                    returncode=process.poll(),
                    reason="shutdown",
                    duration_seconds=time.monotonic() - start,
                )
            if time.monotonic() >= deadline:
                terminate_owned_process(process)
                return ProcessResult(
                    returncode=process.poll(),
                    reason="timeout",
                    duration_seconds=time.monotonic() - start,
                )
            time.sleep(poll_seconds)
