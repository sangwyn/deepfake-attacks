"""Advisory scheduler and per-GPU locks.

These locks coordinate only processes that opt in to this queue. They cannot
reserve a GPU against unrelated users, so every launch also checks nvidia-smi.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path

from .errors import LockUnavailable


class AdvisoryLock:
    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, *, blocking: bool = False) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} label={self.label}\n".encode("utf-8"))
        os.fsync(fd)
        self._fd = fd
        return True

    def require(self) -> "AdvisoryLock":
        if not self.acquire(blocking=False):
            raise LockUnavailable(f"Lock is already held: {self.label}")
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "AdvisoryLock":
        return self.require()

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()


def scheduler_lock(locks_dir: Path) -> AdvisoryLock:
    return AdvisoryLock(locks_dir / "scheduler.lock", "gpuq-scheduler")


def gpu_lock(locks_dir: Path, gpu_uuid: str) -> AdvisoryLock:
    digest = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:24]
    return AdvisoryLock(locks_dir / f"gpu-{digest}.lock", gpu_uuid)
