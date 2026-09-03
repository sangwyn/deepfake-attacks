"""Fixed argv construction for the only supported GPU task kind."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

from .errors import SpecError
from .models import CANONICAL_TASK_KIND, JobSpec


@dataclass(frozen=True, slots=True)
class AttemptCommands:
    """Runner and validator commands plus the attempt's relative output path."""

    run: tuple[str, ...]
    validate: tuple[str, ...]
    attempt_dir: str


def commands_for(spec: JobSpec, attempt: int) -> AttemptCommands:
    """Build argument vectors without accepting executable text from a job."""

    if spec.task_kind != CANONICAL_TASK_KIND:
        raise SpecError(f"No fixed command is registered for {spec.task_kind!r}")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise SpecError("attempt must be a positive integer")
    attempt_dir = (
        PurePosixPath(spec.run_dir) / f"attempt-{attempt:04d}"
    ).as_posix()
    python = sys.executable
    return AttemptCommands(
        run=(
            python,
            "-m",
            "attacklab.cli",
            "run",
            "--config",
            spec.config_path,
            "--run-dir",
            attempt_dir,
        ),
        validate=(
            python,
            "-m",
            "attacklab.cli",
            "verify",
            "--run-dir",
            attempt_dir,
        ),
        attempt_dir=attempt_dir,
    )
