"""Durable, conservative GPU job queue for attack experiments.

Jobs contain data, never shell commands. :mod:`ops.gpuq.commands` is the only
place that maps the canonical task kind to executable argument vectors.
"""

from .db import QueueDatabase
from .errors import GpuQueueError
from .models import JobSpec

__all__ = ["GpuQueueError", "JobSpec", "QueueDatabase"]
__version__ = "0.1.0"
