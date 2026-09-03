"""GPU queue exceptions."""


class GpuQueueError(RuntimeError):
    """Base exception for a safe, expected GPU queue failure."""


class SpecError(GpuQueueError):
    """A submitted job specification is invalid."""


class QueueStateError(GpuQueueError):
    """A requested database state transition is invalid."""


class InventoryError(GpuQueueError):
    """The NVIDIA GPU inventory could not be read reliably."""


class LockUnavailable(GpuQueueError):
    """A singleton scheduler or cooperative GPU lock is already held."""
