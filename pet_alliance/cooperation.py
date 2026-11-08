"""Optional task controls shared by standalone scripts and the scheduler."""

from __future__ import annotations

from .adb import AutomationError


class TaskInterrupted(AutomationError):
    """Leave a task without mistaking a scheduler handoff for a failure."""

    def __init__(self, status: str, reason: str):
        if status not in {"paused", "preempted"}:
            raise ValueError("Task interruption status must be paused or preempted.")
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.report: dict | None = None
        self.run_dir: str = ""


def checkpoint(ctx, name: str, **payload) -> None:
    """Publish a safe task boundary while preserving old script contexts."""
    callback = getattr(ctx, "checkpoint", None)
    if callable(callback):
        callback(name, **payload)
