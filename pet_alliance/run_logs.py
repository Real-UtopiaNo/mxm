"""Keep dated diagnostic runs for the current Beijing day only."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import logging
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Callable

from .dragon_schedule import BEIJING, beijing_now


_RUN_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*_(\d{8}_\d{6}_\d{6})\Z")


def beijing_day(moment: datetime) -> date:
    current = moment.replace(tzinfo=BEIJING) if moment.tzinfo is None else moment.astimezone(BEIJING)
    return current.date()


def _linked(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


@dataclass
class CleanupResult:
    removed: int = 0
    errors: list[str] = field(default_factory=list)


class RunLogRetention:
    """Sweep only recognized run directories, never configuration or progress."""

    def __init__(self, project_root: Path, *, now: Callable[[], datetime] = beijing_now,
                 log: logging.Logger | None = None, protected_paths=()):
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "artifacts" / "runs"
        self.now = now
        self.log = log or logging.getLogger("pet_alliance")
        self._protected = tuple(location for path in protected_paths
                                for location in (Path(path).absolute(), Path(path).resolve()))
        self._lock = threading.RLock()
        self._clean_day: date | None = None
        self._attempt_day: date | None = None
        self._attempt_at = float("-inf")

    def maintain(self, *, force: bool = False, current: datetime | None = None) -> CleanupResult:
        today = beijing_day(self.now() if current is None else current)
        result = CleanupResult()
        with self._lock:
            if not force and (self._clean_day == today or (
                    self._attempt_day == today and time.monotonic() - self._attempt_at < 60)):
                return result
            self._attempt_day, self._attempt_at = today, time.monotonic()
            try:
                # Both the managed root and every recursive-delete target must
                # resolve inside this project's literal diagnostic directory.
                if _linked(self.root.parent) or _linked(self.root):
                    raise OSError("运行记录目录是链接，已跳过自动清理")
                resolved_root = self.root.resolve()
                if resolved_root != self.root or not resolved_root.is_relative_to(self.project_root):
                    raise OSError("运行记录目录超出项目，已跳过自动清理")
                entries = list(self.root.iterdir()) if self.root.exists() else []
                for path in entries:
                    match = _RUN_NAME.fullmatch(path.name)
                    if match is None or not path.is_dir() or _linked(path):
                        continue
                    try:
                        day = datetime.strptime(match[1], "%Y%m%d_%H%M%S_%f").date()
                    except ValueError:
                        continue
                    if day >= today:
                        continue
                    target = path.resolve()
                    if target.parent != resolved_root or any(
                            protected.is_relative_to(target) for protected in self._protected):
                        continue
                    try:
                        shutil.rmtree(path)
                    except OSError as exc:
                        result.errors.append(f"{path.name}: {exc}")
                    else:
                        result.removed += 1
            except OSError as exc:
                result.errors.append(str(exc))
            if not result.errors:
                self._clean_day = today
        if result.removed:
            self.log.info("运行记录仅保留北京时间当天，已清理 %d 个历史目录。", result.removed)
        for error in result.errors:
            self.log.warning("旧运行记录暂未清理，将稍后重试：%s", error)
        return result


class _RunDirectory:
    def __init__(self, root: Path, name: str, retention: RunLogRetention | None):
        self.root, self.name, self.retention = root, name, retention
        self.now = retention.now if retention is not None else beijing_now
        self.day: date | None = None
        self.path: Path | None = None

    def current(self) -> Path:
        moment = self.now()
        today = beijing_day(moment)
        if self.day != today:
            current = moment.replace(tzinfo=BEIJING) if moment.tzinfo is None else moment.astimezone(BEIJING)
            self.path = self.root / f"{self.name}_{current:%Y%m%d_%H%M%S_%f}"
            self.path.mkdir(parents=True, exist_ok=True)
            self.day = today
        if self.retention is not None:
            self.retention.maintain(current=moment)
        return self.path


class RunLogOwner:
    """A running job lazily moves its diagnostics into today's directory."""

    def _init_run_log(self, ctx, name: str) -> None:
        retention = getattr(ctx, "run_logs", None)
        root = retention.root if retention is not None else ctx.project_root / "artifacts" / "runs"
        self._run_log = _RunDirectory(root, name, retention)
        self._run_log.current()

    @property
    def run_dir(self) -> Path:
        if getattr(self, "_run_log", None) is not None:
            return self._run_log.current()
        return self._run_dir

    @run_dir.setter
    def run_dir(self, path: Path) -> None:
        # Explicit overrides are used by replay tools and isolated tests.
        self._run_log = None
        self._run_dir = Path(path)
