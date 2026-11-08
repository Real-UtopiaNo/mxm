"""Keep the scheduler's text log limited to the current Beijing day."""

from __future__ import annotations

from copy import copy
from datetime import date, datetime
import logging
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable

from .dragon_schedule import BEIJING, beijing_now


_TIMESTAMP = re.compile(rb"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2}")


def _beijing(value: datetime) -> datetime:
    return value.replace(tzinfo=BEIJING) if value.tzinfo is None else value.astimezone(BEIJING)


def _today_records(contents: bytes, today: date) -> bytes:
    """Continuation lines, including tracebacks, belong to the preceding record."""
    result = []
    keep = False
    for line in contents.splitlines(keepends=True):
        match = _TIMESTAMP.match(line)
        if match is not None:
            try:
                record_day = datetime.fromisoformat(line[:19].decode("ascii")).date()
            except ValueError:
                keep = False
            else:
                keep = record_day == today
        if keep:
            result.append(line)
    return b"".join(result)


class _BeijingFormatter(logging.Formatter):
    def __init__(self, now: Callable[[], datetime]):
        super().__init__("%(asctime)s %(message)s")
        self.now = now

    def formatTime(self, record, datefmt=None):
        current = getattr(record, "_today_log_time", None)
        current = _beijing(self.now()) if current is None else current
        return current.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


class TodayFileHandler(logging.FileHandler):
    """One UTF-8 log, pruned at startup and whenever the Beijing date changes.

    ``maintain()`` also performs rollover without a new log event, for example
    while the scheduler is paused. Only the exact log and its three legacy
    rotating backups are touched. State files and other artifacts are unrelated.
    """

    def __init__(self, path: str | os.PathLike, *, now: Callable[[], datetime] = beijing_now):
        requested = Path(path).absolute()
        requested.parent.mkdir(parents=True, exist_ok=True)
        self._directory = requested.parent
        if self._directory.resolve() != self._directory:
            raise OSError("Refusing a scheduler log inside a linked directory")
        self._path = requested
        self._now = now
        self._day: date | None = None
        self._checked_file(self._path, base=True)
        super().__init__(requested, mode="a", encoding="utf-8", delay=True, errors="backslashreplace")
        self.setFormatter(_BeijingFormatter(now))
        try:
            self.maintain()
        except BaseException:
            self.close()
            raise

    def _checked_file(self, path: Path, *, base=False):
        """Return an ordinary file's metadata, without following a link."""
        unsafe = path.parent != self._directory or self._directory.resolve() != self._directory
        try:
            value = path.lstat()
        except FileNotFoundError:
            if unsafe:
                raise OSError("Scheduler log path left its directory")
            return None
        reparse = getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        unsafe = unsafe or bool(reparse) or not stat.S_ISREG(value.st_mode)
        if unsafe:
            if base:
                raise OSError("Refusing a linked or non-file scheduler log")
            return None
        return value

    @staticmethod
    def _identity(value):
        if value is None:
            return None
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    def _unchanged(self, path: Path, expected, *, base=False):
        return self._identity(self._checked_file(path, base=base)) == self._identity(expected)

    def _replace(self, contents: bytes, expected):
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self._path.name}.", suffix=".tmp", dir=self._directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(contents)
                output.flush()
            if expected is not None:
                os.chmod(temporary, stat.S_IMODE(expected.st_mode))
            if not self._unchanged(self._path, expected, base=True):
                raise OSError("Scheduler log changed during daily cleanup")
            os.replace(temporary, self._path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _retain_day(self, today: date, *, backups: bool) -> bool:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        base_stat = self._checked_file(self._path, base=True)
        base_contents = self._path.read_bytes() if base_stat is not None else b""
        fragments = []
        legacy = []
        if backups:
            # RotatingFileHandler's highest suffix is the oldest file.
            for suffix in (3, 2, 1):
                candidate = self._path.with_name(f"{self._path.name}.{suffix}")
                metadata = self._checked_file(candidate)
                if metadata is None:
                    continue
                contents = candidate.read_bytes()
                if not self._unchanged(candidate, metadata):
                    raise OSError("Scheduler log backup changed during daily cleanup")
                fragments.append(_today_records(contents, today))
                legacy.append((candidate, metadata))
        fragments.append(_today_records(base_contents, today))
        retained = b""
        for fragment in fragments:
            if fragment:
                if retained and not retained.endswith((b"\n", b"\r")):
                    retained += b"\n"
                retained += fragment
        changed = base_stat is None or retained != base_contents
        if changed:
            self._replace(retained, base_stat)
        elif not self._unchanged(self._path, base_stat, base=True):
            raise OSError("Scheduler log changed during daily cleanup")
        # Delete only ordinary backups whose inspected contents are unchanged,
        # and only after their retained records have reached the main log.
        for candidate, metadata in legacy:
            if self._unchanged(candidate, metadata):
                candidate.unlink()
                changed = True
        return changed

    def _maintain_for(self, current: datetime) -> bool:
        today = current.date()
        if today == self._day:
            return False
        changed = self._retain_day(today, backups=self._day is None)
        self._day = today
        return changed

    def _open(self):
        # A delayed first emit or rollover may occur long after construction.
        self._checked_file(self._path, base=True)
        return super()._open()

    def maintain(self) -> bool:
        """Discard earlier days, even with no emitted messages; return whether changed."""
        self.acquire()
        try:
            if self._closed:
                return False
            return self._maintain_for(_beijing(self._now()))
        finally:
            self.release()

    def emit(self, record):
        self.acquire()
        try:
            if self._closed:
                return
            current = _beijing(self._now())
            self._maintain_for(current)
            local = copy(record)
            local._today_log_time = current
            super().emit(local)
        except Exception:
            self.handleError(record)
        finally:
            self.release()
