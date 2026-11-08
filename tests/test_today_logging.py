from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import stat
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pet_alliance.dragon_schedule import BEIJING
from pet_alliance.today_logging import TodayFileHandler


class Clock:
    def __init__(self, value="2026-11-07T12:00:00+08:00"):
        self.current = datetime.fromisoformat(value)

    def __call__(self):
        return self.current


class TodayFileHandlerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.path = self.directory / "scheduler.log"
        self.clock = Clock()

    def handler(self, **kwargs):
        result = TodayFileHandler(self.path, now=kwargs.get("now", self.clock))
        self.addCleanup(result.close)
        return result

    def emit(self, handler, message):
        record = logging.LogRecord("today-test", logging.INFO, __file__, 1, message, (), None)
        handler.handle(record)

    def text(self):
        return self.path.read_text(encoding="utf-8")

    def test_startup_keeps_only_current_day_records(self):
        self.path.write_text(
            "undated legacy prefix\n"
            "2026-11-06 23:59:59,999 yesterday\nold continuation\n"
            "2026-11-07 00:00:00,001 today\n"
            "2026-11-08 00:00:00 future\nfuture continuation\n"
            "2026-11-07 09:00:00 another today\n",
            encoding="utf-8",
        )
        self.handler()
        self.assertEqual(self.text(), "2026-11-07 00:00:00,001 today\n2026-11-07 09:00:00 another today\n")

    def test_same_day_file_bytes_and_mtime_are_unchanged(self):
        contents = "2026-11-07 10:00:00 中文\r\nTraceback:\r\n  continuation without final newline".encode("utf-8")
        self.path.write_bytes(contents)
        os.utime(self.path, ns=(1_650_000_000_000_000_000, 1_650_000_000_000_000_000))
        before = self.path.stat().st_mtime_ns
        self.handler()
        self.assertEqual(self.path.read_bytes(), contents)
        self.assertEqual(self.path.stat().st_mtime_ns, before)

    def test_legacy_backups_merge_in_rotation_order_and_are_removed(self):
        self.path.write_text("2026-11-07 12:00:00 base\n", encoding="utf-8")
        for suffix, hour in ((1, 11), (2, 10), (3, 9)):
            self.path.with_name(f"scheduler.log.{suffix}").write_text(
                f"2026-11-06 23:59:59 old\n2026-11-07 {hour:02d}:00:00 backup {suffix}\n", encoding="utf-8"
            )
        self.handler()
        self.assertEqual(self.text(),
                         "2026-11-07 09:00:00 backup 3\n"
                         "2026-11-07 10:00:00 backup 2\n"
                         "2026-11-07 11:00:00 backup 1\n"
                         "2026-11-07 12:00:00 base\n")
        self.assertFalse(any(self.path.with_name(f"scheduler.log.{suffix}").exists() for suffix in (1, 2, 3)))

    def test_missing_newline_between_files_does_not_join_records(self):
        self.path.with_name("scheduler.log.1").write_text("2026-11-07 10:00:00 backup", encoding="utf-8")
        self.path.write_text("2026-11-07 11:00:00 base", encoding="utf-8")
        self.handler()
        self.assertEqual(self.text(), "2026-11-07 10:00:00 backup\n2026-11-07 11:00:00 base")

    def test_multiline_record_retains_traceback_only_with_its_date(self):
        self.path.write_text(
            "2026-11-06 23:00:00 failure\nTraceback:\n  old frame\nValueError: old\n"
            "2026-11-07 01:00:00 failure\nTraceback:\n  current frame\nValueError: current\n\n",
            encoding="utf-8",
        )
        self.handler()
        self.assertEqual(self.text(), "2026-11-07 01:00:00 failure\nTraceback:\n  current frame\nValueError: current\n\n")

    def test_emit_crossing_midnight_replaces_yesterday_before_writing(self):
        self.clock.current = datetime(2026, 11, 7, 23, 59, 59, tzinfo=BEIJING)
        handler = self.handler()
        self.emit(handler, "before midnight")
        self.clock.current += timedelta(seconds=1)
        self.emit(handler, "after midnight")
        self.assertEqual(self.text(), "2026-11-08 00:00:00 after midnight\n")

    def test_paused_maintain_clears_file_without_new_log_event(self):
        handler = self.handler()
        self.emit(handler, "last event")
        self.clock.current += timedelta(days=1)
        self.assertTrue(handler.maintain())
        self.assertEqual(self.path.read_bytes(), b"")
        self.assertFalse(handler.maintain())

    def test_emit_after_paused_maintenance_reopens_the_current_file(self):
        handler = self.handler()
        self.emit(handler, "yesterday")
        self.clock.current += timedelta(days=1)
        handler.maintain()
        self.emit(handler, "today")
        self.assertEqual(self.text(), "2026-11-08 12:00:00 today\n")

    def test_utc_clock_is_converted_to_beijing_for_filter_and_output(self):
        self.clock.current = datetime(2026, 11, 7, 16, 0, 0, tzinfo=timezone.utc)
        self.path.write_text("2026-11-07 23:59:59 old\n2026-11-08 00:00:00 kept\n", encoding="utf-8")
        handler = self.handler()
        self.emit(handler, "UTC clock")
        self.assertEqual(self.text(), "2026-11-08 00:00:00 kept\n2026-11-08 00:00:00 UTC clock\n")

    def test_naive_injected_clock_means_beijing_without_host_timezone(self):
        self.clock.current = datetime(2026, 11, 7, 8, 9, 10)
        handler = self.handler()
        record = logging.LogRecord("timezone-test", logging.INFO, __file__, 1, "message %s", ("中文",), None)
        record.created = 0  # A host/record timestamp must never replace the supplied clock.
        with patch("time.localtime", side_effect=AssertionError("host timezone used")):
            handler.handle(record)
        self.assertEqual(self.text(), "2026-11-07 08:09:10 message 中文\n")
        self.assertFalse(hasattr(record, "_today_log_time"))

    def test_only_exact_log_and_three_known_backups_are_touched(self):
        unrelated = ["state.json", "active.lock", "scheduler.log.4", "scheduler.log.old", "other.log"]
        for name in unrelated:
            (self.directory / name).write_bytes(b"preserved")
        self.path.with_name("scheduler.log.1").mkdir()
        self.handler()
        self.assertTrue(self.path.with_name("scheduler.log.1").is_dir())
        for name in unrelated:
            self.assertEqual((self.directory / name).read_bytes(), b"preserved")

    def test_linked_backup_is_skipped_without_reading_or_removing_target(self):
        target = self.directory / "preserved.log"
        target.write_text("2026-11-07 09:00:00 external\n", encoding="utf-8")
        backup = self.path.with_name("scheduler.log.1")
        try:
            backup.symlink_to(target)
        except OSError:
            self.skipTest("Creating symlinks requires an unavailable Windows privilege")
        self.handler()
        self.assertTrue(backup.is_symlink())
        self.assertEqual(self.text(), "")
        self.assertEqual(target.read_text(encoding="utf-8"), "2026-11-07 09:00:00 external\n")

    def test_linked_base_is_refused(self):
        target = self.directory / "preserved.log"
        target.write_bytes(b"preserved")
        try:
            self.path.symlink_to(target)
        except OSError:
            self.skipTest("Creating symlinks requires an unavailable Windows privilege")
        with self.assertRaises(OSError):
            self.handler()
        self.assertEqual(target.read_bytes(), b"preserved")

    def test_windows_reparse_backup_is_skipped(self):
        backup = self.path.with_name("scheduler.log.1")
        backup.write_bytes(b"preserved reparse point")
        original = Path.lstat

        def metadata(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == backup:
                return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return result

        with patch.object(Path, "lstat", metadata):
            self.handler()
        self.assertEqual(backup.read_bytes(), b"preserved reparse point")

    def test_windows_reparse_base_is_refused_without_modifying_it(self):
        self.path.write_bytes(b"preserved reparse point")
        original = Path.lstat

        def metadata(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == self.path:
                return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return result

        with patch.object(Path, "lstat", metadata):
            with self.assertRaises(OSError):
                self.handler()
        self.assertEqual(self.path.read_bytes(), b"preserved reparse point")

    def test_linked_parent_is_refused_before_cleanup(self):
        self.path.write_bytes(b"preserved")
        original = Path.resolve

        def resolved(path, *args, **kwargs):
            if path == self.directory:
                return self.directory.parent / "outside-parent"
            return original(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolved):
            with self.assertRaises(OSError):
                self.handler()
        self.assertEqual(self.path.read_bytes(), b"preserved")

    def test_replacement_failure_preserves_main_log_and_backups(self):
        old = b"2026-11-06 10:00:00 old\n"
        backup = self.path.with_name("scheduler.log.1")
        self.path.write_bytes(old)
        backup.write_bytes(b"2026-11-07 11:00:00 today\n")
        with patch("pet_alliance.today_logging.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.handler()
        self.assertEqual(self.path.read_bytes(), old)
        self.assertTrue(backup.exists())
        self.assertFalse(list(self.directory.glob(".scheduler.log.*.tmp")))

    def test_concurrent_emit_and_maintenance_keep_complete_current_day_records(self):
        handler = self.handler()
        self.emit(handler, "yesterday")
        self.clock.current += timedelta(days=1)
        barrier = threading.Barrier(5)

        def write(worker):
            barrier.wait()
            for sequence in range(20):
                self.emit(handler, f"worker {worker} item {sequence}")

        workers = [threading.Thread(target=write, args=(worker,)) for worker in range(4)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for _ in range(20):
            handler.maintain()
        for worker in workers:
            worker.join(5)
            self.assertFalse(worker.is_alive())
        lines = self.text().splitlines()
        self.assertEqual(len(lines), 80)
        self.assertEqual(len(set(lines)), 80)
        self.assertTrue(all(line.startswith("2026-11-08 12:00:00 worker ") for line in lines))

    def test_closed_handler_does_not_reopen_or_clear_logs(self):
        handler = self.handler()
        self.emit(handler, "before close")
        handler.close()
        self.clock.current += timedelta(days=1)
        self.assertFalse(handler.maintain())
        self.emit(handler, "after close")
        self.assertEqual(self.text(), "2026-11-07 12:00:00 before close\n")


if __name__ == "__main__":
    unittest.main()
