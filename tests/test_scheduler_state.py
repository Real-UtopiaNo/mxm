from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pet_alliance.adb import AutomationError
from pet_alliance.dragon_schedule import BEIJING
from pet_alliance.scheduler_state import SchedulerState


class SchedulerStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "artifacts" / "scheduler" / "state.json"
        self.now = datetime(2026, 11, 6, 8, 30, tzinfo=BEIJING)
        self.state = SchedulerState(self.path, self.now)

    def test_initial_state_is_persisted_without_completed_work(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["date"], "2026-11-06")
        self.assertEqual(self.state.daily_flags(self.now), {"daily_hero": False, "daily_hero_reset": False})
        self.assertIsNone(data["running"])

    def test_new_state_starts_midnight_restart_baseline_today(self):
        self.assertFalse(self.state.midnight_restart_due(self.now))
        self.assertEqual(self.state.snapshot(self.now)["midnight_restart"],
                         {"last_completed_date": "2026-11-06"})
        restarted = SchedulerState(self.path, self.now + timedelta(hours=10))
        self.assertFalse(restarted.midnight_restart_due(self.now + timedelta(hours=10)))

    def test_midnight_restart_becomes_due_at_beijing_midnight(self):
        before = datetime(2026, 11, 6, 15, 59, 59, tzinfo=timezone.utc)
        after = before + timedelta(seconds=1)
        self.assertFalse(self.state.midnight_restart_due(before))
        self.assertTrue(self.state.midnight_restart_due(after))

    def test_ui_snapshot_and_rollover_do_not_consume_midnight_restart(self):
        tomorrow = self.now + timedelta(days=1)
        self.state.snapshot(tomorrow)
        self.state.daily_flags(tomorrow)
        self.state.refresh_day(tomorrow)
        self.assertTrue(self.state.midnight_restart_due(tomorrow))
        restarted = SchedulerState(self.path, tomorrow)
        self.assertTrue(restarted.midnight_restart_due(tomorrow))
        self.assertEqual(restarted.snapshot(tomorrow)["midnight_restart"]["last_completed_date"],
                         "2026-11-06")

    def test_legacy_state_baseline_is_original_date_before_rollover(self):
        data = self.state.snapshot(self.now)
        data.pop("midnight_restart")
        self.path.write_text(json.dumps(data), encoding="utf-8")
        tomorrow = self.now + timedelta(days=1)
        restarted = SchedulerState(self.path, tomorrow)
        self.assertTrue(restarted.midnight_restart_due(tomorrow))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["midnight_restart"],
                         {"last_completed_date": "2026-11-06"})

    def test_same_day_legacy_migration_is_persisted_without_immediate_restart(self):
        data = self.state.snapshot(self.now)
        data.pop("midnight_restart")
        self.path.write_text(json.dumps(data), encoding="utf-8")
        restarted = SchedulerState(self.path, self.now)
        self.assertFalse(restarted.midnight_restart_due(self.now))
        self.assertIn("midnight_restart", json.loads(self.path.read_text(encoding="utf-8")))

    def test_completed_midnight_restart_survives_same_day_reopen(self):
        tomorrow = self.now + timedelta(days=1)
        self.state.complete_midnight_restart("2026-11-07", tomorrow)
        self.assertFalse(self.state.midnight_restart_due(tomorrow))
        restarted = SchedulerState(self.path, tomorrow + timedelta(hours=10))
        self.assertFalse(restarted.midnight_restart_due(tomorrow + timedelta(hours=10)))
        record = restarted.snapshot(tomorrow)["midnight_restart"]
        self.assertEqual(record["last_completed_date"], "2026-11-07")
        self.assertEqual(record["completed_at"], "2026-11-07T08:30:00+08:00")

    def test_restart_failed_or_paused_without_completion_remains_due(self):
        tomorrow = self.now + timedelta(days=1)
        for _ in range(3):
            self.state = SchedulerState(self.path, tomorrow)
            self.state.snapshot(tomorrow)
            self.assertTrue(self.state.midnight_restart_due(tomorrow))
        self.state.complete_midnight_restart("2026-11-07", tomorrow)
        self.assertFalse(self.state.midnight_restart_due(tomorrow))

    def test_one_restart_on_resume_satisfies_several_missed_midnights(self):
        later = self.now + timedelta(days=5)
        self.assertTrue(self.state.midnight_restart_due(later))
        self.state.complete_midnight_restart("2026-11-11", later)
        self.assertFalse(self.state.midnight_restart_due(later))
        self.assertTrue(self.state.midnight_restart_due(later + timedelta(days=1)))

    def test_restart_crossing_next_midnight_does_not_consume_newer_day(self):
        ended = datetime(2026, 11, 8, 0, 0, 5, tzinfo=BEIJING)
        self.state.complete_midnight_restart("2026-11-07", ended)
        self.assertTrue(self.state.midnight_restart_due(ended))
        restarted = SchedulerState(self.path, ended)
        self.assertTrue(restarted.midnight_restart_due(ended))
        self.assertEqual(restarted.snapshot(ended)["midnight_restart"]["last_completed_date"],
                         "2026-11-07")

    def test_duplicate_or_old_restart_acknowledgements_do_not_regress_state(self):
        tomorrow = self.now + timedelta(days=1)
        self.state.complete_midnight_restart("2026-11-07", tomorrow)
        before = self.path.read_bytes()
        self.state.complete_midnight_restart("2026-11-07", tomorrow + timedelta(minutes=1))
        self.state.complete_midnight_restart("2026-11-06", tomorrow + timedelta(minutes=2))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.state.midnight_restart_due(self.now))
        self.assertFalse(self.state.midnight_restart_due(tomorrow))

    def test_invalid_midnight_restart_completion_preserves_state(self):
        before = self.path.read_bytes()
        snapshot = self.state.snapshot(self.now)
        for day in (None, 20261106, "", "20261106", "2026-W45-5", "2026-02-30", "2026-11-07"):
            with self.subTest(day=day), self.assertRaisesRegex(AutomationError, "零点重启完成日期无效"):
                self.state.complete_midnight_restart(day, self.now)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(self.state.snapshot(self.now), snapshot)

    def test_failed_midnight_restart_commit_keeps_saved_and_in_memory_barrier_due(self):
        tomorrow = self.now + timedelta(days=1)
        self.state.refresh_day(tomorrow)
        before = self.path.read_bytes()
        with patch("pet_alliance.scheduler_state.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.state.complete_midnight_restart("2026-11-07", tomorrow)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(self.state.midnight_restart_due(tomorrow))
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_corrupted_midnight_restart_record_is_preserved(self):
        valid = self.state.snapshot(self.now)
        for restart in (None, [], {}, {"last_completed_date": None},
                        {"last_completed_date": "20261106"}, {"last_completed_date": "2026-02-30"}):
            with self.subTest(restart=restart):
                data = {**valid, "midnight_restart": restart}
                text = json.dumps(data)
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(AutomationError, "保留原文件"):
                    SchedulerState(self.path, self.now)
                self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_completed_daily_survives_close_and_resume_same_day(self):
        token = self.state.start_task("daily_hero", self.now)
        self.state.finish_task("daily_hero", {"status": "completed", "completed": [{"region": "west"}]}, self.now, token)
        restarted = SchedulerState(self.path, self.now + timedelta(hours=10))
        self.assertTrue(restarted.daily_flags(self.now)["daily_hero"])
        with self.assertRaisesRegex(AutomationError, "今天已经完成"):
            restarted.start_task("daily_hero", self.now)

    def test_only_completed_marks_daily_done(self):
        for status in ("limited", "paused", "preempted", "stopped", "interrupted", "failed", "preview"):
            with self.subTest(status=status):
                token = self.state.start_task("daily_hero", self.now)
                self.state.finish_task("daily_hero", {"status": status, "completed": [1]}, self.now, token)
                self.assertFalse(self.state.daily_flags(self.now)["daily_hero"])

    def test_midnight_resets_both_daily_flags_and_keeps_history(self):
        for task in ("daily_hero", "daily_hero_reset"):
            token = self.state.start_task(task, self.now)
            self.state.finish_task(task, {"status": "completed"}, self.now, token)
        tomorrow = self.now.replace(day=7, hour=0, minute=0)
        data = self.state.snapshot(tomorrow)
        self.assertFalse(any(record["done"] for record in data["daily"].values()))
        self.assertTrue(all(record["done"] for record in data["history"]["2026-11-06"].values()))

    def test_beijing_midnight_uses_game_date_even_for_utc_clock(self):
        before = datetime(2026, 11, 6, 15, 59, 59, tzinfo=timezone.utc)
        after = before + timedelta(seconds=1)
        self.assertEqual(self.state.snapshot(before)["date"], "2026-11-06")
        self.assertEqual(self.state.snapshot(after)["date"], "2026-11-07")

    def test_cross_midnight_completion_belongs_to_start_date(self):
        started = self.now.replace(hour=23, minute=59)
        token = self.state.start_task("daily_hero_reset", started)
        ended = started + timedelta(minutes=2)
        data = self.state.finish_task("daily_hero_reset", {"status": "completed", "diamonds_spent": 50}, ended, token)
        self.assertEqual(data["date"], "2026-11-07")
        self.assertFalse(data["daily"]["daily_hero_reset"]["done"])
        yesterday = data["history"]["2026-11-06"]["daily_hero_reset"]
        self.assertTrue(yesterday["done"])
        self.assertEqual(yesterday["progress"]["diamonds_spent"], 50)

    def test_daily_progress_survives_pause_and_process_restart(self):
        token = self.state.start_task("daily_hero", self.now)
        progress = {"completed": [{"region": "西部大陆", "marker": [100, 200]}], "skipped": [],
                    "completed_since_arena": 2}
        self.state.record_progress("daily_hero", progress, self.now, attempt_id=token)
        self.state.finish_task("daily_hero", {"status": "paused"}, self.now, token)
        restarted = SchedulerState(self.path, self.now)
        record = restarted.snapshot(self.now)["daily"]["daily_hero"]
        self.assertEqual(record["progress"], progress)
        self.assertFalse(record["done"])
        self.assertEqual(record["status"], "paused")

    def test_process_crash_recovers_incomplete_attempt_without_losing_progress(self):
        token = self.state.start_task("daily_hero_reset", self.now)
        self.state.record_progress("daily_hero_reset", {"diamonds_spent": 50}, self.now, attempt_id=token)
        restarted = SchedulerState(self.path, self.now)
        data = restarted.snapshot(self.now)
        self.assertIsNone(data["running"])
        record = data["daily"]["daily_hero_reset"]
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(record["progress"], {"diamonds_spent": 50})
        self.assertFalse(record["done"])

    def test_crash_then_restart_next_day_does_not_move_old_progress_to_today(self):
        self.state.start_task("daily_hero", self.now)
        restarted = SchedulerState(self.path, self.now + timedelta(days=1))
        data = restarted.snapshot(self.now + timedelta(days=1))
        self.assertEqual(data["daily"]["daily_hero"]["status"], "pending")
        self.assertEqual(data["history"]["2026-11-06"]["daily_hero"]["status"], "interrupted")

    def test_stale_result_cannot_finish_or_modify_new_attempt(self):
        previous = self.state.start_task("daily_hero", self.now)
        self.state.finish_task("daily_hero", {"status": "paused"}, self.now, previous)
        current = self.state.start_task("daily_hero", self.now)
        with self.assertRaisesRegex(AutomationError, "不一致"):
            self.state.finish_task("daily_hero", {"status": "completed"}, self.now, previous)
        with self.assertRaisesRegex(AutomationError, "不一致"):
            self.state.record_progress("daily_hero", {}, self.now, attempt_id=previous)
        self.assertEqual(self.state.snapshot(self.now)["running"]["attempt_id"], current)

    def test_cannot_start_parallel_game_task(self):
        self.state.start_task("slay_dragon", self.now)
        with self.assertRaisesRegex(AutomationError, "已有调度任务"):
            self.state.start_task("check_arena", self.now)

    def test_arena_deadline_and_retry_persist_across_midnight(self):
        expiry = self.now + timedelta(hours=6)
        self.state.set_arena(expiry, self.now + timedelta(minutes=5), {"status": "completed"}, self.now)
        restarted = SchedulerState(self.path, self.now + timedelta(days=1))
        arena = restarted.snapshot(self.now + timedelta(days=1))["arena"]
        self.assertEqual(arena["next_due"], expiry.isoformat(timespec="seconds"))
        self.assertEqual(arena["retry_after"], "2026-11-06T08:35:00+08:00")
        self.assertTrue(arena["checked"])

    def test_arena_completed_report_updates_expiry_and_clears_backoff(self):
        self.state.backoff_arena(now=self.now)
        token = self.state.start_task("check_arena", self.now)
        report = {"status": "completed", "arena_expired_time": "2026-11-06T10:00:00+08:00"}
        data = self.state.finish_task("check_arena", report, self.now, token)
        self.assertEqual(data["arena"]["next_due"], report["arena_expired_time"])
        self.assertIsNone(data["arena"]["retry_after"])
        self.assertEqual(data["arena"]["latest_report"], report)

    def test_arena_backoff_retains_previous_deadline_and_reports(self):
        self.state.set_arena(self.now, report={"status": "exhausted"}, now=self.now)
        self.assertEqual(self.state.backoff_arena(now=self.now), "2026-11-06T08:35:00+08:00")
        arena = self.state.snapshot(self.now)["arena"]
        self.assertEqual(arena["next_due"], self.now.isoformat(timespec="seconds"))
        self.assertEqual(arena["latest_report"], {"status": "exhausted"})

    def test_non_daily_task_has_last_attempt_and_report(self):
        token = self.state.start_task("slay_dragon", self.now)
        self.state.finish_task("slay_dragon", {"status": "preempted", "checks": 5}, self.now, token)
        record = self.state.snapshot(self.now)["tasks"]["slay_dragon"]
        self.assertEqual(record["last_attempt"], self.now.isoformat(timespec="seconds"))
        self.assertEqual(record["latest_report"]["checks"], 5)
        self.assertEqual(record["status"], "preempted")

    def test_snapshot_and_progress_do_not_expose_mutable_state(self):
        token = self.state.start_task("daily_hero", self.now)
        progress = {"completed": [1]}
        self.state.record_progress("daily_hero", progress, self.now, attempt_id=token)
        progress["completed"].append(2)
        data = self.state.snapshot(self.now)
        data["daily"]["daily_hero"]["progress"]["completed"].append(3)
        self.assertEqual(self.state.snapshot(self.now)["daily"]["daily_hero"]["progress"]["completed"], [1])

    def test_corrupted_state_is_preserved_and_does_not_reset_completion(self):
        for text in ("{broken", "{}", '{"version": 2}'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(AutomationError, "保留原文件"):
                    SchedulerState(self.path, self.now)
                self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_inconsistent_completed_record_is_rejected(self):
        value = self.state.snapshot(self.now)
        value["daily"]["daily_hero"]["done"] = True
        value["daily"]["daily_hero"]["status"] = "paused"
        self.path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(AutomationError, "完成状态不一致"):
            SchedulerState(self.path, self.now)

    def test_atomic_replace_failure_leaves_original_state_and_no_temporary_file(self):
        previous = self.path.read_bytes()
        with patch("pet_alliance.scheduler_state.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.state.start_task("daily_hero", self.now)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_invalid_backoff_is_rejected(self):
        for seconds in (0, -1, float("inf"), float("nan"), True, "300"):
            with self.subTest(seconds=seconds), self.assertRaises(AutomationError):
                self.state.backoff_arena(seconds, self.now)

    def test_sidework_records_progress_without_completing_daily_task(self):
        progress = {"completed": [{"region": "魔都", "dungeon": "恶魔城"}],
                    "diamonds_spent": 0, "completed_since_arena": 1}
        self.state.record_daily_sidework("daily_hero", progress, self.now)
        record = self.state.snapshot(self.now)["daily"]["daily_hero"]
        self.assertEqual(record["progress"], progress)
        self.assertFalse(record["done"])
        self.assertEqual(record["status"], "pending")
        self.assertIsNone(record["last_attempt"])
        self.assertIsNone(record["latest_report"])

    def test_sidework_does_not_interrupt_enclosing_arena_attempt(self):
        token = self.state.start_task("check_arena", self.now)
        before = self.state.snapshot(self.now)
        self.state.record_daily_sidework("daily_hero", {"completed_dungeons": 1}, self.now,
                                         attempt_id=token)
        data = self.state.snapshot(self.now)
        self.assertEqual(data["running"], before["running"])
        self.assertEqual(data["arena"], before["arena"])
        finished = self.state.finish_task("check_arena", {"status": "completed"}, self.now, token)
        self.assertEqual(finished["daily"]["daily_hero"]["progress"]["completed_dungeons"], 1)
        self.assertFalse(finished["daily"]["daily_hero"]["done"])

    def test_sidework_cross_midnight_only_updates_explicit_work_date(self):
        token = self.state.start_task("check_arena", self.now.replace(hour=23, minute=59))
        tomorrow = self.now.replace(day=7, hour=0, minute=1)
        self.state.record_daily_sidework("daily_hero", {"completed_dungeons": 1}, tomorrow,
                                         day="2026-11-06", attempt_id=token)
        data = self.state.snapshot(tomorrow)
        self.assertEqual(data["daily"]["daily_hero"]["progress"], {})
        old_record = data["history"]["2026-11-06"]["daily_hero"]
        self.assertEqual(old_record["progress"], {"completed_dungeons": 1})
        self.assertFalse(old_record["done"])
        self.assertEqual(data["running"]["attempt_id"], token)
        self.assertEqual(data["running"]["task"], "check_arena")

    def test_sidework_defaults_to_current_work_date_not_arena_start_date(self):
        token = self.state.start_task("check_arena", self.now)
        tomorrow = self.now + timedelta(days=1)
        self.state.record_daily_sidework("daily_hero", {"completed_dungeons": 1}, tomorrow,
                                         attempt_id=token)
        data = self.state.snapshot(tomorrow)
        self.assertEqual(data["daily"]["daily_hero"]["progress"], {"completed_dungeons": 1})
        self.assertEqual(data["history"]["2026-11-06"]["daily_hero"]["progress"], {})

    def test_sidework_keeps_completed_daily_status_and_report(self):
        token = self.state.start_task("daily_hero_reset", self.now)
        report = {"status": "completed", "diamonds_spent": 50, "completed_dungeons": 1}
        self.state.finish_task("daily_hero_reset", report, self.now, token)
        before = self.state.snapshot(self.now)["daily"]["daily_hero_reset"]
        progress = {"diamonds_spent": 100, "completed_dungeons": 2}
        self.state.record_daily_sidework("daily_hero_reset", progress, self.now)
        record = self.state.snapshot(self.now)["daily"]["daily_hero_reset"]
        self.assertEqual(record, {**before, "progress": progress})
        self.assertTrue(record["done"])
        self.assertEqual(record["status"], "completed")

    def test_sidework_full_progress_replacement_is_idempotent_and_copied(self):
        progress = {"completed": [{"dungeon": "COCO号"}], "diamonds_spent": 50}
        for _ in range(2):
            self.state.record_daily_sidework("daily_hero", progress, self.now)
        progress["completed"].append({"dungeon": "恶魔城"})
        actual = self.state.snapshot(self.now)["daily"]["daily_hero"]["progress"]
        self.assertEqual(actual, {"completed": [{"dungeon": "COCO号"}], "diamonds_spent": 50})
        restarted = SchedulerState(self.path, self.now)
        self.assertEqual(restarted.snapshot(self.now)["daily"]["daily_hero"]["progress"], actual)

    def test_stale_sidework_token_cannot_modify_another_arena_attempt(self):
        previous = self.state.start_task("check_arena", self.now)
        self.state.finish_task("check_arena", {"status": "paused"}, self.now, previous)
        current = self.state.start_task("check_arena", self.now)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(AutomationError, "不一致"):
            self.state.record_daily_sidework("daily_hero", {"completed_dungeons": 1}, self.now,
                                             attempt_id=previous)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.state.snapshot(self.now)["running"]["attempt_id"], current)
        self.assertEqual(self.state.snapshot(self.now)["daily"]["daily_hero"]["progress"], {})

    def test_sidework_token_requires_an_active_attempt(self):
        before = self.path.read_bytes()
        with self.assertRaisesRegex(AutomationError, "不一致"):
            self.state.record_daily_sidework("daily_hero", {}, self.now, attempt_id="old")
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_sidework_does_not_modify_memory_or_saved_state(self):
        before = self.path.read_bytes()
        snapshot = self.state.snapshot(self.now)
        cases = [
            {"task": "check_arena", "progress": {}},
            {"task": "unknown", "progress": {}},
            {"task": "daily_hero", "progress": []},
            {"task": "daily_hero", "progress": {"number": float("nan")}},
            {"task": "daily_hero", "progress": {"nested": {"value": object()}}},
        ]
        cases.extend({"task": "daily_hero", "progress": {}, "day": day}
                     for day in ("", "20261106", "2026-W45-5", "2026-02-30", "2026-11-07", 20261106))
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises((AutomationError, ValueError, TypeError)):
                self.state.record_daily_sidework(now=self.now, **kwargs)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(self.state.snapshot(self.now), snapshot)

    def test_failed_sidework_atomic_replace_preserves_original_file(self):
        before = self.path.read_bytes()
        with patch("pet_alliance.scheduler_state.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.state.record_daily_sidework("daily_hero", {"completed_dungeons": 1}, self.now)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
