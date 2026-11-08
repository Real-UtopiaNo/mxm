from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pet_alliance.daily_hero import DailyHero
from pet_alliance.run_logs import RunLogRetention
from pet_alliance.runtime import ScriptContext
from pet_alliance.scheduler import SchedulerController
from pet_alliance.slay_dragon import SlayDragon


PROJECT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self, value="2026-11-07T11:00:00+08:00"):
        self.current = datetime.fromisoformat(value)

    def __call__(self):
        return self.current

    def advance(self, **delta):
        self.current += timedelta(**delta)


class RunLogRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.retention = RunLogRetention(self.root, now=self.clock)

    def run_directory(self, name):
        path = self.root / "artifacts" / "runs" / name
        path.mkdir(parents=True)
        (path / "events.jsonl").write_text("diagnostics", encoding="utf-8")
        return path

    def test_only_known_valid_past_run_directories_are_removed(self):
        old = self.run_directory("Daily_Hero_20261106_234500_123456")
        keep = [self.run_directory(name) for name in (
            "Slay_Dragon_20261107_000001_123456",
            "Pet_Loadout_20261108_000001_123456",
            "manual_backup", "Daily_Hero_20260230_000001_123456",
            "Daily_Hero_20261106_999999_123456",
        )]
        ordinary_file = old.parent / "Check_Arena_20261106_120000_123456"
        ordinary_file.write_text("manual file", encoding="utf-8")
        result = self.retention.maintain()
        self.assertEqual(result.removed, 1)
        self.assertEqual(result.errors, [])
        self.assertFalse(old.exists())
        self.assertTrue(all(path.exists() for path in keep))
        self.assertTrue(ordinary_file.exists())

    def test_configuration_progress_input_screenshots_and_unknown_files_survive(self):
        self.run_directory("Check_Arena_20261106_120000_123456")
        names = ["automation.toml", "automation.local.toml", "artifacts/scheduler/state.json",
                 "artifacts/scheduler/active.lock", "artifacts/temp/17.png",
                 "artifacts/screenshots/capture_once_20261106_120000_123456.png",
                 "artifacts/diagnostics/reset_cost_20261106_060217_raw.png"]
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        self.retention.maintain()
        for name in names:
            self.assertEqual((self.root / name).read_bytes(), name.encode())

    def test_custom_state_inside_old_run_is_protected(self):
        old = self.run_directory("Check_Arena_20261106_120000_123456")
        state = old / "custom_state.json"
        state.write_bytes(b"confirmed work")
        retention = RunLogRetention(self.root, now=self.clock, protected_paths=(state,))
        self.assertEqual(retention.maintain().removed, 0)
        self.assertEqual(state.read_bytes(), b"confirmed work")

    def test_lexical_state_path_is_protected_even_when_target_resolves_elsewhere(self):
        old = self.run_directory("Check_Arena_20261106_120000_123456")
        state = old / "custom_state.json"
        outside = self.root / "external_state.json"
        resolve = Path.resolve

        def resolved(path, *args, **kwargs):
            return outside if path == state else resolve(path, *args, **kwargs)

        # Model a Windows linked state file without requiring symlink privilege.
        with patch.object(Path, "resolve", resolved):
            retention = RunLogRetention(self.root, now=self.clock, protected_paths=(state,))
            self.assertEqual(retention.maintain().removed, 0)
        self.assertTrue(old.exists())

    def test_repeated_checks_do_not_scan_again_until_beijing_day_changes(self):
        today = self.run_directory("Slay_Dragon_20261107_000001_123456")
        with patch("pet_alliance.run_logs.shutil.rmtree", wraps=__import__("shutil").rmtree) as remove:
            self.retention.maintain()
            self.retention.maintain()
            remove.assert_not_called()
            self.clock.advance(days=1)
            self.retention.maintain()
            remove.assert_called_once_with(today)

    def test_utc_clock_uses_beijing_date(self):
        self.clock.current = datetime.fromisoformat("2026-11-06T16:00:01+00:00")
        old = self.run_directory("Daily_Hero_20261106_235959_123456")
        today = self.run_directory("Daily_Hero_20261107_000001_123456")
        self.retention.maintain()
        self.assertFalse(old.exists())
        self.assertTrue(today.exists())

    def test_delete_failure_preserves_task_and_can_retry(self):
        old = self.run_directory("Daily_Hero_20261106_120000_123456")
        with patch("pet_alliance.run_logs.shutil.rmtree", side_effect=PermissionError("busy")):
            result = self.retention.maintain()
        self.assertEqual(result.removed, 0)
        self.assertEqual(len(result.errors), 1)
        self.assertTrue(old.exists())
        self.assertEqual(self.retention.maintain(force=True).removed, 1)

    def test_linked_run_never_deletes_external_data(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("safe")
        self.retention.root.mkdir(parents=True)
        link = self.retention.root / "Daily_Hero_20261106_120000_123456"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        self.assertEqual(self.retention.maintain().removed, 0)
        self.assertEqual((outside / "keep.txt").read_text(), "safe")

    def test_linked_root_is_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        old = outside / "Daily_Hero_20261106_120000_123456"
        old.mkdir()
        self.retention.root.parent.mkdir(parents=True)
        try:
            self.retention.root.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        result = self.retention.maintain()
        self.assertEqual(result.removed, 0)
        self.assertEqual(len(result.errors), 1)
        self.assertTrue(old.exists())

    def test_hero_and_dragon_roll_diagnostics_without_losing_in_memory_progress(self):
        self.clock.current = datetime.fromisoformat("2026-11-07T23:59:59+08:00")
        context = SimpleNamespace(settings={}, project_root=PROJECT, run_logs=self.retention)
        for job_type in (DailyHero, SlayDragon):
            with self.subTest(job=job_type.__name__):
                self.clock.current = datetime.fromisoformat("2026-11-07T23:59:59+08:00")
                job = job_type(context)
                job.event("yesterday")
                before = job.run_dir
                job.report.reason = "confirmed work retained"
                self.clock.advance(seconds=2)
                # Maintenance may run in a wait before the job writes again.
                self.retention.maintain(force=True)
                job.event("today")
                (job.run_dir / "summary.json").write_text("today summary")
                self.assertFalse(before.exists())
                self.assertIn("_20261108_", job.run_dir.name)
                events = (job.run_dir / "events.jsonl").read_text(encoding="utf-8")
                self.assertIn('"today"', events)
                self.assertNotIn('"yesterday"', events)
                self.assertEqual(job.report.reason, "confirmed work retained")

    def test_directory_and_sweep_share_one_time_sample_at_midnight(self):
        before = datetime.fromisoformat("2026-11-07T23:59:59+08:00")
        after = datetime.fromisoformat("2026-11-08T00:00:00+08:00")
        clock = Mock(side_effect=(before, after, after))
        retention = RunLogRetention(self.root, now=clock)
        context = SimpleNamespace(settings={}, project_root=PROJECT, run_logs=retention)
        job = DailyHero(context)
        directories = list(retention.root.iterdir())
        self.assertEqual(len(directories), 1)
        self.assertTrue(directories[0].exists())
        self.assertEqual(clock.call_count, 1)
        job.event("today")
        self.assertIn("_20261108_", job.run_dir.name)

    def test_standalone_wait_cleans_at_midnight_without_any_device_input(self):
        self.clock.current = datetime.fromisoformat("2026-11-07T23:59:50+08:00")
        today = self.run_directory("Slay_Dragon_20261107_120000_123456")
        context = ScriptContext(None, self.root / "screenshots", run_logs=self.retention)
        with patch("pet_alliance.runtime.time.sleep", side_effect=lambda s: self.clock.advance(seconds=s)) as sleep:
            context.wait(90)
        self.assertFalse(today.exists())
        self.assertEqual(sleep.call_count, 3)

    def test_paused_scheduler_tick_purges_logs_without_changing_completed_work(self):
        config = self.root / "automation.toml"
        config.write_text("[scheduler]\npoll_interval = 1\n", encoding="utf-8")
        old = self.run_directory("Daily_Hero_20261106_120000_123456")
        controller = SchedulerController(config, now=self.clock)
        self.addCleanup(controller.close)
        self.assertFalse(old.exists())
        attempt = controller.state.start_task("daily_hero", self.clock())
        controller.state.finish_task("daily_hero", {"status": "completed"}, self.clock(), attempt_id=attempt)
        self.assertTrue(controller.snapshot()["tasks"]["daily_hero"]["done"])
        today = self.run_directory("Daily_Hero_20261107_120000_123456")
        controller.log.info("yesterday line")
        self.clock.advance(days=1)
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["status"], "paused")
        self.assertFalse(today.exists())
        content = (self.root / "artifacts/scheduler/scheduler.log").read_text(encoding="utf-8")
        self.assertNotIn("yesterday line", content)
        history = controller.state.snapshot(self.clock())["history"]
        self.assertTrue(history["2026-11-07"]["daily_hero"]["done"])


if __name__ == "__main__":
    unittest.main()
