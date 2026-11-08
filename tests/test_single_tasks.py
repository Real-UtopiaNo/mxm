from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from pet_alliance.adb import AutomationError
from pet_alliance.arena_state import persist_arena_state
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.scheduler import SchedulerController
from pet_alliance.scheduler_state import SchedulerState


class Context:
    def __init__(self):
        self.inputs = []
        self.captures = 0

    def capture(self):
        self.control("capture")
        self.captures += 1

    def tap(self, x=1, y=2):
        self.control("tap")
        self.inputs.append((x, y))

    def checkpoint(self, name, **payload):
        self.checkpoint_callback(name, payload)


class SingleTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "automation.toml"
        self.config.write_text('[automation]\nscripts_dir = "jobs"\n'
                               '[daily_hero]\nmin_click_interval = 0.001\n', encoding="utf-8")
        self.directory = self.root / "jobs"
        self.directory.mkdir()
        self.names = ("Daily_Hero", "daily_hero_reset", "check_arena", "slay_dragon", "capture_once")
        for name in self.names:
            (self.directory / f"{name}.py").write_text("raise RuntimeError('must not import catalog')\n", encoding="utf-8")
        self.current = datetime.fromisoformat("2026-11-07T08:00:00+08:00")
        self.ctx = Context()
        self.events = []
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers:
            controller.close()
            if controller._thread:
                controller._thread.join(3)
                self.assertFalse(controller._thread.is_alive())
        self.temp.cleanup()

    def controller(self, runner=None, script_runner=None):
        events = self.events

        class Navigator:
            def __init__(self, ctx):
                self.ctx = ctx

            def ensure_town(self):
                self.ctx.capture()
                events.append("town")

            def restart_to_town(self):
                raise AssertionError("Single runs must not perform midnight restart")

        kwargs = {} if script_runner is None else {"script_runner": script_runner}
        result = SchedulerController(self.config, now=lambda: self.current,
                                     context_factory=lambda controller: self.ctx,
                                     task_runner=runner or self.run_ok, navigator_factory=Navigator, **kwargs)
        self.controllers.append(result)
        return result

    def run_ok(self, name, ctx):
        self.events.append(name)
        return {"status": "completed"}

    def finish(self, controller):
        controller._thread.join(3)
        self.assertFalse(controller._thread.is_alive(), "Single task did not finish")

    @staticmethod
    def row(controller, name):
        return next(row for row in controller.snapshot()["scripts"] if row["name"] == name)

    def test_catalog_discovers_configured_directory_without_import_or_connection(self):
        (self.directory / "custom_job.py").write_text("raise RuntimeError('do not import')", encoding="utf-8")
        with patch("pet_alliance.scheduler.AndroidDevice", side_effect=AssertionError("connected")):
            controller = self.controller()
            snapshot = controller.snapshot()
        self.assertEqual({row["name"] for row in snapshot["scripts"]}, {*self.names, "custom_job"})
        self.assertEqual(self.row(controller, "Daily_Hero")["title"], "英雄级副本")
        self.assertEqual(self.row(controller, "custom_job")["status"], "idle")
        self.assertIsNone(snapshot["mode"])
        self.assertIsNone(controller._thread)

    def test_business_run_executes_once_and_never_chooses_another_task(self):
        controller = self.controller()
        with patch.object(controller, "cycle", side_effect=AssertionError("scheduled")):
            controller.start_script("Daily_Hero")
            self.finish(controller)
        self.assertEqual(self.events, ["town", "daily_hero"])
        self.assertTrue(controller.snapshot()["tasks"]["daily_hero"]["done"])
        self.assertEqual(self.row(controller, "Daily_Hero")["status"], "completed")
        self.assertIsNone(controller.snapshot()["single_script"])

    def test_only_one_worker_and_stop_only_applies_to_matching_script(self):
        ready, release = threading.Event(), threading.Event()

        def run(name, ctx):
            ready.set()
            self.assertTrue(release.wait(2))
            ctx.tap()
            return {"status": "completed"}

        controller = self.controller(run)
        controller.start_script("slay_dragon")
        self.assertTrue(ready.wait(2))
        original = controller._thread
        self.assertEqual(controller.snapshot()["mode"], "single")
        controller.stop_script("check_arena")
        self.assertFalse(controller._pause.is_set())
        with self.assertRaisesRegex(AutomationError, "已有任务"):
            controller.start_script("check_arena")
        controller.start()
        self.assertIs(controller._thread, original)
        controller.stop_script("slay_dragon")
        self.assertEqual(self.row(controller, "slay_dragon")["status"], "stopping")
        with self.assertRaises(AutomationError):
            controller.start_script("capture_once")
        release.set()
        self.finish(controller)
        self.assertEqual(self.ctx.inputs, [])
        self.assertEqual(self.row(controller, "slay_dragon")["status"], "paused")

    def test_capture_script_has_no_town_navigation_or_input(self):
        def capture(name, directory, ctx):
            self.assertEqual(name, "capture_once")
            self.assertEqual(directory, self.directory)
            ctx.capture()

        controller = self.controller(script_runner=capture)
        controller.start_script("capture_once")
        self.finish(controller)
        self.assertEqual(self.events, [])
        self.assertEqual(self.ctx.captures, 1)
        self.assertEqual(self.ctx.inputs, [])

    def test_scheduler_worker_also_blocks_single_start_until_fully_stopped(self):
        ready, release = threading.Event(), threading.Event()
        controller = self.controller()

        def cycle(ctx):
            ready.set()
            self.assertTrue(release.wait(2))

        with patch.object(controller, "cycle", side_effect=cycle):
            controller.start()
            self.assertTrue(ready.wait(2))
            self.assertEqual(controller.snapshot()["mode"], "scheduler")
            with self.assertRaises(AutomationError):
                controller.start_script("Daily_Hero")
            controller.pause()
            with self.assertRaises(AutomationError):
                controller.start_script("capture_once")
            release.set()
            self.finish(controller)
        self.assertIsNone(controller.snapshot()["mode"])

    def test_future_script_is_imported_only_when_started_and_uses_input_gate(self):
        (self.directory / "custom_job.py").write_text(
            "def run(ctx):\n    ctx.capture()\n    ctx.tap(7, 9)\n", encoding="utf-8")
        controller = self.controller()
        controller.start_script("custom_job")
        self.finish(controller)
        self.assertEqual(self.events, [])
        self.assertEqual(self.ctx.inputs, [(7, 9)])
        self.assertEqual(self.row(controller, "custom_job")["status"], "completed")

    def test_unknown_script_is_rejected_without_starting_worker(self):
        controller = self.controller()
        with self.assertRaisesRegex(AutomationError, "找不到脚本"):
            controller.start_script("missing")
        self.assertIsNone(controller._thread)

    def test_single_daily_crosses_midnight_without_preemption_and_records_start_day(self):
        self.current = self.current.replace(hour=23, minute=59)

        def run(name, ctx):
            self.current += timedelta(minutes=2)
            ctx.tap()
            ctx.checkpoint("hero_dungeon", report={"completed": ["A"]}, completed_since_arena=1)
            return {"status": "completed", "completed": ["A"]}

        controller = self.controller(run)
        controller.start_script("Daily_Hero")
        self.finish(controller)
        raw = controller.state.snapshot(self.current)
        self.assertTrue(raw["history"]["2026-11-07"]["daily_hero"]["done"])
        self.assertFalse(raw["daily"]["daily_hero"]["done"])
        self.assertEqual(raw["history"]["2026-11-07"]["daily_hero"]["progress"]["completed"], ["A"])
        self.assertEqual(raw["single_runs"]["Daily_Hero"]["date"], "2026-11-07")
        self.assertTrue(controller.state.midnight_restart_due(self.current))
        self.assertEqual(self.ctx.inputs, [(1, 2)])

    def test_done_daily_can_run_manually_and_pause_keeps_done_valid_on_reload(self):
        ready, release = threading.Event(), threading.Event()

        def run(name, ctx):
            self.assertEqual(ctx.resume_data, {})
            ready.set()
            self.assertTrue(release.wait(2))
            ctx.tap()

        controller = self.controller(run)
        token = controller.state.start_task("daily_hero", self.current)
        controller.state.finish_task("daily_hero", {"status": "completed", "completed": ["old"]},
                                     self.current, token)
        controller.start_script("Daily_Hero")
        self.assertTrue(ready.wait(2))
        running = SchedulerState(controller.state.path, self.current)
        self.assertTrue(running.daily_flags(self.current)["daily_hero"])
        controller.stop_script("Daily_Hero")
        release.set()
        self.finish(controller)
        raw = SchedulerState(controller.state.path, self.current).snapshot(self.current)
        self.assertTrue(raw["daily"]["daily_hero"]["done"])
        self.assertEqual(raw["daily"]["daily_hero"]["status"], "completed")
        self.assertEqual(raw["single_runs"]["Daily_Hero"]["status"], "paused")

    def test_pause_keeps_checkpoint_and_next_single_attempt_resumes_it(self):
        ready, release = threading.Event(), threading.Event()

        def run(name, ctx):
            ctx.checkpoint("hero_dungeon", report={"completed": ["A"]}, completed_since_arena=1)
            ready.set()
            self.assertTrue(release.wait(2))
            ctx.tap()

        controller = self.controller(run)
        controller.start_script("Daily_Hero")
        self.assertTrue(ready.wait(2))
        controller.stop_script("Daily_Hero")
        release.set()
        self.finish(controller)
        saved = controller.snapshot()["tasks"]["daily_hero"]
        self.assertEqual(saved["progress"]["completed"], ["A"])
        self.assertFalse(saved["done"])

        def resume(name, ctx):
            self.assertEqual(ctx.resume_data["completed"], ["A"])
            self.assertEqual(ctx.resume_data["completed_since_arena"], 1)
            return {"status": "completed", "completed": ["B"]}

        controller._task_runner = resume
        controller.start_script("Daily_Hero")
        self.finish(controller)
        self.assertEqual(controller.snapshot()["tasks"]["daily_hero"]["progress"]["completed"], ["A", "B"])

    def test_error_is_saved_and_same_controller_can_start_again(self):
        def fail(name, ctx):
            raise RuntimeError("broken run")

        controller = self.controller(fail)
        controller.start_script("check_arena")
        self.finish(controller)
        self.assertEqual(controller.snapshot()["status"], "error")
        self.assertEqual(self.row(controller, "check_arena")["reason"], "broken run")
        self.assertIsNone(controller.state.snapshot(self.current)["running"])
        controller._task_runner = self.run_ok
        controller.start_script("check_arena")
        self.finish(controller)
        self.assertEqual(self.row(controller, "check_arena")["status"], "completed")

    def test_closing_waits_for_worker_to_save_before_releasing_lease(self):
        ready, release = threading.Event(), threading.Event()

        def run(name, ctx):
            ready.set()
            self.assertTrue(release.wait(2))
            ctx.tap()

        controller = self.controller(run)
        controller.start_script("check_arena")
        self.assertTrue(ready.wait(2))
        controller.close()
        self.assertFalse(controller._lease.handle.closed)
        with self.assertRaisesRegex(AutomationError, "窗口正在关闭"):
            controller.start_script("capture_once")
        release.set()
        self.finish(controller)
        self.assertTrue(controller._lease.handle.closed)
        self.assertEqual(self.ctx.inputs, [])
        self.assertEqual(controller.snapshot()["status"], "closed")

    def test_arena_single_run_preserves_hero_sidework_and_expiry(self):
        expiry = (self.current + timedelta(hours=6, minutes=1)).isoformat()

        def run(name, ctx):
            return {"status": "completed", "arena_expired_time": expiry,
                    "hero_progress_by_day": {"2026-11-07": {"completed": ["魔都/COCO号"]}}}

        controller = self.controller(run)
        controller.start_script("check_arena")
        self.finish(controller)
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["arena_expired_time"], expiry)
        self.assertFalse(snapshot["tasks"]["daily_hero"]["done"])
        self.assertEqual(snapshot["tasks"]["daily_hero"]["progress"]["completed"], ["魔都/COCO号"])

    def test_completed_single_dailies_stay_skipped_by_scheduler_after_reopen(self):
        controller = self.controller()
        controller.state.set_arena(self.current + timedelta(hours=6), now=self.current)
        for name in ("Daily_Hero", "daily_hero_reset"):
            controller.start_script(name)
            self.finish(controller)
        controller.close()
        reopened = self.controller()
        self.assertTrue(reopened.snapshot()["tasks"]["daily_hero"]["done"])
        self.assertTrue(reopened.snapshot()["tasks"]["daily_hero_reset"]["done"])
        with patch.object(reopened, "_task_runner", side_effect=AssertionError("Repeated completed daily")):
            self.assertIsNone(reopened.cycle(reopened._context()))

    def test_partial_single_reset_resumes_in_scheduler_with_costs_and_points_counter(self):
        ready, release = threading.Event(), threading.Event()

        def partial(name, ctx):
            ctx.checkpoint("hero_dungeon", report={
                "completed": ["西部大陆/A"], "diamonds_spent": 50, "resets": 1,
                "diamond_spending": [{"region": "西部大陆", "diamonds": 50}],
                "points_arena_pending": True,
            }, completed_since_arena=2)
            ready.set()
            self.assertTrue(release.wait(2))
            ctx.tap()

        controller = self.controller(partial)
        token = controller.state.start_task("daily_hero", self.current)
        controller.state.finish_task("daily_hero", {"status": "completed"}, self.current, token)
        controller.state.set_arena(self.current + timedelta(hours=6), now=self.current)
        controller.start_script("daily_hero_reset")
        self.assertTrue(ready.wait(2))
        controller.stop_script("daily_hero_reset")
        release.set()
        self.finish(controller)
        controller.close()
        reopened = self.controller()

        def resume(task, ctx):
            self.assertEqual(task, "daily_hero_reset")
            self.assertEqual(ctx.resume_data["completed"], ["西部大陆/A"])
            self.assertEqual(ctx.resume_data["diamonds_spent"], 50)
            self.assertEqual(ctx.resume_data["resets"], 1)
            self.assertEqual(ctx.resume_data["completed_since_arena"], 2)
            self.assertTrue(ctx.resume_data["points_arena_pending"])
            return {"status": "completed", "completed": ["西部大陆/B"],
                    "diamonds_spent": 10, "resets": 1,
                    "diamond_spending": [{"region": "西部大陆", "diamonds": 10}],
                    "points_arena_pending": False}

        reopened._task_runner = resume
        self.assertEqual(reopened.cycle(reopened._context()), "daily_hero_reset")
        record = reopened.snapshot()["tasks"]["daily_hero_reset"]
        self.assertTrue(record["done"])
        self.assertEqual(record["progress"]["completed"], ["西部大陆/A", "西部大陆/B"])
        self.assertEqual(record["progress"]["diamonds_spent"], 60)
        self.assertEqual(record["progress"]["resets"], 2)

    def test_single_guard_expiry_survives_config_reload_and_scheduler_restart(self):
        expiry = self.current + timedelta(hours=6, minutes=1)
        self.ctx.project_root, self.ctx.config_path = self.root, self.config
        self.ctx.settings, self.ctx.use_local_config = {}, False

        def guard(task, ctx):
            persist_arena_state(ctx, expiry, "COCO号")
            return {"status": "completed", "arena_expired_time": expiry.isoformat()}

        controller = self.controller(guard)
        controller.start_script("check_arena")
        self.finish(controller)
        controller._reload()
        self.assertEqual(controller.snapshot()["arena_expired_time"], expiry.isoformat())
        self.assertEqual(controller._choose(), "daily_hero")
        controller.close()
        reopened = self.controller()
        self.assertEqual(reopened.snapshot()["arena_expired_time"], expiry.isoformat())
        self.assertEqual(reopened._choose(), "daily_hero")

    def test_single_arena_no_target_shares_scheduler_retry_interval_after_reopen(self):
        for status in ("blocked", "exhausted", "completed"):
            with self.subTest(status=status):
                controller = self.controller(lambda task, ctx: {
                    "status": status, "returned_to_town": True,
                })
                try:
                    controller.start_script("check_arena")
                    self.finish(controller)
                    expected = self.current + timedelta(minutes=5)
                    self.assertEqual(controller.state.snapshot(self.current)["arena"]["retry_after"],
                                     expected.isoformat())
                    self.assertEqual(controller._choose(), "daily_hero")
                finally:
                    controller.close()
                reopened = self.controller()
                try:
                    self.assertEqual(reopened._choose(), "daily_hero")
                    self.current = expected
                    self.assertEqual(reopened._choose(), "check_arena")
                finally:
                    reopened.close()

    def test_single_arena_failure_before_verified_town_does_not_suppress_retry(self):
        controller = self.controller(lambda task, ctx: {
            "status": "blocked", "returned_to_town": False,
        })
        controller.start_script("check_arena")
        self.finish(controller)
        self.assertIsNone(controller.state.snapshot(self.current)["arena"]["retry_after"])
        self.assertEqual(controller._choose(), "check_arena")


if __name__ == "__main__":
    unittest.main()
