from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pet_alliance.adb import AutomationError
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.scheduler import SchedulerController


class MidnightClock:
    def __init__(self, value="2026-11-06T23:59:50+08:00"):
        self.current = datetime.fromisoformat(value)

    def __call__(self):
        return self.current

    def advance(self, **duration):
        self.current += timedelta(**duration)


class MidnightContext:
    def __init__(self, events):
        self.events = events
        self.inputs = []

    def capture(self):
        self.control("capture")
        self.events.append("capture")

    def tap(self, x=1, y=2):
        self.control("tap")
        self.inputs.append((x, y))
        self.events.append("tap")

    def checkpoint(self, name, **payload):
        self.checkpoint_callback(name, payload)


class MidnightRestartControllerTests(unittest.TestCase):
    """Exercise the persisted midnight gate without a device or live state."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config = self.directory / "automation.toml"
        self.config.write_text(
            "[scheduler]\npoll_interval = 0.01\narena_retry_seconds = 300\n"
            "[daily_hero]\nmin_click_interval = 0.001\n", encoding="utf-8")
        self.clock = MidnightClock()
        self.events = []
        self.context = MidnightContext(self.events)
        self.controllers = []
        self.restart_action = None

    def tearDown(self):
        for controller in reversed(self.controllers):
            controller.close()
            if controller._thread:
                controller._thread.join(3)
                self.assertFalse(controller._thread.is_alive())
        self.temporary.cleanup()

    def controller(self, runner=None):
        test = self

        class Navigator:
            def __init__(self, context):
                self.context = context

            def ensure_town(self):
                test.events.append("ensure_existing_town")
                self.context.capture()

            def restart_to_town(self):
                test.events.append("restart_begin")
                self.context.control("restart_app")
                if test.restart_action:
                    test.restart_action(self.context)
                self.context.capture()
                test.events.append("restart_town_confirmed")

        def default_runner(task, context):
            test.events.append(task)
            return {"status": "completed"}

        controller = SchedulerController(
            self.config, now=self.clock,
            context_factory=lambda owner: self.context,
            task_runner=runner or default_runner,
            navigator_factory=Navigator)
        self.controllers.append(controller)
        controller._context()
        return controller

    def future_arena(self, controller):
        controller.state.set_arena(self.clock() + timedelta(hours=12), now=self.clock())

    def complete_daily(self, controller, task, **report):
        attempt = controller.state.start_task(task, self.clock())
        controller.state.finish_task(task, {"status": "completed", **report},
                                     self.clock(), attempt)

    @staticmethod
    def attach_report(error, **report):
        error.report = {"status": error.status, **report}
        error.run_dir = "midnight_test_run"
        return error

    def test_new_store_does_not_restart_on_its_initial_day(self):
        controller = self.controller()
        self.future_arena(controller)
        self.assertFalse(controller.state.midnight_restart_due(self.clock()))

        self.assertEqual(controller.cycle(self.context), "daily_hero")

        self.assertNotIn("restart_begin", self.events)
        self.assertTrue(controller.snapshot()["tasks"]["daily_hero"]["done"])

    def test_idle_crossing_midnight_restarts_before_any_old_screen_navigation(self):
        controller = self.controller()
        self.future_arena(controller)
        self.complete_daily(controller, "daily_hero")
        self.complete_daily(controller, "daily_hero_reset")
        self.assertIsNone(controller.cycle(self.context))
        self.events.clear()
        self.clock.advance(seconds=20)

        self.assertEqual(controller.cycle(self.context), "daily_hero")

        self.assertEqual(self.events[0], "restart_begin")
        self.assertEqual(self.events.count("restart_begin"), 1)
        self.assertLess(self.events.index("restart_town_confirmed"),
                        self.events.index("daily_hero"))
        self.assertFalse(controller.state.midnight_restart_due(self.clock()))

    def test_hero_is_interrupted_at_next_input_without_waiting_for_safe_checkpoint(self):
        calls = []

        def run(task, context):
            calls.append(task)
            if len(calls) == 1:
                self.clock.advance(seconds=20)
                try:
                    context.tap(30, 40)
                except TaskInterrupted as error:
                    raise self.attach_report(error, completed=["old_day_dungeon"], victories=5)
                self.fail("Midnight allowed another battle input")
            return {"status": "completed"}

        controller = self.controller(run)
        self.future_arena(controller)
        controller.cycle(self.context)

        saved = controller.state.snapshot(self.clock())
        old = saved["history"]["2026-11-06"]["daily_hero"]
        self.assertEqual(old["status"], "preempted")
        self.assertEqual(old["progress"]["completed"], ["old_day_dungeon"])
        self.assertFalse(old["done"])
        self.assertEqual(saved["daily"]["daily_hero"]["progress"], {})
        self.assertEqual(self.context.inputs, [])
        self.assertNotIn("restart_begin", self.events)

        controller.cycle(self.context)
        self.assertEqual(calls, ["daily_hero", "daily_hero"])
        self.assertEqual(self.events.count("restart_begin"), 1)

    def test_arena_is_also_interrupted_at_midnight_and_restart_precedes_due_arena(self):
        calls = []

        def run(task, context):
            calls.append(task)
            self.events.append(task)
            if len(calls) == 1:
                self.clock.advance(seconds=20)
                try:
                    context.tap(70, 80)
                except TaskInterrupted as error:
                    raise self.attach_report(error, checked=["COCO号"])
                self.fail("Arena bypassed the midnight restart barrier")
            self.assertFalse(controller.state.midnight_restart_due(self.clock()))
            return {"status": "completed", "claimed": True}

        controller = self.controller(run)
        self.assertEqual(controller.cycle(self.context), "check_arena")
        saved = controller.state.snapshot(self.clock())
        self.assertEqual(saved["arena"]["status"], "preempted")
        self.assertIsNone(saved["running"])
        self.assertEqual(self.context.inputs, [])
        self.events.clear()

        self.assertEqual(controller.cycle(self.context), "check_arena")

        self.assertEqual(self.events[0], "restart_begin")
        self.assertLess(self.events.index("restart_town_confirmed"),
                        self.events.index("check_arena"))
        self.assertEqual(calls, ["check_arena", "check_arena"])

    def test_paused_midnight_is_deferred_and_persisted_across_same_day_recreation(self):
        controller = self.controller()
        self.future_arena(controller)
        controller.pause()
        self.clock.advance(seconds=20)
        # UI refresh must not acknowledge the still-pending restart.
        self.assertEqual(controller.snapshot()["date"], "2026-11-07")
        self.assertTrue(controller.state.midnight_restart_due(self.clock()))
        with self.assertRaises(TaskInterrupted) as stopped:
            controller.cycle(self.context)
        self.assertEqual(stopped.exception.status, "paused")
        self.assertEqual(self.events, [])
        controller.close()

        replacement = self.controller()
        replacement.cycle(self.context)
        self.assertEqual(self.events.count("restart_begin"), 1)
        replacement.close()

        same_day = self.controller()
        same_day.cycle(self.context)
        self.assertEqual(self.events.count("restart_begin"), 1)
        self.assertFalse(same_day.state.midnight_restart_due(self.clock()))

    def test_restart_failure_leaves_barrier_pending_and_never_starts_a_task(self):
        controller = self.controller()
        self.future_arena(controller)
        self.clock.advance(seconds=20)

        def fail_restart(context):
            raise AutomationError("登录尚未到达天星镇")

        self.restart_action = fail_restart
        with self.assertRaisesRegex(AutomationError, "尚未到达"):
            controller.cycle(self.context)
        self.assertTrue(controller.state.midnight_restart_due(self.clock()))
        self.assertNotIn("restart_town_confirmed", self.events)
        self.assertNotIn("ensure_existing_town", self.events)
        self.assertNotIn("daily_hero", self.events)
        self.assertIsNone(controller.state.snapshot(self.clock())["running"])

        self.restart_action = None
        controller.cycle(self.context)
        self.assertEqual(self.events.count("restart_begin"), 2)
        self.assertFalse(controller.state.midnight_restart_due(self.clock()))

    def test_pause_during_restart_does_not_commit_or_allow_another_input(self):
        controller = self.controller()
        self.future_arena(controller)
        self.clock.advance(seconds=20)

        def pause_during_login(context):
            controller.pause()
            context.tap()

        self.restart_action = pause_during_login
        with self.assertRaises(TaskInterrupted) as stopped:
            controller.cycle(self.context)
        self.assertEqual(stopped.exception.status, "paused")
        self.assertTrue(controller.state.midnight_restart_due(self.clock()))
        self.assertEqual(self.context.inputs, [])
        self.assertNotIn("restart_town_confirmed", self.events)
        self.assertNotIn("daily_hero", self.events)

        controller._pause.clear()
        self.restart_action = None
        controller.cycle(self.context)
        self.assertEqual(self.events.count("restart_begin"), 2)
        self.assertFalse(controller.state.midnight_restart_due(self.clock()))

    def test_acknowledges_only_after_town_and_preserves_daily_history_and_costs(self):
        controller = self.controller()
        self.future_arena(controller)
        self.complete_daily(controller, "daily_hero", completed=["old_hero"])
        self.complete_daily(controller, "daily_hero_reset", resets=1, diamonds_spent=50)
        old_daily = deepcopy(controller.state.snapshot(self.clock())["daily"])
        self.clock.advance(seconds=20)
        # Simulate confirmed work saved while this midnight barrier was pending.
        self.complete_daily(controller, "daily_hero", completed=["today_hero"])
        self.complete_daily(controller, "daily_hero_reset", resets=2, diamonds_spent=20)
        today_daily = deepcopy(controller.state.snapshot(self.clock())["daily"])

        def inspect_unconfirmed(context):
            self.assertTrue(controller.state.midnight_restart_due(self.clock()))
            saved = controller.state.snapshot(self.clock())
            self.assertEqual(saved["daily"], today_daily)
            self.assertEqual(saved["history"]["2026-11-06"], old_daily)

        self.restart_action = inspect_unconfirmed
        self.assertIsNone(controller.cycle(self.context))

        saved = controller.state.snapshot(self.clock())
        self.assertEqual(self.events[0], "restart_begin")
        self.assertIn("restart_town_confirmed", self.events)
        self.assertFalse(controller.state.midnight_restart_due(self.clock()))
        self.assertEqual(saved["daily"], today_daily)
        self.assertEqual(saved["history"]["2026-11-06"], old_daily)
        self.assertIsNone(saved["running"])

    def test_ordinary_town_navigation_crossing_midnight_stops_old_input_and_restarts_same_cycle(self):
        controller = self.controller()
        self.future_arena(controller)
        test = self

        class CrossingNavigator:
            def __init__(self, context):
                self.context = context

            def ensure_town(self):
                test.events.append("ensure_before_midnight")
                test.clock.advance(seconds=20)
                self.context.tap(90, 100)
                test.fail("Old-page navigation continued after midnight")

            def restart_to_town(self):
                test.events.append("restart_begin")
                self.context.control("restart_app")
                self.context.capture()
                test.events.append("restart_town_confirmed")

        controller._navigator_factory = CrossingNavigator

        self.assertEqual(controller.cycle(self.context), "daily_hero")

        self.assertEqual(self.context.inputs, [])
        self.assertEqual(self.events[0], "ensure_before_midnight")
        self.assertEqual(self.events.count("restart_begin"), 1)
        self.assertLess(self.events.index("restart_town_confirmed"),
                        self.events.index("daily_hero"))
        self.assertFalse(controller.state.midnight_restart_due(self.clock()))
        self.assertTrue(controller.snapshot()["tasks"]["daily_hero"]["done"])

    def test_restart_crossing_another_midnight_completes_each_start_date_once(self):
        controller = self.controller()
        controller.state.set_arena(self.clock() + timedelta(days=3), now=self.clock())
        self.clock.current = datetime.fromisoformat("2026-11-07T23:59:50+08:00")
        started_days = []

        def cross_another_midnight(context):
            started_days.append(self.clock().date().isoformat())
            self.assertTrue(controller.state.midnight_restart_due(self.clock()))
            if len(started_days) == 1:
                self.clock.advance(seconds=20)
            else:
                self.assertEqual(
                    controller.state.snapshot(self.clock())["midnight_restart"]["last_completed_date"],
                    "2026-11-07")

        self.restart_action = cross_another_midnight
        with patch.object(controller.state, "complete_midnight_restart",
                          wraps=controller.state.complete_midnight_restart) as completed:
            self.assertEqual(controller.cycle(self.context), "daily_hero")

        self.assertEqual(started_days, ["2026-11-07", "2026-11-08"])
        self.assertEqual([call.args[0] for call in completed.call_args_list], started_days)
        self.assertEqual(self.events.count("restart_begin"), 2)
        self.assertEqual(self.events.count("restart_town_confirmed"), 2)
        self.assertLess(max(index for index, event in enumerate(self.events)
                            if event == "restart_town_confirmed"),
                        self.events.index("daily_hero"))
        self.assertFalse(controller.state.midnight_restart_due(self.clock()))
        self.assertEqual(
            controller.state.snapshot(self.clock())["midnight_restart"]["last_completed_date"],
            "2026-11-08")


if __name__ == "__main__":
    unittest.main()
