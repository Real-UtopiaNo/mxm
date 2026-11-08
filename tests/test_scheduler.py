from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from pet_alliance.adb import AndroidDevice, AutomationError
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.dragon_schedule import BEIJING
from pet_alliance.scheduler import SchedulerController


class Clock:
    def __init__(self, value: str = "2026-11-06T08:00:00+08:00"):
        self.current = datetime.fromisoformat(value)

    def __call__(self):
        return self.current

    def advance(self, **duration):
        self.current += timedelta(**duration)


class Context:
    def __init__(self):
        self.inputs = []
        self.observations = 0

    def capture(self):
        self.control("capture")
        self.observations += 1

    def tap(self, x=1, y=2):
        self.control("tap")
        self.inputs.append((x, y))

    def checkpoint(self, name, **payload):
        self.checkpoint_callback(name, payload)


class SchedulerEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config = self.directory / "automation.toml"
        self.config.write_text("[scheduler]\npoll_interval = 0.01\n"
                               "arena_retry_seconds = 300\n"
                               "[daily_hero]\nmin_click_interval = 0.001\n", encoding="utf-8")
        self.clock = Clock()
        self.context = Context()
        self.events = []
        self.controllers = []

    def tearDown(self):
        for controller in reversed(self.controllers):
            controller.close()
            if controller._thread:
                controller._thread.join(3)
                self.assertFalse(controller._thread.is_alive(), "Worker did not stop after close")
        self.temporary.cleanup()

    def controller(self, runner=None):
        events = self.events

        class Navigator:
            def __init__(self, context):
                self.context = context

            def ensure_town(self):
                self.context.capture()
                events.append("town")

            def restart_to_town(self):
                self.context.capture()
                events.append("midnight_restart")

        result = SchedulerController(
            self.config, now=self.clock,
            context_factory=lambda controller: self.context,
            task_runner=runner or (lambda task, ctx: {"status": "completed"}),
            navigator_factory=Navigator,
        )
        self.controllers.append(result)
        result._context()
        return result

    def future_arena(self, controller, hours=12):
        controller.state.set_arena(self.clock() + timedelta(hours=hours), now=self.clock())

    @staticmethod
    def attach_report(error, **report):
        error.report = {"status": error.status, **report}
        error.run_dir = "diagnostic_run"
        return error

    def test_due_arena_interrupts_dragon_then_returns_to_town_before_each_job(self):
        self.clock.current = datetime(2026, 11, 6, 9, 30, tzinfo=BEIJING)
        counts = {"slay_dragon": 0}

        def run(task, ctx):
            self.events.append(task)
            if task == "slay_dragon":
                counts[task] += 1
                if counts[task] == 1:
                    self.clock.advance(minutes=2)
                    try:
                        ctx.capture()
                    except TaskInterrupted as error:
                        raise self.attach_report(error, observations=1)
            if task == "check_arena":
                return {"status": "completed", "claimed": True,
                        "arena_expired_time": (self.clock() + timedelta(hours=6)).isoformat()}
            return {"status": "completed"}

        controller = self.controller(run)
        controller.state.set_arena(self.clock() + timedelta(minutes=1), now=self.clock())
        self.assertEqual(controller.cycle(self.context), "slay_dragon")
        self.assertEqual(controller.state.snapshot(self.clock())["tasks"]["slay_dragon"]["status"], "preempted")
        self.assertEqual(controller.cycle(self.context), "check_arena")
        self.assertEqual(controller.cycle(self.context), "slay_dragon")
        self.assertEqual(self.events, ["town", "slay_dragon", "town", "check_arena", "town", "slay_dragon"])

    def test_dragon_closing_at_eleven_yields_to_unfinished_hero(self):
        self.clock.current = datetime(2026, 11, 6, 10, 59, tzinfo=BEIJING)

        def run(task, ctx):
            self.events.append(task)
            if task == "slay_dragon":
                self.clock.advance(minutes=2)
                ctx.capture()
            return {"status": "completed"}

        controller = self.controller(run)
        self.future_arena(controller)
        controller.cycle(self.context)
        controller.cycle(self.context)
        self.assertEqual(self.events, ["town", "slay_dragon", "town", "daily_hero"])
        self.assertTrue(controller.snapshot()["tasks"]["daily_hero"]["done"])

    def test_hero_yields_at_checkpoint_after_dragon_opens_without_interrupting_battle_io(self):
        self.clock.current = datetime(2026, 11, 6, 8, 59, tzinfo=BEIJING)

        def run(task, ctx):
            self.assertEqual(task, "daily_hero")
            self.clock.advance(minutes=2)
            ctx.capture()
            ctx.tap(30, 40)
            try:
                ctx.checkpoint("hero_dungeon", report={"completed": ["first"]}, completed_since_arena=1)
            except TaskInterrupted as error:
                raise self.attach_report(error, completed=["first"])
            self.fail("Higher priority dragon should interrupt at the safe checkpoint")

        controller = self.controller(run)
        self.future_arena(controller)
        controller.cycle(self.context)
        saved = controller.snapshot()["tasks"]["daily_hero"]
        self.assertEqual(self.context.inputs, [(30, 40)])
        self.assertFalse(saved["done"])
        self.assertEqual(saved["status"], "preempted")
        self.assertEqual(saved["progress"]["completed"], ["first"])
        self.assertEqual(saved["progress"]["completed_since_arena"], 1)

    def test_pause_stops_next_input_immediately_without_automatic_town_navigation(self):
        ready, release = threading.Event(), threading.Event()

        def run(task, ctx):
            ready.set()
            self.assertTrue(release.wait(2), "Test did not release the runner")
            try:
                ctx.tap()
            except TaskInterrupted as error:
                raise self.attach_report(error, victories=2)
            self.fail("An input was allowed after Pause")

        controller = self.controller(run)
        self.future_arena(controller)
        controller.start()
        self.assertTrue(ready.wait(2))
        controller.pause()
        release.set()
        controller._thread.join(3)
        self.assertFalse(controller._thread.is_alive())
        self.assertEqual(self.context.inputs, [])
        self.assertEqual(self.events, ["town"])
        saved = controller.snapshot()
        self.assertEqual(saved["status"], "paused")
        self.assertEqual(saved["tasks"]["daily_hero"]["status"], "paused")
        self.assertEqual(saved["tasks"]["daily_hero"]["progress"]["victories"], 2)
        self.assertFalse(saved["tasks"]["daily_hero"]["done"])

    def test_completed_daily_job_is_skipped_after_controller_is_recreated(self):
        calls = []
        controller = self.controller(lambda task, ctx: calls.append(task) or {"status": "completed"})
        self.future_arena(controller)
        controller.cycle(self.context)
        controller.close()
        replacement = self.controller(lambda task, ctx: calls.append(task) or {"status": "completed"})
        replacement.cycle(self.context)
        self.assertEqual(calls, ["daily_hero", "daily_hero_reset"])
        self.assertTrue(replacement.snapshot()["tasks"]["daily_hero"]["done"])
        self.assertTrue(replacement.snapshot()["tasks"]["daily_hero_reset"]["done"])

    def test_midnight_checkpoint_keeps_old_work_on_old_day_and_starts_new_day_hero(self):
        self.clock.current = datetime(2026, 11, 6, 23, 59, tzinfo=BEIJING)
        calls = []

        def run(task, ctx):
            calls.append(task)
            if len(calls) == 1:
                self.clock.advance(minutes=2)
                try:
                    ctx.checkpoint("hero_dungeon", report={"completed": ["old_day"]}, completed_since_arena=1)
                except TaskInterrupted as error:
                    raise self.attach_report(error, completed=["old_day"])
            self.assertEqual(ctx.resume_data, {})
            return {"status": "completed", "completed": ["new_day"]}

        controller = self.controller(run)
        self.future_arena(controller)
        controller.cycle(self.context)
        saved = controller.state.snapshot(self.clock())
        self.assertEqual(saved["date"], "2026-11-07")
        self.assertFalse(saved["daily"]["daily_hero"]["done"])
        self.assertEqual(saved["daily"]["daily_hero"]["progress"], {})
        self.assertEqual(saved["history"]["2026-11-06"]["daily_hero"]["progress"]["completed"], ["old_day"])
        controller.cycle(self.context)
        self.assertEqual(calls, ["daily_hero", "daily_hero"])
        self.assertTrue(controller.snapshot()["tasks"]["daily_hero"]["done"])

    def test_complete_report_finishing_after_midnight_never_marks_new_day_done(self):
        self.clock.current = datetime(2026, 11, 6, 23, 59, tzinfo=BEIJING)

        def run(task, ctx):
            self.clock.advance(minutes=2)
            return {"status": "completed", "completed": ["last_old_day_dungeon"]}

        controller = self.controller(run)
        self.future_arena(controller)
        controller.cycle(self.context)
        saved = controller.state.snapshot(self.clock())
        self.assertTrue(saved["history"]["2026-11-06"]["daily_hero"]["done"])
        self.assertFalse(saved["daily"]["daily_hero"]["done"])

    def test_paused_reset_restarts_with_partial_progress_and_accumulates_confirmed_costs(self):
        calls = []

        def first(task, ctx):
            self.assertEqual(task, "daily_hero_reset")
            error = TaskInterrupted("paused", "暂停")
            raise self.attach_report(error, completed=["west_a"], resets=1, diamonds_spent=50,
                                     diamond_spending=[{"dungeon": "west_a", "diamonds": 50}])

        controller = self.controller(first)
        self.future_arena(controller)
        token = controller.state.start_task("daily_hero", self.clock())
        controller.state.finish_task("daily_hero", {"status": "completed"}, self.clock(), token)
        with self.assertRaises(TaskInterrupted):
            controller.cycle(self.context)
        controller.close()

        def second(task, ctx):
            calls.append(task)
            self.assertEqual(ctx.resume_data["completed"], ["west_a"])
            self.assertEqual(ctx.resume_data["diamonds_spent"], 50)
            return {"status": "completed", "completed": ["ring_a"], "resets": 1, "diamonds_spent": 10,
                    "diamond_spending": [{"dungeon": "ring_a", "diamonds": 10}]}

        replacement = self.controller(second)
        replacement.cycle(self.context)
        record = replacement.snapshot()["tasks"]["daily_hero_reset"]
        self.assertEqual(calls, ["daily_hero_reset"])
        self.assertTrue(record["done"])
        self.assertEqual(record["progress"]["completed"], ["west_a", "ring_a"])
        self.assertEqual(record["progress"]["resets"], 2)
        self.assertEqual(record["progress"]["diamonds_spent"], 60)
        self.assertEqual(record["progress"]["diamond_spending"], [
            {"dungeon": "west_a", "diamonds": 50}, {"dungeon": "ring_a", "diamonds": 10}])

    def test_checkpoint_progress_is_not_double_counted_when_attempt_finishes(self):
        def run(task, ctx):
            ctx.checkpoint("hero_dungeon", report={"completed": ["a"], "victories": 5}, completed_since_arena=1)
            ctx.checkpoint("hero_dungeon", report={"completed": ["a", "b"], "victories": 10}, completed_since_arena=2)
            return {"status": "completed", "completed": ["a", "b"], "victories": 10}

        controller = self.controller(run)
        self.future_arena(controller)
        controller.cycle(self.context)
        progress = controller.snapshot()["tasks"]["daily_hero"]["progress"]
        self.assertEqual(progress["completed"], ["a", "b"])
        self.assertEqual(progress["victories"], 10)
        self.assertEqual(progress["completed_since_arena"], 2)

    def test_no_vacant_arena_backs_off_five_minutes_and_allows_hero(self):
        def run(task, ctx):
            self.events.append(task)
            if task == "check_arena":
                error = AutomationError("均由其他玩家守护")
                error.report = {"status": "exhausted", "returned_to_town": True}
                raise error
            return {"status": "completed"}

        controller = self.controller(run)
        controller.cycle(self.context)
        saved = controller.state.snapshot(self.clock())
        self.assertEqual(datetime.fromisoformat(saved["arena"]["retry_after"]), self.clock() + timedelta(minutes=5))
        self.assertEqual(controller._choose(), "daily_hero")
        controller.cycle(self.context)
        self.assertEqual(self.events, ["town", "check_arena", "town", "daily_hero"])
        self.clock.advance(minutes=5)
        self.assertEqual(controller._choose(), "check_arena")

    def test_exhausted_report_without_verified_return_is_a_failure(self):
        def run(task, ctx):
            error = AutomationError("回城失败")
            error.report = {"status": "exhausted", "returned_to_town": False}
            raise error

        controller = self.controller(run)
        with self.assertRaisesRegex(AutomationError, "回城失败"):
            controller.cycle(self.context)

    def test_worker_unexpected_failure_saves_diagnostic_report_and_stops_with_error(self):
        def run(task, ctx):
            error = AutomationError("副本识别失败")
            error.report = {"status": "stopped", "victories": 3}
            error.run_dir = "failed_run"
            raise error

        controller = self.controller(run)
        self.future_arena(controller)
        controller.start()
        controller._thread.join(3)
        saved = controller.snapshot()
        self.assertFalse(saved["worker_alive"])
        self.assertEqual(saved["status"], "error")
        self.assertIn("副本识别失败", saved["reason"])
        self.assertFalse(saved["tasks"]["daily_hero"]["done"])
        self.assertEqual(saved["tasks"]["daily_hero"]["progress"]["run_dir"], "failed_run")
        self.assertEqual(saved["tasks"]["daily_hero"]["progress"]["victories"], 3)

    def test_limited_run_stops_worker_without_marking_daily_job_completed(self):
        controller = self.controller(lambda task, ctx: {"status": "limited", "completed": ["one"]})
        self.future_arena(controller)
        controller.start()
        controller._thread.join(3)
        saved = controller.snapshot()
        self.assertEqual(saved["status"], "error")
        self.assertFalse(saved["tasks"]["daily_hero"]["done"])
        self.assertEqual(saved["tasks"]["daily_hero"]["status"], "limited")

    def test_two_windows_cannot_own_same_directory_and_close_releases_lease(self):
        controller = self.controller()
        with self.assertRaisesRegex(AutomationError, "已有调度器"):
            self.controller()
        controller.close()
        replacement = self.controller()
        self.assertEqual(replacement.snapshot()["status"], "paused")

    def test_opening_controller_and_reading_snapshot_does_not_connect_or_input(self):
        calls = []
        controller = SchedulerController(self.config, now=self.clock,
                                         context_factory=lambda ctrl: calls.append("connected"))
        self.controllers.append(controller)
        snapshot = controller.snapshot()
        self.assertEqual(calls, [])
        self.assertEqual(self.context.inputs, [])
        self.assertEqual(snapshot["status"], "paused")
        self.assertFalse(snapshot["worker_alive"])

    def test_idle_does_not_repeat_navigation_but_new_due_job_rechecks_town_first(self):
        def run(task, ctx):
            self.events.append(task)
            return {"status": "completed", "arena_expired_time":
                    (self.clock() + timedelta(hours=6)).isoformat()}

        controller = self.controller(run)
        controller.state.set_arena(self.clock() + timedelta(minutes=10), now=self.clock())
        for task in ("daily_hero", "daily_hero_reset"):
            token = controller.state.start_task(task, self.clock())
            controller.state.finish_task(task, {"status": "completed"}, self.clock(), token)
        self.assertIsNone(controller.cycle(self.context))
        self.assertEqual(self.events, ["town"])
        self.assertIsNone(controller.cycle(self.context))
        self.assertEqual(self.events, ["town"])
        # A reconnect may be required after the game has sat idle. The navigator
        # must get a new opportunity to verify/recover town before any job runs.
        self.clock.advance(minutes=10)
        self.assertEqual(controller.cycle(self.context), "check_arena")
        self.assertEqual(self.events, ["town", "town", "check_arena"])

    def test_microsecond_config_deadline_does_not_clear_persisted_arena_backoff_on_reload(self):
        with self.config.open("a", encoding="utf-8") as stream:
            stream.write('\n[check_arena]\narena_expired_time = "2026-11-06T07:59:30.123456+08:00"\n')
        controller = self.controller()
        deadline = controller.state.snapshot(self.clock())["arena"]["next_due"]
        self.assertEqual(deadline, "2026-11-06T07:59:30+08:00")
        retry = controller.state.backoff_arena(300, self.clock())
        for _ in range(2):
            controller._reload()
            self.assertEqual(controller.state.snapshot(self.clock())["arena"]["retry_after"], retry)
            self.assertEqual(controller._choose(), "daily_hero")

    def test_cross_job_click_interval_waits_before_fresh_capture_and_never_after_location_is_chosen(self):
        controller = self.controller()
        controller._min_click = 1
        elapsed, waits = [0.0], []
        phase = ["previous_job"]

        def wait(delay):
            waits.append((phase[0], delay))
            elapsed[0] += delay
            return False

        with patch("pet_alliance.scheduler.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(controller._pause, "wait", side_effect=wait):
            self.context.tap(1, 2)
            elapsed[0] = .25
            phase[0] = "new_job_capture"
            self.context.capture()
            captured_at = elapsed[0]
            # Model a moving target. Its coordinate is chosen from the fresh frame.
            target_x = int(captured_at * 100)
            phase[0] = "new_job_tap"
            self.context.tap(target_x, 40)
            tapped_at = elapsed[0]

        self.assertEqual(waits, [("new_job_capture", .75)])
        self.assertEqual(captured_at, 1.0)
        self.assertEqual(tapped_at, captured_at)
        self.assertEqual(self.context.inputs[-1], (100, 40))

    def test_pause_after_restart_force_stop_blocks_the_following_start_adb_command(self):
        controller = self.controller()
        device = AndroidDevice("fake-adb", "emulator-test")
        device.before_command = lambda: controller._control("adb")
        commands = []

        def adb(command, **kwargs):
            commands.append(command)
            if "resolve-activity" in command:
                output = b"com.example.game/.MainActivity\n"
            elif "force-stop" in command:
                controller.pause()
                output = b""
            else:
                self.fail(f"Unexpected ADB command after pause: {command}")
            return subprocess.CompletedProcess(command, 0, output, b"")

        with patch("pet_alliance.adb.subprocess.run", side_effect=adb):
            with self.assertRaises(TaskInterrupted) as raised:
                device.restart_app("com.example.game")
        self.assertEqual(raised.exception.status, "paused")
        self.assertEqual(len(commands), 2)
        self.assertIn("resolve-activity", commands[0])
        self.assertIn("force-stop", commands[1])
        self.assertFalse(any("start" in command for command in commands))

    def test_arena_prerequisite_hero_work_remains_pending_and_resumes_daily_counter(self):
        calls = []

        def run(task, ctx):
            calls.append(task)
            self.assertEqual(ctx.hero_resume_date, "2026-11-06")
            if task == "check_arena":
                self.assertEqual(ctx.hero_resume_data, {})
                return {"status": "completed", "returned_to_town": True,
                        "arena_expired_time": (self.clock() + timedelta(hours=6)).isoformat(),
                        "hero_progress_by_day": {"2026-11-06": {
                            "completed": ["COCO号"], "victories": 5, "completed_since_arena": 1}}}
            self.assertEqual(task, "daily_hero")
            self.assertEqual(ctx.resume_data["completed"], ["COCO号"])
            self.assertEqual(ctx.resume_data["victories"], 5)
            self.assertEqual(ctx.resume_data["completed_since_arena"], 1)
            self.assertEqual(ctx.hero_resume_data, ctx.resume_data)
            ctx.checkpoint("hero_dungeon", report={"completed": ["恶魔城"], "victories": 5}, completed_since_arena=2)
            return {"status": "completed", "completed": ["恶魔城"], "victories": 5}

        controller = self.controller(run)
        controller.cycle(self.context)
        hero = controller.snapshot()["tasks"]["daily_hero"]
        self.assertFalse(hero["done"])
        self.assertEqual(hero["status"], "pending")
        self.assertEqual(hero["progress"]["completed"], ["COCO号"])
        self.assertEqual(hero["progress"]["victories"], 5)
        self.assertEqual(hero["progress"]["completed_since_arena"], 1)
        controller.cycle(self.context)
        hero = controller.snapshot()["tasks"]["daily_hero"]
        self.assertEqual(calls, ["check_arena", "daily_hero"])
        self.assertTrue(hero["done"])
        self.assertEqual(hero["progress"]["completed"], ["COCO号", "恶魔城"])
        self.assertEqual(hero["progress"]["victories"], 10)
        self.assertEqual(hero["progress"]["completed_since_arena"], 2)

    def test_multiple_arena_runs_merge_only_each_new_hero_increment(self):
        names = ["COCO号", "恶魔城"]
        calls = []

        def run(task, ctx):
            self.assertEqual(task, "check_arena")
            index = len(calls)
            self.assertEqual(ctx.hero_resume_date, "2026-11-06")
            self.assertEqual(ctx.hero_resume_data["completed"], ["already_done", *names[:index]])
            self.assertEqual(ctx.hero_resume_data["victories"], 5 * (index + 1))
            calls.append(task)
            return {"status": "completed", "returned_to_town": True,
                    "arena_expired_time": (self.clock() + timedelta(hours=6)).isoformat(),
                    "hero_progress_by_day": {"2026-11-06": {
                        "completed": [names[index]], "victories": 5,
                        "completed_since_arena": (index + 2) % 3}}}

        controller = self.controller(run)
        controller.state.record_daily_sidework("daily_hero", {
            "completed": ["already_done"], "victories": 5, "completed_since_arena": 1}, self.clock())
        controller.cycle(self.context)
        controller.state.set_arena(None, now=self.clock())
        controller.cycle(self.context)
        hero = controller.snapshot()["tasks"]["daily_hero"]
        self.assertFalse(hero["done"])
        self.assertEqual(hero["progress"]["completed"], ["already_done", "COCO号", "恶魔城"])
        self.assertEqual(hero["progress"]["victories"], 15)
        self.assertEqual(hero["progress"]["completed_since_arena"], 0)

    def test_arena_pause_and_failure_both_persist_completed_child_hero_work(self):
        for index, status in enumerate(("paused", "stopped"), 1):
            with self.subTest(status=status):
                def run(task, ctx):
                    self.assertEqual(task, "check_arena")
                    self.assertEqual(len(ctx.hero_resume_data.get("completed", [])), index - 1)
                    error = (TaskInterrupted("paused", "用户暂停") if status == "paused"
                             else AutomationError("擂台页面识别失败"))
                    error.report = {"status": status, "hero_progress_by_day": {"2026-11-06": {
                        "completed": [f"dungeon_{index}"], "victories": 5,
                        "completed_since_arena": index}}}
                    raise error

                controller = self.controller(run)
                controller.start()
                controller._thread.join(3)
                saved = controller.snapshot()
                self.assertFalse(saved["worker_alive"])
                self.assertEqual(saved["status"], "paused" if status == "paused" else "error")
                hero = saved["tasks"]["daily_hero"]
                self.assertFalse(hero["done"])
                self.assertEqual(hero["status"], "pending")
                self.assertEqual(hero["progress"]["completed"], [f"dungeon_{number}" for number in range(1, index + 1)])
                self.assertEqual(hero["progress"]["victories"], 5 * index)
                self.assertEqual(hero["progress"]["completed_since_arena"], index)
                controller.close()

    def test_arena_hero_work_before_midnight_updates_history_without_completing_new_day(self):
        self.clock.current = datetime(2026, 11, 6, 23, 59, tzinfo=BEIJING)

        def run(task, ctx):
            self.assertEqual(task, "check_arena")
            self.assertEqual(ctx.hero_resume_date, "2026-11-06")
            self.clock.advance(minutes=2)
            return {"status": "completed", "returned_to_town": True,
                    "arena_expired_time": (self.clock() + timedelta(hours=6)).isoformat(),
                    "hero_progress_by_day": {"2026-11-06": {
                        "completed": ["COCO号"], "victories": 5, "completed_since_arena": 1}}}

        controller = self.controller(run)
        controller.cycle(self.context)
        saved = controller.state.snapshot(self.clock())
        self.assertEqual(saved["date"], "2026-11-07")
        old_hero = saved["history"]["2026-11-06"]["daily_hero"]
        self.assertFalse(old_hero["done"])
        self.assertEqual(old_hero["progress"]["completed"], ["COCO号"])
        self.assertEqual(old_hero["progress"]["victories"], 5)
        new_hero = saved["daily"]["daily_hero"]
        self.assertFalse(new_hero["done"])
        self.assertEqual(new_hero["status"], "pending")
        self.assertEqual(new_hero["progress"], {})

    def test_arena_child_hero_work_spanning_midnight_is_split_by_actual_work_day(self):
        self.clock.current = datetime(2026, 11, 6, 23, 59, tzinfo=BEIJING)

        def run(task, ctx):
            self.clock.advance(minutes=2)
            return {"status": "completed", "returned_to_town": True,
                    "arena_expired_time": (self.clock() + timedelta(hours=6)).isoformat(),
                    "hero_progress_by_day": {
                        "2026-11-06": {"completed": ["old_day"], "victories": 5, "completed_since_arena": 1},
                        "2026-11-07": {"completed": ["new_day"], "victories": 5, "completed_since_arena": 1}}}

        controller = self.controller(run)
        controller.cycle(self.context)
        saved = controller.state.snapshot(self.clock())
        self.assertEqual(saved["history"]["2026-11-06"]["daily_hero"]["progress"]["completed"], ["old_day"])
        self.assertEqual(saved["daily"]["daily_hero"]["progress"]["completed"], ["new_day"])
        self.assertFalse(saved["daily"]["daily_hero"]["done"])

    def test_blocked_arena_with_confirmed_town_backs_off_and_allows_hero_or_dragon(self):
        for hour, next_task in ((8, "daily_hero"), (9, "slay_dragon")):
            with self.subTest(next_task=next_task):
                self.clock.current = datetime(2026, 11, 6, hour, 30, tzinfo=BEIJING)
                self.events.clear()

                def run(task, ctx):
                    self.events.append(task)
                    if task == "check_arena":
                        error = AutomationError("目标副本的英雄级挑战未完成")
                        error.report = {"status": "blocked", "returned_to_town": True}
                        raise error
                    return {"status": "completed"}

                controller = self.controller(run)
                controller.cycle(self.context)
                self.assertEqual(controller._choose(), next_task)
                saved = controller.state.snapshot(self.clock())
                self.assertEqual(datetime.fromisoformat(saved["arena"]["retry_after"]), self.clock() + timedelta(minutes=5))
                controller.cycle(self.context)
                self.assertEqual(self.events, ["town", "check_arena", "town", next_task])
                controller.close()

    def test_blocked_arena_without_confirmed_town_stops_worker_for_returned_or_raised_reports(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                def run(task, ctx):
                    report = {"status": "blocked", "returned_to_town": False}
                    if raises:
                        error = AutomationError("前置英雄失败后回城失败")
                        error.report = report
                        raise error
                    return report

                controller = self.controller(run)
                controller.start()
                controller._thread.join(3)
                saved = controller.snapshot()
                self.assertFalse(saved["worker_alive"])
                self.assertEqual(saved["status"], "error")
                self.assertIsNone(controller.state.snapshot(self.clock())["arena"]["retry_after"])
                self.assertFalse(saved["tasks"]["daily_hero"]["done"])
                controller.close()

    def test_arena_child_pending_points_visit_is_resumed_before_daily_hero_run(self):
        calls = []

        def run(task, ctx):
            calls.append(task)
            if task == "check_arena":
                if len(calls) == 2:
                    self.assertTrue(ctx.hero_resume_data["points_arena_pending"])
                    self.assertEqual(ctx.hero_resume_data["completed_since_arena"], 2)
                    self.assertEqual(ctx.hero_resume_date, "2026-11-06")
                    return {"status": "completed", "returned_to_town": True,
                            "arena_expired_time": (self.clock() + timedelta(hours=6)).isoformat(),
                            "hero_progress_by_day": {"2026-11-06": {
                                "points_arena_pending": False, "completed_since_arena": 0, "arena_visits": 1}}}
                return {"status": "completed", "returned_to_town": True,
                        "arena_expired_time": (self.clock() + timedelta(hours=6)).isoformat(),
                        "hero_progress_by_day": {"2026-11-06": {
                            "completed": ["third"], "victories": 5,
                            "completed_since_arena": 2, "points_arena_pending": True}}}
            self.assertEqual(task, "daily_hero")
            self.assertFalse(ctx.resume_data["points_arena_pending"])
            self.assertFalse(ctx.hero_resume_data["points_arena_pending"])
            self.assertEqual(ctx.resume_data["completed_since_arena"], 0)
            self.assertEqual(ctx.hero_resume_date, "2026-11-06")
            return {"status": "completed", "points_arena_pending": False}

        controller = self.controller(run)
        controller.cycle(self.context)
        hero = controller.snapshot()["tasks"]["daily_hero"]
        self.assertFalse(hero["done"])
        self.assertTrue(hero["progress"]["points_arena_pending"])
        controller.cycle(self.context)
        hero = controller.snapshot()["tasks"]["daily_hero"]
        self.assertFalse(hero["done"])
        self.assertFalse(hero["progress"]["points_arena_pending"])
        self.assertEqual(hero["progress"]["completed_since_arena"], 0)
        self.assertEqual(hero["progress"]["arena_visits"], 1)
        controller.cycle(self.context)
        self.assertEqual(calls, ["check_arena", "check_arena", "daily_hero"])


if __name__ == "__main__":
    unittest.main()
