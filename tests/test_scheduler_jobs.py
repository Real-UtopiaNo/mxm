from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pet_alliance.adb import AutomationError
from pet_alliance.check_arena import CheckArena, CheckArenaReport
from pet_alliance.cooperation import TaskInterrupted, checkpoint
from pet_alliance.daily_hero import HeroReport
from pet_alliance.runtime import ScriptContext
from pet_alliance.scheduler_jobs import ManagedCheckArena, create_job, run_task
from pet_alliance.slay_dragon import SlayDragon
from pet_alliance.dragon_schedule import BEIJING
from tests.test_arena_schedule import ScheduleScene
from tests.test_check_arena import (
    NOW, Page as GuardianPage, ReplayContext as GuardianContext,
    isolate_hero_preparation,
)
from tests.test_daily_hero_reset import ResetScene


ROOT = Path(__file__).resolve().parent.parent


class ScriptControlTests(unittest.TestCase):
    def test_pause_prevents_each_device_operation(self):
        for operation, args in (
            ("capture", ()), ("tap", (10, 20)),
            ("swipe", (1, 2, 3, 4)), ("key", ("back",)),
            ("restart_app", ("com.example.game",)),
        ):
            with self.subTest(operation=operation):
                device = Mock()
                pause = Mock(side_effect=TaskInterrupted("paused", "用户暂停"))
                ctx = ScriptContext(device, Path("unused"), control=pause)
                with self.assertRaises(TaskInterrupted) as raised:
                    getattr(ctx, operation)(*args)
                self.assertEqual(raised.exception.status, "paused")
                pause.assert_called_once_with(operation)
                self.assertEqual(device.mock_calls, [])

    @patch("pet_alliance.runtime.time.sleep")
    def test_wait_observes_pause_within_a_quarter_second(self, sleep):
        control = Mock(side_effect=[None, TaskInterrupted("paused", "暂停")])
        ctx = ScriptContext(Mock(), Path("unused"), control=control)
        with self.assertRaises(TaskInterrupted):
            ctx.wait(30)
        sleep.assert_called_once_with(.25)
        self.assertEqual(control.call_count, 2)

    @patch("pet_alliance.runtime.time.sleep")
    def test_controlled_wait_keeps_total_duration_without_long_sleeps(self, sleep):
        control = Mock()
        ctx = ScriptContext(Mock(), Path("unused"), control=control)
        ctx.wait(.6)
        delays = [call.args[0] for call in sleep.call_args_list]
        self.assertAlmostEqual(sum(delays), .6)
        self.assertTrue(all(0 < delay <= .25 for delay in delays))
        self.assertEqual(control.call_count, len(delays) + 1)

    @patch("pet_alliance.runtime.time.sleep")
    def test_standalone_wait_preserves_original_single_sleep(self, sleep):
        ctx = ScriptContext(Mock(), Path("unused"))
        ctx.wait(30)
        sleep.assert_called_once_with(30)

    def test_checkpoint_is_optional_and_passes_an_independent_payload(self):
        checkpoint(SimpleNamespace(), "map", count=2)
        callback = Mock()
        ctx = ScriptContext(Mock(), Path("unused"), checkpoint_callback=callback)
        checkpoint(ctx, "map", count=2)
        callback.assert_called_once_with("map", {"count": 2})

    def test_interruption_rejects_failure_status(self):
        with self.assertRaises(ValueError):
            TaskInterrupted("stopped", "不是协作中断")


class HeroCooperationTests(unittest.TestCase):
    def test_checkpoint_after_third_dungeon_waits_for_points_arena(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 4)])
            boundaries = []

            def saved(name, **data):
                boundaries.append((name, data, len(scene.arena_calls), scene.view))

            scene.ctx.checkpoint = saved
            scene.run_points_arena = lambda region: scene.simulated_arena(region)
            scene.run()
            dungeons = [item for item in boundaries if item[0] == "hero_dungeon"]
            self.assertEqual([item[1]["completed_since_arena"] for item in dungeons], [1, 2, 0, 1])
            self.assertEqual([item[2] for item in dungeons], [0, 0, 1, 1])
            self.assertTrue(all(item[3] == "map" for item in dungeons))
            self.assertEqual([len(item[1]["report"]["completed"]) for item in dungeons], [1, 2, 3, 4])

    def test_restored_two_dungeons_trigger_points_arena_after_next_success(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True])])
            scene.ctx.resume_data = {"completed_since_arena": 2}
            scene.run_points_arena = lambda region: scene.simulated_arena(region)
            scene.run()
            self.assertEqual(scene.arena_calls, [("first", 1, 1)])

    def test_invalid_resume_counter_starts_at_zero(self):
        for value in (3, -1, True, "2", None):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                scene = ScheduleScene(directory, [("first", [True])])
                scene.ctx.resume_data = {"completed_since_arena": value}
                scene.run_points_arena = lambda region: scene.simulated_arena(region)
                scene.run()
                self.assertEqual(scene.arena_calls, [])

    def test_preemption_saves_progress_and_does_not_mark_daily_task_done(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 4)])

            def interrupt(name, **payload):
                if name == "hero_dungeon":
                    raise TaskInterrupted("preempted", "屠龙时段开始")

            scene.ctx.checkpoint = interrupt
            with self.assertRaises(TaskInterrupted):
                scene.run()
            saved = json.loads((scene.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "preempted")
            self.assertEqual(len(saved["completed"]), 1)
            self.assertEqual(saved["reason"], "屠龙时段开始")
            self.assertEqual(len(scene.attempts), 1)

    def test_pause_keeps_confirmed_reset_charges_in_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [], "ring": [], "cursed": []})
            scene.report.resets = 1
            scene.report.diamonds_spent = 50
            scene.report.diamond_spending.append({"region": "西部大陆", "diamonds": 50})
            scene.ctx.checkpoint = Mock(side_effect=TaskInterrupted("paused", "暂停"))
            with self.assertRaises(TaskInterrupted):
                scene.run()
            saved = json.loads((scene.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paused")
            self.assertEqual(saved["diamonds_spent"], 50)
            self.assertEqual(saved["resets"], 1)
            self.assertEqual(len(saved["diamond_spending"]), 1)


class ManagedJobTests(unittest.TestCase):
    def test_new_claim_with_one_second_guard_time_does_not_use_ocr(self):
        pages = [GuardianPage("19", "challenge_guardian"),
                 GuardianPage("claim_success", "dismiss_claim_success"),
                 GuardianPage("guardian_one_second", "guardian_close"),
                 GuardianPage("18", "dungeon_close"), GuardianPage("17")]
        with tempfile.TemporaryDirectory() as directory:
            ctx = GuardianContext(pages)
            job = ManagedCheckArena(ctx)
            job.run_dir.rmdir()
            job.run_dir = Path(directory)
            job.now = lambda: NOW + timedelta(seconds=ctx.clock.elapsed)
            job.started = 0
            isolate_hero_preparation(job)

            def save_deadline(*args):
                self.assertEqual(ctx.index, 4)
                self.assertTrue(job.report.claimed)
                self.assertEqual(job.vision.map_owner(job.frame, "COCO号", "1111"), "self")
                self.assertEqual(args[1], datetime.fromisoformat(job.report.claimed_at)
                                 + timedelta(hours=6, minutes=1))
                return Path(directory) / "automation.local.toml"

            with patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic), \
                    patch.object(job.time_reader, "read", side_effect=AssertionError("New claims must not OCR even a visible one-second guard time")) as read, \
                    patch("pet_alliance.check_arena.persist_arena_state", side_effect=save_deadline) as persist:
                owner = job.claim_empty("COCO号")

            self.assertEqual(owner, "self")
            self.assertTrue(job.report.claimed)
            self.assertIsNone(job.report.guard_seconds)
            self.assertEqual(datetime.fromisoformat(job.report.claimed_at),
                             (NOW + timedelta(seconds=ctx.input_times[0])).replace(microsecond=0))
            self.assertEqual(datetime.fromisoformat(job.report.arena_expired_time)
                             - datetime.fromisoformat(job.report.claimed_at),
                             timedelta(hours=6, minutes=1))
            read.assert_not_called()
            persist.assert_called_once()
            self.assertEqual(persist.call_args.args[2], "COCO号")
            self.assertEqual([item[1] for item in ctx.inputs],
                             ["challenge_guardian", "dismiss_claim_success", "guardian_close", "dungeon_close"])

    def test_new_claim_with_blank_guard_time_saves_only_after_map_confirms_self(self):
        pages = [GuardianPage("19", "challenge_guardian"),
                 GuardianPage("claim_success", "dismiss_claim_success"),
                 GuardianPage("guardian_no_duration", "guardian_close"),
                 GuardianPage("18", "dungeon_close"), GuardianPage("17")]
        with tempfile.TemporaryDirectory() as directory:
            ctx = GuardianContext(pages)
            ctx.settings["daily_hero"].update(click_retry_interval=1.4, transition_timeout=4)
            ctx.dropped[0] = 1
            job = ManagedCheckArena(ctx)
            job.run_dir.rmdir()
            job.run_dir = Path(directory)
            job.now = lambda: NOW + timedelta(seconds=ctx.clock.elapsed)
            job.started = 0
            isolate_hero_preparation(job)
            challenge_times = []
            original_tap = ctx.tap

            def tap(x, y):
                if ctx.pages[ctx.index].control == "challenge_guardian":
                    challenge_times.append(job.now())
                original_tap(x, y)

            def save_deadline(*args):
                self.assertEqual(ctx.index, 4)
                self.assertTrue(job.report.claimed)
                self.assertEqual(job.vision.map_owner(job.frame, "COCO号", "1111"), "self")
                self.assertEqual(args[1].replace(microsecond=0),
                                 (challenge_times[0] + timedelta(hours=6, minutes=1)).replace(microsecond=0))
                return Path(directory) / "automation.local.toml"

            ctx.tap = tap
            with patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic), \
                    patch.object(job.time_reader, "read", side_effect=AssertionError("New claims must not OCR the blank guard time")) as read, \
                    patch("pet_alliance.check_arena.persist_arena_state", side_effect=save_deadline) as persist:
                owner = job.claim_empty("COCO号")

            self.assertEqual(owner, "self")
            self.assertTrue(job.report.claimed)
            self.assertEqual(len(challenge_times), 2)
            self.assertGreaterEqual(challenge_times[1] - challenge_times[0], timedelta(seconds=1))
            self.assertEqual(datetime.fromisoformat(job.report.claimed_at),
                             challenge_times[0].replace(microsecond=0))
            self.assertIsNone(job.report.guard_seconds)
            read.assert_not_called()
            persist.assert_called_once()
            self.assertEqual(persist.call_args.args[2], "COCO号")
            self.assertEqual([item[1] for item in ctx.inputs],
                             ["challenge_guardian", "challenge_guardian", "dismiss_claim_success",
                              "guardian_close", "dungeon_close"])

    def test_claim_success_notice_is_dismissed_before_saving_six_hours_one_minute_deadline(self):
        pages = [GuardianPage("17", "绝望监狱"), GuardianPage("18", "arena_button"),
                 GuardianPage("19", "challenge_guardian"),
                 GuardianPage("claim_success", "dismiss_claim_success"),
                 GuardianPage("guardian_no_duration", "guardian_close"),
                 GuardianPage("18", "dungeon_close"), GuardianPage("17")]
        with tempfile.TemporaryDirectory() as directory:
            ctx = GuardianContext(pages, arena_list=["绝望监狱"])
            # A lost first tap must not trigger another claim or an early OCR.
            ctx.dropped[3] = 1
            job = ManagedCheckArena(ctx)
            job.run_dir.rmdir()
            job.run_dir = Path(directory)
            job.world = lambda: None
            job.enter_region = lambda region: None
            job.now = lambda: NOW + timedelta(seconds=ctx.clock.elapsed)
            job.started = 0
            isolate_hero_preparation(job)
            timeline = []
            original_map_owner = job.vision.map_owner

            def map_owner(frame, name, player):
                # The supplied map has no owner on this dungeon. Simulate only
                # its post-claim label; all previous pages use real vision.
                if ctx.index == 6 and name == "绝望监狱":
                    return "self"
                return original_map_owner(frame, name, player)

            def save_deadline(*args):
                self.assertEqual(ctx.index, 6)
                self.assertEqual(timeline, [])
                timeline.append("save_deadline")
                return Path(directory) / "automation.local.toml"

            def return_to_town():
                self.assertEqual(ctx.index, 6)
                self.assertEqual(timeline, ["save_deadline"])
                timeline.append("return_to_town")
                job.observe()

            job.return_to_town = return_to_town
            with patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic), \
                    patch.object(job.vision, "map_owner", side_effect=map_owner), \
                    patch.object(job.time_reader, "read", side_effect=AssertionError("New claims must not OCR the blank guard time")) as read, \
                    patch("pet_alliance.check_arena.persist_arena_state", side_effect=save_deadline) as persist:
                report = job.run()

            self.assertEqual(report.status, "completed")
            self.assertTrue(report.claimed)
            self.assertTrue(report.returned_to_town)
            self.assertIsNone(report.guard_seconds)
            self.assertEqual(datetime.fromisoformat(report.arena_expired_time)
                             - datetime.fromisoformat(report.claimed_at), timedelta(hours=6, minutes=1))
            read.assert_not_called()
            persist.assert_called_once()
            self.assertEqual(persist.call_args.args[2], "绝望监狱")
            self.assertEqual(timeline, ["save_deadline", "return_to_town"])
            self.assertEqual([item[1] for item in ctx.inputs],
                             ["绝望监狱", "arena_button", "challenge_guardian",
                              "dismiss_claim_success", "dismiss_claim_success", "guardian_close", "dungeon_close"])

    def test_empty_claim_replay_persists_six_hours_one_minute_deadline_before_return(self):
        pages = [GuardianPage("17", "绝望监狱"), GuardianPage("18", "arena_button"),
                 GuardianPage("19", "challenge_guardian"), GuardianPage("20", "guardian_close"),
                 GuardianPage("18", "dungeon_close"), GuardianPage("17")]
        with tempfile.TemporaryDirectory() as directory:
            ctx = GuardianContext(pages, arena_list=["绝望监狱"])
            job = ManagedCheckArena(ctx)
            job.run_dir.rmdir()
            job.run_dir = Path(directory)
            job.world = lambda: None
            job.enter_region = lambda region: None
            job.return_to_town = lambda: job.observe()
            job.now = lambda: NOW + timedelta(seconds=ctx.clock.elapsed)
            job.started = 0
            isolate_hero_preparation(job)
            original_map_owner = job.vision.map_owner

            def map_owner(frame, name, player):
                if ctx.index == 5 and name == "绝望监狱":
                    return "self"
                return original_map_owner(frame, name, player)

            with patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic), \
                    patch.object(job.vision, "map_owner", side_effect=map_owner), \
                    patch.object(job.time_reader, "read", side_effect=AssertionError("New claims must not OCR guard time")) as read, \
                    patch("pet_alliance.check_arena.persist_arena_state",
                          return_value=Path(directory) / "automation.local.toml") as persist:
                report = job.run()
            self.assertTrue(report.claimed)
            self.assertTrue(report.returned_to_town)
            self.assertEqual(report.status, "completed")
            job.prepare_claim.assert_called_once_with("绝望监狱")
            self.assertIsNone(report.guard_seconds)
            self.assertEqual(datetime.fromisoformat(report.arena_expired_time)
                             - datetime.fromisoformat(report.claimed_at), timedelta(hours=6, minutes=1))
            read.assert_not_called()
            persist.assert_called_once()
            self.assertEqual(persist.call_args.args[2], "绝望监狱")
            self.assertEqual([item[1] for item in ctx.inputs],
                             ["绝望监狱", "arena_button", "challenge_guardian", "guardian_close", "dungeon_close"])

    def test_managed_empty_claim_saves_expiration_only_after_confirmed_new_owner(self):
        for owner, claimed, expected in (("self", True, 1), ("self", False, 0), ("other", False, 0)):
            with self.subTest(owner=owner, claimed=claimed):
                job = object.__new__(ManagedCheckArena)
                job.report = CheckArenaReport(claimed=claimed)
                job.record_expiration = Mock()
                job.record_new_claim_expiration = Mock()
                with patch.object(CheckArena, "claim_empty", return_value=owner):
                    self.assertEqual(job.claim_empty("COCO号"), owner)
                self.assertEqual(job.record_new_claim_expiration.call_count, expected)
                job.record_expiration.assert_not_called()

    def test_existing_owned_arena_still_uses_observed_guard_time(self):
        pages = [GuardianPage("17", "COCO号"), GuardianPage("18", "arena_button"),
                 GuardianPage("20", "guardian_close"), GuardianPage("17")]
        with tempfile.TemporaryDirectory() as directory:
            ctx = GuardianContext(pages, arena_list=["COCO号"])
            job = ManagedCheckArena(ctx)
            job.run_dir.rmdir()
            job.run_dir = Path(directory)
            job.world = lambda: None
            job.enter_region = lambda region: None
            job.return_to_town = lambda: job.observe()
            job.now = lambda: NOW + timedelta(seconds=ctx.clock.elapsed)
            job.started = 0
            job.record_new_claim_expiration = Mock()

            def read_guard_time(frame):
                self.assertEqual(ctx.index, 2)
                self.assertEqual(job.guardian_owner(frame), "self")
                return 1198

            with patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic), \
                    patch.object(job.time_reader, "read", side_effect=read_guard_time) as read, \
                    patch("pet_alliance.check_arena.persist_arena_state",
                          return_value=Path(directory) / "automation.local.toml") as persist:
                report = job.run()

            self.assertEqual(report.status, "completed")
            self.assertFalse(report.claimed)
            self.assertTrue(report.returned_to_town)
            self.assertEqual(report.guard_seconds, 1198)
            self.assertEqual(datetime.fromisoformat(report.arena_expired_time)
                             - datetime.fromisoformat(report.observed_at),
                             timedelta(hours=6, seconds=-1198))
            read.assert_called_once()
            persist.assert_called_once()
            job.record_new_claim_expiration.assert_not_called()

    def test_task_dispatch_is_explicit(self):
        for task, constructor in (
            ("check_arena", "ManagedCheckArena"), ("slay_dragon", "SlayDragon"),
            ("daily_hero", "DailyHero"), ("daily_hero_reset", "DailyHeroReset"),
        ):
            with self.subTest(task=task), patch(f"pet_alliance.scheduler_jobs.{constructor}") as make:
                ctx = object()
                self.assertIs(create_job(task, ctx), make.return_value)
                make.assert_called_once_with(ctx)
        with self.assertRaises(AutomationError):
            create_job("unknown", object())

    def test_run_task_returns_report_and_diagnostic_directory(self):
        job = SimpleNamespace(report=HeroReport(status="completed"), run_dir=Path("run"), run=Mock())
        with patch("pet_alliance.scheduler_jobs.create_job", return_value=job), \
                patch("pet_alliance.scheduler_jobs.run_with_loadout", side_effect=lambda job, group: job.run()):
            result = run_task("daily_hero", object())
        self.assertEqual(result, {**asdict(job.report), "run_dir": "run"})

    def test_run_task_attaches_interrupted_report_before_reraising(self):
        interruption = TaskInterrupted("paused", "用户暂停")
        job = SimpleNamespace(report=HeroReport(status="paused", victories=7),
                              run_dir=Path("run"), run=Mock(side_effect=interruption))
        with patch("pet_alliance.scheduler_jobs.create_job", return_value=job), \
                patch("pet_alliance.scheduler_jobs.run_with_loadout", side_effect=lambda job, group: job.run()):
            with self.assertRaises(TaskInterrupted) as raised:
                run_task("daily_hero", object())
        self.assertIs(raised.exception, interruption)
        self.assertEqual(interruption.report["victories"], 7)
        self.assertEqual(interruption.run_dir, "run")

    def test_run_task_preserves_failure_type_and_paid_reset_statistics(self):
        from pet_alliance.daily_hero_reset import ResetReport

        error = AutomationError("下一页面识别失败")
        job = SimpleNamespace(report=ResetReport(status="stopped", resets=2, diamonds_spent=100),
                              run_dir=Path("failed_run"), run=Mock(side_effect=error))
        with patch("pet_alliance.scheduler_jobs.create_job", return_value=job), \
                patch("pet_alliance.scheduler_jobs.run_with_loadout", side_effect=lambda job, group: job.run()):
            with self.assertRaises(AutomationError) as raised:
                run_task("daily_hero_reset", object())
        self.assertIs(raised.exception, error)
        self.assertEqual(error.report["diamonds_spent"], 100)
        self.assertEqual(error.report["resets"], 2)
        self.assertEqual(error.run_dir, "failed_run")

    def test_guardian_and_dragon_save_scheduler_status(self):
        for task in ("check_arena", "slay_dragon"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                ctx = SimpleNamespace(project_root=ROOT, settings={}, dry_run=False,
                                      log=logging.getLogger("managed-test"))
                job = create_job(task, ctx)
                job.run_dir.rmdir()
                job.run_dir = Path(directory)
                interruption = TaskInterrupted("paused", "用户暂停")
                if isinstance(job, CheckArena):
                    job.prepare_map = Mock(side_effect=interruption)
                else:
                    # Pet selection has its own real-frame replay tests; this
                    # test isolates saving an interruption during monitoring.
                    job.prepare_dragon_loadout = Mock()
                    job.step = Mock(side_effect=interruption)
                with patch("pet_alliance.slay_dragon.beijing_now",
                           return_value=datetime(2026, 11, 6, 9, 30, tzinfo=BEIJING)):
                    with self.assertRaises(TaskInterrupted):
                        job.run()
                saved = json.loads((job.run_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "paused")
                self.assertEqual(saved["reason"], "用户暂停")


if __name__ == "__main__":
    unittest.main()
