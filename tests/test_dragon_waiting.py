from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch

from pet_alliance import AutomationError
from tests import test_slay_dragon as replay_helpers


BEIJING = timezone(timedelta(hours=8))


def local_time(day=5, hour=11, minute=59, second=58):
    return datetime(2026, 11, day, hour, minute, second, tzinfo=BEIJING)


class DragonWaitingTests(unittest.TestCase):
    @contextmanager
    def replay(self, pages, **options):
        # Reuse the strict input replay without inheriting/re-running its tests.
        with replay_helpers.SlayDragonTests().replay(pages, **options) as pair:
            yield pair

    def events(self, dragon):
        return [json.loads(line) for line in (dragon.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    def summary(self, dragon):
        return json.loads((dragon.run_dir / "summary.json").read_text(encoding="utf-8"))

    def recovery_pages(self, **options):
        return replay_helpers.SlayDragonTests().recovery_pages(**options)

    def test_wait_until_noon_uses_beijing_clock_without_capture_or_input(self):
        with self.replay([replay_helpers.Page()]) as (dragon, ctx):
            base = local_time()
            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=lambda: base + timedelta(seconds=ctx.clock.now)):
                dragon.wait_for_start()
            self.assertEqual(ctx.clock.waits, [2])
            self.assertEqual(ctx.captures, [])
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(dragon.report.scheduled_waits, 1)
            self.assertEqual(dragon.report.next_start, local_time(hour=12, minute=0, second=0).isoformat())
            self.assertEqual([event["event"] for event in self.events(dragon)],
                             ["dragon_wait_start", "dragon_wait_complete"])

    def test_22_oclock_waits_until_next_day_nine_in_short_interruptible_chunks(self):
        with self.replay([replay_helpers.Page()]) as (dragon, ctx):
            def wall_clock():
                if ctx.clock.now < 30:
                    return local_time(hour=22, minute=0, second=0)
                # Simulate elapsed overnight time/system clock adjustment.
                return local_time(day=6, hour=8, minute=59, second=59) + timedelta(seconds=ctx.clock.now - 30)

            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock):
                dragon.wait_for_start()
            self.assertEqual(ctx.clock.waits, [30, 1])
            self.assertTrue(all(0 < seconds <= 30 for seconds in ctx.clock.waits))
            self.assertEqual(dragon.report.next_start, local_time(day=6, hour=9, minute=0, second=0).isoformat())
            self.assertEqual(ctx.captures, [])
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])

    def test_clock_moving_back_does_not_trigger_restart_before_start(self):
        with self.replay([replay_helpers.Page()]) as (dragon, ctx):
            base = local_time(second=40)

            def wall_clock():
                adjusted_seconds = ctx.clock.now - (10 if ctx.clock.now >= 20 else 0)
                return base + timedelta(seconds=adjusted_seconds)

            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock):
                dragon.wait_for_start()
                self.assertEqual(wall_clock(), local_time(hour=12, minute=0, second=0))
            self.assertEqual(ctx.clock.waits, [20, 10])
            self.assertEqual(ctx.inputs, [])

    def test_clock_moving_forward_ends_wait_without_stale_countdown(self):
        with self.replay([replay_helpers.Page()]) as (dragon, ctx):
            base = local_time(second=0)

            def wall_clock():
                return base + timedelta(seconds=ctx.clock.now + (50 if ctx.clock.now >= 30 else 0))

            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock):
                dragon.wait_for_start()
            self.assertEqual(ctx.clock.waits, [30])
            self.assertEqual(ctx.inputs, [])

    def test_exact_start_time_returns_without_waiting_for_next_session(self):
        with self.replay([replay_helpers.Page()]) as (dragon, ctx):
            with patch("pet_alliance.slay_dragon.beijing_now", return_value=local_time(hour=12, minute=0, second=0)):
                dragon.wait_for_start()
            self.assertEqual(ctx.clock.waits, [])
            self.assertEqual(ctx.inputs, [])

    def test_startup_inside_active_interval_observes_hosting_without_wait_or_restart(self):
        pages = [replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})]
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            with patch("pet_alliance.slay_dragon.beijing_now", return_value=local_time(hour=9, minute=30, second=0)):
                report = dragon.run()
            self.assertEqual(report.checks, 1)
            self.assertEqual(report.scheduled_waits, 0)
            self.assertEqual(ctx.captures, [(0, 0)])
            self.assertEqual(ctx.clock.waits, [])
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(self.events(dragon)[0]["event"], "startup_schedule")

    def test_startup_outside_interval_waits_before_capture_then_runs_full_recovery(self):
        pages = self.recovery_pages()
        pages[8] = replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            base = local_time()
            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=lambda: base + timedelta(seconds=ctx.clock.now)):
                report = dragon.run()
            self.assertEqual(ctx.clock.waits[0], 2)
            self.assertEqual(ctx.restarts, [(2, replay_helpers.PACKAGE)])
            self.assertTrue(ctx.captures)
            self.assertTrue(all(at >= 2 for at, _ in ctx.captures))
            self.assertEqual(ctx.captures[0], (2, 1))
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "notice", "dragon_entry", "dragon_npc", "dragon_npc", "quest_mark"])
            self.assertTrue(all(at >= 2 for at, _, _, _ in ctx.inputs))
            self.assertEqual(report.scheduled_waits, 1)
            self.assertEqual(report.next_start, local_time(hour=12, minute=0, second=0).isoformat())
            self.assertEqual(report.restarts, 1)
            self.assertEqual(report.checks, 1)
            kinds = [event["event"] for event in self.events(dragon)]
            self.assertLess(kinds.index("dragon_wait_complete"), kinds.index("recovery_start"))
            self.assertLess(kinds.index("recovery_complete"), kinds.index("hosting_active"))

    def test_startup_at_interval_end_waits_for_next_start_including_next_day(self):
        for end_hour, target in ((11, local_time(hour=12, minute=0, second=0)),
                                 (16, local_time(hour=18, minute=0, second=0)),
                                 (22, local_time(day=6, hour=9, minute=0, second=0))):
            with self.subTest(end_hour=end_hour):
                pages = self.recovery_pages()
                pages[8] = replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})
                with self.replay(pages, max_checks=1) as (dragon, ctx):
                    base = local_time(hour=end_hour, minute=0, second=0)
                    # Jump the wall clock after its first bounded sleep; game
                    # captures and inputs must remain untouched before then.
                    def wall_clock():
                        return base if ctx.clock.now == 0 else target

                    with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock):
                        report = dragon.run()
                    self.assertEqual(ctx.clock.waits[0], 30)
                    self.assertEqual(ctx.restarts, [(30, replay_helpers.PACKAGE)])
                    self.assertEqual(report.next_start, target.isoformat())
                    self.assertEqual(report.scheduled_waits, 1)
                    self.assertTrue(all(at >= 30 for at, _ in ctx.captures))
                    self.assertTrue(all(at >= 30 for at, _, _, _ in ctx.inputs))

    def test_startup_exact_start_runs_monitor_without_forced_restart(self):
        pages = [replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})]
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            with patch("pet_alliance.slay_dragon.beijing_now", return_value=local_time(hour=12, minute=0, second=0)):
                report = dragon.run()
            self.assertEqual(report.scheduled_waits, 0)
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(ctx.clock.waits, [])
            self.assertEqual(ctx.inputs, [])

    def test_startup_crossing_start_does_not_skip_to_the_following_session(self):
        pages = self.recovery_pages()
        pages[8] = replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            before = local_time(second=59)
            after = local_time(hour=12, minute=0, second=1)
            reads = iter((before, after))
            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=lambda: next(reads, after)):
                report = dragon.run()
            self.assertEqual(report.next_start, local_time(hour=12, minute=0, second=0).isoformat())
            self.assertEqual(report.scheduled_waits, 1)
            self.assertEqual(ctx.restarts, [(0, replay_helpers.PACKAGE)])
            # Only transition polling is allowed after the immediate restart.
            self.assertLess(ctx.clock.now, 10)
            self.assertEqual(ctx.captures[0], (0, 1))

    def test_wait_respects_max_runtime_without_restart_or_game_input(self):
        pages = [replay_helpers.Page()]
        with self.replay(pages, max_runtime=7, max_checks=1) as (dragon, ctx):
            base = local_time(hour=22, minute=0, second=0)
            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=lambda: base + timedelta(seconds=ctx.clock.now)):
                with self.assertRaisesRegex(AutomationError, "最长运行时间"):
                    dragon.run()
            self.assertLessEqual(ctx.clock.now, 7)
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(ctx.captures, [])
            summary = self.summary(dragon)
            self.assertEqual(summary["status"], "stopped")
            self.assertIn("最长运行时间", summary["reason"])
            self.assertEqual(summary["scheduled_waits"], 1)

    def test_interrupt_during_wait_writes_interrupted_report(self):
        pages = [replay_helpers.Page()]
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            base = local_time(hour=22, minute=0, second=0)

            def interrupt(_seconds):
                raise KeyboardInterrupt()

            ctx.wait = interrupt
            with patch("pet_alliance.slay_dragon.beijing_now", return_value=base):
                with self.assertRaises(KeyboardInterrupt):
                    dragon.run()
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(ctx.captures, [])
            self.assertEqual(self.summary(dragon)["status"], "interrupted")
            self.assertEqual(self.summary(dragon)["scheduled_waits"], 1)

    def test_outside_interval_preview_does_not_wait_or_restart(self):
        pages = [replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})]
        with self.replay(pages, dry_run=True) as (dragon, ctx):
            with patch("pet_alliance.slay_dragon.beijing_now", return_value=local_time()):
                report = dragon.run()
            self.assertEqual(report.status, "preview")
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(ctx.clock.waits, [])
            self.assertEqual(len(ctx.captures), 1)
            planned = next(event["planned"] for event in self.events(dragon) if event["event"] == "preview")
            self.assertIn("12:00:00", planned)

    def test_monitor_crossing_eleven_waits_then_runs_full_recovery(self):
        pages = self.recovery_pages()
        pages[0] = replay_helpers.Page(
            {"stop_hosting": replay_helpers.control("stop_hosting")}, after=(2, 9))
        pages[8] = replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})
        pages.append(replay_helpers.Page(
            {"start_hosting": replay_helpers.control("start_hosting")}, {"start_hosting": 9}))
        with self.replay(pages, max_checks=2) as (dragon, ctx):
            before = local_time(hour=10, minute=59, second=58)
            noon = local_time(hour=12, minute=0, second=0)

            def wall_clock():
                return (before + timedelta(seconds=ctx.clock.now) if ctx.clock.now < 35
                        else noon + timedelta(seconds=ctx.clock.now - 35))

            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock):
                report = dragon.run()
            self.assertEqual(ctx.restarts, [(35, replay_helpers.PACKAGE)])
            self.assertEqual(report.scheduled_waits, 1)
            self.assertEqual(report.next_start, noon.isoformat())
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "notice", "dragon_entry", "dragon_npc", "dragon_npc", "quest_mark"])
            self.assertTrue(all(at >= 35 for at, _, _, _ in ctx.inputs))
            self.assertFalse(any(5 <= at < 35 for at, _ in ctx.captures))
            kinds = [event["event"] for event in self.events(dragon)]
            self.assertLess(kinds.index("dragon_wait_complete"), kinds.index("recovery_start"))

    def test_monitor_crossing_sixteen_and_twenty_two_waits_for_next_opening(self):
        for hour, target in ((16, local_time(hour=18, minute=0, second=0)),
                             (22, local_time(day=6, hour=9, minute=0, second=0))):
            with self.subTest(end_hour=hour):
                pages = [replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})]
                with self.replay(pages, max_checks=2) as (dragon, ctx):
                    before = local_time(hour=hour - 1, minute=59, second=58)

                    def wall_clock():
                        return (before + timedelta(seconds=ctx.clock.now) if ctx.clock.now < 35
                                else target + timedelta(seconds=ctx.clock.now - 35))

                    recoveries = []
                    with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock), \
                            patch.object(dragon, "recover", side_effect=lambda: recoveries.append(wall_clock())):
                        report = dragon.run()
                    self.assertEqual(recoveries, [target])
                    self.assertEqual(report.scheduled_waits, 1)
                    self.assertEqual(report.next_start, target.isoformat())
                    self.assertEqual(ctx.captures, [(0, 0), (35, 0)])
                    self.assertEqual(ctx.inputs, [])
                    self.assertEqual(ctx.clock.waits, [5, 30])

    def test_cloud_entry_crossing_end_discards_old_deadline_and_restarts_after_wait(self):
        pages = self.recovery_pages(cloud_seconds=240)
        pages[0] = replay_helpers.Page({"town_eye": replay_helpers.control("town_eye")})
        pages[8] = replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})
        with self.replay(pages, poll_interval=30, transition_timeout=90, max_checks=2) as (dragon, ctx):
            before = local_time(hour=10, minute=58, second=0)
            noon = local_time(hour=12, minute=0, second=0)

            def wall_clock():
                return (before + timedelta(seconds=ctx.clock.now) if ctx.clock.now < 150
                        else noon + timedelta(seconds=ctx.clock.now - 150))

            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock):
                report = dragon.run()
            self.assertEqual(ctx.restarts, [(150, replay_helpers.PACKAGE)])
            self.assertEqual(report.status, "limited")
            self.assertEqual(report.scheduled_waits, 1)
            self.assertTrue(all(at >= 150 for at, _, _, _ in ctx.inputs))
            self.assertFalse(any(120 <= at < 150 for at, _ in ctx.captures))
            entrance = next(item for item in ctx.inputs if item[2] == "dragon_entry")
            # The old entrance deadline was t=300. A fresh five-minute window
            # after relogin lets this new cloud clear at t=420 successfully.
            self.assertEqual(entrance[0], 420)
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "notice", "dragon_entry", "dragon_npc", "dragon_npc", "quest_mark"])

    def test_capture_or_recognition_crossing_end_blocks_stale_start_hosting_tap(self):
        for delayed_operation in ("capture", "recognition"):
            with self.subTest(operation=delayed_operation):
                pages = [replay_helpers.Page(
                    {"start_hosting": replay_helpers.control("start_hosting")}, {"start_hosting": 0})]
                with self.replay(pages, max_checks=2) as (dragon, ctx):
                    before = local_time(hour=10, minute=59, second=59)
                    noon = local_time(hour=12, minute=0, second=0)

                    def wall_clock():
                        return (before + timedelta(seconds=ctx.clock.now) if ctx.clock.now < 31
                                else noon + timedelta(seconds=ctx.clock.now - 31))

                    capture = ctx.capture
                    find = dragon.find
                    delayed = False

                    def slow_capture():
                        nonlocal delayed
                        frame = capture()
                        if delayed_operation == "capture" and not delayed:
                            ctx.clock.now += 1
                            delayed = True
                        return frame

                    def slow_find(name, frame=None):
                        nonlocal delayed
                        match = find(name, frame)
                        if delayed_operation == "recognition" and name == "start_hosting" and not delayed:
                            ctx.clock.now += 1
                            delayed = True
                        return match

                    recoveries = []
                    with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock), \
                            patch.object(ctx, "capture", side_effect=slow_capture), \
                            patch.object(dragon, "find", side_effect=slow_find), \
                            patch.object(dragon, "recover", side_effect=lambda: recoveries.append(wall_clock())):
                        report = dragon.run()
                    self.assertEqual(recoveries, [noon])
                    self.assertEqual(report.scheduled_waits, 1)
                    self.assertEqual(report.start_clicks, 1)
                    self.assertEqual([(at, name) for at, _, name, _ in ctx.inputs], [(31, "start_hosting")])
                    self.assertEqual(ctx.restarts, [])

    def test_recovery_crossing_end_waits_before_restart_command(self):
        pages = self.recovery_pages()
        pages[8] = replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})
        with self.replay(pages, max_checks=2) as (dragon, ctx):
            before = local_time(hour=10, minute=59, second=59)
            noon = local_time(hour=12, minute=0, second=0)

            def wall_clock():
                return (before + timedelta(seconds=ctx.clock.now) if ctx.clock.now < 31
                        else noon + timedelta(seconds=ctx.clock.now - 31))

            town_control = dragon.town_control
            delayed = False

            def slow_unknown_page(frame):
                nonlocal delayed
                result = town_control(frame)
                if not delayed:
                    ctx.clock.now += 1
                    delayed = True
                return result

            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock), \
                    patch.object(dragon, "town_control", side_effect=slow_unknown_page):
                report = dragon.run()
            self.assertEqual(ctx.restarts, [(31, replay_helpers.PACKAGE)])
            self.assertEqual(report.scheduled_waits, 1)
            self.assertTrue(all(at >= 31 for at, _, _, _ in ctx.inputs))
            kinds = [event["event"] for event in self.events(dragon)]
            self.assertEqual(kinds.count("recovery_start"), 1)
            self.assertLess(kinds.index("dragon_wait_complete"), kinds.index("recovery_start"))

    def test_login_or_town_recognition_expiring_at_end_waits_instead_of_timing_out(self):
        for waiting_for in ("login", "town"):
            with self.subTest(waiting_for=waiting_for):
                pages = self.recovery_pages()
                pages[8] = replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})
                with self.replay(pages, login_timeout=1, town_timeout=1, max_checks=2) as (dragon, ctx):
                    before = local_time(hour=10, minute=59, second=59)
                    noon = local_time(hour=12, minute=0, second=0)

                    def wall_clock():
                        return (before + timedelta(seconds=ctx.clock.now) if ctx.clock.now < 31
                                else noon + timedelta(seconds=ctx.clock.now - 31))

                    find = dragon.find
                    delayed = False

                    def slow_missing_control(name, frame=None):
                        nonlocal delayed
                        delayed_name = "login_enter" if waiting_for == "login" else "notice"
                        if ctx.index == 1 and name == delayed_name and not delayed:
                            # Finish the unsuccessful predicate exactly when
                            # both its ordinary deadline and the active slot end.
                            ctx.clock.now += 1
                            delayed = True
                            return None
                        return find(name, frame)

                    with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock), \
                            patch.object(dragon, "find", side_effect=slow_missing_control):
                        report = dragon.run()
                    self.assertTrue(delayed)
                    self.assertEqual(report.status, "limited")
                    self.assertEqual(report.scheduled_waits, 1)
                    self.assertEqual(report.next_start, noon.isoformat())
                    self.assertEqual(ctx.restarts, [(0, replay_helpers.PACKAGE), (31, replay_helpers.PACKAGE)])
                    self.assertTrue(all(at >= 31 for at, _, _, _ in ctx.inputs))
                    self.assertEqual([item[2] for item in ctx.inputs],
                                     ["login_enter", "notice", "dragon_entry", "dragon_npc", "dragon_npc", "quest_mark"])
                    kinds = [event["event"] for event in self.events(dragon)]
                    self.assertEqual(kinds.count("recovery_start"), 2)
                    self.assertEqual(kinds.count("recovery_complete"), 1)

    def test_adjacent_active_hours_do_not_wait_or_restart(self):
        for hour in (9, 12, 13, 14, 18, 19, 20):
            with self.subTest(hour=hour):
                pages = [replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})]
                with self.replay(pages, max_checks=2) as (dragon, ctx):
                    before = local_time(hour=hour, minute=59, second=58)
                    with patch("pet_alliance.slay_dragon.beijing_now",
                               side_effect=lambda: before + timedelta(seconds=ctx.clock.now)):
                        report = dragon.run()
                    self.assertEqual(report.scheduled_waits, 0)
                    self.assertEqual(ctx.restarts, [])
                    self.assertEqual(ctx.inputs, [])
                    self.assertEqual(ctx.captures, [(0, 0), (5, 0)])
                    self.assertEqual(ctx.clock.waits, [5])

    def test_running_from_morning_to_evening_pauses_only_at_two_real_gaps(self):
        pages = [replay_helpers.Page({"stop_hosting": replay_helpers.control("stop_hosting")})]
        with self.replay(pages, max_checks=8) as (dragon, ctx):
            # Fast-forward wall time between monitoring checks. Simulated
            # game time stays short; waiting still uses its real 30s chunks.
            phases = ((0, local_time(hour=9, minute=5, second=0)),
                      (5, local_time(hour=10, minute=59, second=59)),
                      (10, local_time(hour=11, minute=0, second=0)),
                      (40, local_time(hour=12, minute=0, second=0)),
                      (45, local_time(hour=15, minute=59, second=59)),
                      (50, local_time(hour=16, minute=0, second=0)),
                      (80, local_time(hour=18, minute=0, second=0)),
                      (85, local_time(hour=19, minute=0, second=0)),
                      (90, local_time(hour=20, minute=0, second=0)))

            def wall_clock():
                return next(value for at, value in reversed(phases) if ctx.clock.now >= at)

            recoveries = []
            with patch("pet_alliance.slay_dragon.beijing_now", side_effect=wall_clock), \
                    patch.object(dragon, "recover", side_effect=lambda: recoveries.append(wall_clock())):
                report = dragon.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(report.scheduled_waits, 2)
            self.assertEqual(recoveries, [local_time(hour=12, minute=0, second=0),
                                          local_time(hour=18, minute=0, second=0)])
            self.assertEqual(ctx.captures, [(0, 0), (5, 0), (40, 0), (45, 0),
                                           (80, 0), (85, 0), (90, 0), (95, 0)])
            self.assertEqual(ctx.clock.waits, [5, 5, 30, 5, 5, 30, 5, 5, 5])
            self.assertEqual(ctx.inputs, [])


if __name__ == "__main__":
    unittest.main()
