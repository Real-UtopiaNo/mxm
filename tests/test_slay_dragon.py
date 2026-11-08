from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from pet_alliance import AutomationError, Screenshot
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.dragon_vision import DragonVision
from pet_alliance.slay_dragon import SlayDragon, _OutsideDragonTime
from pet_alliance.vision import Match, VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "slay_dragon"
PACKAGE = "com.example.dragon"


def control(name, x=100, y=100):
    return Match(name, x, y, 20, 20, 1.0)


@dataclass
class Page:
    controls: dict[str, Match] = field(default_factory=dict)
    taps: dict[str, int] = field(default_factory=dict)
    after: tuple[float, int] | None = None
    fixture: str | None = None
    # Some labels locate a control whose clickable object is elsewhere.
    # Keep the observed label box and the independent interaction point apart.
    tap_points: dict[str, tuple[int, int]] = field(default_factory=dict)


class ReplayClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def monotonic(self):
        return self.now

    def wait(self, seconds):
        if seconds < 0:
            raise AssertionError("A replay must never wait a negative duration")
        self.waits.append(seconds)
        self.now += seconds


class ReplayContext:
    """Reject taps outside the control visible in the current captured state."""

    def __init__(self, directory, pages, *, restart_page=1, size=(1920, 1080),
                 dry_run=False, daily_options=None, **options):
        self.project_root = Path(directory)
        self.settings = {
            "daily_hero": {"poll_interval": .5, "min_click_interval": .1,
                           "click_retry_interval": 1, "max_click_retries": 3, **(daily_options or {})},
            "slay_dragon": {"package": PACKAGE, "transition_timeout": 10, **options},
        }
        self.log = logging.getLogger("dragon-replay")
        self.dry_run = dry_run
        # Production task entry selects this group before starting the job.
        self.active_pet_loadout = "屠龙"
        self.pages = pages
        self.restart_page = restart_page
        self.size = size
        self.index = 0
        self.entered_at = 0.0
        self.clock = ReplayClock()
        self.wait = self.clock.wait
        self.inputs = []
        self.restarts = []
        self.captures = []
        self.screenshots = {}
        self.capture_error = None

    def enter(self, index):
        self.index = index
        self.entered_at = self.clock.now

    def visible_page(self):
        while self.pages[self.index].after:
            delay, destination = self.pages[self.index].after
            if self.clock.now - self.entered_at < delay:
                break
            self.enter(destination)
        return self.pages[self.index]

    def capture(self):
        if self.capture_error:
            raise self.capture_error
        page = self.visible_page()
        self.captures.append((self.clock.now, self.index))
        if self.index not in self.screenshots:
            if page.fixture:
                source = cv2.imdecode(np.fromfile(str(FIXTURES / page.fixture), np.uint8), cv2.IMREAD_COLOR)
                if source is None:
                    raise AssertionError(f"Cannot read dragon fixture {page.fixture}")
                image = cv2.resize(source, self.size)
            else:
                image = np.full((self.size[1], self.size[0], 3), self.index + 1, np.uint8)
            self.screenshots[self.index] = Screenshot(cv2.imencode(".png", image)[1].tobytes(), *self.size)
        return self.screenshots[self.index]

    def tap(self, x, y):
        if self.dry_run:
            raise AssertionError("Preview must not call tap")
        page = self.visible_page()
        point = (x, y)
        visible = {
            name: (round(page.tap_points.get(name, match.center)[0] * self.size[0] / 1920),
                   round(page.tap_points.get(name, match.center)[1] * self.size[1] / 1080))
            for name, match in page.controls.items()
        }
        possible = [name for name, expected in visible.items() if expected == point and name in page.taps]
        if len(possible) != 1:
            raise AssertionError(f"Unexpected/stale tap {point} on page {self.index}: {visible}")
        name = possible[0]
        self.inputs.append((self.clock.now, self.index, name, point))
        self.enter(page.taps[name])

    def restart_app(self, package):
        if self.dry_run:
            raise AssertionError("Preview must not restart the game")
        self.restarts.append((self.clock.now, package))
        if self.restart_page is not None:
            self.enter(self.restart_page)


class MarkerVision:
    def __init__(self, ctx):
        self.ctx = ctx

    def find(self, frame, name):
        index = int(frame.image[0, 0, 0]) - 1
        return self.ctx.pages[index].controls.get(name)


class SlayDragonTests(unittest.TestCase):
    @contextmanager
    def replay(self, pages, *, real_vision=False, **options):
        with tempfile.TemporaryDirectory() as directory:
            ctx = ReplayContext(directory, pages, **options)
            # Keep every replay on an active slot irrespective of the real
            # system time. Tests of waiting may override this wall-clock stub.
            clock_start = datetime(2026, 11, 5, 9, 30, tzinfo=timezone(timedelta(hours=8)))
            clock_now = lambda: clock_start + timedelta(seconds=ctx.clock.now)
            select_group = lambda context, group: setattr(context, "active_pet_loadout", group)
            if real_vision:
                shutil.copytree(ROOT / "assets" / "slay_dragon", Path(directory) / "assets" / "slay_dragon")
                with patch("pet_alliance.slay_dragon.time.monotonic", ctx.clock.monotonic), \
                        patch("pet_alliance.slay_dragon.beijing_now", side_effect=clock_now), \
                        patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select_group):
                    yield SlayDragon(ctx), ctx
            else:
                with patch("pet_alliance.slay_dragon.DragonVision", return_value=MarkerVision(ctx)), \
                        patch("pet_alliance.slay_dragon.time.monotonic", ctx.clock.monotonic), \
                        patch("pet_alliance.slay_dragon.beijing_now", side_effect=clock_now), \
                        patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select_group):
                    yield SlayDragon(ctx), ctx

    def read_summary(self, dragon):
        return json.loads((dragon.run_dir / "summary.json").read_text(encoding="utf-8"))

    def recovery_pages(self, *, cloud_seconds=0, both_npc_and_mark=False):
        pages = [
            Page(),
            Page({"login_enter": control("login_enter", 900, 900)}, {"login_enter": 2}),
            # The notification is dismissed at its lower edge, never the body.
            Page({"notice": control("notice", 940, 1000)}, {"notice": 3}),
            Page({"town": control("town", 800, 300)}, after=(cloud_seconds, 4)),
            Page({"town": control("town", 800, 300), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 5}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"dragon_npc": control("dragon_npc", 700, 300)}, {"dragon_npc": 6}),
            Page({"dragon_npc": control("dragon_npc", 1000, 450)}, {"dragon_npc": 7}),
            Page({"quest_mark": control("quest_mark", 900, 250)}, {"quest_mark": 8}),
            Page({"start_hosting": control("start_hosting", 1600, 650)}, {"start_hosting": 8}),
        ]
        if both_npc_and_mark:
            pages[6].controls["quest_mark"] = control("quest_mark", 900, 250)
        return pages

    def test_stop_hosting_is_observed_without_input(self):
        with self.replay([Page({"stop_hosting": control("stop_hosting")})]) as (dragon, ctx):
            dragon.step()
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(len(ctx.captures), 1)

    def test_recovery_selects_dragon_group_after_notice_is_closed(self):
        with self.replay(self.recovery_pages(cloud_seconds=1)) as (dragon, ctx):
            selected_on = []

            def select(context, group):
                selected_on.append((ctx.index, group))
                self.assertTrue(dragon.uncovered_town(dragon.observe()))
                context.active_pet_loadout = group
                context.pet_loadout_last_click_at = ctx.clock.now

            with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select) as switch:
                dragon.recover()
            switch.assert_called_once_with(ctx, "屠龙")
            self.assertEqual(selected_on, [(3, "屠龙")])
            self.assertEqual([item[2] for item in ctx.inputs][:2], ["login_enter", "notice"])
            self.assertEqual(ctx.active_pet_loadout, "屠龙")

    def test_script_entry_outside_hours_waits_before_pet_preparation(self):
        from pet_alliance.pet_loadouts import run_with_loadout

        with self.replay(self.recovery_pages(), max_checks=1) as (dragon, ctx):
            selected_at = []
            base = datetime(2026, 11, 5, 11, 59, 58, tzinfo=timezone(timedelta(hours=8)))

            def select(context, group):
                selected_at.append(ctx.clock.now)
                context.active_pet_loadout = group

            with patch("pet_alliance.slay_dragon.beijing_now",
                       side_effect=lambda: base + timedelta(seconds=ctx.clock.now)), \
                    patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select):
                run_with_loadout(dragon, "屠龙")
            self.assertEqual(ctx.clock.waits[0], 2)
            self.assertTrue(selected_at)
            self.assertTrue(all(at >= 2 for at in selected_at))
            self.assertTrue(all(at >= 2 for at, _ in ctx.captures))
            self.assertEqual(ctx.restarts[0][0], 2)

    def test_new_active_run_reselects_group_even_with_previous_cached_value(self):
        from pet_alliance.pet_loadouts import run_with_loadout

        with self.replay([Page({"stop_hosting": control("stop_hosting")})], max_checks=1) as (dragon, ctx):
            ctx.active_pet_loadout = "屠龙"
            with patch("pet_alliance.pet_loadouts.switch_pet_loadout") as select:
                run_with_loadout(dragon, "屠龙")
            select.assert_called_once_with(ctx, "屠龙")

    def test_session_closes_during_pet_preparation_before_next_game_input(self):
        with self.replay([Page()]) as (dragon, ctx):
            base = datetime(2026, 11, 5, 10, 59, 59, tzinfo=timezone(timedelta(hours=8)))
            original_control = Mock()
            ctx.control = original_control

            def select(context, group):
                context.control("capture")
                ctx.clock.wait(1)
                context.control("tap")
                self.fail("时段结束后不应继续换组输入")

            with patch("pet_alliance.slay_dragon.beijing_now",
                       side_effect=lambda: base + timedelta(seconds=ctx.clock.now)), \
                    patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select):
                with self.assertRaises(_OutsideDragonTime):
                    dragon.prepare_dragon_loadout(force=True)
            original_control.assert_called_once_with("capture")
            self.assertIs(ctx.control, original_control)
            self.assertEqual(ctx.inputs, [])

    def test_pet_preparation_preserves_existing_pause_callback(self):
        with self.replay([Page()]) as (dragon, ctx):
            original_control = Mock(side_effect=TaskInterrupted("paused", "暂停"))
            ctx.control = original_control
            with patch("pet_alliance.pet_loadouts.switch_pet_loadout",
                       side_effect=lambda context, group: context.control("tap")):
                with self.assertRaises(TaskInterrupted):
                    dragon.prepare_dragon_loadout(force=True)
            original_control.assert_called_once_with("tap")
            self.assertIs(ctx.control, original_control)
            self.assertEqual(ctx.inputs, [])

    def test_shared_town_deadline_stops_pet_input_and_restores_callback(self):
        pages = [Page({"town_eye": control("town_eye")})]
        with self.replay(pages, cloud_timeout=5) as (dragon, ctx):
            ctx.active_pet_loadout = None
            original_control = Mock()
            ctx.control = original_control

            def select(context, group):
                ctx.clock.wait(4.9)
                context.control("capture")
                ctx.clock.wait(.2)
                context.control("tap")
                self.fail("共享时限结束后不应继续换组输入")

            with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select):
                with self.assertRaisesRegex(AutomationError, "共享 5 秒"):
                    dragon.enter_dragon()
            original_control.assert_called_once_with("capture")
            self.assertIs(ctx.control, original_control)
            self.assertEqual(ctx.inputs, [])

    def test_clouded_dragon_label_does_not_delay_pet_selection(self):
        pages = self.recovery_pages(cloud_seconds=2)[3:]
        # Rebase the destinations after taking only the town-to-dragon branch.
        pages[0].after = (2, 1)
        pages[1].taps = {"dragon_entry": 2}
        pages[2].taps = {"dragon_npc": 3}
        pages[3].taps = {"dragon_npc": 4}
        pages[4].taps = {"quest_mark": 5}
        pages[5].taps = {"start_hosting": 5}
        with self.replay(pages) as (dragon, ctx):
            ctx.active_pet_loadout = None
            selected_on = []

            def select(context, group):
                selected_on.append(ctx.index)
                self.assertIsNone(dragon.find("dragon_entry", dragon.observe()))
                context.active_pet_loadout = group

            with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select):
                dragon.enter_dragon()
            self.assertEqual(selected_on, [0])
            self.assertEqual([item[2] for item in ctx.inputs][0], "dragon_entry")

    def test_pet_selection_keeps_original_shared_town_deadline(self):
        pages = [Page({"town_eye": control("town_eye"),
                       "dragon_entry": control("dragon_entry", 1200, 350)},
                      {"dragon_entry": 0}, tap_points={"dragon_entry": (1210, 230)})]
        with self.replay(pages, cloud_timeout=5) as (dragon, ctx):
            ctx.active_pet_loadout = None

            def select(context, group):
                ctx.clock.wait(5)
                context.active_pet_loadout = group

            with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=select):
                with self.assertRaisesRegex(AutomationError, "共享 5 秒"):
                    dragon.enter_dragon()
            self.assertEqual(ctx.inputs, [])

    def test_pause_during_pet_selection_never_enters_dragon(self):
        pages = [Page({"town_eye": control("town_eye"),
                       "dragon_entry": control("dragon_entry", 1200, 350)},
                      {"dragon_entry": 0}, tap_points={"dragon_entry": (1210, 230)})]
        with self.replay(pages) as (dragon, ctx):
            ctx.active_pet_loadout = None
            with patch("pet_alliance.pet_loadouts.switch_pet_loadout",
                       side_effect=TaskInterrupted("paused", "暂停")):
                with self.assertRaises(TaskInterrupted):
                    dragon.enter_dragon()
            self.assertEqual(ctx.inputs, [])

    def test_start_hosting_taps_once_without_confirmation_or_retry(self):
        pages = [Page({"start_hosting": control("start_hosting")}, {"start_hosting": 0})]
        with self.replay(pages) as (dragon, ctx):
            dragon.step()
            self.assertEqual([item[2] for item in ctx.inputs], ["start_hosting"])
            self.assertEqual(len(ctx.captures), 1)
            self.assertEqual(ctx.clock.waits, [])
            self.assertEqual(dragon.report.start_clicks, 1)

    def test_town_eye_start_enters_dragon_without_restart_or_login(self):
        pages = [
            Page({"town_eye": control("town_eye", 1200, 200),
                  "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 1}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"dragon_npc": control("dragon_npc", 700, 300)}, {"dragon_npc": 2}),
            Page({"quest_mark": control("quest_mark", 900, 250)}, {"quest_mark": 3}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            report = dragon.run()
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(report.restarts, 0)
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["dragon_entry", "dragon_npc", "quest_mark"])
            self.assertEqual(ctx.inputs[0][3], (1210, 230))
            self.assertEqual(ctx.index, 3)

    def test_town_eye_remains_anchor_while_entry_text_is_hidden_by_cloud(self):
        pages = [
            Page({"town_eye": control("town_eye", 1200, 200)}, after=(45, 1)),
            Page({"town_eye": control("town_eye", 1200, 200),
                  "dragon_entry": control("dragon_entry", 1400, 450)},
                 {"dragon_entry": 2}, tap_points={"dragon_entry": (1410, 330)}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual(ctx.restarts, [])
            self.assertEqual([item[2] for item in ctx.inputs], ["dragon_entry"])
            self.assertGreaterEqual(ctx.inputs[0][0], 45)
            self.assertLess(ctx.inputs[0][0], 300)
            self.assertEqual(ctx.inputs[0][3], (1410, 330))
            self.assertFalse(any(index == 0 for _, index, *_ in ctx.inputs))

    def test_town_start_notice_and_cloud_detection_share_one_deadline(self):
        pages = [
            Page({"town_eye": control("town_eye")}, after=(120, 1)),
            Page({"town_eye": control("town_eye"), "notice": control("notice", 940, 1000),
                  "dragon_entry": control("dragon_entry", 1200, 350)}, {"notice": 2}),
            Page({"town_eye": control("town_eye")}, after=(120, 3)),
            Page({"town_eye": control("town_eye"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 4}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual(ctx.restarts, [])
            self.assertEqual([item[2] for item in ctx.inputs], ["notice", "dragon_entry"])
            self.assertGreaterEqual(ctx.inputs[1][0], 240)
            self.assertLess(ctx.inputs[1][0], 300)
            self.assertEqual(ctx.index, 4)

    def test_town_start_notice_does_not_reset_five_minute_timeout(self):
        pages = [
            Page({"town_eye": control("town_eye")}, after=(200, 1)),
            Page({"notice": control("notice", 940, 1000)}, {"notice": 2}),
            Page({"town_eye": control("town_eye")}),
        ]
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            with self.assertRaisesRegex(AutomationError, "300 秒"):
                dragon.run()
            self.assertEqual(ctx.restarts, [])
            self.assertEqual([item[2] for item in ctx.inputs], ["notice"])
            self.assertGreaterEqual(ctx.clock.now, 300)
            self.assertLessEqual(ctx.clock.now, 305)

    def test_notice_at_start_is_dismissed_without_restart_before_town_entry(self):
        pages = [
            Page({"notice": control("notice", 940, 1000)}, {"notice": 1}),
            Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 2}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual(ctx.restarts, [])
            self.assertEqual([item[2] for item in ctx.inputs], ["notice", "dragon_entry"])
            self.assertEqual(ctx.index, 2)

    def test_recovery_closes_announcement_before_old_notice_and_town_entry(self):
        pages = [
            Page(),
            Page({"login_enter": control("login_enter")}, {"login_enter": 2}),
            # The town and its entrance may be visible behind the announcement.
            # Neither is clickable until both covering notifications close.
            Page({"announcement_close": control("announcement_close", 1800, 150),
                  "notice": control("notice", 940, 1000),
                  "town_eye": control("town_eye"),
                  "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"announcement_close": 3}),
            Page({"notice": control("notice", 940, 1000),
                  "town_eye": control("town_eye"),
                  "dragon_entry": control("dragon_entry", 1200, 350)}, {"notice": 4}),
            Page({"town_eye": control("town_eye"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 5}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            report = dragon.run()
            self.assertEqual(ctx.restarts, [(0, PACKAGE)])
            self.assertEqual(report.restarts, 1)
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "announcement_close", "notice", "dragon_entry"])
            self.assertEqual(ctx.inputs[1][3], (1810, 160))
            self.assertEqual(ctx.index, 5)

    def test_announcement_page_anchor_waits_for_close_control_without_restart(self):
        pages = [
            Page({"announcement_page": control("announcement_page", 1100, 150)}, after=(3, 1)),
            Page({"announcement_page": control("announcement_page", 1100, 150),
                  "announcement_close": control("announcement_close", 1800, 150)},
                 {"announcement_close": 2}),
            Page({"town_eye": control("town_eye"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 3}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual(ctx.restarts, [])
            self.assertEqual([item[2] for item in ctx.inputs], ["announcement_close", "dragon_entry"])
            self.assertGreaterEqual(ctx.inputs[0][0], 3)
            self.assertFalse(any(index == 0 for _, index, *_ in ctx.inputs))

    def test_both_notification_orders_at_start_need_no_restart(self):
        for first, second in (("announcement_close", "notice"), ("notice", "announcement_close")):
            with self.subTest(order=(first, second)):
                pages = [
                    Page({first: control(first, 1800, 150)}, {first: 1}),
                    Page({second: control(second, 940, 1000)}, {second: 2}),
                    Page({"town_eye": control("town_eye"),
                          "dragon_entry": control("dragon_entry", 1200, 350)},
                         {"dragon_entry": 3}, tap_points={"dragon_entry": (1210, 230)}),
                    Page({"stop_hosting": control("stop_hosting")}),
                ]
                with self.replay(pages, max_checks=1) as (dragon, ctx):
                    dragon.run()
                    self.assertEqual(ctx.restarts, [])
                    self.assertEqual([item[2] for item in ctx.inputs], [first, second, "dragon_entry"])
                    self.assertEqual(ctx.index, 3)

    def test_late_announcement_and_old_notice_relocate_each_retry_with_separate_budgets(self):
        for size, close_points in (((1920, 1080), [(1810, 160), (1770, 190)]),
                                   ((1280, 720), [(1207, 107), (1180, 127)])):
            with self.subTest(size=size):
                pages = [
                    Page({"town_eye": control("town_eye"),
                          "dragon_entry": control("dragon_entry", 1200, 350)},
                         {"dragon_entry": 1}, tap_points={"dragon_entry": (1210, 230)}),
                    Page({"announcement_close": control("announcement_close", 1800, 150),
                          "dragon_entry": control("dragon_entry", 1200, 350)},
                         {"announcement_close": 2}),
                    Page({"announcement_close": control("announcement_close", 1760, 180)},
                         {"announcement_close": 3}),
                    Page({"notice": control("notice", 600, 1000)}, {"notice": 4}),
                    Page({"notice": control("notice", 1000, 1000)}, {"notice": 5}),
                    Page({"town_eye": control("town_eye"),
                          "dragon_entry": control("dragon_entry", 1400, 450)},
                         {"dragon_entry": 6}, tap_points={"dragon_entry": (1410, 330)}),
                    Page({"stop_hosting": control("stop_hosting")}),
                ]
                with self.replay(pages, size=size, max_checks=1, max_click_retries=1) as (dragon, ctx):
                    dragon.run()
                    self.assertEqual(ctx.restarts, [])
                    self.assertEqual([item[2] for item in ctx.inputs],
                                     ["dragon_entry", "announcement_close", "announcement_close",
                                      "notice", "notice", "dragon_entry"])
                    self.assertEqual([item[3] for item in ctx.inputs[1:3]], close_points)
                    for name in ("announcement_close", "notice", "dragon_entry"):
                        clicks = [item for item in ctx.inputs if item[2] == name]
                        self.assertEqual(len(clicks), 2)
                        self.assertGreaterEqual(clicks[1][0] - clicks[0][0], 1)
                    self.assertEqual(ctx.index, 6)

    def test_announcement_and_old_notice_share_original_five_minute_deadline(self):
        pages = [
            Page({"town_eye": control("town_eye")}, after=(200, 1)),
            Page({"announcement_close": control("announcement_close", 1800, 150)},
                 {"announcement_close": 2}),
            Page(after=(50, 3)),
            Page({"notice": control("notice", 940, 1000)}, {"notice": 4}),
            Page({"town_eye": control("town_eye")}),
        ]
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            with self.assertRaisesRegex(AutomationError, "300 秒"):
                dragon.run()
            self.assertEqual(ctx.restarts, [])
            self.assertEqual([item[2] for item in ctx.inputs], ["announcement_close", "notice"])
            self.assertGreaterEqual(ctx.inputs[0][0], 200)
            self.assertGreaterEqual(ctx.inputs[1][0], 250)
            self.assertGreaterEqual(ctx.clock.now, 300)
            self.assertLessEqual(ctx.clock.now, 305)

    def test_monitoring_rechecks_schedule_during_active_slot_without_game_input(self):
        pages = [Page({"stop_hosting": control("stop_hosting")})]
        with self.replay(pages, max_checks=3) as (dragon, ctx):
            current = datetime(2026, 11, 5, 9, 30, tzinfo=timezone(timedelta(hours=8)))
            with patch("pet_alliance.slay_dragon.beijing_now",
                       side_effect=lambda: current + timedelta(seconds=ctx.clock.now)) as wall_clock:
                report = dragon.run()
            self.assertGreaterEqual(wall_clock.call_count, 3)
            self.assertEqual(report.checks, 3)
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(ctx.clock.waits, [5, 5])

    def test_start_button_recognition_crossing_closing_time_waits_before_any_tap(self):
        pages = self.recovery_pages()
        pages[0] = Page({"start_hosting": control("start_hosting")}, {"start_hosting": 0})
        pages[-1] = Page({"stop_hosting": control("stop_hosting")})
        with self.replay(pages, max_checks=2) as (dragon, ctx):
            current = datetime(2026, 11, 5, 10, 59, 59, tzinfo=timezone(timedelta(hours=8)))
            find = dragon.vision.find
            crossed = False

            def cross_boundary_during_recognition(frame, name):
                nonlocal crossed
                match = find(frame, name)
                if name == "start_hosting" and ctx.index == 0 and match and not crossed:
                    # A capture began during the slot, but recognizing the
                    # visible button finishes after it has closed.
                    ctx.clock.wait(2)
                    crossed = True
                return match

            with patch("pet_alliance.slay_dragon.beijing_now",
                       side_effect=lambda: current + timedelta(seconds=ctx.clock.now)), \
                    patch.object(dragon.vision, "find", side_effect=cross_boundary_during_recognition):
                report = dragon.run()

            self.assertTrue(crossed)
            self.assertEqual(report.scheduled_waits, 1)
            self.assertEqual(report.restarts, 1)
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "notice", "dragon_entry", "dragon_npc", "dragon_npc", "quest_mark"])
            self.assertEqual(report.start_clicks, 0)
            # Noon is 3,601 seconds after the initial 10:59:59. The stale
            # start button must not be clicked between 11:00 and noon.
            self.assertEqual(ctx.restarts, [(3601, PACKAGE)])
            self.assertTrue(all(when >= 3601 for when, *_ in ctx.inputs))
            self.assertEqual(ctx.index, len(pages) - 1)

    def test_options_inherit_shared_click_timing_and_allow_dragon_overrides(self):
        shared = {"poll_interval": .8, "min_click_interval": 1.2,
                  "click_retry_interval": 2.1, "max_click_retries": 4}
        with self.replay([Page()], daily_options=shared) as (dragon, _):
            for name, value in shared.items():
                self.assertEqual(getattr(dragon.options, name), value)
            self.assertEqual(dragon.options.monitor_interval, 5)
            self.assertEqual(dragon.options.login_timeout, 90)
            self.assertEqual(dragon.options.town_timeout, 90)
            self.assertEqual(dragon.options.cloud_timeout, 300)
            self.assertEqual(dragon.options.max_checks, 0)
            self.assertEqual(dragon.options.max_runtime, 0)
        with self.replay([Page()], daily_options=shared, poll_interval=.2,
                         click_retry_interval=3.5) as (dragon, _):
            self.assertEqual(dragon.options.poll_interval, .2)
            self.assertEqual(dragon.options.click_retry_interval, 3.5)
            self.assertEqual(dragon.options.min_click_interval, 1.2)

    def test_monitor_checks_and_unconfirmed_start_clicks_are_five_seconds_apart(self):
        pages = [Page({"start_hosting": control("start_hosting")}, {"start_hosting": 0})]
        with self.replay(pages, max_checks=3) as (dragon, ctx):
            report = dragon.run()
            self.assertEqual(report.checks, 3)
            self.assertEqual(report.start_clicks, 3)
            self.assertEqual([item[0] for item in ctx.inputs], [0, 5, 10])
            self.assertEqual(ctx.clock.waits, [5, 5])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(self.read_summary(dragon)["checks"], 3)
            self.assertTrue((dragon.run_dir / "last_frame.jpg").is_file())

    def test_unknown_screen_runs_entire_recovery_and_relocates_npc(self):
        with self.replay(self.recovery_pages(), max_checks=1) as (dragon, ctx):
            report = dragon.run()
            self.assertEqual(ctx.restarts, [(0, PACKAGE)])
            self.assertEqual(report.restarts, 1)
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "notice", "dragon_entry", "dragon_npc", "dragon_npc", "quest_mark"])
            npc_points = [item[3] for item in ctx.inputs if item[2] == "dragon_npc"]
            self.assertEqual(npc_points, [(710, 310), (1010, 460)])
            self.assertEqual(ctx.inputs[1][3], (950, 1010))
            self.assertEqual(report.start_clicks, 0)
            self.assertEqual(ctx.index, 8)

    def test_cloud_hiding_entry_longer_than_normal_transition_is_waited_out(self):
        pages = self.recovery_pages(cloud_seconds=45)
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            dragon.run()
            entry_time = next(item[0] for item in ctx.inputs if item[2] == "dragon_entry")
            self.assertGreaterEqual(entry_time, 45)
            self.assertLess(entry_time, 300)
            self.assertFalse(any(item[1] == 3 for item in ctx.inputs))

    def test_delayed_notice_after_town_has_priority_over_visible_dragon_entry(self):
        pages = [
            Page(),
            Page({"login_enter": control("login_enter")}, {"login_enter": 2}),
            Page({"town": control("town")}, after=(120, 3)),
            # A later notification must still be handled after town was seen.
            # The dragon label may be visible through the notice, but cannot
            # receive input until the notice has been dismissed at its edge.
            Page({"town": control("town"), "notice": control("notice", 940, 1000),
                  "dragon_entry": control("dragon_entry", 1200, 350)}, {"notice": 4}),
            Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 5}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual([item[2] for item in ctx.inputs], ["login_enter", "notice", "dragon_entry"])
            self.assertGreaterEqual(ctx.inputs[1][0], 120)
            self.assertLess(ctx.inputs[2][0], 300)
            self.assertEqual(ctx.inputs[1][3], (950, 1010))
            self.assertEqual(ctx.inputs[2][3], (1210, 230))
            self.assertEqual(ctx.index, 5)

    def test_notice_after_missed_entry_resumes_at_relocated_icon_with_independent_retry_budget(self):
        for size, expected in (((1920, 1080), [(1210, 230), (610, 1010), (1410, 330)]),
                               ((1280, 720), [(807, 153), (407, 673), (940, 220)])):
            with self.subTest(size=size):
                pages = [
                    Page(),
                    Page({"login_enter": control("login_enter")}, {"login_enter": 2}),
                    Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1200, 350)},
                         {"dragon_entry": 3}, tap_points={"dragon_entry": (1210, 230)}),
                    Page({"town": control("town"), "notice": control("notice", 600, 1000),
                          "dragon_entry": control("dragon_entry", 1200, 350)}, {"notice": 4}),
                    Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1400, 450)},
                         {"dragon_entry": 5}, tap_points={"dragon_entry": (1410, 330)}),
                    Page({"stop_hosting": control("stop_hosting")}),
                ]
                with self.replay(pages, size=size, max_checks=1, max_click_retries=1) as (dragon, ctx):
                    dragon.run()
                    self.assertEqual([item[2] for item in ctx.inputs],
                                     ["login_enter", "dragon_entry", "notice", "dragon_entry"])
                    self.assertEqual([item[3] for item in ctx.inputs[1:]], expected)
                    entries = [item for item in ctx.inputs if item[2] == "dragon_entry"]
                    self.assertGreaterEqual(entries[1][0] - entries[0][0], 1)
                    self.assertEqual(ctx.index, 5)

    def test_notice_and_entry_cycles_share_one_five_minute_deadline(self):
        pages = [
            Page(),
            Page({"login_enter": control("login_enter")}, {"login_enter": 2}),
            Page({"town": control("town")}, after=(100, 3)),
            Page({"notice": control("notice", 600, 1000)}, {"notice": 4}),
            Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 5}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"town": control("town")}, after=(100, 6)),
            Page({"notice": control("notice", 1000, 1000)}, {"notice": 7}),
            Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1400, 450)},
                 {"dragon_entry": 8}, tap_points={"dragon_entry": (1410, 330)}),
            Page({"town": control("town")}),
        ]
        with self.replay(pages, poll_interval=5, max_checks=1, max_click_retries=1) as (dragon, ctx):
            with self.assertRaisesRegex(AutomationError, "300 秒"):
                dragon.run()
            stage_start = next(at for at, index in ctx.captures if index == 2)
            self.assertGreaterEqual(ctx.clock.now - stage_start, 300)
            self.assertLessEqual(ctx.clock.now - stage_start, 305)
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "notice", "dragon_entry", "notice", "dragon_entry"])
            self.assertEqual(ctx.index, 8)
            self.assertIn("300 秒", self.read_summary(dragon)["reason"])

    def test_initial_notice_starts_shared_timeout_and_later_notice_may_arrive_after_ninety_seconds(self):
        pages = [
            Page(),
            Page({"login_enter": control("login_enter")}, {"login_enter": 2}),
            Page({"notice": control("notice", 600, 1000)}, {"notice": 3}),
            # Dismissal/loading hides the town controls; a second notice is
            # still allowed after the old 90-second town stage would expire.
            Page(after=(120, 4)),
            Page({"notice": control("notice", 1000, 1000)}, {"notice": 5}),
            Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 6}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"stop_hosting": control("stop_hosting")}),
        ]
        with self.replay(pages, poll_interval=5, max_checks=1, max_click_retries=1) as (dragon, ctx):
            dragon.run()
            notices = [item for item in ctx.inputs if item[2] == "notice"]
            self.assertEqual(len(notices), 2)
            self.assertGreaterEqual(notices[1][0] - notices[0][0], 120)
            self.assertLess(ctx.clock.now - notices[0][0], 300)
            self.assertEqual([item[2] for item in ctx.inputs],
                             ["login_enter", "notice", "notice", "dragon_entry"])
            self.assertEqual(ctx.index, 6)

    def test_town_entry_and_retry_hit_icon_above_fresh_label_at_each_resolution(self):
        for size, expected in (((1920, 1080), [(1210, 230), (1410, 330)]),
                               ((1280, 720), [(807, 153), (940, 220)])):
            with self.subTest(size=size):
                pages = [
                    Page(),
                    Page({"login_enter": control("login_enter", 900, 900)}, {"login_enter": 2}),
                    Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1200, 350)},
                         {"dragon_entry": 3}, tap_points={"dragon_entry": (1210, 230)}),
                    # First click leaves town visible, then the label moves.
                    # Only the newly observed icon can advance recovery.
                    Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1400, 450)},
                         {"dragon_entry": 4}, tap_points={"dragon_entry": (1410, 330)}),
                    Page({"dragon_npc": control("dragon_npc", 700, 300)}, {"dragon_npc": 5}),
                    Page({"quest_mark": control("quest_mark", 900, 250)}, {"quest_mark": 6}),
                    Page({"stop_hosting": control("stop_hosting", 1600, 650)}),
                ]
                with self.replay(pages, size=size, max_checks=1) as (dragon, ctx):
                    dragon.run()
                    entries = [item for item in ctx.inputs if item[2] == "dragon_entry"]
                    self.assertEqual([item[3] for item in entries], expected)
                    self.assertGreaterEqual(entries[1][0] - entries[0][0], 1)
                    self.assertEqual(ctx.index, 6)

    def test_npc_disappearing_before_mark_causes_observation_not_stale_clicks(self):
        pages = self.recovery_pages()
        pages[6] = Page(after=(3, 7))
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual(len([item for item in ctx.inputs if item[2] == "dragon_npc"]), 1)
            self.assertFalse(any(item[1] == 6 for item in ctx.inputs))
            self.assertEqual(ctx.index, 8)

    def test_quest_mark_does_not_short_circuit_still_visible_npc(self):
        pages = self.recovery_pages(both_npc_and_mark=True)
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual([item[2] for item in ctx.inputs][-3:],
                             ["dragon_npc", "dragon_npc", "quest_mark"])

    def test_npc_approach_may_require_more_clicks_than_normal_retry_budget(self):
        pages = self.recovery_pages()[:5]
        for number in range(6):
            pages.append(Page({"dragon_npc": control("dragon_npc", 500 + number * 90, 300)},
                              {"dragon_npc": len(pages) + 1}))
        pages.append(Page({"quest_mark": control("quest_mark", 900, 250)}, {"quest_mark": 12}))
        pages.append(Page({"stop_hosting": control("stop_hosting", 1600, 650)}))
        with self.replay(pages, max_checks=1) as (dragon, ctx):
            dragon.run()
            clicks = [item for item in ctx.inputs if item[2] == "dragon_npc"]
            self.assertEqual(len(clicks), 6)
            self.assertEqual(len({item[3] for item in clicks}), 6)
            self.assertTrue(all(second[0] - first[0] >= 1 for first, second in zip(clicks, clicks[1:])))

    def test_enter_and_notice_have_independent_retry_budgets_and_fresh_coordinates(self):
        pages = [
            Page(),
            Page({"login_enter": control("login_enter", 600, 800)}, {"login_enter": 2}),
            Page({"login_enter": control("login_enter", 900, 800)}, {"login_enter": 3}),
            Page({"notice": control("notice", 600, 1000)}, {"notice": 4}),
            Page({"notice": control("notice", 1000, 1000)}, {"notice": 5}),
            Page({"town": control("town"), "dragon_entry": control("dragon_entry", 1200, 350)},
                 {"dragon_entry": 6}, tap_points={"dragon_entry": (1210, 230)}),
            Page({"dragon_npc": control("dragon_npc", 700, 300)}, {"dragon_npc": 7}),
            Page({"quest_mark": control("quest_mark", 900, 250)}, {"quest_mark": 8}),
            Page({"stop_hosting": control("stop_hosting", 1600, 650)}),
        ]
        with self.replay(pages, max_checks=1, max_click_retries=1) as (dragon, ctx):
            dragon.run()
            self.assertEqual([item[2] for item in ctx.inputs[:4]],
                             ["login_enter", "login_enter", "notice", "notice"])
            self.assertNotEqual(ctx.inputs[0][3], ctx.inputs[1][3])
            self.assertNotEqual(ctx.inputs[2][3], ctx.inputs[3][3])
            self.assertGreaterEqual(ctx.inputs[1][0] - ctx.inputs[0][0], 1)
            self.assertGreaterEqual(ctx.inputs[3][0] - ctx.inputs[2][0], 1)

    def test_login_enter_absent_times_out_at_ninety_seconds_without_taps(self):
        with self.replay([Page(), Page()], poll_interval=5, max_checks=1) as (dragon, ctx):
            with self.assertRaises(AutomationError):
                dragon.run()
            self.assertGreaterEqual(ctx.clock.now, 90)
            self.assertLessEqual(ctx.clock.now, 95)
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(len(ctx.restarts), 1)
            summary = self.read_summary(dragon)
            self.assertTrue(summary["reason"])
            self.assertNotEqual(summary["status"], "running")

    def test_town_not_reached_times_out_ninety_seconds_after_enter_click(self):
        pages = [Page(), Page({"login_enter": control("login_enter")}, {"login_enter": 2}), Page()]
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            with self.assertRaises(AutomationError):
                dragon.run()
            self.assertGreaterEqual(ctx.clock.now - ctx.inputs[0][0], 90)
            self.assertLessEqual(ctx.clock.now - ctx.inputs[0][0], 95)
            self.assertEqual([item[2] for item in ctx.inputs], ["login_enter"])

    def test_cloud_entry_wait_stops_at_five_minutes(self):
        pages = self.recovery_pages(cloud_seconds=1000)
        with self.replay(pages, poll_interval=5, max_checks=1) as (dragon, ctx):
            with self.assertRaises(AutomationError):
                dragon.run()
            # The shared town loop starts as soon as either the town or its
            # covering login notice appears, before that notice is clicked.
            stage_start = next(at for at, index in ctx.captures if index == 2)
            self.assertGreaterEqual(ctx.clock.now - stage_start, 300)
            self.assertLessEqual(ctx.clock.now - stage_start, 305)
            self.assertFalse(any(item[2] in {"dragon_entry", "dragon_npc", "quest_mark"} for item in ctx.inputs))

    def test_npc_missing_without_mark_stops_at_transition_timeout(self):
        pages = self.recovery_pages()
        pages[6] = Page()
        with self.replay(pages, max_checks=1, transition_timeout=4) as (dragon, ctx):
            with self.assertRaises(AutomationError):
                dragon.run()
            self.assertEqual(ctx.index, 6)
            self.assertEqual(len([item for item in ctx.inputs if item[2] == "dragon_npc"]), 1)
            self.assertFalse(any(item[1] == 6 for item in ctx.inputs))

    def test_start_click_uses_device_resolution(self):
        pages = [Page({"start_hosting": control("start_hosting", 1600, 650)}, {"start_hosting": 0})]
        with self.replay(pages, size=(1280, 720)) as (dragon, ctx):
            dragon.step()
            self.assertEqual(ctx.inputs[0][3], (1073, 440))

    def test_capture_failure_writes_stopped_summary_and_preserves_exception(self):
        with self.replay([Page()], max_checks=1) as (dragon, ctx):
            ctx.capture_error = AutomationError("ADB capture failed")
            with self.assertRaisesRegex(AutomationError, "ADB capture failed"):
                dragon.run()
            summary = self.read_summary(dragon)
            self.assertIn("ADB capture failed", summary["reason"])
            self.assertNotEqual(summary["status"], "running")
            self.assertEqual(ctx.restarts, [])

    def test_keyboard_interrupt_writes_interrupted_summary(self):
        with self.replay([Page()], max_checks=1) as (dragon, ctx):
            ctx.capture_error = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                dragon.run()
            self.assertEqual(self.read_summary(dragon)["status"], "interrupted")
            self.assertEqual(ctx.restarts, [])

    def test_unknown_preview_is_single_observation_without_recovery_or_wait(self):
        with self.replay([Page()], dry_run=True) as (dragon, ctx):
            report = dragon.run()
            self.assertEqual(report.status, "preview")
            self.assertEqual(len(ctx.captures), 1)
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(ctx.clock.waits, [])

    def test_start_button_preview_does_not_call_tap(self):
        pages = [Page({"start_hosting": control("start_hosting")}, {"start_hosting": 0})]
        with self.replay(pages, dry_run=True) as (dragon, ctx):
            dragon.run()
            self.assertEqual(len(ctx.captures), 1)
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertEqual(ctx.clock.waits, [])

    def test_monitor_runtime_limit_terminates_unbounded_check_loop(self):
        pages = [Page({"stop_hosting": control("stop_hosting")})]
        with self.replay(pages, max_runtime=7) as (dragon, ctx):
            with self.assertRaises(AutomationError):
                dragon.run()
            self.assertGreaterEqual(ctx.clock.now, 7)
            self.assertLessEqual(ctx.clock.now, 12)
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.restarts, [])
            self.assertTrue(self.read_summary(dragon)["reason"])

    def test_real_login_notice_town_npc_and_assistant_screens_replay(self):
        vision = DragonVision(ROOT / "assets" / "slay_dragon")
        for assistant, button in (("8.png", "stop_hosting"), ("9.png", "start_hosting")):
            with self.subTest(assistant=assistant):
                pages = [Page()]
                expected = [("10.png", "login_enter"), ("11.png", "notice"),
                            ("12.png", "dragon_entry"), ("13.png", "dragon_npc"),
                            (assistant, button)]
                for index, (filename, name) in enumerate(expected, start=1):
                    image = cv2.imdecode(np.fromfile(str(FIXTURES / filename), np.uint8), cv2.IMREAD_COLOR)
                    frame = VisualFrame(cv2.resize(image, (1920, 1080)))
                    visible = {key: match for key in vision.profile["templates"]
                               if (match := vision.find(frame, key)) is not None}
                    self.assertIn(name, visible)
                    target = index + 1 if index < len(expected) else index
                    actions = {name: target} if name != "stop_hosting" else {}
                    # Figure 12's purple dragon occupies the object above its
                    # text label; (1212, 243) is inside that observed icon.
                    tap_points = {"dragon_entry": (1212, 243)} if name == "dragon_entry" else {}
                    pages.append(Page(visible, actions, fixture=filename, tap_points=tap_points))
                self.assertGreater(pages[2].controls["notice"].center[1], 950)
                with self.replay(pages, real_vision=True, max_checks=2) as (dragon, ctx):
                    report = dragon.run()
                    expected_taps = ["login_enter", "notice", "dragon_entry", "dragon_npc"]
                    if button == "start_hosting":
                        expected_taps.append("start_hosting")
                    self.assertEqual([item[2] for item in ctx.inputs], expected_taps)
                    self.assertEqual(next(item[3] for item in ctx.inputs if item[2] == "dragon_entry"),
                                     (1212, 243))
                    self.assertEqual(report.restarts, 1)
                    self.assertEqual(report.checks, 2)
                    # A visible assistant already reached during NPC walking
                    # must not be toggled shut by clicking its quest mark.
                    self.assertFalse(any(item[2] == "quest_mark" for item in ctx.inputs))

    def test_real_announcement_and_old_notice_both_orders_recover_at_two_resolutions(self):
        vision = DragonVision(ROOT / "assets" / "slay_dragon")
        for size in ((1920, 1080), (1280, 720)):
            for notices in (("announcement.png", "11.png"), ("11.png", "announcement.png")):
                with self.subTest(size=size, order=notices):
                    expected = [("10.png", "login_enter")]
                    expected.extend((name, "announcement_close" if name == "announcement.png" else "notice")
                                    for name in notices)
                    expected.extend((("12.png", "dragon_entry"), ("13.png", "dragon_npc"),
                                     ("8.png", "stop_hosting")))
                    pages = [Page()]
                    for index, (filename, name) in enumerate(expected, start=1):
                        image = cv2.imdecode(np.fromfile(str(FIXTURES / filename), np.uint8), cv2.IMREAD_COLOR)
                        frame = VisualFrame(cv2.resize(image, size))
                        visible = {key: match for key in vision.profile["templates"]
                                   if (match := vision.find(frame, key)) is not None}
                        self.assertIn(name, visible)
                        points = {}
                        if name == "announcement_close":
                            # Independently observed close shield near (907, 84)
                            # in the original 956x540 capture, not matcher output.
                            visible[name] = control(name, 1811, 159)
                            points[name] = (1821, 169)
                        elif name == "dragon_entry":
                            points[name] = (1212, 243)
                        actions = {name: index + 1} if name != "stop_hosting" else {}
                        pages.append(Page(visible, actions, fixture=filename, tap_points=points))
                    with self.replay(pages, real_vision=True, size=size, max_checks=1) as (dragon, ctx):
                        report = dragon.run()
                        self.assertEqual(report.restarts, 1)
                        self.assertEqual([item[2] for item in ctx.inputs],
                                         [name for _, name in expected if name != "stop_hosting"])
                        announcement_tap = next(item[3] for item in ctx.inputs if item[2] == "announcement_close")
                        self.assertEqual(announcement_tap,
                                         (round(1821 * size[0] / 1920), round(169 * size[1] / 1080)))
                        self.assertEqual(ctx.index, len(pages) - 1)

    def test_real_failed_native_entry_confirms_scene_and_approaches_npc(self):
        for size in ((1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                # These interaction points are measured independently from the
                # screenshots. Do not build this page from find(): the failure
                # was a scene guard rejecting an otherwise visible NPC.
                pages = [
                    Page({"dragon_entry": control("dragon_entry", 1202, 233)},
                         {"dragon_entry": 1}, fixture="12.png",
                         tap_points={"dragon_entry": (1212, 243)}),
                    # The saved NPC label occupies [308,102,369,119] on the
                    # 752x423 profile. Convert its bounds to reference space
                    # before taking the integer center: (864,281).
                    Page({"dragon_npc": Match("dragon_npc", 786, 260, 156, 43, 1.0)},
                         {"dragon_npc": 2}, fixture="native_entry_20261107.jpg"),
                    Page(fixture="8.png"),
                ]
                with self.replay(pages, real_vision=True, size=size, max_checks=1,
                                 cloud_timeout=5) as (dragon, ctx):
                    report = dragon.run()
                    self.assertEqual(report.status, "limited")
                    self.assertEqual(ctx.restarts, [])
                    self.assertEqual(report.restarts, 0)
                    self.assertEqual([item[2] for item in ctx.inputs],
                                     ["dragon_entry", "dragon_npc"])
                    self.assertEqual([item[3] for item in ctx.inputs], [
                        (round(x * size[0] / 1920), round(y * size[1] / 1080))
                        for x, y in ((1212, 243), (864, 281))
                    ])
                    self.assertEqual(ctx.index, 2)
                    self.assertTrue(dragon.find("stop_hosting", dragon.frame))

    def test_real_failed_entry_with_visible_mark_still_approaches_npc(self):
        vision = DragonVision(ROOT / "assets" / "slay_dragon")
        for size in ((1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                pages = []
                for index, filename in enumerate((
                        "12.png", "native_entry_mark_20261107.jpg", "8.png")):
                    source = cv2.imdecode(np.fromfile(str(FIXTURES / filename), np.uint8),
                                          cv2.IMREAD_COLOR)
                    current = VisualFrame(cv2.resize(source, size))
                    visible = {name: match for name in vision.profile["templates"]
                               if (match := vision.find(current, name)) is not None}
                    actions, points = {}, {}
                    if index == 0:
                        actions = {"dragon_entry": 1}
                        points = {"dragon_entry": (1212, 243)}
                    elif index == 1:
                        # Measured from the real failure, independently of
                        # find(): [308,117,369,134] at the 752x423 profile.
                        # The yellow mark is also visible. Its presence must
                        # not skip approaching the still visible full name.
                        visible["dragon_npc"] = Match("dragon_npc", 786, 299, 156, 43, 1.0)
                        actions = {"dragon_npc": 2}
                    pages.append(Page(visible, actions, fixture=filename, tap_points=points))
                with self.replay(pages, real_vision=True, size=size, max_checks=1,
                                 cloud_timeout=5) as (dragon, ctx):
                    report = dragon.run()
                    self.assertEqual(report.status, "limited")
                    self.assertEqual(ctx.restarts, [])
                    self.assertEqual(report.restarts, 0)
                    self.assertEqual([item[2] for item in ctx.inputs],
                                     ["dragon_entry", "dragon_npc"])
                    self.assertEqual([item[3] for item in ctx.inputs], [
                        (round(x * size[0] / 1920), round(y * size[1] / 1080))
                        for x, y in ((1212, 243), (864, 320))
                    ])
                    self.assertEqual(ctx.index, 2)
                    self.assertTrue(dragon.find("stop_hosting", dragon.frame))

    def test_real_town_eye_fixture_starts_at_town_without_login_or_restart(self):
        vision = DragonVision(ROOT / "assets" / "slay_dragon")
        for size in ((1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                pages = []
                for index, (filename, name) in enumerate((
                        ("12.png", "dragon_entry"), ("13.png", "dragon_npc"),
                        ("8.png", "stop_hosting"))):
                    image = cv2.imdecode(np.fromfile(str(FIXTURES / filename), np.uint8), cv2.IMREAD_COLOR)
                    frame = VisualFrame(cv2.resize(image, size))
                    visible = {key: match for key in vision.profile["templates"]
                               if (match := vision.find(frame, key)) is not None}
                    self.assertIn(name, visible)
                    if filename == "12.png":
                        self.assertIn("town_eye", visible)
                    actions = {name: index + 1} if name != "stop_hosting" else {}
                    points = {"dragon_entry": (1212, 243)} if name == "dragon_entry" else {}
                    pages.append(Page(visible, actions, fixture=filename, tap_points=points))
                with self.replay(pages, real_vision=True, size=size, max_checks=1) as (dragon, ctx):
                    report = dragon.run()
                    self.assertEqual(ctx.restarts, [])
                    self.assertEqual(report.restarts, 0)
                    self.assertEqual([item[2] for item in ctx.inputs], ["dragon_entry", "dragon_npc"])
                    self.assertEqual(ctx.index, 2)


if __name__ == "__main__":
    unittest.main()
