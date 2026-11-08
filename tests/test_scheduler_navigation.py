from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pet_alliance.adb import AutomationError, Screenshot
from pet_alliance.scheduler_navigation import SchedulerNavigator
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
PATHS = {
    "dragon": FIXTURES / "scheduler" / "dragon_town_exit.png",
    "town": FIXTURES / "points_arena" / "2.png",
    "world": FIXTURES / "daily_hero" / "world.jpg",
    "local": FIXTURES / "daily_hero" / "region_after.jpg",
    "normal": FIXTURES / "daily_hero" / "normal.jpg",
    "defeat": FIXTURES / "daily_hero" / "defeat.jpg",
    "battle": FIXTURES / "daily_hero" / "battle.jpg",
    "guardian": FIXTURES / "check_arena" / "20.png",
    "empty_guardian": FIXTURES / "check_arena" / "19.png",
    "guardian_dungeon": FIXTURES / "check_arena" / "18.png",
    "roster": FIXTURES / "points_arena" / "6.png",
    "lobby": FIXTURES / "points_arena" / "4.png",
    "points_result": FIXTURES / "points_arena" / "7.png",
    "arena": FIXTURES / "points_arena" / "3.png",
    "login": FIXTURES / "slay_dragon" / "10.png",
    "notice": FIXTURES / "slay_dragon" / "11.png",
    "announcement": FIXTURES / "slay_dragon" / "announcement.png",
    "town_fresh": FIXTURES / "pet_loadouts" / "town_daily_badge_20261108.png",
    "town_clouded": FIXTURES / "points_arena" / "town_clouds_20261108.jpg",
}

# Independent screen hit areas in the reference viewport. In particular the
# dragon destination name is around y=710, while only its ring at y=840 is hit.
BOUNDS = {
    "portal": (65, 800, 170, 890),
    "guardian_close": (1380, 195, 1480, 295),
    "panel_close": (1370, 190, 1480, 300),
    "world_button": (1640, 780, 1890, 930),
    "town": (120, 455, 280, 595),
    "roster_close": (1550, 105, 1650, 210),
    "lobby_close": (1530, 195, 1640, 300),
    "arena_exit": (175, 420, 345, 785),
    "exit_battle": (1010, 825, 1225, 920),
    "login": (850, 820, 1050, 970),
    "notice": (800, 970, 1110, 1060),
    # User's 956 x 538 screenshot: the cross is around (904, 83).
    # Scale its independently observed hit area into the 1920 x 1080 viewport.
    "announcement": (1783, 127, 1847, 207),
}


@dataclass
class Page:
    image: str
    control: str | None = None
    after: tuple[float, int] | None = None


class Replay:
    def __init__(self, pages, *, size=(1374, 772), dry_run=False):
        self.project_root = ROOT
        self.settings = {"daily_hero": {
            "poll_interval": .1, "min_click_interval": .1,
            "click_retry_interval": .4, "transition_timeout": 2,
        }, "slay_dragon": {"login_timeout": 2, "town_timeout": 2}}
        self.dry_run = dry_run
        self.log = logging.getLogger("scheduler-navigation-replay")
        self.pages = pages
        self.size = size
        self.index = 0
        self.elapsed = 0.0
        self.entered = 0.0
        self.inputs = []
        self.restarts = []
        self.dropped = {}
        self.images = {}
        for page in pages:
            if page.image == "unknown":
                image = np.zeros((772, 1374, 3), np.uint8)
            else:
                image = cv2.imdecode(np.fromfile(str(PATHS[page.image]), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, size)
            self.images[page.image] = Screenshot(cv2.imencode(".png", image)[1].tobytes(), *size)

    def monotonic(self):
        return self.elapsed

    def wait(self, seconds):
        self.elapsed += seconds

    def enter(self, index):
        self.index = index
        self.entered = self.elapsed

    def capture(self):
        page = self.pages[self.index]
        if page.after and self.elapsed - self.entered >= page.after[0]:
            self.enter(page.after[1])
        return self.images[self.pages[self.index].image]

    def tap(self, x, y):
        if self.dry_run:
            raise AssertionError("Preview must not send input")
        page = self.pages[self.index]
        if page.control is None:
            raise AssertionError(f"Unexpected tap on {page.image}")
        left, top, right, bottom = BOUNDS[page.control]
        rx, ry = x * 1920 / self.size[0], y * 1080 / self.size[1]
        if not (left <= rx <= right and top <= ry <= bottom):
            raise AssertionError(f"Unexpected/stale tap {(rx, ry)} for {page.control}")
        self.inputs.append((self.index, page.control))
        if self.dropped.get(self.index, 0):
            self.dropped[self.index] -= 1
        else:
            self.enter(self.index + 1)

    def restart_app(self, package):
        if self.dry_run:
            raise AssertionError("Preview must not restart")
        self.restarts.append(package)
        self.enter(self.index + 1)


class SchedulerNavigationTests(unittest.TestCase):
    def navigation(self, ctx, directory):
        nav = SchedulerNavigator(ctx)
        nav.run_dir = Path(directory)
        return nav

    def run_pages(self, pages, **options):
        ctx = Replay(pages, **options)
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            nav = self.navigation(ctx, directory)
            nav.ensure_town()
            self.assertTrue(nav.is_town(nav.frame))
        return ctx

    def test_already_town_does_not_click(self):
        ctx = self.run_pages([Page("town")])
        self.assertEqual(ctx.inputs, [])
        self.assertEqual(ctx.restarts, [])

    def test_fresh_town_with_daily_badges_and_eye_edge_cloud_needs_no_restart(self):
        for size in ((1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                ctx = self.run_pages([Page("town_fresh")], size=size)
                self.assertEqual(ctx.inputs, [])
                self.assertEqual(ctx.restarts, [])

    def test_town_with_cloud_over_entire_eye_needs_no_click_or_restart(self):
        for size in ((1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                ctx = self.run_pages([Page("town_clouded")], size=size)
                self.assertEqual(ctx.inputs, [])
                self.assertEqual(ctx.restarts, [])

    def test_forced_restart_from_town_logs_in_and_retries_missed_controls(self):
        ctx = Replay([
            Page("town"), Page("login", "login"),
            Page("notice", "notice"), Page("town"),
        ])
        ctx.dropped = {1: 1, 2: 1}
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            nav = self.navigation(ctx, directory)
            nav.restart_to_town()
            self.assertTrue(nav.is_town(nav.frame))
        self.assertEqual(ctx.restarts, ["com.monster.mxm1old"])
        self.assertEqual(ctx.inputs, [
            (1, "login"), (1, "login"), (2, "notice"), (2, "notice"),
        ])

    def test_forced_restart_does_not_navigate_old_dragon_scene(self):
        ctx = Replay([Page("dragon", "portal"), Page("town")])
        with tempfile.TemporaryDirectory() as directory:
            nav = self.navigation(ctx, directory)
            nav.restart_to_town()
            self.assertTrue(nav.is_town(nav.frame))
        self.assertEqual(len(ctx.restarts), 1)
        self.assertEqual(ctx.inputs, [])

    def test_restart_dismisses_both_login_notices_in_either_order(self):
        for notices in (("announcement", "notice"), ("notice", "announcement")):
            with self.subTest(notices=notices):
                ctx = Replay([
                    Page("town"), Page("login", "login"),
                    *(Page(name, name) for name in notices), Page("town"),
                ])
                with tempfile.TemporaryDirectory() as directory, \
                        patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
                    nav = self.navigation(ctx, directory)
                    nav.restart_to_town()
                    self.assertTrue(nav.is_town(nav.frame))
                self.assertEqual(ctx.restarts, ["com.monster.mxm1old"])
                self.assertEqual([name for _, name in ctx.inputs], ["login", *notices])

    def test_login_and_each_notice_have_independent_missed_click_allowance(self):
        ctx = Replay([
            Page("town"), Page("login", "login"),
            Page("announcement", "announcement"), Page("notice", "notice"), Page("town"),
        ])
        ctx.settings["daily_hero"]["max_click_retries"] = 1
        ctx.settings["slay_dragon"]["town_timeout"] = 5
        ctx.dropped = {1: 1, 2: 1, 3: 1}
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            nav = self.navigation(ctx, directory)
            nav.restart_to_town()
            self.assertTrue(nav.is_town(nav.frame))
        self.assertEqual(ctx.inputs, [
            (1, "login"), (1, "login"),
            (2, "announcement"), (2, "announcement"),
            (3, "notice"), (3, "notice"),
        ])

    def test_repeated_alternating_notices_share_one_login_timeout(self):
        ctx = Replay([
            Page("town"), Page("login", "login"),
            *(Page(name, name) for name in ("announcement", "notice") * 12), Page("town"),
        ])
        ctx.settings["daily_hero"]["max_click_retries"] = 1
        ctx.settings["slay_dragon"]["town_timeout"] = 1.5
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            with self.assertRaisesRegex(AutomationError, "未确认目标页面"):
                self.navigation(ctx, directory).restart_to_town()
        self.assertGreaterEqual(ctx.elapsed, 1.5)
        self.assertLess(ctx.elapsed, 2)
        self.assertEqual(len(ctx.restarts), 1)
        self.assertLess(len(ctx.inputs), 25)
        self.assertTrue(any(name == "announcement" for _, name in ctx.inputs))

    def test_forced_restart_honors_package_configuration(self):
        ctx = Replay([Page("town"), Page("town")])
        ctx.settings["slay_dragon"]["package"] = "com.example.monster"
        with tempfile.TemporaryDirectory() as directory:
            self.navigation(ctx, directory).restart_to_town()
        self.assertEqual(ctx.restarts, ["com.example.monster"])
        self.assertEqual(ctx.inputs, [])

    def test_forced_restart_waits_for_loading_then_login_without_second_restart(self):
        ctx = Replay([
            Page("town"), Page("unknown", after=(.7, 2)),
            Page("login", "login"), Page("town"),
        ])
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            nav = self.navigation(ctx, directory)
            nav.restart_to_town()
            self.assertTrue(nav.is_town(nav.frame))
        self.assertEqual(len(ctx.restarts), 1)
        self.assertEqual(ctx.inputs, [(2, "login")])

    def test_forced_restart_timeout_saves_frame_and_does_not_confirm_town(self):
        ctx = Replay([Page("town"), Page("unknown")])
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            nav = self.navigation(ctx, directory)
            with self.assertRaisesRegex(AutomationError, "登录页面"):
                nav.restart_to_town()
            self.assertTrue((Path(directory) / "navigation_last_frame.jpg").is_file())
            self.assertNotIn("scheduler_town_confirmed",
                             (Path(directory) / "events.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(len(ctx.restarts), 1)
        self.assertEqual(ctx.inputs, [])

    def test_forced_restart_login_timeout_never_claims_town_success(self):
        ctx = Replay([Page("town"), Page("login", "login"), Page("unknown")])
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            nav = self.navigation(ctx, directory)
            with self.assertRaisesRegex(AutomationError, "登录并处理通知"):
                nav.restart_to_town()
            self.assertNotIn("scheduler_town_confirmed",
                             (Path(directory) / "events.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(len(ctx.restarts), 1)
        self.assertEqual(ctx.inputs, [(1, "login")])

    def test_forced_restart_propagates_pause_before_restart(self):
        ctx = Replay([Page("town")])
        ctx.restart_app = lambda package: (_ for _ in ()).throw(InterruptedError("paused"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                self.navigation(ctx, directory).restart_to_town()
        self.assertEqual(ctx.restarts, [])
        self.assertEqual(ctx.inputs, [])

    def test_forced_restart_propagates_pause_during_login(self):
        ctx = Replay([Page("town"), Page("login")])
        ctx.capture = lambda: (_ for _ in ()).throw(InterruptedError("paused"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                self.navigation(ctx, directory).restart_to_town()
        self.assertEqual(len(ctx.restarts), 1)
        self.assertEqual(ctx.inputs, [])

    def test_dragon_exit_clicks_ring_at_each_resolution(self):
        for size in ((1374, 772), (1920, 1080), (752, 423)):
            with self.subTest(size=size):
                ctx = self.run_pages([Page("dragon", "portal"), Page("town")], size=size)
                self.assertEqual(ctx.inputs, [(0, "portal")])
                self.assertEqual(ctx.restarts, [])

    def test_dragon_exit_works_outside_dragon_hours(self):
        with patch("pet_alliance.slay_dragon.is_dragon_active", return_value=False):
            ctx = self.run_pages([Page("dragon", "portal"), Page("town")])
        self.assertEqual(ctx.inputs, [(0, "portal")])

    def test_guardian_closes_only_and_returns_via_world(self):
        ctx = self.run_pages([
            Page("guardian", "guardian_close"), Page("guardian_dungeon", "panel_close"),
            Page("local", "world_button"), Page("world", "town"), Page("town"),
        ])
        self.assertEqual([control for _, control in ctx.inputs],
                         ["guardian_close", "panel_close", "world_button", "town"])

    def test_empty_guardian_never_challenges_on_navigation(self):
        ctx = self.run_pages([
            Page("empty_guardian", "guardian_close"), Page("local", "world_button"),
            Page("world", "town"), Page("town"),
        ])
        self.assertEqual(ctx.inputs[0], (0, "guardian_close"))

    def test_points_roster_exits_all_panels(self):
        ctx = self.run_pages([
            Page("roster", "roster_close"), Page("lobby", "lobby_close"),
            Page("arena", "arena_exit"), Page("town"),
        ])
        self.assertEqual([control for _, control in ctx.inputs],
                         ["roster_close", "lobby_close", "arena_exit"])

    def test_dungeon_panel_returns_via_world(self):
        ctx = self.run_pages([
            Page("normal", "panel_close"), Page("local", "world_button"),
            Page("world", "town"), Page("town"),
        ])
        self.assertEqual(ctx.inputs[0], (0, "panel_close"))

    def test_ended_battle_exits_without_next_stage(self):
        ctx = self.run_pages([
            Page("defeat", "exit_battle"), Page("normal", "panel_close"),
            Page("local", "world_button"), Page("world", "town"), Page("town"),
        ])
        self.assertEqual(ctx.inputs[0], (0, "exit_battle"))

    def test_unknown_restarts_and_handles_login_notice(self):
        ctx = self.run_pages([
            Page("unknown"), Page("login", "login"),
            Page("notice", "notice"), Page("town"),
        ])
        self.assertEqual(ctx.restarts, ["com.monster.mxm1old"])
        self.assertEqual([control for _, control in ctx.inputs], ["login", "notice"])

    def test_unfinished_battle_restarts_without_challenging(self):
        ctx = self.run_pages([
            Page("battle"), Page("login", "login"), Page("town"),
        ])
        self.assertEqual(len(ctx.restarts), 1)
        self.assertEqual(ctx.inputs, [(1, "login")])

    def test_points_settlement_restarts_without_confirming_rewards(self):
        ctx = self.run_pages([
            Page("points_result"), Page("login", "login"), Page("town"),
        ])
        self.assertEqual(len(ctx.restarts), 1)
        self.assertEqual(ctx.inputs, [(1, "login")])

    def test_notice_at_start_is_dismissed_without_restart(self):
        ctx = self.run_pages([Page("notice", "notice"), Page("town")])
        self.assertEqual(ctx.restarts, [])

    def test_mixed_notices_at_start_are_dismissed_without_restart(self):
        for notices in (("announcement", "notice"), ("notice", "announcement")):
            for size in ((1374, 772), (1920, 1080), (752, 423)):
                with self.subTest(notices=notices, size=size):
                    ctx = self.run_pages([
                        *(Page(name, name) for name in notices), Page("town"),
                    ], size=size)
                    self.assertEqual(ctx.restarts, [])
                    self.assertEqual([name for _, name in ctx.inputs], list(notices))

    def test_notification_loop_waits_through_unknown_frame_without_blind_input(self):
        ctx = self.run_pages([
            Page("announcement", "announcement"), Page("unknown", after=(.7, 2)),
            Page("notice", "notice"), Page("town"),
        ])
        self.assertEqual(ctx.restarts, [])
        self.assertEqual(ctx.inputs, [(0, "announcement"), (2, "notice")])

    def test_announcement_title_waits_for_delayed_cross_without_restart(self):
        ctx = Replay([
            Page("announcement", "announcement"), Page("notice", "notice"), Page("town"),
        ])
        tap_times = []
        original_tap = ctx.tap

        def record_tap(x, y):
            tap_times.append(ctx.elapsed)
            original_tap(x, y)

        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            nav = self.navigation(ctx, directory)
            original_find = nav.dragon.find

            def delayed_close(frame, name):
                if name == "announcement_close" and ctx.elapsed < 1:
                    return None
                return original_find(frame, name)

            with patch.object(nav.dragon, "find", side_effect=delayed_close), \
                    patch.object(ctx, "tap", side_effect=record_tap):
                nav.ensure_town()
                self.assertTrue(nav.is_town(nav.frame))
        self.assertEqual(ctx.restarts, [])
        self.assertEqual(ctx.inputs, [(0, "announcement"), (1, "notice")])
        self.assertGreaterEqual(tap_times[0], 1)

    def test_town_is_not_confirmed_while_either_login_notice_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            nav = self.navigation(Replay([Page("town")]), directory)
            for name in ("announcement", "notice"):
                frame = VisualFrame.from_screenshot(Replay([Page(name)]).capture())
                with self.subTest(name=name):
                    self.assertFalse(nav.is_town(frame))

    def test_missed_ring_tap_is_relocated_and_retried(self):
        ctx = Replay([Page("dragon", "portal"), Page("town")])
        ctx.dropped[0] = 1
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            self.navigation(ctx, directory).ensure_town()
        self.assertEqual(ctx.inputs, [(0, "portal"), (0, "portal")])
        self.assertGreaterEqual(ctx.elapsed, .4)

    def test_loading_after_ring_never_reuses_old_coordinates(self):
        ctx = self.run_pages([
            Page("dragon", "portal"), Page("unknown", after=(.7, 2)), Page("town"),
        ])
        self.assertEqual(ctx.inputs, [(0, "portal")])
        self.assertEqual(ctx.restarts, [])

    def test_unknown_recovery_times_out_without_random_clicks(self):
        ctx = Replay([Page("unknown"), Page("unknown")])
        with tempfile.TemporaryDirectory() as directory, \
                patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
            with self.assertRaisesRegex(AutomationError, "登录页面"):
                self.navigation(ctx, directory).ensure_town()
        self.assertEqual(ctx.inputs, [])
        self.assertEqual(len(ctx.restarts), 1)

    def test_pause_propagates_without_restart(self):
        ctx = Replay([Page("dragon", "portal")])
        ctx.capture = lambda: (_ for _ in ()).throw(InterruptedError("paused"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                self.navigation(ctx, directory).ensure_town()
        self.assertEqual(ctx.inputs, [])
        self.assertEqual(ctx.restarts, [])

    def test_preview_does_not_click_or_restart(self):
        ctx = Replay([Page("dragon", "portal")], dry_run=True)
        with tempfile.TemporaryDirectory() as directory:
            self.navigation(ctx, directory).ensure_town()
        self.assertEqual(ctx.inputs, [])
        self.assertEqual(ctx.restarts, [])

    def test_portal_variants_and_guard_fail_closed(self):
        ctx = Replay([Page("town")])
        with tempfile.TemporaryDirectory() as directory:
            nav = self.navigation(ctx, directory)
            for filename in ("8.png", "9.png", "13.png", "native_stop.png"):
                path = FIXTURES / "slay_dragon" / filename
                frame = VisualFrame(cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR))
                with self.subTest(filename=filename):
                    self.assertIsNotNone(nav.dragon_exit(frame))
            for name in ("town", "local", "world", "guardian", "roster"):
                frame = VisualFrame.from_screenshot(Replay([Page(name)]).capture())
                with self.subTest(name=name):
                    self.assertIsNone(nav.dragon_exit(frame))


if __name__ == "__main__":
    unittest.main()
