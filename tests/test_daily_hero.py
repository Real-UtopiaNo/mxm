from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pet_alliance import AutomationError, Screenshot
from pet_alliance.daily_hero import ClickTimeoutError, DailyHero, HeroOptions, RetryBudget
from pet_alliance.vision import Match, TemplateBank, VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "daily_hero"


def frame(name: str, size=None) -> VisualFrame:
    image = cv2.imdecode(np.fromfile(str(FIXTURES / f"{name}.jpg"), np.uint8), cv2.IMREAD_COLOR)
    if size:
        image = cv2.resize(image, size)
    return VisualFrame(image)


class ReplayClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


class ReplayContext:
    """The UI advances only after an input, using real captured game images."""

    def __init__(self, root: Path, states: list[str], **options):
        self.project_root = root
        self.settings = {"daily_hero": {
            "transition_timeout": 2, "poll_interval": .1,
            "min_click_interval": .1, "click_retry_interval": .4, **options,
        }}
        self.log = logging.getLogger("replay")
        self.dry_run = False
        self.states = states
        self.index = 0
        self.inputs = []
        self.input_times = []
        self.clock = ReplayClock()
        self.wait = self.clock.wait
        self.fixtures = {}
        self.delayed_result = False
        self.pending_index = None
        self.stale_captures = 0
        self.dropped_clicks = {}
        self.delay_seconds = {}
        self.pending_until = None
        for name in set(states):
            image = frame(name).image
            data = cv2.imencode(".png", image)[1].tobytes()
            self.fixtures[name] = Screenshot(data, 1920, 1080)

    def capture(self):
        if self.pending_index is not None:
            if self.pending_until is not None:
                ready = self.clock.now >= self.pending_until
            else:
                ready = self.stale_captures == 0
                self.stale_captures -= 1
            if ready:
                self.index = self.pending_index
                self.pending_index = None
                self.pending_until = None
        return self.fixtures[self.states[self.index]]

    def tap(self, x, y):
        state = self.states[self.index]
        self.inputs.append((state, x, y))
        self.input_times.append(self.clock.now)
        if self.dry_run:
            return
        if self.dropped_clicks.get(state, 0):
            self.dropped_clicks[state] -= 1
            return
        if self.delay_seconds.get(state):
            if self.pending_index is not None:
                raise AssertionError("A slow successful click was repeated before confirmation")
            self.pending_index = self.index + 1
            self.pending_until = self.clock.now + self.delay_seconds[state]
        elif self.delayed_result and state == "victory":
            if self.pending_index is not None:
                raise AssertionError("Previous result received a duplicate Next click")
            self.pending_index = self.index + 1
            self.stale_captures = 2
        else:
            self.index += 1
        if self.index >= len(self.states):
            raise AssertionError("Unexpected extra input after the replay ended")


class DailyHeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = TemplateBank(ROOT / "assets" / "daily_hero")

    def test_real_world_regions_require_recommended_label_and_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["world"])
            regions = {region["key"] for region, _ in hero.regions(frame("world"))}
            self.assertEqual(regions, {"west", "cursed", "flame", "sky", "maelstrom", "dream", "demon", "etsam", "otherworld"})
            self.assertNotIn("sumeru", regions)
            self.assertNotIn("palaces", regions)

    def test_region_entry_does_not_click_noninteractive_level_labels(self):
        # Real-device misses: Dream (907, 581), Otherworld (1367, 777).
        # Their icon/name hit areas are above the recommended-level text.
        for key, destination, bounds in (
            ("dream", "region_dream", (860, 470, 945, 550)),
            ("otherworld", "region_otherworld", (1290, 650, 1420, 750)),
        ):
            with self.subTest(region=key), tempfile.TemporaryDirectory() as directory:
                hero = self.hero(directory, ["world", destination, "world"])
                region = next(r for r in hero.bank.profile["regions"] if r["key"] == key)
                original_tap = hero.ctx.tap

                def tap(x, y):
                    left, top, right, bottom = bounds
                    if hero.ctx.index == 0 and not (left <= x <= right and top <= y <= bottom):
                        hero.ctx.dropped_clicks["world"] = 1
                    original_tap(x, y)

                hero.ctx.tap = tap
                with patch.object(hero, "regions", side_effect=lambda f: [(region, hero.find(region["template"], f))]
                                  if hero.ctx.index == 0 else []), \
                        patch.object(hero, "current_region", return_value=None), \
                        patch.object(hero, "markers", return_value=[]):
                    report = hero.run()
                self.assertEqual(report.status, "completed")
                self.assertEqual(sum(state == "world" for state, _, _ in hero.ctx.inputs), 1)
                self.assertTrue(hero.is_map(frame(destination)))

    def hero(self, directory: str, states: list[str], **options) -> DailyHero:
        root = Path(directory)
        # The profile remains in the project; only generated evidence goes into
        # the temporary directory.
        ctx = ReplayContext(ROOT, states, **options)
        hero = DailyHero(ctx)
        clock_patch = patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        hero.started = ctx.clock.now
        hero.run_dir.rmdir()
        hero.run_dir = root / "run"
        hero.run_dir.mkdir()
        return hero

    def test_completed_dungeon_marker_disappears_in_real_images(self):
        before = self.bank.find_all(frame("region_before"), "marker")
        after = self.bank.find_all(frame("region_after"), "marker")
        self.assertEqual(len(before), 8)
        self.assertEqual(len(after), 7)
        self.assertTrue(any(abs(m.center[0] - 308) < 3 for m in before))
        self.assertFalse(any(abs(m.center[0] - 308) < 3 for m in after))

    def test_reported_missing_dungeon_flags_are_detected_in_real_maps(self):
        expected = {
            "dream_missing": [(911, 251), (1366, 431), (718, 469), (1620, 540),
                              (433, 638), (1082, 728), (1515, 815), (646, 819)],
            "sky_missing": [(391, 599)],
            "cursed_missing": [(1694, 322), (582, 932)],
        }
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home"])
            for name, points in expected.items():
                with self.subTest(map=name):
                    detected = hero.markers(frame(name))
                    self.assertEqual(len(detected), len(points), [m.center for m in detected])
                    for x, y in points:
                        self.assertTrue(any(abs(m.center[0] - x) <= 5 and abs(m.center[1] - y) <= 5
                                            for m in detected), (name, (x, y)))

    def test_obscured_freyes_requires_both_landmark_and_pending_badge(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home"])
            pending = frame("cursed_missing")
            self.assertTrue(any(m.name == "marker_freyes_obscured" for m in hero.markers(pending)))
            completed = frame("cursed_missing")
            completed.image[915:950, 565:600] = completed.image[915:950, 530:565].copy()
            self.assertFalse(any(m.name == "marker_freyes_obscured" for m in hero.markers(completed)))
            wrong_region = frame("dream_missing")
            wrong_region.image[915:950, 565:600] = pending.image[915:950, 565:600]
            self.assertFalse(any(m.name == "marker_freyes_obscured" for m in hero.markers(wrong_region)))

    def test_current_region_is_found_on_different_diamond_backgrounds(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["world"])
            for key in ("dream", "sky", "cursed"):
                with self.subTest(region=key):
                    current = hero.current_region(frame(f"world_current_{key}"))
                    self.assertIsNotNone(current)
                    self.assertEqual(current[0]["key"], key)

    def test_entering_already_selected_region_closes_world_map(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["world_current_cursed", "cursed_missing"])
            original_tap = hero.ctx.tap

            def tap(x, y):
                # The game ignores selecting the current region again.
                if y < 940:
                    hero.ctx.dropped_clicks["world_current_cursed"] = 1
                original_tap(x, y)

            hero.ctx.tap = tap
            region = next(r for r in hero.bank.profile["regions"] if r["key"] == "cursed")
            hero.enter_region(region)
            self.assertEqual(hero.ctx.index, 1)
            self.assertEqual(len(hero.ctx.inputs), 1)
            self.assertGreater(hero.ctx.inputs[0][2], 940)

    def test_result_states_and_disabled_next_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home"])
            for name, expected in (("battle", "battle"), ("victory", "victory"), ("complete", "complete"), ("defeat", "defeat")):
                with self.subTest(name=name):
                    self.assertEqual(hero.battle_state(frame(name))[0], expected)
            self.assertIsNone(self.bank.find(frame("complete"), "next_stage"))
            self.assertIsNone(self.bank.find(frame("victory"), "next_disabled"))

    def test_hero_and_normal_modes_are_visually_distinct(self):
        self.assertIsNotNone(self.bank.find(frame("normal"), "normal_dragon"))
        self.assertIsNone(self.bank.find(frame("normal"), "hero_dragon"))
        self.assertIsNotNone(self.bank.find(frame("hero"), "hero_dragon"))
        self.assertIsNone(self.bank.find(frame("hero"), "normal_dragon"))

    def test_dimmed_background_control_is_rejected(self):
        image = frame("region_after").image
        dimmed = VisualFrame((image.astype(float) * .45).astype(np.uint8))
        self.assertIsNone(self.bank.find(dimmed, "world_button"))

    def test_screen_resolution_is_normalized_and_click_is_scaled_back(self):
        smaller = frame("home", (1280, 720))
        match = self.bank.find(smaller, "adventure")
        self.assertIsNotNone(match)
        self.assertEqual(smaller.device_point(match.center), (1089, 497))
        with self.assertRaises(AutomationError):
            VisualFrame(np.zeros((1920, 1080, 3), np.uint8))

    def test_default_allows_ten_retries_after_initial_attempt(self):
        budget = RetryBudget(HeroOptions().max_retries)
        self.assertEqual([budget.failed() for _ in range(11)], [True] * 10 + [False])
        self.assertEqual(budget.failures, 11)

    def test_missed_adventure_click_is_retried_before_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home", "region_before", "world"])
            hero.ctx.dropped_clicks["home"] = 1
            hero.world()
            self.assertEqual(hero.ctx.states[hero.ctx.index], "world")
            self.assertEqual(sum(s == "home" for s, _, _ in hero.ctx.inputs), 2)

    def test_world_helper_still_opens_world_and_retries_a_missed_map_click(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home", "region_before", "world"])
            hero.ctx.dropped_clicks["region_before"] = 1
            hero.world()
            self.assertEqual(hero.ctx.states[hero.ctx.index], "world")
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["home", "region_before", "region_before"])
            expected = hero.find("world_button", frame("region_before")).center
            self.assertEqual([entry[1:] for entry in hero.ctx.inputs[1:]], [expected, expected])

    def test_every_navigation_and_battle_control_recovers_from_a_missed_click(self):
        states = ["home", "region_before", "normal", "hero",
                  "battle", "victory", "battle", "complete", "reward", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_dungeons=1)
            hero.ctx.dropped_clicks = {name: 1 for name in set(states[:-1])}
            report = hero.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(report.completed), 1)
            self.assertEqual(report.victories, 2)
            self.assertEqual(report.failures, 0)
            self.assertTrue(all(count == 0 for count in hero.ctx.dropped_clicks.values()))
            self.assertTrue(all(b - a >= .1 - 1e-9 for a, b in zip(hero.ctx.input_times, hero.ctx.input_times[1:])))

    def test_slow_skip_waits_for_result_without_rapid_repeat_clicks(self):
        states = ["region_before", "normal", "hero", "battle", "complete", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, click_retry_interval=1, min_click_interval=.2)
            hero.ctx.delay_seconds["battle"] = .6
            hero.observe()
            self.assertTrue(hero.fight_dungeon({"name": "test"}, hero.markers(hero.frame)[0]))
            self.assertEqual(sum(s == "battle" for s, _, _ in hero.ctx.inputs), 1)
            self.assertTrue(all(b - a >= .2 - 1e-9 for a, b in zip(hero.ctx.input_times, hero.ctx.input_times[1:])))

    def test_click_retries_stop_at_limit_without_spending_battle_retry_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home", "region_before", "world"], max_click_retries=2)
            hero.ctx.dropped_clicks["home"] = 99
            with self.assertRaisesRegex(AutomationError, "补点 2 次"):
                hero.world()
            self.assertEqual(len(hero.ctx.inputs), 3)
            self.assertEqual(hero.ctx.index, 0)
            self.assertEqual(hero.report.failures, 0)
            self.assertTrue(all(b - a >= .4 - 1e-9 for a, b in zip(hero.ctx.input_times, hero.ctx.input_times[1:])))

    def test_loading_screen_does_not_receive_clicks_at_stale_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home", "region_before"])
            original_capture = hero.ctx.capture
            encoded = cv2.imencode(".png", np.zeros((1080, 1920, 3), np.uint8))[1].tobytes()
            loading = Screenshot(encoded, 1920, 1080)

            def capture():
                if hero.ctx.index == 1 and hero.ctx.clock.now < .8:
                    return loading
                return original_capture()

            hero.ctx.capture = capture
            hero.click_until(lambda f: hero.find("adventure", f), hero.is_map, "test")
            self.assertEqual(len(hero.ctx.inputs), 1)
            self.assertGreaterEqual(hero.ctx.clock.now, .8)

    def test_retry_relocates_a_control_that_moved_after_the_missed_click(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home", "region_before", "world"])
            hero.ctx.dropped_clicks["home"] = 1
            original_capture = hero.ctx.capture
            image = frame("home").image.copy()
            image[716:774, 1575:1691] = 0
            image[716:774, 1615:1731] = hero.bank.images["adventure"]
            moved = Screenshot(cv2.imencode(".png", image)[1].tobytes(), 1920, 1080)

            def capture():
                if hero.ctx.index == 0 and hero.ctx.inputs:
                    return moved
                return original_capture()

            hero.ctx.capture = capture
            hero.world()
            points = [(x, y) for state, x, y in hero.ctx.inputs if state == "home"]
            self.assertEqual(points, [(1633, 745), (1673, 745)])

    def test_end_to_end_replay_from_town_until_dungeon_completed(self):
        states = ["home", "region_before", "normal", "hero", "battle", "victory", "battle", "complete", "reward", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_dungeons=1)
            hero.ctx.delayed_result = True
            report = hero.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(report.completed), 1)
            self.assertEqual(report.victories, 2)
            self.assertEqual([s for s, _, _ in hero.ctx.inputs], states[:-1])
            self.assertTrue((hero.run_dir / "summary.json").is_file())

    def test_current_map_flags_are_processed_before_opening_world_map(self):
        states = ["home", "region_before", "normal", "hero", "battle", "complete", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_dungeons=1)
            report = hero.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(report.completed), 1)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], states[:-1])
            _, x, y = hero.ctx.inputs[1]
            first_marker = hero.markers(frame("region_before"))[0]
            self.assertEqual((x, y), (first_marker.center[0] - 65, first_marker.center[1] + 10))

    def test_current_map_without_flags_opens_world_and_selects_another_flagged_region(self):
        states = ["home", "region_after", "world", "region_dream", "normal", "hero",
                  "battle", "complete", "region_dream"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_dungeons=1)
            dream = next(r for r in hero.bank.profile["regions"] if r["key"] == "dream")
            original_markers = hero.markers
            with patch.object(hero, "markers", side_effect=lambda f: [] if hero.ctx.index == 1 else original_markers(f)), \
                    patch.object(hero, "regions", side_effect=lambda f: [(dream, hero.find(dream["template"], f))]
                                 if hero.is_world(f) else []):
                report = hero.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], states[:-1])
            self.assertEqual(report.completed[0]["region"], dream["name"])
            self.assertEqual(hero.ctx.inputs[1][1:], hero.find("world_button", frame("region_after")).center)

    def test_region_map_and_dungeon_panel_start_with_current_region_flags(self):
        for initial in ([], ["normal"]):
            states = initial + ["region_before", "normal", "hero", "battle", "complete", "region_after"]
            with self.subTest(initial=initial or ["region_before"]), tempfile.TemporaryDirectory() as directory:
                hero = self.hero(directory, states, max_dungeons=1)
                report = hero.run()
                self.assertEqual(report.status, "limited")
                self.assertEqual(len(report.completed), 1)
                self.assertEqual([state for state, _, _ in hero.ctx.inputs], states[:-1])
                if initial:
                    self.assertEqual(hero.ctx.inputs[0][1:], hero.find("panel_close", frame("normal")).center)

    def test_cleared_current_region_is_resolved_and_not_reentered_from_world(self):
        states = ["region_before", "normal", "hero", "battle", "complete", "region_after", "world"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            original_markers = hero.markers
            with patch.object(hero, "markers", side_effect=lambda f: [] if hero.ctx.index == 5 else original_markers(f)), \
                    patch.object(hero, "regions", return_value=[]):
                report = hero.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(report.completed), 1)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], states[:-1])

    def test_adventure_map_keeps_an_already_open_world_map_without_input(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["world"])
            self.assertFalse(hero.adventure_map())
            self.assertEqual(hero.ctx.inputs, [])

    def test_adventure_waits_past_normal_timeout_for_clouds_to_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home", "region_before"], poll_interval=5)
            town, destination = frame("home"), frame("region_before")
            adventure = hero.find("adventure", town)
            scans = []

            def observe():
                hero.frame = destination if hero.ctx.index else town
                return hero.frame

            def find(name, current):
                if name == "adventure":
                    scans.append(hero.ctx.clock.now)
                    return adventure if hero.ctx.clock.now >= 45 else None
                return None

            with patch.object(hero, "observe", side_effect=observe), \
                    patch.object(hero, "find", side_effect=find), \
                    patch.object(hero, "is_world", return_value=False), \
                    patch.object(hero, "is_map", side_effect=lambda f: f is destination):
                self.assertTrue(hero.adventure_map())

            self.assertEqual(len(hero.ctx.inputs), 1)
            self.assertEqual(hero.ctx.inputs[0][1:], adventure.center)
            self.assertEqual(hero.ctx.input_times, [45])
            self.assertTrue(any(0 < when < 45 for when in scans))
            self.assertGreater(hero.ctx.clock.now, hero.options.transition_timeout)

    def test_adventure_missing_for_five_minutes_times_out_without_input(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home"], poll_interval=10)
            town = frame("home")
            hero.frame = town
            with patch.object(hero, "observe", return_value=town), \
                    patch.object(hero, "find", return_value=None), \
                    patch.object(hero, "is_world", return_value=False), \
                    patch.object(hero, "is_map", return_value=False):
                with self.assertRaises(ClickTimeoutError):
                    hero.adventure_map()
            self.assertEqual(hero.ctx.clock.now, 300)
            self.assertEqual(hero.ctx.inputs, [])

    def test_adventure_retry_waits_for_clouds_then_relocates_visible_control(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home", "region_before"], poll_interval=5,
                             min_click_interval=7, click_retry_interval=11,
                             max_click_retries=1)
            hero.ctx.dropped_clicks["home"] = 1
            town, destination = frame("home"), frame("region_before")
            original = hero.find("adventure", town)
            moved = Match(original.name, original.x + 40, original.y,
                          original.width, original.height, original.score)
            clear_again = VisualFrame(town.image.copy())
            clouded = VisualFrame(np.zeros_like(town.image))

            def observe():
                if hero.ctx.index:
                    hero.frame = destination
                elif not hero.ctx.inputs:
                    hero.frame = town
                else:
                    hero.frame = clear_again if hero.ctx.clock.now >= 50 else clouded
                return hero.frame

            def find(name, current):
                if name == "adventure":
                    return original if current is town else moved if current is clear_again else None
                return None

            with patch.object(hero, "observe", side_effect=observe), \
                    patch.object(hero, "find", side_effect=find), \
                    patch.object(hero, "is_world", return_value=False), \
                    patch.object(hero, "is_map", side_effect=lambda f: f is destination):
                self.assertTrue(hero.adventure_map())

            self.assertEqual([entry[1:] for entry in hero.ctx.inputs], [original.center, moved.center])
            self.assertEqual(hero.ctx.input_times, [0, 50])
            self.assertGreaterEqual(hero.ctx.input_times[1] - hero.ctx.input_times[0],
                                    max(hero.options.min_click_interval, hero.options.click_retry_interval))

    def test_click_until_timeout_override_keeps_default_deadline_for_other_controls(self):
        for timeout, expected in ((None, 2), (7, 7)):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory:
                hero = self.hero(directory, ["home"], poll_interval=1)
                town = frame("home")
                kwargs = {} if timeout is None else {"timeout": timeout}
                with patch.object(hero, "observe", return_value=town):
                    with self.assertRaises(ClickTimeoutError):
                        hero.click_until(lambda f: None, lambda f: False, "invisible control", **kwargs)
                self.assertEqual(hero.ctx.clock.now, expected)
                self.assertEqual(hero.ctx.inputs, [])

    def test_same_map_ignores_completed_flag_and_recognizes_different_region(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home"])
            self.assertTrue(hero.same_map(frame("region_before"), frame("region_after")))
            self.assertFalse(hero.same_map(frame("region_before"), frame("region_dream")))

    def test_adventure_entry_retries_and_waits_through_loading_before_local_dungeon(self):
        states = ["home", "region_before", "normal", "hero", "battle", "complete", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_dungeons=1)
            hero.ctx.dropped_clicks["home"] = 1
            original_capture = hero.ctx.capture
            encoded = cv2.imencode(".png", np.zeros((1080, 1920, 3), np.uint8))[1].tobytes()
            loading = Screenshot(encoded, 1920, 1080)

            def capture():
                if hero.ctx.index == 1 and hero.ctx.clock.now < .8:
                    return loading
                return original_capture()

            hero.ctx.capture = capture
            report = hero.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["home"] + states[:-1])
            self.assertGreaterEqual(hero.ctx.input_times[2], .8)

    def test_defeat_exits_and_reopens_same_dungeon_without_clicking_replay(self):
        cycle = ["region_before", "normal", "hero", "battle", "defeat"]
        states = cycle * 3 + ["region_before"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_retries=2)
            hero.observe()
            target = hero.markers(hero.frame)[0]
            completed = hero.fight_dungeon({"name": "test"}, target)
            self.assertFalse(completed)
            self.assertEqual(hero.report.failures, 3)
            self.assertEqual(len(hero.report.skipped), 1)
            exits = [(x, y) for state, x, y in hero.ctx.inputs if state == "defeat"]
            self.assertEqual(len(exits), 3)
            self.assertTrue(all(x > 1000 for x, _ in exits))

    def test_unknown_page_stops_and_saves_evidence_without_input(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["battle"], poll_interval=30)
            with self.assertRaises(ClickTimeoutError):
                hero.run()
            self.assertEqual(hero.ctx.clock.now, 300)
            self.assertEqual(hero.ctx.inputs, [])
            saved = json.loads((hero.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "stopped")
            self.assertTrue((hero.run_dir / "last_frame.jpg").is_file())

    def test_defeat_can_return_directly_to_hero_panel_preserving_progress(self):
        states = ["region_before", "normal", "hero", "battle", "defeat",
                  "hero_progress", "battle", "defeat", "hero_progress", "region_before"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_retries=1)
            hero.observe()
            target = hero.markers(hero.frame)[0]
            self.assertFalse(hero.fight_dungeon({"name": "test"}, target))
            self.assertEqual(hero.report.failures, 2)
            self.assertEqual(sum(s == "normal" for s, _, _ in hero.ctx.inputs), 1)
            self.assertEqual(sum(s == "hero_progress" for s, _, _ in hero.ctx.inputs), 2)

    def test_dry_run_plans_one_action_without_waiting_for_ui_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["home"])
            hero.ctx.dry_run = True
            report = hero.run()
            self.assertEqual(report.status, "preview")
            self.assertEqual(hero.ctx.index, 0)
            self.assertEqual(len(hero.ctx.inputs), 1)

    def test_invalid_runtime_options_are_rejected(self):
        for settings in ({"max_retries": -1}, {"max_retries": 1.5}, {"poll_interval": float("nan")}, {"max_runtime": 0},
                         {"max_click_retries": -1}, {"min_click_interval": 0}, {"click_retry_interval": float("inf")}):
            with self.subTest(settings=settings), self.assertRaises(AutomationError):
                HeroOptions(**settings)

    def test_target_players_requires_nonempty_string_list(self):
        for players in ("Hello", [], [""], ["   "], [123], None):
            with self.subTest(players=players), self.assertRaisesRegex(AutomationError, "target_players"):
                HeroOptions(target_players=players)
        options = HeroOptions(target_players=["coco", "Hell"])
        self.assertEqual(options.target_players, ["coco", "Hell"])
        first, second = HeroOptions(), HeroOptions()
        first.target_players.append("extra")
        self.assertEqual(second.target_players, ["Hello", "Hell", "coco"])

    def test_current_region_is_inspected_when_diamond_covers_its_flag(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["world", "region_after", "world"])
            self.assertEqual(hero.current_region(frame("world"))[0]["key"], "ring")
            with patch.object(hero, "regions", return_value=[]), patch.object(hero, "markers", return_value=[]):
                report = hero.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(hero.ctx.inputs), 2)
            self.assertEqual(hero.ctx.inputs[0][1:], (960, 1012))

    def test_incomplete_world_render_is_not_treated_as_no_remaining_work(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["world"])
            world = frame("world")
            self.assertTrue(hero.world_ready(world))
            world.image[735:819, 1301:1434] = 0
            self.assertTrue(hero.is_world(world))
            self.assertFalse(hero.world_ready(world))


if __name__ == "__main__":
    unittest.main()
