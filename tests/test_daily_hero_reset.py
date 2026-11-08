from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2

from pet_alliance.adb import AutomationError
from pet_alliance.daily_hero import DailyHero, HeroOptions
from pet_alliance.daily_hero_reset import DailyHeroReset, HeroResetOptions, ResetUnavailable
from pet_alliance.hero_reset_vision import HeroResetVision
from pet_alliance.vision import Match, VisualFrame
from tests.test_points_arena import ArenaReplayContext, Page


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGIONS = ["西部大陆", "环形领地", "诅咒荒野"]


class ReplayClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


class SceneFrame:
    def __init__(self, name):
        self.name = name

    def device_point(self, point):
        return point

    def save(self, path, matches=()):
        path.write_bytes(f"simulated screenshot: {self.name}".encode())
        return path


class ResetScene(DailyHeroReset):
    """Keep every entrance visible, including completed and skipped ones."""

    def __init__(self, directory, outcomes, *, current="flame", max_dungeons=0):
        ctx = SimpleNamespace(
            project_root=ROOT, settings={"daily_hero_reset": {"max_dungeons": max_dungeons}},
            log=logging.getLogger("reset-scene"), dry_run=False,
        )
        with patch("pet_alliance.daily_hero_reset.HeroResetVision"):
            super().__init__(ctx)
        self.run_dir.rmdir()
        self.run_dir = Path(directory) / "run"
        self.run_dir.mkdir()
        self.scene_regions = self.bank.profile["regions"]
        self.outcomes = outcomes
        self.scene_markers = {
            key: [Match(f"{key}:{i}", 100 + 100 * i, 200, 60, 24, 1)
                  for i in range(len(values))]
            for key, values in outcomes.items()
        }
        self.view = "map"
        self.selected = current
        self.attempts = []
        self.history = []
        self.arena_calls = []

    def observe(self):
        self.frame = SceneFrame(self.view)
        return self.view

    def adventure_map(self):
        return self.view != "world"

    def world(self):
        self.history.append(("world", self.selected))
        self.view = "world"

    def find(self, name, frame=None):
        if (frame or self.view) == "world":
            if name == "world_anchor":
                return Match(name, 0, 0, 10, 10, 1)
            if any(region["template"] == name for region in self.scene_regions):
                return Match(name, 100, 100, 80, 30, 1)
        return None

    def current_region(self, frame):
        if frame != "world":
            raise AssertionError("Region identity must come from the world map")
        region = next(r for r in self.scene_regions if r["key"] == self.selected)
        return region, self.find(region["template"], frame)

    def enter_region(self, region):
        if self.view != "world":
            raise AssertionError("Entering regions requires the world map")
        self.view = "map"
        self.selected = region["key"]
        self.history.append(("enter", self.selected))

    def markers(self, frame):
        if frame != "map":
            raise AssertionError("Dungeon discovery requires the local map")
        return self.scene_markers.get(self.selected, [])

    def fight_dungeon(self, region, target):
        if self.view != "map" or region["key"] != self.selected:
            raise AssertionError("The chosen region must be confirmed before any dungeon")
        identity = self.selected, target.name
        if identity in self.attempts:
            raise AssertionError(f"A completed or skipped dungeon was repeated: {identity}")
        self.attempts.append(identity)
        self.history.append(("dungeon", self.selected))
        index = self.scene_markers[self.selected].index(target)
        success = self.outcomes[self.selected][index]
        entry = {"region": region["name"], "marker": target.center, "failures": 0}
        (self.report.completed if success else self.report.skipped).append(entry)
        return success

    def run_points_arena(self, region):
        self.report.arena_visits += 1
        self.arena_calls.append((region["key"], len(self.report.completed), len(self.attempts)))
        # A return through town must retain the run loop's visited entrances.
        self.view = "town"
        self.world()
        self.enter_region(region)

    def return_to_town(self):
        self.history.append(("return_to_town", self.selected))
        self.view = "town"
        self.observe()


class DailyHeroResetTests(unittest.TestCase):
    def hero(self, directory, *, settings=None):
        ctx = SimpleNamespace(
            project_root=ROOT, settings=settings or {},
            log=logging.getLogger("reset-test"), dry_run=False,
        )
        with patch("pet_alliance.daily_hero_reset.HeroResetVision"):
            hero = DailyHeroReset(ctx)
        hero.run_dir.rmdir()
        hero.run_dir = Path(directory) / "run"
        hero.run_dir.mkdir()
        return hero

    def workflow(self, directory, states, **options):
        hero = self.hero(directory, settings={"daily_hero": {
            "poll_interval": .1, "min_click_interval": .1,
            "click_retry_interval": .4, "transition_timeout": 2, **options,
        }})
        clock = ReplayClock()
        clock_patch = patch("pet_alliance.daily_hero.time.monotonic", clock.monotonic)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        hero.started = 0
        hero.ctx.wait = clock.wait
        hero.ctx.clock = clock
        hero.ctx.states = states
        hero.ctx.index = 0
        hero.ctx.inputs = []
        hero.ctx.dropped = {}
        frames = {name: SceneFrame(name) for name in set(states)}
        target = Match("entrance", 400, 300, 100, 30, 1)
        controls = {
            "map": {"world_button"},
            "normal": {"panel_close", "normal_dragon"},
            "completed": {"panel_close", "hero_dragon"},
            "disabled": {"panel_close", "hero_dragon"},
            # A challenge-looking control behind the dialog must never be
            # mistaken for a successful confirmation.
            "confirmation": {"panel_close", "hero_dragon", "challenge"},
            "ready": {"panel_close", "hero_dragon", "challenge"},
            "battle": {"skip"},
            "defeat": {"defeat", "exit_battle"},
            "complete": {"victory", "next_disabled", "exit_battle"},
        }
        reset_controls = {
            "completed": {"reset"}, "disabled": {"reset_disabled"},
            "confirmation": {"reset_dialog", "confirm_reset"},
        }

        def observe():
            hero.frame = frames[states[hero.ctx.index]]
            return hero.frame

        def find(name, frame=None):
            current = frame or hero.frame
            if name in controls.get(current.name, set()):
                return Match(name, 700, 650, 100, 40, 1)
            return None

        def reset_find(frame, name):
            if name in reset_controls.get(frame.name, set()):
                return Match(name, 700, 650, 100, 40, 1)
            return None

        def tap(x, y):
            state = states[hero.ctx.index]
            hero.ctx.inputs.append((state, x, y))
            if hero.ctx.dropped.get(state, 0):
                hero.ctx.dropped[state] -= 1
                return
            hero.ctx.index += 1
            if hero.ctx.index >= len(states):
                raise AssertionError("Unexpected input after the replay ended")

        hero.observe = observe
        hero.find = find
        hero.is_map = lambda frame: frame.name == "map"
        hero.reset_vision.find.side_effect = reset_find
        hero.reset_vision.reset_cost.return_value = 50
        hero.reset_vision.entrances.side_effect = lambda frame: [target] if frame.name == "map" else []
        hero.ctx.tap = tap
        return hero, target

    def town_replay(self, directory, pages, *, size=(1920, 1080), **options):
        ctx = ArenaReplayContext(pages, size=size, **options)
        hero = DailyHeroReset(ctx)
        clock_patch = patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        hero.started = ctx.clock.now
        hero.run_dir.rmdir()
        hero.run_dir = Path(directory) / "run"
        hero.run_dir.mkdir()
        return hero

    def real_reset_replay(self, directory, fixture_paths, *, dropped=None):
        screenshots = [SimpleNamespace(data=path.read_bytes()) for path in fixture_paths]
        clock = ReplayClock()
        clock_patch = patch("pet_alliance.daily_hero.time.monotonic", clock.monotonic)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        hero = self.hero(directory, settings={"daily_hero": {
            "poll_interval": .1, "min_click_interval": .1,
            "click_retry_interval": .4, "transition_timeout": 2,
        }})
        hero.reset_vision = HeroResetVision(ROOT / "assets/daily_hero_reset")
        hero.ctx.index = 0
        hero.ctx.inputs = []
        hero.ctx.capture = lambda: screenshots[hero.ctx.index]
        hero.ctx.wait = clock.wait
        pending_drops = dict(dropped or {})

        def tap(x, y):
            if hero.ctx.index == len(screenshots) - 1:
                raise AssertionError("Unexpected click after entering the battle")
            hero.ctx.inputs.append((x, y))
            if pending_drops.get(hero.ctx.index, 0):
                pending_drops[hero.ctx.index] -= 1
                return
            hero.ctx.index += 1

        hero.ctx.tap = tap
        return hero

    def test_reset_options_inherit_hero_options_and_have_independent_region_lists(self):
        first, second = HeroResetOptions(), HeroResetOptions()
        self.assertIsInstance(first, HeroOptions)
        self.assertEqual(first.regions, DEFAULT_REGIONS)
        first.regions.append("天域")
        self.assertEqual(second.regions, DEFAULT_REGIONS)

    def test_reset_regions_require_unique_nonempty_string_list(self):
        for regions in (None, "西部大陆", [], [""], ["   "], [123], [True], ["西部大陆", "西部大陆"]):
            with self.subTest(regions=regions), self.assertRaisesRegex(AutomationError, "regions"):
                HeroResetOptions(regions=regions)

    def test_unknown_regions_are_rejected_before_any_input(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(
                project_root=Path(directory),
                settings={"daily_hero_reset": {"regions": ["不存在的区域"]}},
                log=logging.getLogger("invalid-reset-config"), dry_run=False,
            )
            bank = SimpleNamespace(profile={"regions": [{"name": "西部大陆", "key": "west"}]})
            with patch("pet_alliance.daily_hero.TemplateBank", return_value=bank):
                with self.assertRaisesRegex(AutomationError, "不存在的区域"):
                    DailyHeroReset(ctx)

    def test_reset_options_override_shared_values_without_mutating_settings(self):
        settings = {
            "daily_hero": {"poll_interval": .8, "click_retry_interval": 1.2,
                           "target_players": ["Hell", "coco"], "max_retries": 8},
            "daily_hero_reset": {"poll_interval": .5, "regions": ["环形领地"]},
        }
        expected = deepcopy(settings)
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, settings=settings)
            self.assertEqual(hero.options.poll_interval, .5)
            self.assertEqual(hero.options.click_retry_interval, 1.2)
            self.assertEqual(hero.options.target_players, ["Hell", "coco"])
            self.assertEqual(hero.options.max_retries, 8)
            self.assertEqual(hero.options.regions, ["环形领地"])
            self.assertEqual(settings, expected)

    def test_completed_dungeon_switches_hero_resets_confirms_then_challenges(self):
        with tempfile.TemporaryDirectory() as directory:
            states = ["map", "normal", "completed", "confirmation", "ready", "battle"]
            hero, target = self.workflow(directory, states)
            hero.open_dungeon(target)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], states[:-1])
            self.assertEqual(hero.ctx.inputs[0][1:], target.center)
            self.assertEqual(hero.report.resets, 1)
            self.assertEqual(hero.report.diamonds_spent, 50)
            self.assertEqual(len(hero.report.diamond_spending), 1)
            self.assertEqual(hero.report.diamond_spending[0]["marker"], target.center)
            self.assertEqual(hero.report.diamond_spending[0]["diamonds"], 50)

    def test_real_reset_prompt_integrates_vision_clicks_and_spending(self):
        for cost, fixture, confirm_point in (
                (50, "reset_confirm.png", (575, 546)),
                (10, "reset_confirm_10.jpg", (803, 764)),
                (10, "reset_confirm_10_raw.png", (803, 764)),
                (300, "reset_confirm_300_20261108.jpg", (803, 764))):
            for dropped_clicks in (False, True):
                with self.subTest(cost=cost, fixture=fixture, dropped_clicks=dropped_clicks), \
                        tempfile.TemporaryDirectory() as directory:
                    fixture_paths = [
                        ROOT / "tests/fixtures/daily_hero_reset/15.png",
                        ROOT / "tests/fixtures/daily_hero_reset" / fixture,
                        ROOT / "tests/fixtures/daily_hero/hero.jpg",
                        ROOT / "tests/fixtures/daily_hero/battle.jpg",
                    ]
                    hero = self.real_reset_replay(
                        directory, fixture_paths, dropped={0: 1, 1: 1} if dropped_clicks else {},
                    )
                    target = Match("entrance", 400, 300, 100, 30, 1)
                    # Begin with the completed hero panel already open. All later
                    # observations and controls use the production image matchers.
                    with patch.object(hero, "open_dungeon_panel"):
                        hero.open_dungeon(target)
                    repeats = 2 if dropped_clicks else 1
                    self.assertEqual(hero.ctx.inputs,
                                     [(683, 640)] * repeats + [confirm_point] * repeats + [(960, 907)])
                    events = [json.loads(line) for line in
                              (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual([event["template"] for event in events if event["event"] == "tap"],
                                     ["reset"] * repeats + ["confirm_reset"] * repeats + ["challenge"])
                    self.assertEqual(hero.ctx.index, 3)
                    self.assertEqual(hero.report.resets, 1)
                    self.assertEqual(hero.report.diamonds_spent, cost)
                    self.assertEqual(hero.report.diamond_spending[0]["diamonds"], cost)
                    charges = [event for event in events if event["event"] == "dungeon_reset"]
                    self.assertEqual(len(charges), 1)
                    self.assertEqual(charges[0]["diamonds"], cost)
                    quote_bytes = (hero.run_dir / "reset_1_quote.png").read_bytes()
                    self.assertTrue(quote_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
                    saved_quote = VisualFrame.from_screenshot(SimpleNamespace(data=quote_bytes))
                    source_quote = VisualFrame.from_screenshot(
                        SimpleNamespace(data=fixture_paths[1].read_bytes()),
                    )
                    self.assertEqual(cv2.norm(saved_quote.image, source_quote.image, cv2.NORM_INF), 0)

    def test_real_three_hundred_quote_is_counted_only_after_confirmed_transition(self):
        fixtures = ROOT / "tests/fixtures"
        fixture_paths = [
            fixtures / "daily_hero_reset/15.png",
            fixtures / "daily_hero_reset/reset_confirm_300_20261108.jpg",
            fixtures / "daily_hero/hero.jpg",
            fixtures / "daily_hero/battle.jpg",
        ]
        for confirmation_succeeds in (False, True):
            with self.subTest(confirmation_succeeds=confirmation_succeeds), \
                    tempfile.TemporaryDirectory() as directory:
                hero = self.real_reset_replay(
                    directory, fixture_paths,
                    dropped={1: 1 if confirmation_succeeds else 100},
                )
                original_tap = hero.ctx.tap
                confirmation_attempts = []

                def check_charge_boundary(x, y):
                    if hero.ctx.index == 1:
                        confirmation_attempts.append((x, y))
                        # Seeing this real 300-diamond dialog, or retrying
                        # its button, does not prove a purchase succeeded.
                        self.assertEqual(hero.report.resets, 0)
                        self.assertEqual(hero.report.diamonds_spent, 0)
                        self.assertEqual(hero.report.diamond_spending, [])
                        self.assertEqual(hero.reset_vision.reset_cost(hero.frame), 300)
                    elif hero.ctx.index == 2:
                        # The next challenge is allowed only after the dialog
                        # has disappeared and its cost was recorded once.
                        self.assertEqual(hero.report.resets, 1)
                        self.assertEqual(hero.report.diamonds_spent, 300)
                    original_tap(x, y)

                hero.ctx.tap = check_charge_boundary
                target = Match("entrance", 400, 300, 100, 30, 1)
                with patch.object(hero, "open_dungeon_panel"):
                    if confirmation_succeeds:
                        hero.open_dungeon(target)
                    else:
                        with self.assertRaises(AutomationError):
                            hero.open_dungeon(target)
                self.assertGreaterEqual(len(confirmation_attempts), 2)
                self.assertTrue(all(point == (803, 764) for point in confirmation_attempts))
                self.assertEqual(hero.ctx.index, 3 if confirmation_succeeds else 1)
                self.assertEqual(hero.report.resets, int(confirmation_succeeds))
                self.assertEqual(hero.report.diamonds_spent, 300 if confirmation_succeeds else 0)
                events = [json.loads(line) for line in
                          (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                charges = [event for event in events if event["event"] == "dungeon_reset"]
                self.assertEqual(len(charges), int(confirmation_succeeds))
                if confirmation_succeeds:
                    self.assertEqual(charges[0]["diamonds"], 300)

    def test_unreadable_real_cost_keeps_lossless_quote_before_stopping(self):
        fixtures = ROOT / "tests/fixtures"
        for fixture in ("reset_confirm_10_raw.png", "reset_confirm_300_20261108.jpg"):
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory() as directory:
                fixture_paths = [
                    fixtures / "daily_hero_reset/15.png",
                    fixtures / "daily_hero_reset" / fixture,
                    fixtures / "daily_hero/hero.jpg",
                    fixtures / "daily_hero/battle.jpg",
                ]
                hero = self.real_reset_replay(directory, fixture_paths)
                target = Match("entrance", 400, 300, 100, 30, 1)
                # Keep the real dialog/button matchers. An unreadable amount must
                # save precisely the pixels used for recognition before aborting.
                with patch.object(hero, "open_dungeon_panel"), \
                        patch.object(hero.reset_vision, "reset_cost", return_value=None), \
                        self.assertRaisesRegex(AutomationError, "无法识别重置所需的钻石数量"):
                    hero.open_dungeon(target)
                self.assertEqual(hero.ctx.inputs, [(683, 640)])
                self.assertEqual(hero.ctx.index, 1)
                self.assertEqual(hero.report.resets, 0)
                self.assertEqual(hero.report.diamonds_spent, 0)
                self.assertEqual(hero.report.diamond_spending, [])
                quote_bytes = (hero.run_dir / "reset_1_quote.png").read_bytes()
                self.assertTrue(quote_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
                saved_quote = VisualFrame.from_screenshot(SimpleNamespace(data=quote_bytes))
                source_quote = VisualFrame.from_screenshot(
                    SimpleNamespace(data=fixture_paths[1].read_bytes()),
                )
                self.assertEqual(cv2.norm(saved_quote.image, source_quote.image, cv2.NORM_INF), 0)

    def test_real_fifty_then_ten_diamond_resets_accumulate_only_confirmed_costs(self):
        fixtures = ROOT / "tests/fixtures"
        fixture_paths = [
            fixtures / "daily_hero_reset/15.png",
            fixtures / "daily_hero_reset/reset_confirm.png",
            fixtures / "daily_hero/hero.jpg",
            fixtures / "daily_hero/battle.jpg",
            fixtures / "daily_hero_reset/15.png",
            fixtures / "daily_hero_reset/reset_confirm_10.jpg",
            fixtures / "daily_hero/hero.jpg",
            fixtures / "daily_hero/battle.jpg",
        ]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.real_reset_replay(directory, fixture_paths, dropped={5: 1})
            first = Match("first entrance", 400, 300, 100, 30, 1)
            second = Match("second entrance", 900, 300, 100, 30, 1)
            with patch.object(hero, "open_dungeon_panel"):
                hero.open_dungeon(first)
                # The first dungeon has completed and the second panel is open.
                hero.ctx.index = 4
                hero.open_dungeon(second)
            self.assertEqual(hero.ctx.index, 7)
            self.assertEqual(hero.report.resets, 2)
            self.assertEqual(hero.report.diamonds_spent, 60)
            self.assertEqual([entry["diamonds"] for entry in hero.report.diamond_spending], [50, 10])
            self.assertEqual([entry["marker"] for entry in hero.report.diamond_spending],
                             [first.center, second.center])
            events = [json.loads(line) for line in
                      (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            charges = [event for event in events if event["event"] == "dungeon_reset"]
            self.assertEqual([event["diamonds_spent"] for event in charges], [50, 60])

    def test_reset_and_confirmation_share_dropped_click_retry_logic(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "completed", "confirmation", "ready", "battle"])
            hero.ctx.dropped = {"completed": 1, "confirmation": 1}
            hero.open_dungeon(target)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs],
                             ["map", "completed", "completed", "confirmation", "confirmation", "ready"])
            self.assertEqual(hero.report.resets, 1)
            self.assertEqual(hero.report.diamonds_spent, 50)
            self.assertEqual(len(hero.report.diamond_spending), 1)

    def test_each_successful_reset_adds_the_displayed_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            states = ["map", "completed", "confirmation", "ready", "battle"] * 2
            hero, first = self.workflow(directory, states)
            second = Match("second entrance", 900, 300, 100, 30, 1)
            hero.reset_vision.entrances.side_effect = (
                lambda frame: [first, second] if frame.name == "map" else []
            )
            hero.reset_vision.reset_cost.side_effect = (
                lambda frame: 50 if hero.ctx.index < 5 else 75
            )
            hero.open_dungeon(first)
            # The first battle has completed and navigation returned to the map.
            hero.ctx.index = 5
            hero.open_dungeon(second)
            self.assertEqual(hero.report.resets, 2)
            self.assertEqual(hero.report.diamonds_spent, 125)
            self.assertEqual([entry["diamonds"] for entry in hero.report.diamond_spending], [50, 75])
            self.assertEqual([entry["marker"] for entry in hero.report.diamond_spending],
                             [first.center, second.center])

    def test_unreadable_cost_stops_before_confirming_or_recording_a_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "completed", "confirmation", "ready", "battle"])
            hero.reset_vision.reset_cost.return_value = None
            with self.assertRaises(AutomationError):
                hero.open_dungeon(target)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["map", "completed"])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)
            self.assertEqual(hero.report.diamond_spending, [])

    def test_confirmation_timeout_does_not_record_an_unconfirmed_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "completed", "confirmation", "ready", "battle"],
                                         max_click_retries=100)
            hero.ctx.dropped["confirmation"] = 100
            with self.assertRaises(AutomationError):
                hero.open_dungeon(target)
            self.assertTrue(any(state == "confirmation" for state, _, _ in hero.ctx.inputs))
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)
            self.assertEqual(hero.report.diamond_spending, [])

    def test_confirm_retry_stops_if_displayed_cost_changes_or_becomes_unreadable(self):
        for changed_cost in (10, 75, None):
            with self.subTest(changed_cost=changed_cost), tempfile.TemporaryDirectory() as directory:
                hero, target = self.workflow(directory,
                                             ["map", "completed", "confirmation", "ready", "battle"])
                hero.ctx.dropped["confirmation"] = 1
                hero.reset_vision.reset_cost.side_effect = (
                    lambda frame: changed_cost if any(state == "confirmation" for state, _, _
                                                      in hero.ctx.inputs) else 50
                )
                with self.assertRaises(AutomationError):
                    hero.open_dungeon(target)
                self.assertEqual([state for state, _, _ in hero.ctx.inputs],
                                 ["map", "completed", "confirmation"])
                self.assertEqual(hero.report.diamonds_spent, 0)
                self.assertEqual(hero.report.diamond_spending, [])

    def test_gray_reset_is_not_clicked_and_is_reported_as_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "disabled", "map"])
            self.assertFalse(hero.fight_dungeon({"key": "ring", "name": "环形领地"}, target))
            self.assertEqual(hero.report.failures, 0)
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)
            self.assertEqual(hero.report.diamond_spending, [])
            self.assertEqual(len(hero.report.skipped), 1)
            self.assertEqual(hero.report.skipped[0]["failures"], 0)
            self.assertEqual(hero.report.skipped[0]["reason"], "已重置过（重置按钮为灰色）")
            self.assertEqual(hero.report.skipped[0]["reason_code"], "already_reset")
            # The only input on the disabled panel closes it.
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["map", "disabled"])
            taps = [event for event in (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                    if '"event": "tap"' in event]
            self.assertFalse(any('"template": "reset_disabled"' in event for event in taps))

    def test_gray_reset_open_raises_specific_unavailable_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "disabled"])
            with self.assertRaises(ResetUnavailable):
                hero.open_dungeon(target)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["map"])

    def test_gray_reset_on_resume_is_also_treated_as_already_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["disabled"])
            with self.assertRaises(ResetUnavailable):
                hero.open_dungeon(target, resume=True)
            self.assertEqual(hero.ctx.inputs, [])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)

    def test_failed_battle_returning_to_gray_panel_is_skipped_once_with_its_failures(self):
        for returned_to_map in (False, True):
            with self.subTest(returned_to_map=returned_to_map), tempfile.TemporaryDirectory() as directory:
                states = ["map", "completed", "confirmation", "ready", "battle", "defeat"]
                states += (["map", "normal", "disabled"] if returned_to_map else ["disabled"])
                states += ["map"]
                hero, target = self.workflow(directory, states)
                # Failures in earlier dungeons must not appear on this entry.
                hero.report.failures = 7
                self.assertFalse(hero.fight_dungeon({"key": "west", "name": "西部大陆"}, target))
                self.assertEqual([state for state, _, _ in hero.ctx.inputs], states[:-1])
                self.assertEqual(hero.report.failures, 8)
                self.assertEqual(hero.report.skipped, [{
                    "region": "西部大陆", "marker": target.center, "failures": 1,
                    "reason": "已重置过（重置按钮为灰色）", "reason_code": "already_reset",
                }])
                self.assertEqual(hero.report.completed, [])
                self.assertEqual(hero.report.resets, 1)
                self.assertEqual(hero.report.diamonds_spent, 50)
                self.assertEqual(len(hero.report.diamond_spending), 1)
                events = [json.loads(line) for line in
                          (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(sum(event["event"] == "dungeon_reset_unavailable" for event in events), 1)

    def test_real_gray_reset_panel_closes_and_skips_without_reset_or_challenge(self):
        fixture_paths = [
            ROOT / "tests/fixtures/daily_hero_reset/16.png",
            ROOT / "tests/fixtures/daily_hero/region_before.jpg",
        ]
        screenshots = [SimpleNamespace(data=path.read_bytes()) for path in fixture_paths]
        clock = ReplayClock()
        with tempfile.TemporaryDirectory() as directory, patch(
                "pet_alliance.daily_hero.time.monotonic", clock.monotonic):
            hero = self.hero(directory, settings={"daily_hero": {
                "poll_interval": .1, "min_click_interval": .1,
                "click_retry_interval": .4, "transition_timeout": 2,
            }})
            hero.reset_vision = HeroResetVision(ROOT / "assets/daily_hero_reset")
            hero.ctx.index = 0
            hero.ctx.inputs = []
            hero.ctx.capture = lambda: screenshots[hero.ctx.index]
            hero.ctx.wait = clock.wait

            def tap(x, y):
                if hero.ctx.index:
                    raise AssertionError("Only closing the already-reset panel is allowed")
                hero.ctx.inputs.append((x, y))
                hero.ctx.index += 1

            hero.ctx.tap = tap
            target = Match("entrance", 400, 300, 100, 30, 1)
            with patch.object(hero, "open_dungeon_panel"):
                completed = hero.fight_dungeon({"key": "ring", "name": "环形领地"}, target)
            self.assertFalse(completed)
            self.assertEqual(hero.ctx.inputs, [(1015, 180)])
            events = [json.loads(line) for line in
                      (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["template"] for event in events if event["event"] == "tap"],
                             ["panel_close"])
            self.assertEqual(hero.report.skipped[0]["reason_code"], "already_reset")
            self.assertEqual(hero.report.skipped[0]["failures"], 0)
            self.assertEqual(hero.report.completed, [])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)
            self.assertEqual(hero.report.diamond_spending, [])

    def test_failed_west_normal_panel_switches_to_hero_and_skips_gray_reset(self):
        fixture_paths = [
            ROOT / "tests/fixtures/daily_hero_reset/failed_west_normal.jpg",
            ROOT / "tests/fixtures/daily_hero_reset/16.png",
            ROOT / "tests/fixtures/daily_hero/region_before.jpg",
        ]
        screenshots = [SimpleNamespace(data=path.read_bytes()) for path in fixture_paths]
        clock = ReplayClock()
        with tempfile.TemporaryDirectory() as directory, patch(
                "pet_alliance.daily_hero.time.monotonic", clock.monotonic):
            hero = self.hero(directory, settings={"daily_hero": {
                "poll_interval": .1, "min_click_interval": .1,
                "click_retry_interval": .4, "transition_timeout": 2,
            }})
            hero.reset_vision = HeroResetVision(ROOT / "assets/daily_hero_reset")
            hero.ctx.index = 0
            hero.ctx.inputs = []
            hero.ctx.capture = lambda: screenshots[hero.ctx.index]
            hero.ctx.wait = clock.wait

            def tap(x, y):
                if hero.ctx.index == 0:
                    self.assertTrue(450 <= x <= 610 and 190 <= y <= 355,
                                    "The normal-mode dragon icon must be clicked first")
                elif hero.ctx.index == 1:
                    self.assertEqual((x, y), (1015, 180))
                else:
                    raise AssertionError("No input is allowed after the gray panel closes")
                hero.ctx.inputs.append((x, y))
                hero.ctx.index += 1

            hero.ctx.tap = tap
            target = Match("entrance", 400, 300, 100, 30, 1)
            with patch.object(hero, "open_dungeon_panel"):
                completed = hero.fight_dungeon({"key": "west", "name": "西部大陆"}, target)
            self.assertFalse(completed)
            self.assertEqual(len(hero.ctx.inputs), 2)
            events = [json.loads(line) for line in
                      (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["template"] for event in events if event["event"] == "tap"],
                             ["normal_dragon", "panel_close"])
            self.assertEqual(hero.report.skipped[0]["reason_code"], "already_reset")
            self.assertEqual(hero.report.skipped[0]["failures"], 0)
            self.assertEqual(hero.report.completed, [])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)
            self.assertEqual(hero.report.diamond_spending, [])

    def test_normal_mode_control_accepts_only_normal_panels(self):
        fixtures = [
            ("daily_hero/normal.jpg", True),
            ("daily_hero_reset/failed_west_normal.jpg", True),
            ("daily_hero/hero.jpg", False),
            ("daily_hero_reset/15.png", False),
            ("daily_hero_reset/16.png", False),
            ("daily_hero_reset/reset_confirm.png", False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory)
            for name, expected in fixtures:
                with self.subTest(fixture=name):
                    screenshot = SimpleNamespace(data=(ROOT / "tests/fixtures" / name).read_bytes())
                    frame = VisualFrame.from_screenshot(screenshot)
                    control = hero.normal_mode_control(frame)
                    self.assertEqual(control is not None, expected)
                    if expected:
                        self.assertEqual(control.name, "normal_dragon")

    def test_dimmed_normal_panel_does_not_click_dragon(self):
        screenshot = SimpleNamespace(
            data=(ROOT / "tests/fixtures/daily_hero/normal.jpg").read_bytes())
        normal = VisualFrame.from_screenshot(screenshot)
        dimmed = VisualFrame((normal.image * .75).astype("uint8"))
        ok, encoded = cv2.imencode(".png", dimmed.image)
        self.assertTrue(ok)
        clock = ReplayClock()
        with tempfile.TemporaryDirectory() as directory, patch(
                "pet_alliance.daily_hero.time.monotonic", clock.monotonic):
            hero = self.hero(directory, settings={"daily_hero": {
                "poll_interval": .1, "min_click_interval": .1,
                "click_retry_interval": .4, "transition_timeout": 2,
            }})
            hero.reset_vision = HeroResetVision(ROOT / "assets/daily_hero_reset")
            hero.ctx.capture = lambda: SimpleNamespace(data=encoded.tobytes())
            hero.ctx.wait = clock.wait
            hero.ctx.inputs = []
            hero.ctx.tap = lambda x, y: hero.ctx.inputs.append((x, y))
            self.assertIsNotNone(hero.find("normal_dragon", dimmed))
            self.assertIsNone(hero.find("panel_close", dimmed))
            self.assertIsNone(hero.normal_mode_control(dimmed))
            with patch.object(hero, "open_dungeon_panel"), self.assertRaises(AutomationError):
                hero.open_dungeon(Match("entrance", 400, 300, 100, 30, 1))
            self.assertEqual(hero.ctx.inputs, [])

    def test_gray_skip_does_not_count_towards_three_dungeon_arena_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            states = ["map", "disabled", "map", "ready", "battle", "complete", "map",
                      "ready", "battle", "complete", "map"]
            hero, first = self.workflow(directory, states, max_dungeons=3)
            targets = [first, Match("second", 800, 300, 100, 30, 1),
                       Match("third", 1200, 300, 100, 30, 1)]
            hero.starting_region = lambda: {"key": "west", "name": "西部大陆"}
            hero.markers = lambda frame: targets if frame.name == "map" else []
            hero.reset_vision.entrances.side_effect = hero.markers
            with patch.object(hero, "run_points_arena") as arena, patch.object(hero, "return_to_town") as town:
                report = hero.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(report.completed), 2)
            self.assertEqual(len(report.skipped), 1)
            self.assertEqual(report.skipped[0]["reason_code"], "already_reset")
            arena.assert_not_called()
            town.assert_called_once_with()
            self.assertEqual(report.resets, 0)
            self.assertEqual(report.diamonds_spent, 0)

    def test_resume_challenges_without_reset_when_returned_to_existing_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["ready", "battle"])
            hero.open_dungeon(target, resume=True)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["ready"])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)

    def test_resume_reopens_map_dungeon_without_another_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "normal", "ready", "battle"])
            hero.open_dungeon(target, resume=True)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["map", "normal", "ready"])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)

    def test_full_fight_failure_retries_keep_reset_once_for_both_exit_destinations(self):
        for returned_to_map in (False, True):
            with self.subTest(returned_to_map=returned_to_map), tempfile.TemporaryDirectory() as directory:
                resume_states = ["map", "normal", "ready"] if returned_to_map else ["ready"]
                states = ["map", "normal", "completed", "confirmation", "ready", "battle", "defeat"]
                states += resume_states + ["battle", "complete", "map"]
                hero, target = self.workflow(directory, states)
                self.assertTrue(hero.fight_dungeon({"key": "west", "name": "西部大陆"}, target))
                inputs = [state for state, _, _ in hero.ctx.inputs]
                self.assertEqual(inputs, states[:-1])
                self.assertEqual(inputs.count("completed"), 1)
                self.assertEqual(inputs.count("confirmation"), 1)
                self.assertEqual(hero.report.resets, 1)
                self.assertEqual(hero.report.diamonds_spent, 50)
                self.assertEqual(hero.report.diamond_spending,
                                 [{"region": "西部大陆", "marker": target.center, "diamonds": 50}])
                self.assertEqual(hero.report.failures, 1)
                self.assertEqual(len(hero.report.completed), 1)
                self.assertEqual(hero.report.skipped, [])

    def test_resume_unexpected_completed_panel_stops_without_clicking_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["completed"])
            with self.assertRaises(AutomationError):
                hero.open_dungeon(target, resume=True)
            self.assertEqual(hero.ctx.inputs, [])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)

    def test_unfinished_hero_panel_challenges_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "normal", "ready", "battle"])
            hero.open_dungeon(target)
            self.assertEqual([state for state, _, _ in hero.ctx.inputs], ["map", "normal", "ready"])
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)

    def test_run_summary_records_paid_reset_even_when_battle_later_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory,
                                         ["map", "completed", "confirmation", "ready", "battle", "defeat", "map"],
                                         max_retries=0, max_dungeons=1)
            hero.starting_region = lambda: {"key": "west", "name": "西部大陆"}
            hero.markers = lambda frame: [target] if frame.name == "map" else []
            with patch.object(hero, "return_to_town") as town:
                report = hero.run()
            summary = json.loads((hero.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(report.skipped), 1)
            self.assertEqual(summary["diamonds_spent"], 50)
            self.assertEqual(summary["resets"], 1)
            self.assertTrue(summary["returned_to_town"])
            town.assert_called_once_with()
            self.assertEqual(summary["diamond_spending"],
                             [{"region": "西部大陆", "marker": list(target.center), "diamonds": 50}])

    def test_error_and_ctrl_c_summaries_keep_previous_confirmed_spending(self):
        for failure, status in ((AutomationError("battle capture failed"), "stopped"),
                                (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                states = ["map", "completed", "confirmation", "ready", "battle"]
                hero, target = self.workflow(directory, states)
                hero.starting_region = lambda: {"key": "west", "name": "西部大陆"}
                hero.markers = lambda frame: [target] if frame.name == "map" else []
                original_tap = hero.ctx.tap

                def interrupted_battle(x, y):
                    if states[hero.ctx.index] == "battle":
                        raise failure
                    original_tap(x, y)

                hero.ctx.tap = interrupted_battle
                with patch.object(hero, "return_to_town") as town, self.assertRaises(type(failure)):
                    hero.run()
                summary = json.loads((hero.run_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], status)
                self.assertEqual(summary["diamonds_spent"], 50)
                self.assertEqual(summary["resets"], 1)
                self.assertEqual(summary["diamond_spending"][0]["region"], "西部大陆")
                self.assertFalse(summary["returned_to_town"])
                town.assert_not_called()

    def test_spending_starts_at_zero_for_each_new_run(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first, target = self.workflow(first_dir, ["map", "completed", "confirmation", "ready", "battle"])
            first.open_dungeon(target)
            second = self.hero(second_dir)
            self.assertEqual(first.report.diamonds_spent, 50)
            self.assertEqual(second.report.resets, 0)
            self.assertEqual(second.report.diamonds_spent, 0)
            self.assertEqual(second.report.diamond_spending, [])
            self.assertIsNot(first.report.diamond_spending, second.report.diamond_spending)

    def test_nameplate_entry_retries_at_its_center(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "ready", "battle"])
            hero.ctx.dropped["map"] = 1
            hero.open_dungeon(target)
            self.assertEqual(hero.ctx.inputs[:2], [("map", *target.center), ("map", *target.center)])

    def test_nameplate_retry_uses_a_freshly_recognized_position(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.workflow(directory, ["map", "ready", "battle"])
            hero.ctx.dropped["map"] = 1
            moved = Match(target.name, target.x + 20, target.y, target.width, target.height, 1)
            hero.reset_vision.entrances.side_effect = (
                lambda frame: [moved if hero.ctx.inputs else target] if frame.name == "map" else []
            )
            hero.open_dungeon(target)
            self.assertEqual(hero.ctx.inputs[:2], [("map", *target.center), ("map", *moved.center)])

    def test_unexpected_error_is_not_silently_converted_to_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory)
            target = Match("entrance", 400, 300, 100, 30, 1)
            with patch.object(DailyHero, "fight_dungeon", side_effect=AutomationError("unexpected page")):
                with self.assertRaisesRegex(AutomationError, "unexpected page"):
                    hero.fight_dungeon({"key": "ring", "name": "环形领地"}, target)
            self.assertEqual(hero.report.skipped, [])

    def test_full_run_confirms_world_before_dungeons_and_only_visits_selected_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [True], "ring": [True], "cursed": [True],
                                           "flame": [True]})
            report = scene.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.history[0], ("world", "flame"))
            self.assertEqual([key for key, _ in scene.attempts], ["west", "ring", "cursed"])
            self.assertEqual(scene.arena_calls, [("cursed", 3, 3)])
            self.assertEqual(scene.history[-1], ("return_to_town", "cursed"))
            self.assertTrue(report.returned_to_town)
            summary = json.loads((scene.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["returned_to_town"])
            self.assertEqual((scene.run_dir / "last_frame.jpg").read_bytes(), b"simulated screenshot: town")
            events = [json.loads(line) for line in
                      (scene.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["event"] == "reset_returned_to_town" for event in events), 1)

    def test_processed_selected_regions_never_fall_back_to_unselected_current_region(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [], "ring": [], "cursed": [], "flame": [True]})
            scene.view = "world"
            processed = {"west", "ring", "cursed"}
            self.assertEqual(scene.region_candidates("world", processed), [])
            self.assertEqual(scene.attempts, [])

    def test_selected_regions_are_candidates_without_world_red_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {})
            scene.view = "world"
            candidates = scene.region_candidates("world", {"west"})
            self.assertEqual([region["key"] for region, _ in candidates], ["ring", "cursed"])

    def test_missing_selected_region_label_stops_instead_of_silently_skipping(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {})
            scene.view = "world"
            original = scene.find
            scene.find = lambda name, frame=None: None if name == "region_ring" else original(name, frame)
            with self.assertRaisesRegex(AutomationError, "环形领地"):
                scene.region_candidates("world", set())
            self.assertEqual(scene.attempts, [])

    def test_arena_counts_only_completed_dungeons_and_preserves_visited_entrances(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [False, True, True, True, True],
                                           "ring": [True], "cursed": [True]})
            report = scene.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(report.completed), 6)
            self.assertEqual(len(report.skipped), 1)
            self.assertEqual(scene.arena_calls, [("west", 3, 4), ("cursed", 6, 7)])
            self.assertEqual(len(scene.attempts), 7)

    def test_max_dungeons_runs_third_completion_arena_before_return(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [True] * 5}, max_dungeons=3)
            report = scene.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(scene.attempts), 3)
            self.assertEqual(scene.arena_calls, [("west", 3, 3)])
            self.assertEqual(scene.history[-1], ("return_to_town", "west"))
            self.assertTrue(report.returned_to_town)
            summary = json.loads((scene.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "limited")
            self.assertTrue(summary["returned_to_town"])
            self.assertEqual((scene.run_dir / "last_frame.jpg").read_bytes(), b"simulated screenshot: town")

    def test_all_gray_skipped_dungeons_still_return_to_town(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [False], "ring": [False], "cursed": [False]})
            report = scene.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(report.skipped), 3)
            self.assertEqual(report.completed, [])
            self.assertEqual(report.diamonds_spent, 0)
            self.assertEqual(scene.arena_calls, [])
            self.assertTrue(report.returned_to_town)
            self.assertEqual(scene.history[-1], ("return_to_town", "cursed"))

    def test_preview_does_not_return_to_town(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, _ = self.workflow(directory, ["map"])
            hero.ctx.dry_run = True
            with patch.object(hero, "return_to_town") as town:
                report = hero.run()
            self.assertEqual(report.status, "preview")
            self.assertFalse(report.returned_to_town)
            town.assert_not_called()
            self.assertEqual(hero.ctx.inputs, [])

    def test_return_to_town_replays_real_map_and_retries_dropped_first_tap(self):
        for size in ((849, 477), (1920, 1080)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                hero = self.town_replay(directory,
                                        [Page("region_before", "world_button"), Page("1", "town"), Page("2")],
                                        size=size)
                hero.ctx.dropped_clicks[1] = 1
                hero.finish()
                self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                                 ["world_button", "town", "town"])
                self.assertGreaterEqual(hero.ctx.input_times[2] - hero.ctx.input_times[1], .4 - 1e-9)
                self.assertEqual(hero.ctx.index, 2)
                self.assertIsNotNone(hero.town_find("town_page"))
                self.assertTrue(hero.report.returned_to_town)

    def test_already_in_town_uses_fresh_screenshot_and_sends_no_input(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.town_replay(directory, [Page("1"), Page("2")])
            stale_world = hero.observe()
            hero.ctx.index = 1
            self.assertIsNotNone(hero.find("world_anchor", stale_world))
            hero.finish()
            self.assertEqual(hero.ctx.inputs, [])
            self.assertIsNotNone(hero.town_find("town_page"))
            self.assertTrue(hero.report.returned_to_town)

    def test_success_summary_and_last_frame_are_saved_after_real_return_to_town(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.town_replay(directory, [Page("1", "town"), Page("2")])
            hero.starting_region = lambda: None
            hero.region_candidates = lambda frame, processed: []
            report = hero.run()
            self.assertEqual(report.status, "completed")
            self.assertTrue(report.returned_to_town)
            summary = json.loads((hero.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["returned_to_town"])
            last_frame = VisualFrame.from_screenshot(
                SimpleNamespace(data=(hero.run_dir / "last_frame.jpg").read_bytes()))
            self.assertIsNotNone(hero.town_find("town_page", last_frame))
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs], ["town"])

    def test_return_to_town_timeout_stops_run_and_keeps_diamond_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.town_replay(directory, [Page("1", "town"), Page("2")], poll_interval=5)
            hero.starting_region = lambda: None
            hero.region_candidates = lambda frame, processed: []
            # The town click leaves the source screen, but loading never
            # completes within the new five-minute arrival budget.
            hero.ctx.loading_delays[0] = 600
            hero.report.resets = 1
            hero.report.diamonds_spent = 50
            hero.report.diamond_spending = [{"region": "西部大陆", "marker": [400, 300], "diamonds": 50}]
            with self.assertRaises(AutomationError):
                hero.run()
            summary = json.loads((hero.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "stopped")
            self.assertFalse(summary["returned_to_town"])
            self.assertEqual(summary["resets"], 1)
            self.assertEqual(summary["diamonds_spent"], 50)
            self.assertEqual(summary["diamond_spending"], hero.report.diamond_spending)
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs], ["town"])
            self.assertGreaterEqual(hero.ctx.clock.now, 300)
            self.assertLess(hero.ctx.clock.now, 305)


if __name__ == "__main__":
    unittest.main()
