from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pet_alliance.adb import AutomationError
from pet_alliance.daily_hero import DailyHero
from pet_alliance.daily_hero_reset import DailyHeroReset, HeroResetOptions
from pet_alliance.run_logs import RunLogRetention
from pet_alliance.vision import Match
from tests import test_daily_hero_reset as reset_helpers


ROOT = Path(__file__).resolve().parent.parent
DREAM_TARGET = "奇幻梦境-龙萌幻境"


def region_name(region):
    return region["name"] if isinstance(region, dict) else region


class TargetResetScene(reset_helpers.ResetScene):
    """Replay real work selection with fake scenes and a named-entry registry."""

    def __init__(self, directory, outcomes, *, regions=None, dungeons=None,
                 named=None, points_enabled=True, max_dungeons=0):
        reset_settings = {"max_dungeons": max_dungeons}
        if regions is not None:
            reset_settings["regions"] = regions
        if dungeons is not None:
            reset_settings["dungeons"] = dungeons
        self.named_indices = dict(named or {})
        self.named_lookups = []
        self.vision = Mock()

        def validate(region, name):
            if (region_name(region), name) not in self.named_indices:
                raise AutomationError(f"尚未支持的指定副本：{region_name(region)}-{name}")

        self.vision.validate_dungeon.side_effect = validate
        ctx = SimpleNamespace(
            project_root=ROOT,
            settings={"daily_hero": {"points_arena_enabled": points_enabled},
                      "daily_hero_reset": reset_settings},
            log=logging.getLogger("reset-target-scene"), dry_run=False,
            run_logs=RunLogRetention(Path(directory)), tap=Mock(),
        )
        with patch("pet_alliance.daily_hero_reset.HeroResetVision", return_value=self.vision):
            DailyHeroReset.__init__(self, ctx)
        self.scene_regions = self.bank.profile["regions"]
        self.outcomes = outcomes
        self.scene_markers = {
            key: [Match(f"{key}:{index}", 100 + 220 * index, 200, 60, 24, 1)
                  for index in range(len(values))]
            for key, values in outcomes.items()
        }
        self.view = "map"
        self.selected = "flame"
        self.attempts = []
        self.history = []
        self.arena_calls = []

        def named_entrance(frame, region, name):
            key = region_name(region), name
            self.named_lookups.append(key)
            if frame != "map":
                raise AssertionError("Named entry recognition requires the local map")
            expected_region = next(r for r in self.scene_regions if r["name"] == key[0])
            if expected_region["key"] != self.selected:
                raise AssertionError("Named target was searched in another region")
            index = self.named_indices[key]
            return None if index is None else self.scene_markers[self.selected][index]

        self.vision.named_entrance.side_effect = named_entrance


class ResetDungeonTargetTests(unittest.TestCase):
    def replay(self, directory, states, **options):
        """Reuse click replay helpers while keeping all diagnostics temporary."""
        helper = reset_helpers.DailyHeroResetTests()
        self.addCleanup(helper.doCleanups)

        def initialize(owner, _ctx, _name):
            owner._run_log = None
            owner._run_dir = Path(directory) / "initial"
            owner._run_dir.mkdir()

        with patch.object(DailyHero, "_init_run_log", initialize):
            return helper.workflow(directory, states, **options)

    def test_dungeon_options_default_is_empty_and_not_shared(self):
        first, second = HeroResetOptions(), HeroResetOptions()
        self.assertEqual(first.dungeons, [])
        first.dungeons.append(DREAM_TARGET)
        self.assertEqual(second.dungeons, [])

    def test_dungeon_only_options_are_allowed_and_trimmed(self):
        options = HeroResetOptions(regions=[], dungeons=[" 奇幻梦境 - 龙萌幻境 "])
        self.assertEqual(options.regions, [])
        self.assertEqual(options.dungeons, [DREAM_TARGET])

    def test_invalid_dungeon_lists_are_rejected(self):
        for targets in (None, DREAM_TARGET, [""], ["   "], [123], [True],
                        ["奇幻梦境"], ["-龙萌幻境"], ["奇幻梦境-"],
                        [DREAM_TARGET, " 奇幻梦境 - 龙萌幻境 "]):
            with self.subTest(targets=targets), self.assertRaisesRegex(AutomationError, "dungeons"):
                HeroResetOptions(dungeons=targets)

    def test_no_regions_and_no_dungeons_is_rejected(self):
        with self.assertRaises(AutomationError):
            HeroResetOptions(regions=[], dungeons=[])

    def test_full_regions_run_first_then_only_the_named_entrance(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {"west": [True], "ring": [True], "cursed": [True],
                            "dream": [True, True, True]}, dungeons=[DREAM_TARGET],
                named={("奇幻梦境", "龙萌幻境"): 1},
            )
            report = scene.run()
            self.assertEqual(scene.attempts,
                             [("west", "west:0"), ("ring", "ring:0"),
                              ("cursed", "cursed:0"), ("dream", "dream:1")])
            self.assertEqual(report.status, "completed")
            self.assertTrue(report.returned_to_town)
            self.assertEqual(scene.history[-1], ("return_to_town", "dream"))
            self.assertEqual(scene.arena_calls, [("cursed", 3, 3)])
            self.assertTrue(scene.named_lookups)
            scene.ctx.tap.assert_not_called()

    def test_individual_items_preserve_order_with_repeated_nonadjacent_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            targets = [DREAM_TARGET, "魔都-绝望监狱", "奇幻梦境-第二梦境"]
            scene = TargetResetScene(
                directory, {"dream": [True, True, True], "demon": [True, True]},
                regions=[], dungeons=targets,
                named={("奇幻梦境", "龙萌幻境"): 1, ("魔都", "绝望监狱"): 0,
                       ("奇幻梦境", "第二梦境"): 2},
            )
            report = scene.run()
            self.assertEqual(scene.attempts, [("dream", "dream:1"),
                                             ("demon", "demon:0"),
                                             ("dream", "dream:2")])
            self.assertEqual(scene.arena_calls, [("dream", 3, 3)])
            self.assertEqual(report.status, "completed")
            self.assertTrue(report.returned_to_town)

    def test_points_count_crosses_the_region_and_individual_work_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {"west": [True, True], "dream": [True, True]},
                regions=["西部大陆"], dungeons=[DREAM_TARGET, "奇幻梦境-第二梦境"],
                named={("奇幻梦境", "龙萌幻境"): 0, ("奇幻梦境", "第二梦境"): 1},
            )
            report = scene.run()
            self.assertEqual(len(report.completed), 4)
            self.assertEqual(scene.arena_calls, [("dream", 3, 3)])
            self.assertEqual(scene.attempts[-1], ("dream", "dream:1"))
            self.assertTrue(report.returned_to_town)

    def test_skipped_named_target_does_not_advance_points_count(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {"west": [True, True], "dream": [False, True]},
                regions=["西部大陆"], dungeons=[DREAM_TARGET, "奇幻梦境-第二梦境"],
                named={("奇幻梦境", "龙萌幻境"): 0, ("奇幻梦境", "第二梦境"): 1},
            )
            report = scene.run()
            self.assertEqual(len(report.completed), 3)
            self.assertEqual(len(report.skipped), 1)
            self.assertEqual(scene.arena_calls, [("dream", 3, 4)])

    def test_points_disabled_applies_to_both_work_types(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {"west": [True, True], "dream": [True, True]},
                regions=["西部大陆"], dungeons=[DREAM_TARGET, "奇幻梦境-第二梦境"],
                named={("奇幻梦境", "龙萌幻境"): 0, ("奇幻梦境", "第二梦境"): 1},
                points_enabled=False,
            )
            report = scene.run()
            self.assertEqual(len(report.completed), 4)
            self.assertEqual(scene.arena_calls, [])
            self.assertFalse(report.points_arena_pending)

    def test_global_limit_can_stop_before_any_individual_target(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {"west": [True, True], "dream": [True]},
                regions=["西部大陆"], dungeons=[DREAM_TARGET],
                named={("奇幻梦境", "龙萌幻境"): 0}, max_dungeons=2,
            )
            report = scene.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(scene.attempts, [("west", "west:0"), ("west", "west:1")])
            self.assertEqual(scene.named_lookups, [])
            self.assertTrue(report.returned_to_town)

    def test_global_limit_runs_third_completion_points_before_stopping_individual_work(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {"west": [True, True], "dream": [True, True]},
                regions=["西部大陆"], dungeons=[DREAM_TARGET, "奇幻梦境-第二梦境"],
                named={("奇幻梦境", "龙萌幻境"): 0, ("奇幻梦境", "第二梦境"): 1},
                max_dungeons=3,
            )
            report = scene.run()
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(scene.attempts), 3)
            self.assertEqual(scene.arena_calls, [("dream", 3, 3)])
            self.assertTrue(report.returned_to_town)

    def test_missing_named_target_stops_without_selecting_other_visible_dungeons(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {"west": [True], "dream": [True, True]},
                regions=["西部大陆"], dungeons=[DREAM_TARGET],
                named={("奇幻梦境", "龙萌幻境"): None},
            )
            with self.assertRaisesRegex(AutomationError, "龙萌幻境"):
                scene.run()
            self.assertEqual(scene.attempts, [("west", "west:0")])
            self.assertEqual(scene.report.status, "stopped")
            self.assertFalse(scene.report.returned_to_town)
            scene.ctx.tap.assert_not_called()

    def test_unknown_region_or_unsupported_dungeon_is_rejected_before_input(self):
        for target in ("不存在的区域-龙萌幻境", "奇幻梦境-不存在的副本"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                ctx = SimpleNamespace(
                    project_root=ROOT, settings={"daily_hero_reset": {"dungeons": [target]}},
                    log=logging.getLogger("invalid-reset-target"), dry_run=False,
                    run_logs=RunLogRetention(Path(directory)), tap=Mock(), capture=Mock(),
                )
                vision = Mock()
                vision.validate_dungeon.side_effect = AutomationError("指定副本尚不支持")
                with patch("pet_alliance.daily_hero_reset.HeroResetVision", return_value=vision):
                    with self.assertRaises(AutomationError):
                        DailyHeroReset(ctx)
                ctx.tap.assert_not_called()
                ctx.capture.assert_not_called()

    def test_dungeon_work_keeps_the_real_region_key_and_distinct_work_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = TargetResetScene(
                directory, {}, regions=[], dungeons=[DREAM_TARGET, "奇幻梦境-第二梦境"],
                named={("奇幻梦境", "龙萌幻境"): 0, ("奇幻梦境", "第二梦境"): 1},
            )
            first, second = scene.selected_dungeons
            self.assertEqual(first["key"], "dream")
            self.assertEqual(second["key"], "dream")
            self.assertNotEqual(scene.region_work_key(first), scene.region_work_key(second))
            scene.view = "world"
            candidates = scene.region_candidates("world", {scene.region_work_key(first)})
            self.assertEqual([r["_reset_dungeon"] for r, _ in candidates], ["第二梦境"])

    def test_named_reset_cost_and_completion_include_dungeon_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.replay(
                directory, ["map", "completed", "confirmation", "ready", "battle", "complete", "map"],
            )
            region = deepcopy(next(r for r in hero.bank.profile["regions"] if r["key"] == "dream"))
            region["_reset_dungeon"] = "龙萌幻境"
            hero.reset_vision.named_entrance.return_value = target
            self.assertTrue(hero.fight_dungeon(region, target))
            self.assertEqual(hero.report.diamonds_spent, 50)
            self.assertEqual(hero.report.diamond_spending[0]["dungeon"], "龙萌幻境")
            self.assertEqual(hero.report.completed[0]["dungeon"], "龙萌幻境")
            events = [json.loads(line) for line in
                      (hero.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(next(e for e in events if e["event"] == "dungeon_reset")["dungeon"],
                             "龙萌幻境")

    def test_named_gray_reset_is_skipped_without_spending_and_keeps_dungeon_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.replay(directory, ["map", "disabled", "map"])
            region = deepcopy(next(r for r in hero.bank.profile["regions"] if r["key"] == "dream"))
            region["_reset_dungeon"] = "龙萌幻境"
            hero.reset_vision.named_entrance.return_value = target
            self.assertFalse(hero.fight_dungeon(region, target))
            self.assertEqual(hero.report.skipped[0]["dungeon"], "龙萌幻境")
            self.assertEqual(hero.report.skipped[0]["reason_code"], "already_reset")
            self.assertEqual(hero.report.diamonds_spent, 0)
            self.assertEqual(hero.report.resets, 0)

    def test_named_entrance_retry_uses_fresh_named_position(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.replay(directory, ["map", "disabled", "map"])
            shifted = Match("named entrance", target.x + 20, target.y + 5,
                            target.width, target.height, 1)
            hero.ctx.dropped["map"] = 1
            hero.reset_vision.named_entrance.side_effect = (
                lambda frame, _region, _name:
                shifted if any(state == "map" for state, _, _ in hero.ctx.inputs) else target
            )
            region = deepcopy(next(r for r in hero.bank.profile["regions"] if r["key"] == "dream"))
            region["_reset_dungeon"] = "龙萌幻境"
            self.assertFalse(hero.fight_dungeon(region, target))
            map_taps = [(x, y) for state, x, y in hero.ctx.inputs if state == "map"]
            self.assertEqual(map_taps, [target.center, shifted.center])
            self.assertGreaterEqual(hero.reset_vision.named_entrance.call_count, 2)
            hero.reset_vision.entrances.assert_not_called()

    def test_missing_named_entrance_during_retry_does_not_click_a_generic_plate(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, target = self.replay(directory, ["map"])
            hero.ctx.dropped["map"] = 1
            hero.reset_vision.named_entrance.side_effect = (
                lambda frame, _region, _name: None if hero.ctx.inputs else target
            )
            region = deepcopy(next(r for r in hero.bank.profile["regions"] if r["key"] == "dream"))
            region["_reset_dungeon"] = "龙萌幻境"
            with self.assertRaises(AutomationError):
                hero.fight_dungeon(region, target)
            self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs], [target.center])
            hero.reset_vision.entrances.assert_not_called()
            self.assertEqual(hero.report.resets, 0)


if __name__ == "__main__":
    unittest.main()
