from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pet_alliance.daily_hero import DailyHero
from pet_alliance.vision import Match


ROOT = Path(__file__).resolve().parent.parent


class ScheduleScene(DailyHero):
    """A map simulation whose markers remain visible after every attempt."""

    def __init__(self, directory: str, regions: list[tuple[str, list[bool]]], *, max_dungeons=0):
        ctx = SimpleNamespace(
            project_root=ROOT,
            settings={
                "daily_hero": {
                    "max_dungeons": max_dungeons,
                    "target_players": ["Hello", "Hell", "coco"],
                },
            },
            log=logging.getLogger("arena-schedule"),
            dry_run=False,
        )
        super().__init__(ctx)
        self.run_dir.rmdir()
        self.run_dir = Path(directory) / "run"
        self.run_dir.mkdir()
        self.scene_regions = [{"key": key, "name": key} for key, _ in regions]
        self.outcomes = {key: outcomes for key, outcomes in regions}
        self.scene_markers = {
            key: [Match(f"{key}:{index}", 100 + 100 * index, 100, 20, 20, 1.0)
                  for index in range(len(outcomes))]
            for key, outcomes in regions
        }
        self.view = "world"
        self.selected_region = None
        self.adventure_return_region = self.scene_regions[0]["key"]
        self.attempts = []
        self.fight_regions = []
        self.history = []
        self.arena_calls = []

    def observe(self):
        return self.view

    def adventure_map(self):
        if self.view == "world":
            return False
        if self.view == "town":
            self.selected_region = self.adventure_return_region
            self.view = "map"
            self.history.append(("adventure", self.selected_region))
        return True

    def world(self):
        self.history.append(("world", self.selected_region))
        self.view = "world"

    def regions(self, frame):
        if frame != "world":
            raise AssertionError("Region selection requires the world map")
        return [(region, None) for region in self.scene_regions]

    def current_region(self, frame):
        if frame != "world":
            raise AssertionError("Current-region lookup requires the world map")
        return next(((region, None) for region in self.scene_regions
                     if region["key"] == self.selected_region), None)

    def enter_region(self, region):
        if self.view != "world":
            raise AssertionError("Region entry requires the world map")
        self.selected_region = region["key"]
        self.view = "map"
        self.history.append(("enter", self.selected_region))

    def markers(self, frame):
        if frame != "map":
            raise AssertionError("Dungeon lookup requires the restored region map")
        return self.scene_markers[self.selected_region]

    def fight_dungeon(self, region, target):
        actual_key = self.selected_region if region["key"] == "__current__" else region["key"]
        if self.view != "map" or self.selected_region != actual_key:
            raise AssertionError("Dungeon attempts must resume in the original region")
        identity = (actual_key, target.name)
        if identity in self.attempts:
            raise AssertionError(f"A previously attempted dungeon was repeated: {identity}")
        self.attempts.append(identity)
        self.fight_regions.append(region)
        index = self.scene_markers[actual_key].index(target)
        completed = self.outcomes[actual_key][index]
        entry = {"region": region["name"], "marker": target.center, "failures": 0}
        if completed:
            self.report.completed.append(entry)
            # Each dungeon contains several victories; these must not affect
            # the three-dungeon schedule.
            self.report.victories += 4
        else:
            self.report.skipped.append(entry)
        self.history.append(("dungeon", identity, completed))
        return completed

    def simulated_arena(self, region, *, no_tickets=False):
        if region["key"] == "__current__":
            self.resolve_region(region)
        self.arena_calls.append((region["key"], len(self.report.completed), len(self.attempts)))
        self.report.arena_visits += 1
        if no_tickets:
            self.report.arena_no_tickets += 1
        # The real wrapper must restore the caller's region after going to
        # town. Preserve all markers so only visited prevents repeated work.
        self.selected_region = None
        self.view = "town"
        self.world()
        self.enter_region(region)


class LocalFirstScene(ScheduleScene):
    def __init__(self, directory, regions, *, current_region, max_dungeons=0):
        super().__init__(directory, regions, max_dungeons=max_dungeons)
        self.view = "town"
        self.adventure_return_region = current_region


class ArenaScheduleTests(unittest.TestCase):
    def run_scene(self, scene, *, no_tickets=False):
        with patch.object(scene, "run_points_arena", create=True,
                          side_effect=lambda region: scene.simulated_arena(region, no_tickets=no_tickets)):
            return scene.run()

    def test_every_third_completed_dungeon_interrupts_then_resumes_same_region(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 7)])
            report = self.run_scene(scene)
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.arena_calls, [("first", 3, 3), ("first", 6, 6)])
            self.assertEqual(len(scene.attempts), 7)
            self.assertEqual([event[1] for event in scene.history if event[0] == "enter"],
                             ["first", "first", "first"])

    def test_skipped_dungeon_does_not_advance_the_arena_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True, True, False, True])])
            report = self.run_scene(scene)
            self.assertEqual(report.status, "completed")
            self.assertEqual(len(report.completed), 3)
            self.assertEqual(len(report.skipped), 1)
            self.assertEqual(scene.arena_calls, [("first", 3, 4)])

    def test_completion_count_crosses_regions_and_triggers_at_region_end(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [
                ("first", [True, True]), ("second", [True]), ("third", [True]),
            ])
            report = self.run_scene(scene)
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.arena_calls, [("second", 3, 3)])
            self.assertEqual([event[1] for event in scene.history if event[0] == "enter"],
                             ["first", "second", "second", "third"])
            self.assertEqual([region for region, _ in scene.attempts],
                             ["first", "first", "second", "third"])

    def test_third_completion_runs_arena_before_max_dungeons_return(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 5)], max_dungeons=3)
            report = self.run_scene(scene)
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(scene.attempts), 3)
            self.assertEqual(scene.arena_calls, [("first", 3, 3)])
            self.assertEqual(scene.view, "map")
            self.assertEqual(scene.selected_region, "first")

    def test_max_dungeons_before_three_completions_does_not_run_arena(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 5)], max_dungeons=2)
            report = self.run_scene(scene)
            self.assertEqual(report.status, "limited")
            self.assertEqual(len(scene.attempts), 2)
            self.assertEqual(scene.arena_calls, [])

    def test_no_tickets_still_checks_again_after_three_more_completions(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 7)])
            report = self.run_scene(scene, no_tickets=True)
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.arena_calls, [("first", 3, 3), ("first", 6, 6)])
            self.assertEqual(report.arena_no_tickets, 2)
            self.assertEqual(len(scene.attempts), 7)

    def test_return_from_arena_preserves_visited_skipped_dungeons_and_processed_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [
                ("first", [False, True, True, True, True]), ("second", [True]),
            ])
            report = self.run_scene(scene)
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.arena_calls, [("first", 3, 4)])
            self.assertEqual(len(report.skipped), 1)
            self.assertEqual(len(scene.attempts), 6)
            self.assertEqual([event[1] for event in scene.history if event[0] == "enter"],
                             ["first", "first", "second"])
            self.assertEqual(scene.attempts[-2:], [("first", "first:4"), ("second", "second:0")])

    def test_current_region_is_cleared_before_opening_world(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = LocalFirstScene(directory, [
                ("other", [True]), ("local", [True, True]),
            ], current_region="local")
            report = self.run_scene(scene)
            self.assertEqual(report.status, "completed")
            self.assertEqual([region for region, _ in scene.attempts], ["local", "local", "other"])
            first_world = next(index for index, event in enumerate(scene.history) if event[0] == "world")
            before_world = [event for event in scene.history[:first_world] if event[0] == "dungeon"]
            self.assertEqual(len(before_world), 2)
            self.assertEqual(scene.arena_calls, [("other", 3, 3)])
            self.assertEqual([entry["region"] for entry in report.completed], ["local", "local", "other"])

    def test_current_region_without_targets_opens_world_before_other_dungeons(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = LocalFirstScene(directory, [
                ("other", [True]), ("local", []),
            ], current_region="local")
            report = self.run_scene(scene)
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.attempts, [("other", "other:0")])
            first_world = next(index for index, event in enumerate(scene.history) if event[0] == "world")
            self.assertFalse(any(event[0] == "dungeon" for event in scene.history[:first_world]))
            self.assertNotIn(("enter", "local"), scene.history)

    def test_current_region_skipped_markers_are_not_retried_before_world(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = LocalFirstScene(directory, [
                ("other", []), ("local", [False, True]),
            ], current_region="local")
            report = self.run_scene(scene)
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.attempts, [("local", "local:0"), ("local", "local:1")])
            self.assertEqual(len(report.skipped), 1)
            self.assertEqual(report.skipped[0]["region"], "local")
            self.assertEqual(report.completed[0]["region"], "local")
            self.assertEqual(scene.arena_calls, [])
            self.assertNotIn(("enter", "local"), scene.history)

    def test_current_region_limit_returns_without_opening_world(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = LocalFirstScene(directory, [("local", [True, True])],
                                    current_region="local", max_dungeons=1)
            report = self.run_scene(scene)
            self.assertEqual(report.status, "limited")
            self.assertEqual(scene.attempts, [("local", "local:0")])
            self.assertFalse(any(event[0] == "world" for event in scene.history))

    def test_unknown_current_region_is_resolved_at_third_completion_then_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = LocalFirstScene(directory, [
                ("other", []), ("local", [True] * 5),
            ], current_region="local")
            arena_completed_counts = []

            def arena_body(_arena):
                arena_completed_counts.append(len(scene.report.completed))
                scene.history.append(("arena", scene.selected_region))
                scene.view = "town"
                scene.selected_region = None
                scene.adventure_return_region = "other"

            with patch("pet_alliance.points_arena.PointsArena.run", autospec=True,
                       side_effect=arena_body), patch.object(scene, "same_map", create=True, return_value=False):
                report = scene.run()

            self.assertEqual(report.status, "completed")
            self.assertEqual(arena_completed_counts, [3])
            self.assertEqual(report.arena_visits, 1)
            self.assertEqual(scene.attempts, [("local", f"local:{index}") for index in range(5)])
            self.assertEqual([entry["region"] for entry in report.completed], ["local"] * 5)
            self.assertTrue(all(region is scene.fight_regions[0] for region in scene.fight_regions))
            self.assertEqual(scene.fight_regions[0]["key"], "local")
            first_world = next(index for index, event in enumerate(scene.history) if event[0] == "world")
            self.assertEqual(sum(event[0] == "dungeon" for event in scene.history[:first_world]), 3)
            self.assertEqual(sum(event == ("enter", "local") for event in scene.history), 1)

    def test_arena_wrapper_counts_visit_and_continues_when_adventure_returns_to_same_map(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True])])
            region = scene.scene_regions[0]
            scene.view = "map"
            scene.selected_region = "first"

            def arena_body(_arena):
                self.assertEqual(scene.report.arena_visits, 1)
                scene.history.append(("arena", "first"))
                scene.view = "town"
                scene.selected_region = None

            with patch("pet_alliance.points_arena.PointsArena.run", autospec=True,
                       side_effect=arena_body) as arena_run, \
                    patch.object(scene, "same_map", create=True, return_value=True) as same_map:
                scene.run_points_arena(region)

            arena_run.assert_called_once()
            same_map.assert_called_once()
            self.assertEqual(scene.history, [("arena", "first"), ("adventure", "first")])
            self.assertEqual(scene.report.arena_visits, 1)
            self.assertEqual(scene.view, "map")
            self.assertEqual(scene.selected_region, "first")

    def test_arena_wrapper_restores_original_region_when_adventure_returns_elsewhere(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True]), ("second", [])])
            region = scene.scene_regions[0]
            scene.view = "map"
            scene.selected_region = "first"
            scene.adventure_return_region = "second"

            def arena_body(_arena):
                self.assertEqual(scene.report.arena_visits, 1)
                scene.history.append(("arena", "first"))
                scene.view = "town"
                scene.selected_region = None

            with patch("pet_alliance.points_arena.PointsArena.run", autospec=True,
                       side_effect=arena_body) as arena_run, \
                    patch.object(scene, "same_map", create=True, return_value=False) as same_map:
                scene.run_points_arena(region)

            arena_run.assert_called_once()
            same_map.assert_called_once()
            self.assertEqual(scene.history, [
                ("arena", "first"), ("adventure", "second"), ("world", "second"), ("enter", "first"),
            ])
            self.assertEqual(scene.report.arena_visits, 1)
            self.assertEqual(scene.view, "map")
            self.assertEqual(scene.selected_region, "first")


if __name__ == "__main__":
    unittest.main()
