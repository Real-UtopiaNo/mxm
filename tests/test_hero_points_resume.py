from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from pet_alliance.cooperation import TaskInterrupted
from tests.test_arena_schedule import ScheduleScene
from tests.test_daily_hero_reset import ResetScene


class HeroPointsResumeTests(unittest.TestCase):
    def summary(self, scene):
        return json.loads((scene.run_dir / "summary.json").read_text(encoding="utf-8"))

    def recording_arena(self, scene):
        def run(region):
            scene.history.append(("points_started", region["key"]))
            scene.simulated_arena(region)
            scene.history.append(("points_finished", region["key"]))

        return run

    def test_pause_after_third_clear_saves_pending_before_any_points_arena_input(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 4)])
            saved = []
            scene.run_points_arena = self.recording_arena(scene)

            def checkpoint(name, **payload):
                if name == "hero_points_pending":
                    saved.append(deepcopy(payload))
                    raise TaskInterrupted("paused", "第三本结束时暂停")

            scene.ctx.checkpoint = checkpoint
            with self.assertRaises(TaskInterrupted):
                scene.run()
            self.assertEqual(len(scene.attempts), 3)
            self.assertEqual(scene.arena_calls, [])
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0]["report"]["points_arena_pending"])
            self.assertEqual(saved[0]["completed_since_arena"], 2)
            report = self.summary(scene)
            self.assertEqual(report["status"], "paused")
            self.assertTrue(report["points_arena_pending"])
            self.assertEqual(len(report["completed"]), 3)

    def test_resumed_pending_arena_finishes_before_another_dungeon_and_resets_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("demon", [True])])
            scene.ctx.resume_data = {"points_arena_pending": True, "completed_since_arena": 2}
            scene.run_points_arena = self.recording_arena(scene)
            boundaries = []

            def checkpoint(name, **payload):
                boundaries.append((name, deepcopy(payload), len(scene.attempts)))

            scene.ctx.checkpoint = checkpoint
            report = scene.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(scene.arena_calls, [("demon", 0, 0)])
            arena_finished = next(index for index, event in enumerate(scene.history) if event[0] == "points_finished")
            first_dungeon = next(index for index, event in enumerate(scene.history) if event[0] == "dungeon")
            self.assertLess(arena_finished, first_dungeon)
            start = next(payload for name, payload, _ in boundaries if name == "hero_start")
            self.assertTrue(start["report"]["points_arena_pending"])
            self.assertEqual(start["completed_since_arena"], 2)
            resumed = [(payload, attempts) for name, payload, attempts in boundaries if name == "hero_points_resumed"]
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0][1], 0)
            self.assertFalse(resumed[0][0]["report"]["points_arena_pending"])
            self.assertEqual(resumed[0][0]["completed_since_arena"], 0)
            dungeon = next(payload for name, payload, _ in boundaries if name == "hero_dungeon")
            self.assertEqual(dungeon["completed_since_arena"], 1)
            self.assertFalse(report.points_arena_pending)
            self.assertFalse(self.summary(scene)["points_arena_pending"])

    def test_pausing_again_during_resumed_arena_retains_pending_for_next_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("demon", [True])])
            scene.ctx.resume_data = {"points_arena_pending": True, "completed_since_arena": 2}
            boundaries = []
            scene.ctx.checkpoint = lambda name, **payload: boundaries.append((name, deepcopy(payload)))

            def interrupted_arena(region):
                self.assertTrue(scene.report.points_arena_pending)
                scene.arena_calls.append((region["key"], len(scene.report.completed), len(scene.attempts)))
                raise TaskInterrupted("paused", "积分赛中暂停")

            scene.run_points_arena = interrupted_arena
            with self.assertRaises(TaskInterrupted):
                scene.run()
            self.assertEqual(scene.arena_calls, [("demon", 0, 0)])
            self.assertEqual(scene.attempts, [])
            self.assertNotIn("hero_points_resumed", [name for name, _ in boundaries])
            self.assertTrue(scene.report.points_arena_pending)
            self.assertTrue(self.summary(scene)["points_arena_pending"])
            self.assertEqual(self.summary(scene)["status"], "paused")

    def test_normal_third_clear_clears_pending_only_after_points_arena_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 3)])
            boundaries = []

            def checkpoint(name, **payload):
                boundaries.append((name, deepcopy(payload)))

            def arena(region):
                self.assertTrue(scene.report.points_arena_pending)
                self.assertEqual(boundaries[-1][0], "hero_points_pending")
                self.assertEqual(boundaries[-1][1]["completed_since_arena"], 2)
                scene.simulated_arena(region)

            scene.ctx.checkpoint = checkpoint
            scene.run_points_arena = arena
            report = scene.run()
            pending = [payload for name, payload in boundaries if name == "hero_points_pending"]
            self.assertEqual(len(pending), 1)
            self.assertTrue(pending[0]["report"]["points_arena_pending"])
            last_dungeon = [payload for name, payload in boundaries if name == "hero_dungeon"][-1]
            self.assertEqual(last_dungeon["completed_since_arena"], 0)
            self.assertFalse(last_dungeon["report"]["points_arena_pending"])
            self.assertFalse(report.points_arena_pending)

    def test_reset_resumes_points_in_demon_then_reapplies_configured_region_whitelist(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [True], "ring": [True], "cursed": [False],
                                           "demon": [True], "flame": [True]})
            scene.ctx.resume_data = {"points_arena_pending": True, "completed_since_arena": 2}
            boundaries = []
            scene.ctx.checkpoint = lambda name, **payload: boundaries.append((name, deepcopy(payload)))
            report = scene.run()
            self.assertEqual(report.status, "completed")
            self.assertTrue(report.returned_to_town)
            self.assertEqual(scene.arena_calls, [("demon", 0, 0)])
            self.assertEqual([key for key, _ in scene.attempts], ["west", "ring", "cursed"])
            self.assertFalse(any(key in {"demon", "flame"} for key, _ in scene.attempts))
            resumed_index = next(index for index, (name, _) in enumerate(boundaries) if name == "hero_points_resumed")
            first_region = next(payload for name, payload in boundaries[resumed_index + 1:] if name == "hero_region")
            self.assertEqual(first_region["region"]["key"], "west")
            self.assertFalse(report.points_arena_pending)


if __name__ == "__main__":
    unittest.main()
