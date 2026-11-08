from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pet_alliance.adb import AutomationError
from pet_alliance.arena_hero import ArenaHeroPrerequisite
from pet_alliance.check_arena import CheckArena, CheckArenaReport
from pet_alliance.daily_hero import DailyHero, HeroOptions
from pet_alliance.daily_hero_reset import DailyHeroReset
from tests.test_arena_hero import HeroHarness, NOW
from tests.test_arena_schedule import ScheduleScene
from tests.test_daily_hero_reset import ResetScene


ROOT = Path(__file__).resolve().parent.parent


class HeroPointsToggleTests(unittest.TestCase):
    def set_enabled(self, scene, enabled):
        scene.options = replace(scene.options, points_arena_enabled=enabled)

    def boundaries(self, scene):
        saved = []
        scene.ctx.checkpoint = lambda name, **payload: saved.append((name, deepcopy(payload)))
        return saved

    def assert_disabled_progress(self, saved):
        self.assertTrue(saved)
        for name, payload in saved:
            self.assertEqual(payload["completed_since_arena"], 0, name)
            self.assertFalse(payload["report"]["points_arena_pending"], name)
        self.assertNotIn("hero_points_pending", [name for name, _ in saved])

    def test_default_enabled_and_invalid_non_boolean_values_rejected(self):
        self.assertTrue(HeroOptions().points_arena_enabled)
        self.assertFalse(HeroOptions(points_arena_enabled=False).points_arena_enabled)
        for value in (0, 1, "false", "true", None, [], {}):
            with self.subTest(value=value), self.assertRaises(AutomationError):
                HeroOptions(points_arena_enabled=value)

    def test_enabled_runs_two_visits_for_six_completed_dungeons(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 6)])
            self.set_enabled(scene, True)
            scene.run_points_arena = scene.simulated_arena
            report = scene.run()
            self.assertEqual(len(report.completed), 6)
            self.assertEqual(scene.arena_calls, [("first", 3, 3), ("first", 6, 6)])

    def test_disabled_discards_pending_and_never_accumulates_six_clears(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ScheduleScene(directory, [("first", [True] * 6)])
            self.set_enabled(scene, False)
            scene.ctx.resume_data = {"completed_since_arena": 2, "points_arena_pending": True}
            saved = self.boundaries(scene)
            with patch.object(scene, "run_points_arena") as arena:
                report = scene.run()
            arena.assert_not_called()
            self.assertEqual(len(report.completed), 6)
            self.assertFalse(report.points_arena_pending)
            self.assert_disabled_progress(saved)

    def test_reenabling_starts_new_count_after_disabled_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ScheduleScene(directory, [("first", [True] * 2)])
            self.set_enabled(first, False)
            first.ctx.resume_data = {"completed_since_arena": 2, "points_arena_pending": True}
            saved = self.boundaries(first)
            with patch.object(first, "run_points_arena") as arena:
                first.run()
            arena.assert_not_called()
            last = saved[-1][1]
            resumed_directory = Path(directory) / "resumed"
            resumed_directory.mkdir()
            resumed = ScheduleScene(str(resumed_directory), [("first", [True] * 3)])
            self.set_enabled(resumed, True)
            resumed.ctx.resume_data = {"completed_since_arena": last["completed_since_arena"],
                                       "points_arena_pending": last["report"]["points_arena_pending"]}
            resumed.run_points_arena = resumed.simulated_arena
            resumed.run()
            self.assertEqual(resumed.arena_calls, [("first", 3, 3)])

    def test_reset_inherits_daily_hero_switch(self):
        ctx = SimpleNamespace(project_root=ROOT, settings={"daily_hero": {"points_arena_enabled": False}},
                              log=logging.getLogger("reset-points-toggle"))
        scene = DailyHeroReset(ctx)
        self.addCleanup(scene.run_dir.rmdir)
        self.assertFalse(scene.options.points_arena_enabled)

    def test_reset_and_arena_section_overrides_cannot_change_global_points_switch(self):
        for job_class, section in ((DailyHeroReset, "daily_hero_reset"), (CheckArena, "check_arena")):
            for enabled in (True, False):
                with self.subTest(section=section, enabled=enabled):
                    ctx = SimpleNamespace(
                        project_root=ROOT, log=logging.getLogger("global-points-toggle"),
                        settings={"daily_hero": {"points_arena_enabled": enabled},
                                  section: {"points_arena_enabled": not enabled, "max_dungeons": 7}},
                    )
                    job = job_class(ctx)
                    self.addCleanup(job.run_dir.rmdir)
                    self.assertEqual(job.options.points_arena_enabled, enabled)
                    self.assertEqual(job.options.max_dungeons, 7)

    def test_prerequisite_constructor_discards_pending_when_disabled(self):
        ctx = SimpleNamespace(project_root=ROOT, settings={"daily_hero": {"points_arena_enabled": False}},
                              log=logging.getLogger("prerequisite-points-toggle"),
                              hero_resume_date=NOW.date().isoformat(),
                              hero_resume_data={"completed_since_arena": 2, "points_arena_pending": True})
        helper = ArenaHeroPrerequisite(ctx, now=lambda: NOW)
        self.addCleanup(helper.run_dir.rmdir)
        self.assertEqual(helper.completed_since_arena, 0)
        self.assertFalse(helper.report.points_arena_pending)
        helper.sync_progress()
        self.assertEqual(ctx.hero_resume_data, {"completed_since_arena": 0, "points_arena_pending": False})

    def test_disabled_reset_ignores_pending_and_counts_no_points_visits(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = ResetScene(directory, {"west": [True] * 3, "ring": [True] * 3, "cursed": []})
            self.set_enabled(scene, False)
            scene.ctx.resume_data = {"completed_since_arena": 2, "points_arena_pending": True}
            saved = self.boundaries(scene)
            report = scene.run()
            self.assertEqual(len(report.completed), 6)
            self.assertEqual(scene.arena_calls, [])
            self.assertEqual(scene.view, "town")
            self.assert_disabled_progress(saved)

    def test_prerequisite_disabled_drops_old_pending_and_still_verifies_current_day_hero(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "cleared"], counter=2, pending=True)
            self.set_enabled(helper, False)
            with patch.object(DailyHero, "fight_dungeon", side_effect=helper.battle) as battle, \
                    patch.object(helper, "run_points_arena") as arena:
                self.assertTrue(helper.ensure_cleared("COCO号", True))
            battle.assert_called_once()
            arena.assert_not_called()
            self.assertEqual(helper.verified_date, NOW.date().isoformat())
            self.assertEqual(len(helper.report.completed), 1)
            self.assertEqual(helper.ctx.hero_resume_data["completed_since_arena"], 0)
            self.assertFalse(helper.ctx.hero_resume_data["points_arena_pending"])
            self.assertEqual(helper.progress_by_day[NOW.date().isoformat()]["completed_since_arena"], 0)
            self.assertFalse(helper.progress_by_day[NOW.date().isoformat()]["points_arena_pending"])

    def test_check_arena_disabled_does_not_restore_independent_pending_visit(self):
        checker = object.__new__(CheckArena)
        checker.ctx = SimpleNamespace(hero_resume_date=NOW.date().isoformat(),
                                      hero_resume_data={"completed_since_arena": 2, "points_arena_pending": True})
        checker.options = HeroOptions(points_arena_enabled=False)
        checker.report = CheckArenaReport()
        checker.now = lambda: NOW
        with patch("pet_alliance.check_arena.ArenaHeroPrerequisite") as helper:
            checker.resume_hero_points()
        helper.assert_not_called()
        self.assertEqual(checker.ctx.hero_resume_data["completed_since_arena"], 0)
        self.assertFalse(checker.ctx.hero_resume_data["points_arena_pending"])
        self.assertEqual(checker.report.hero_progress["completed_since_arena"], 0)
        self.assertFalse(checker.report.hero_progress["points_arena_pending"])


if __name__ == "__main__":
    unittest.main()
