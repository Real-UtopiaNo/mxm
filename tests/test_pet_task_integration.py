from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from pet_alliance.adb import AutomationError
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.daily_hero import HeroReport
from pet_alliance.runtime import run_script
from pet_alliance.scheduler_jobs import run_task


ROOT = Path(__file__).resolve().parent.parent
LOADOUTS = (
    ("daily_hero", "Daily_Hero", "pet_alliance.daily_hero.DailyHero", "刷图"),
    ("daily_hero_reset", "daily_hero_reset", "pet_alliance.daily_hero_reset.DailyHeroReset", "刷图"),
    ("check_arena", "check_arena", "pet_alliance.check_arena.CheckArena", "PVP"),
    ("slay_dragon", "slay_dragon", "pet_alliance.slay_dragon.SlayDragon", "屠龙"),
)


class PetTaskEntryTests(unittest.TestCase):
    def test_scheduler_dispatches_each_task_with_its_pet_loadout(self):
        for task, _script, _constructor, group in LOADOUTS:
            with self.subTest(task=task):
                ctx = object()
                job = SimpleNamespace(
                    report=HeroReport(status="completed"),
                    run_dir=Path("isolated_run"),
                    run=Mock(),
                )
                with patch("pet_alliance.scheduler_jobs.create_job", return_value=job) as create, \
                        patch("pet_alliance.scheduler_jobs.run_with_loadout") as run:
                    result = run_task(task, ctx)
                create.assert_called_once_with(task, ctx)
                run.assert_called_once_with(job, group)
                job.run.assert_not_called()
                self.assertEqual(result, {**asdict(job.report), "run_dir": "isolated_run"})

    def test_loadout_failure_retains_run_report_and_diagnostic_directory(self):
        for error in (AutomationError("宠物组确认失败"), TaskInterrupted("paused", "用户暂停")):
            with self.subTest(error=type(error).__name__):
                job = SimpleNamespace(
                    report=HeroReport(status="paused", victories=3),
                    run_dir=Path("loadout_failure_run"),
                    run=Mock(),
                )
                with patch("pet_alliance.scheduler_jobs.create_job", return_value=job), \
                        patch("pet_alliance.scheduler_jobs.run_with_loadout", side_effect=error), \
                        self.assertRaises(type(error)) as raised:
                    run_task("daily_hero", object())
                self.assertIs(raised.exception, error)
                self.assertEqual(error.report, asdict(job.report))
                self.assertEqual(error.run_dir, "loadout_failure_run")
                job.run.assert_not_called()

    def test_each_standalone_script_uses_the_same_pet_loadout_as_scheduler(self):
        for _task, script, constructor, group in LOADOUTS:
            with self.subTest(script=script):
                ctx = object()
                with patch(constructor) as create, \
                        patch("pet_alliance.pet_loadouts.run_with_loadout") as run:
                    run_script(script, ROOT / "scripts", ctx)
                create.assert_called_once_with(ctx)
                run.assert_called_once_with(create.return_value, group)
                create.return_value.run.assert_not_called()

    def test_capture_once_keeps_its_screenshot_only_behavior(self):
        ctx = Mock()
        with patch("pet_alliance.pet_loadouts.run_with_loadout") as run:
            run_script("capture_once", ROOT / "scripts", ctx)
        run.assert_not_called()
        ctx.screenshot.assert_called_once_with("capture_once")
        self.assertEqual(ctx.mock_calls, [call.screenshot("capture_once")])


if __name__ == "__main__":
    unittest.main()
