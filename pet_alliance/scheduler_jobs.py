"""Construct existing automation jobs with scheduler-specific persistence."""

from __future__ import annotations

from dataclasses import asdict

from .adb import AutomationError
from .check_arena import CheckArena
from .daily_hero import DailyHero
from .daily_hero_reset import DailyHeroReset
from .pet_loadouts import run_with_loadout
from .runtime import ScriptContext
from .slay_dragon import SlayDragon


class ManagedCheckArena(CheckArena):
    """Save the next guardian deadline after a newly claimed empty arena too."""

    allow_hero_battle = True

    def prepare_map(self) -> None:
        super().prepare_map()
        self.resume_hero_points()

    def claim_empty(self, name: str) -> str:
        owner = super().claim_empty(name)
        if owner == "self" and self.report.claimed:
            self.record_new_claim_expiration(name)
        return owner


def create_job(task: str, ctx: ScriptContext):
    jobs = {
        "check_arena": ManagedCheckArena,
        "slay_dragon": SlayDragon,
        "daily_hero": DailyHero,
        "daily_hero_reset": DailyHeroReset,
    }
    try:
        job = jobs[task]
    except KeyError:
        raise AutomationError(f"未知调度任务：{task}。") from None
    return job(ctx)


def run_task(task: str, ctx: ScriptContext) -> dict:
    job = create_job(task, ctx)
    try:
        group = {"check_arena": "PVP", "slay_dragon": "屠龙",
                 "daily_hero": "刷图", "daily_hero_reset": "刷图"}[task]
        run_with_loadout(job, group)
    except Exception as exc:
        exc.report = asdict(job.report)
        exc.run_dir = str(job.run_dir)
        raise
    return {**asdict(job.report), "run_dir": str(job.run_dir)}
