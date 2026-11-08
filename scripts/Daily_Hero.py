"""视觉识别并完成各区域尚未完成的英雄级副本。"""

from pet_alliance import ScriptContext


def run(ctx: ScriptContext) -> None:
    from pet_alliance.daily_hero import DailyHero
    from pet_alliance.pet_loadouts import run_with_loadout

    run_with_loadout(DailyHero(ctx), "刷图")
