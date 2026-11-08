"""重置并重打选定区域的英雄级副本。"""

from pet_alliance import ScriptContext


def run(ctx: ScriptContext) -> None:
    from pet_alliance.daily_hero_reset import DailyHeroReset
    from pet_alliance.pet_loadouts import run_with_loadout

    run_with_loadout(DailyHeroReset(ctx), "刷图")
