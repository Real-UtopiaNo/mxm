"""视觉识别并完成各区域尚未完成的英雄级副本。"""

from pet_alliance import ScriptContext


def run(ctx: ScriptContext) -> None:
    from pet_alliance.daily_hero import DailyHero

    DailyHero(ctx).run()
