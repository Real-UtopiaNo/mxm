"""每五秒检查屠龙托管，异常时重启游戏并恢复。"""

from pet_alliance import ScriptContext


def run(ctx: ScriptContext) -> None:
    from pet_alliance.slay_dragon import SlayDragon
    from pet_alliance.pet_loadouts import run_with_loadout

    run_with_loadout(SlayDragon(ctx), "屠龙")
