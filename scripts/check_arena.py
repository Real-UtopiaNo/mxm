"""按优先级检查魔都擂主，占领空位或保存本人的守护结束时间。"""

from pet_alliance import ScriptContext


def run(ctx: ScriptContext) -> None:
    from pet_alliance.check_arena import CheckArena
    from pet_alliance.pet_loadouts import run_with_loadout

    run_with_loadout(CheckArena(ctx), "PVP")
