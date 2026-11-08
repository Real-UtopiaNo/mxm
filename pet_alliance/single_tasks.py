"""Discover runnable scripts without importing them or connecting a device."""

from pathlib import Path

from .runtime import list_scripts


BUSINESS_TASKS = {
    "daily_hero": ("英雄级副本", "完成英雄级副本，按原设置穿插积分赛。"),
    "daily_hero_reset": ("英雄级副本重置", "重置选定区域的英雄级副本并挑战，记录钻石消耗。"),
    "check_arena": ("检查擂台", "按配置顺序检查、占擂，并保存守护结束时间。"),
    "slay_dragon": ("屠龙托管", "按屠龙时段持续托管；停止后结束本任务。"),
}


def task_for_script(name: str) -> str | None:
    key = name.casefold()
    return key if key in BUSINESS_TASKS else None


def script_catalog(directory: Path) -> list[dict]:
    result = []
    for name in list_scripts(directory):
        task = task_for_script(name)
        if task:
            title, description = BUSINESS_TASKS[task]
        elif name == "capture_once":
            title, description = "单次截图", "仅保存模拟器截图，不点击或导航。"
        else:
            title, description = name, "运行此目录内脚本的 run(ctx) 入口。"
        result.append({"name": name, "title": title, "description": description})
    return result
