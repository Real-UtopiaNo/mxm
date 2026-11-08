from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .adb import AndroidDevice, AutomationError, find_adb
from .configuration import load_config
from .runtime import ScriptContext, list_scripts, run_script


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="雷电模拟器：截图、输入与可扩展任务脚本")
    root.add_argument("--config", type=Path, help="显式配置文件；默认加载 automation.toml 及本机覆盖文件")
    root.add_argument("--adb", help="覆盖配置中的 adb.exe 路径")
    root.add_argument("--serial", help="覆盖配置中的设备编号")
    root.add_argument("--dry-run", action="store_true", help="记录输入动作，截图和等待照常执行")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="列出设备和连接状态")
    commands.add_parser("doctor", help="检查视觉依赖、模板、设备和截图，不发送点击")
    capture = commands.add_parser("capture", help="抓取游戏原始画面")
    capture.add_argument("--output", type=Path, help="PNG 保存位置；默认按时间命名")
    tap = commands.add_parser("tap", help="点击截图坐标")
    tap.add_argument("x", type=int)
    tap.add_argument("y", type=int)
    swipe = commands.add_parser("swipe", help="滑动截图坐标")
    for coordinate in ("x1", "y1", "x2", "y2"):
        swipe.add_argument(coordinate, type=int)
    swipe.add_argument("--duration", type=int, default=300, help="滑动时长（毫秒）")
    key = commands.add_parser("key", help="模拟系统按键")
    key.add_argument("name", choices=("back", "home", "enter"))
    commands.add_parser("scripts", help="列出可用任务脚本")
    scheduler = commands.add_parser("scheduler", help="打开全天托管面板；点击启动后才操作游戏")
    # Accept both the common global-option order and options after scheduler.
    # Suppressed defaults preserve any value already parsed on the root parser.
    scheduler.add_argument("--config", type=Path, default=argparse.SUPPRESS, help="显式配置文件")
    scheduler.add_argument("--adb", default=argparse.SUPPRESS, help="ADB 可执行文件路径")
    scheduler.add_argument("--serial", default=argparse.SUPPRESS, help="模拟器设备编号")
    scheduler.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                           help="调度器不支持预演；此选项会报错退出")
    task = commands.add_parser("run", help="执行 scripts 目录中的独立脚本")
    task.add_argument("name", help="脚本文件名，不含 .py")
    return root


def main(argv: list[str] | None = None) -> int:
    # Windows redirected output may otherwise use GBK and garble Chinese logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "scheduler":
            if args.dry_run:
                raise AutomationError("调度器不支持 --dry-run；可打开面板查看状态，点击启动后才会操作游戏。")
            from .scheduler_ui import main as scheduler_main

            scheduler_args = []
            for option, value in (("--config", args.config), ("--adb", args.adb), ("--serial", args.serial)):
                if value is not None:
                    scheduler_args.extend((option, str(value)))
            return scheduler_main(scheduler_args)
        config_path = (args.config or PROJECT_ROOT / "automation.toml").resolve()
        config = load_config(config_path, include_local=args.config is None)
        base = config_path.parent
        settings = config.get("automation", {})
        scripts_dir = base / settings.get("scripts_dir", "scripts")
        screenshots_dir = base / settings.get("screenshots_dir", "artifacts/screenshots")
        if args.command == "scripts":
            print("\n".join(list_scripts(scripts_dir)) or "尚无任务脚本。")
            return 0
        settings = config.get("device", {})
        device = AndroidDevice(
            find_adb(args.adb or settings.get("adb_path")),
            args.serial or settings.get("serial") or None,
            float(settings.get("timeout", 30)),
        )
        if args.command == "devices":
            devices = device.devices()
            print("\n".join(f"{serial}\t{state}" for serial, state in devices) or "未发现设备。")
            return 0
        device.connect()
        from .run_logs import RunLogRetention

        state_path = base / config.get("scheduler", {}).get("state_file", "artifacts/scheduler/state.json")
        run_logs = RunLogRetention(base, protected_paths=(state_path, config_path)) if args.command == "run" else None
        if run_logs is not None:
            run_logs.maintain()
        ctx = ScriptContext(device, screenshots_dir, args.dry_run, settings=config, project_root=base,
                            config_path=config_path, use_local_config=args.config is None,
                            run_logs=run_logs)
        if args.command == "doctor":
            from .arena_vision import PointsArenaVision
            from .dragon_vision import DragonVision
            from .hero_reset_vision import HeroResetVision
            from .check_arena_vision import CheckArenaVision
            from .vision import TemplateBank, VisualFrame

            bank = TemplateBank(base / "assets" / "daily_hero")
            PointsArenaVision(base / "assets" / "points_arena")
            DragonVision(base / "assets" / "slay_dragon")
            HeroResetVision(base / "assets" / "daily_hero_reset")
            CheckArenaVision(base / "assets" / "check_arena")
            scripts = list_scripts(scripts_dir)
            if "Daily_Hero" not in scripts:
                raise AutomationError("缺少 scripts/Daily_Hero.py，请完整 clone 仓库。")
            screenshot = ctx.capture()
            VisualFrame.from_screenshot(screenshot)
            path = screenshot.save(screenshots_dir / "doctor.png")
            print(f"检查通过：{len(bank.images)} 个视觉模板，{screenshot.width}×{screenshot.height} 横屏。")
            print("积分赛视觉素材：检查通过。")
            print("屠龙战视觉素材：检查通过。")
            print("英雄副本重置视觉素材：检查通过。")
            print("关卡擂主检查视觉素材：检查通过。")
            print(f"ADB：{device.adb_path}\n设备：{device.serial}\n截图：{path}")
        elif args.command == "capture":
            if args.output:
                if args.output.suffix.lower() != ".png":
                    raise AutomationError("截图输出文件必须使用 .png 扩展名。")
                frame = ctx.capture()
                path = frame.save(args.output)
                print(f"截图 {frame.width}×{frame.height}：{path}")
            else:
                ctx.screenshot()
        elif args.command == "tap":
            ctx.tap(args.x, args.y)
        elif args.command == "swipe":
            ctx.swipe(args.x1, args.y1, args.x2, args.y2, args.duration)
        elif args.command == "key":
            ctx.key(args.name)
        elif args.command == "run":
            run_script(args.name, scripts_dir, ctx)
        return 0
    except KeyboardInterrupt:
        logging.info("已停止。")
        return 130
    except (AutomationError, OSError, ValueError) as exc:
        logging.error("错误：%s", exc)
        return 1
