from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import importlib.util
import logging
from pathlib import Path
import re
import sys
import time

from .adb import AndroidDevice, AutomationError, Screenshot


class ScriptContext:
    """Stable entry point for independently developed task scripts."""

    def __init__(
        self, device: AndroidDevice, screenshots_dir: Path, dry_run: bool = False,
        *, settings: dict | None = None, project_root: Path | None = None,
        config_path: Path | None = None, use_local_config: bool = True,
        control: Callable[[str], None] | None = None,
        checkpoint_callback: Callable[[str, dict], None] | None = None,
        run_logs=None,
    ):
        self.device = device
        self.screenshots_dir = screenshots_dir
        self.dry_run = dry_run
        self.log = logging.getLogger("pet_alliance")
        self.settings = settings or {}
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.config_path = config_path or self.project_root / "automation.toml"
        self.use_local_config = use_local_config
        self.control = control
        self.checkpoint_callback = checkpoint_callback
        self.run_logs = run_logs

    def _cooperate(self, operation: str) -> None:
        if self.run_logs is not None:
            self.run_logs.maintain()
        if self.control is not None:
            self.control(operation)

    def checkpoint(self, name: str, **payload) -> None:
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(name, payload)

    def capture(self) -> Screenshot:
        """Return original PNG bytes and dimensions for future image recognition."""
        self._cooperate("capture")
        return self.device.capture()

    def screenshot(self, name: str = "capture") -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise AutomationError("截图名称仅支持字母、数字、下划线和短横线。")
        frame = self.capture()
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        path = frame.save(self.screenshots_dir / f"{name}_{timestamp}.png")
        self.log.info("截图 %s×%s：%s", frame.width, frame.height, path)
        return path

    def tap(self, x: int, y: int) -> None:
        self._cooperate("tap")
        self.device._check_coordinates(x, y)
        self.log.info("%s点击 (%d, %d)", "[预演] " if self.dry_run else "", x, y)
        if not self.dry_run:
            self.device.tap(x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._cooperate("swipe")
        self.device._check_coordinates(x1, y1, x2, y2)
        if type(duration_ms) is not int or duration_ms < 1:
            raise AutomationError("滑动时长必须是正整数（毫秒）。")
        self.log.info(
            "%s滑动 (%d, %d) → (%d, %d)，%d ms",
            "[预演] " if self.dry_run else "", x1, y1, x2, y2, duration_ms,
        )
        if not self.dry_run:
            self.device.swipe(x1, y1, x2, y2, duration_ms)

    def key(self, name: str) -> None:
        self._cooperate("key")
        if name not in {"back", "home", "enter"}:
            raise AutomationError("按键仅支持 back、home、enter。")
        self.log.info("%s按键 %s", "[预演] " if self.dry_run else "", name)
        if not self.dry_run:
            self.device.key(name)

    def restart_app(self, package: str) -> None:
        self._cooperate("restart_app")
        self.device._check_package(package)
        self.log.info("%s重启游戏 %s", "[预演] " if self.dry_run else "", package)
        if not self.dry_run:
            self.device.restart_app(package)

    def wait(self, seconds: float) -> None:
        if seconds < 0:
            raise AutomationError("等待时间不能为负数。")
        if self.control is None and self.run_logs is None:
            time.sleep(seconds)
            return
        self._cooperate("wait")
        remaining = seconds
        while remaining > 0:
            delay = min(remaining, .25 if self.control is not None else 30)
            time.sleep(delay)
            remaining -= delay
            self._cooperate("wait")


def list_scripts(directory: Path) -> list[str]:
    return sorted(
        path.stem for path in directory.glob("*.py")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", path.stem)
    )


def run_script(name: str, directory: Path, context: ScriptContext) -> None:
    if name not in list_scripts(directory):
        raise AutomationError(f"找不到脚本 {name}，可用脚本：{', '.join(list_scripts(directory)) or '无'}")
    spec = importlib.util.spec_from_file_location(f"pet_alliance_task_{name}", directory / f"{name}.py")
    if spec is None or spec.loader is None:
        raise AutomationError(f"无法加载脚本 {name}。")
    module = importlib.util.module_from_spec(spec)
    # Decorators such as dataclass resolve their module during import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    entry = getattr(module, "run", None)
    if not callable(entry):
        raise AutomationError(f"脚本 {name} 需要定义 run(ctx) 函数。")
    entry(context)
