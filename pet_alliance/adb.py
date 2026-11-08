from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile


class AutomationError(RuntimeError):
    """An actionable device, configuration, or script error."""


def _windows_adb_candidates() -> list[str]:
    """Find custom LDPlayer installs without recursively scanning disks."""
    if os.name != "nt":
        return []
    candidates = []
    # A running emulator (or its ADB server) also covers nonstandard drive paths.
    powershell = shutil.which("powershell.exe")
    if powershell:
        try:
            result = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command",
                 "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
                 "@(Get-CimInstance Win32_Process -Filter "
                 "\"Name='dnplayer.exe' OR Name='ldplayer.exe' OR Name='ldconsole.exe' "
                 "OR Name='dnconsole.exe' OR Name='adb.exe'\" | "
                 "Select-Object -ExpandProperty ExecutablePath) | ConvertTo-Json -Compress"],
                capture_output=True, timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                paths = json.loads(result.stdout.decode("utf-8-sig"))
                if isinstance(paths, str):
                    paths = [paths]
                for path in paths or []:
                    if path:
                        candidates.append(str(Path(path).parent / "adb.exe"))
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    # Installed but not running: use LDPlayer's uninstall registration.
    import winreg

    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(
                    hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                    0, winreg.KEY_READ | view,
                ) as root:
                    for index in range(winreg.QueryInfoKey(root)[0]):
                        try:
                            with winreg.OpenKey(root, winreg.EnumKey(root, index)) as app:
                                name = winreg.QueryValueEx(app, "DisplayName")[0]
                                if "ldplayer" not in name.lower() and "雷电" not in name:
                                    continue
                                location = winreg.QueryValueEx(app, "InstallLocation")[0]
                                candidates.append(str(Path(location) / "adb.exe"))
                        except OSError:
                            continue
            except OSError:
                continue
    return candidates


def find_adb(configured: str | None = None) -> str:
    if configured:
        resolved = shutil.which(os.path.expandvars(os.path.expanduser(configured)))
        if resolved:
            return resolved
        raise AutomationError(f"找不到 ADB：{configured}，请修改 automation.local.toml 或使用 --adb。")
    candidates = [os.environ.get("ADB_PATH"), shutil.which("adb")]
    if os.environ.get("LDPLAYER_HOME"):
        candidates.append(str(Path(os.environ["LDPLAYER_HOME"]) / "adb.exe"))
    # Prefer an explicit environment/PATH choice before running discovery.
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    candidates = _windows_adb_candidates()
    for drive in "CDEFG":
        for parent in ("LDPlayer", "leidian", "Program Files/LDPlayer", "Program Files/leidian"):
            root = Path(f"{drive}:/") / parent
            candidates.extend(str(path) for path in root.glob("LDPlayer*/adb.exe"))
            candidates.append(str(root / "adb.exe"))
    for candidate in candidates:
        if Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise AutomationError(
        "未找到 ADB。请启动雷电，或复制 automation.local.example.toml 为 "
        "automation.local.toml，填写雷电安装目录下的 adb.exe 路径。"
    )


def png_dimensions(data: bytes) -> tuple[int, int]:
    if (
        len(data) < 33
        or data[:8] != b"\x89PNG\r\n\x1a\n"
        or data[8:12] != b"\x00\x00\x00\r"
        or data[12:16] != b"IHDR"
        or not data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
    ):
        raise AutomationError("截图数据不是完整 PNG，未保存；请检查模拟器连接。")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise AutomationError("截图分辨率无效。")
    return width, height


@dataclass(frozen=True)
class Screenshot:
    data: bytes
    width: int
    height: int

    def save(self, path: Path) -> Path:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(self.data)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path


class AndroidDevice:
    def __init__(self, adb_path: str, serial: str | None = None, timeout: float = 30):
        if timeout <= 0:
            raise AutomationError("device.timeout 必须大于 0。")
        self.adb_path = adb_path
        self.serial = serial
        self.timeout = timeout
        # A scheduler can stop between the individual commands of a restart.
        self.before_command: Callable[[], None] | None = None

    def _adb(
        self, *arguments: str, targeted: bool = True, check_shell_result: bool = False,
    ) -> bytes:
        if self.before_command is not None:
            self.before_command()
        command = [self.adb_path]
        if targeted:
            if self.serial is None:
                raise AutomationError("尚未选择设备，请先调用 connect()。")
            command += ["-s", self.serial]
        command.extend(arguments)
        try:
            result = subprocess.run(
                command, capture_output=True, timeout=self.timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise AutomationError(f"ADB 操作超时（{self.timeout:g} 秒），请检查模拟器。") from exc
        except OSError as exc:
            raise AutomationError(f"无法启动 ADB：{exc}") from exc
        if result.returncode:
            detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
            raise AutomationError(f"ADB 操作失败：{detail}")
        if check_shell_result:
            # Android's am/cmd can report an error while adb itself exits with 0.
            detail = (result.stdout + b"\n" + result.stderr).decode(
                "utf-8", errors="replace",
            ).strip()
            if re.search(
                r"(?im)^\s*(?:error\b|exception\b|security exception\b|failure\b|"
                r"(?:java|android)\.[\w.]*Exception\b)", detail,
            ):
                raise AutomationError(f"Android 命令失败：{detail}")
        return result.stdout

    def devices(self) -> list[tuple[str, str]]:
        output = self._adb("devices", targeted=False).decode("utf-8", errors="replace")
        devices = []
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 2 and not line.startswith(("List of devices", "*")):
                devices.append((fields[0], fields[1]))
        return devices

    def connect(self) -> None:
        devices = self.devices()
        available = [serial for serial, state in devices if state == "device"]
        if self.serial is not None:
            if self.serial not in available:
                states = dict(devices)
                raise AutomationError(
                    f"设备 {self.serial} 不可用（{states.get(self.serial, '未连接')}）。"
                    "请启动雷电并开启本地 ADB 调试，或通过 --serial 选择设备。"
                )
        elif len(available) == 1:
            self.serial = available[0]
        elif not available:
            raise AutomationError("没有可用设备，请启动雷电并开启本地 ADB 调试。")
        else:
            raise AutomationError(f"有多个设备，请用 --serial 指定：{', '.join(available)}")

    def capture(self) -> Screenshot:
        data = self._adb("exec-out", "screencap", "-p")
        width, height = png_dimensions(data)
        return Screenshot(data, width, height)

    @staticmethod
    def _check_package(package: str) -> None:
        if not isinstance(package, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+", package,
        ):
            raise AutomationError("游戏包名无效，必须是以点分隔的 Android 包名。")

    def resolve_launcher(self, package: str) -> str:
        """Resolve the selected app's launcher before making any app changes."""
        self._check_package(package)
        output = self._adb(
            "shell", "cmd", "package", "resolve-activity", "--brief",
            "-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER",
            package, check_shell_result=True,
        ).decode("utf-8", errors="replace").strip()
        component_pattern = (
            re.escape(package) + r"/\.?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
        )
        components = [line.strip() for line in output.splitlines()
                      if re.fullmatch(component_pattern, line.strip())]
        if len(components) != 1:
            raise AutomationError(f"无法确认游戏 {package} 的启动入口：{output or 'ADB 返回为空'}")
        return components[0]

    def restart_app(self, package: str) -> None:
        """Restart only this package on the selected Android device."""
        component = self.resolve_launcher(package)
        self._adb("shell", "am", "force-stop", package, check_shell_result=True)
        output = self._adb(
            "shell", "am", "start", "-W", "-n", component, check_shell_result=True,
        ).decode("utf-8", errors="replace").strip()
        if not re.search(r"(?im)^Status:\s*ok\s*$", output):
            raise AutomationError(f"游戏 {package} 启动未确认成功：{output or 'ADB 返回为空'}")

    @staticmethod
    def _check_coordinates(*coordinates: int) -> None:
        if any(type(value) is not int or value < 0 for value in coordinates):
            raise AutomationError("点击和滑动坐标必须是非负整数。")

    def tap(self, x: int, y: int) -> None:
        self._check_coordinates(x, y)
        self._adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._check_coordinates(x1, y1, x2, y2)
        if type(duration_ms) is not int or duration_ms < 1:
            raise AutomationError("滑动时长必须是正整数（毫秒）。")
        self._adb("shell", "input", "swipe", *map(str, (x1, y1, x2, y2, duration_ms)))

    def key(self, name: str) -> None:
        keys = {"back": "4", "home": "3", "enter": "66"}
        if name not in keys:
            raise AutomationError("按键仅支持 back、home、enter。")
        self._adb("shell", "input", "keyevent", keys[name])
