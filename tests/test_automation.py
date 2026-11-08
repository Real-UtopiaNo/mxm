from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

from pet_alliance import AndroidDevice, AutomationError, ScriptContext
from pet_alliance.runtime import list_scripts, run_script


def png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\r\n\x00"))
        + chunk(b"IEND", b"")
    )


class AutomationTests(unittest.TestCase):
    def device(self, serial="emulator-5554"):
        return AndroidDevice("adb.exe", serial)

    @patch("pet_alliance.adb.subprocess.run")
    def test_capture_preserves_binary_png_and_saves_it(self, run):
        data = png()
        run.return_value = subprocess.CompletedProcess([], 0, data, b"")
        frame = self.device().capture()
        self.assertEqual((frame.width, frame.height), (1, 1))
        self.assertEqual(frame.data, data)
        with tempfile.TemporaryDirectory() as directory:
            saved = frame.save(Path(directory) / "nested" / "capture.png")
            self.assertEqual(saved.read_bytes(), data)
        self.assertEqual(run.call_args.args[0], ["adb.exe", "-s", "emulator-5554", "exec-out", "screencap", "-p"])

    @patch("pet_alliance.adb.subprocess.run")
    def test_invalid_or_truncated_capture_fails(self, run):
        for data in (b"error: offline", png()[:-8]):
            with self.subTest(data=data):
                run.return_value = subprocess.CompletedProcess([], 0, data, b"")
                with self.assertRaises(AutomationError):
                    self.device().capture()

    @patch("pet_alliance.adb.subprocess.run")
    def test_multiple_devices_require_explicit_selection(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, b"List of devices attached\na\tdevice\nb\tdevice\n", b"")
        with self.assertRaisesRegex(AutomationError, "多个设备"):
            self.device(None).connect()

    @patch("pet_alliance.adb.subprocess.run")
    def test_auto_selects_only_online_device(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, b"List of devices attached\na\toffline\nb\tdevice\n", b"")
        device = self.device(None)
        device.connect()
        self.assertEqual(device.serial, "b")

    @patch("pet_alliance.adb.subprocess.run")
    def test_selected_offline_device_fails(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, b"List of devices attached\nemulator-5554\toffline\n", b"")
        with self.assertRaisesRegex(AutomationError, "offline"):
            self.device().connect()

    @patch("pet_alliance.adb.subprocess.run")
    def test_adb_timeout_and_error_are_actionable(self, run):
        run.side_effect = subprocess.TimeoutExpired(["adb.exe"], 30)
        with self.assertRaisesRegex(AutomationError, "超时"):
            self.device().capture()
        run.side_effect = None
        run.return_value = subprocess.CompletedProcess([], 1, b"", b"device offline")
        with self.assertRaisesRegex(AutomationError, "device offline"):
            self.device().capture()

    @patch("pet_alliance.adb.subprocess.run")
    def test_dry_run_suppresses_all_input_but_keeps_capture(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, png(), b"")
        ctx = ScriptContext(self.device(), Path("unused"), dry_run=True)
        ctx.tap(12, 34)
        ctx.swipe(10, 20, 30, 40)
        ctx.key("back")
        run.assert_not_called()
        ctx.capture()
        self.assertEqual(run.call_args.args[0][3:], ["exec-out", "screencap", "-p"])

    @patch("pet_alliance.adb.subprocess.run")
    def test_input_targets_selected_device_and_validates_arguments(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, b"", b"")
        device = self.device()
        device.tap(12, 34)
        self.assertEqual(run.call_args.args[0], ["adb.exe", "-s", "emulator-5554", "shell", "input", "tap", "12", "34"])
        run.reset_mock()
        for x in (-1, "1; reboot", 1.5):
            with self.assertRaises(AutomationError):
                device.tap(x, 0)
        with self.assertRaises(AutomationError):
            device.swipe(0, 0, 1, 1, 0)
        with self.assertRaises(AutomationError):
            device.key("back; reboot")
        run.assert_not_called()

    @patch("pet_alliance.adb.subprocess.run")
    def test_restart_only_selected_game_and_device_after_resolving_launcher(self, run):
        package = "com.monster.mxm1old"
        component = f"{package}/org.cocos2dx.lua.AppActivity"
        run.side_effect = [
            subprocess.CompletedProcess([], 0, f"priority=0\n{component}\n".encode(), b""),
            subprocess.CompletedProcess([], 0, b"", b""),
            subprocess.CompletedProcess([], 0, b"Starting: Intent\nStatus: ok\nComplete\n", b""),
        ]
        AndroidDevice("adb.exe", "emulator-5570").restart_app(package)
        prefix = ["adb.exe", "-s", "emulator-5570", "shell"]
        self.assertEqual([call.args[0] for call in run.call_args_list], [
            prefix + ["cmd", "package", "resolve-activity", "--brief", "-a",
                      "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER", package],
            prefix + ["am", "force-stop", package],
            prefix + ["am", "start", "-W", "-n", component],
        ])

    @patch("pet_alliance.adb.subprocess.run")
    def test_invalid_package_rejected_without_adb_including_dry_run(self, run):
        device = self.device()
        ctx = ScriptContext(device, Path("unused"), dry_run=True)
        for package in (None, "", "android", "com.game;reboot", "com.game$(reboot)",
                        "com.game\nreboot", "com.game other", "--user", "com..game"):
            with self.subTest(package=package):
                for action in (device.resolve_launcher, device.restart_app, ctx.restart_app):
                    with self.assertRaises(AutomationError):
                        action(package)
        run.assert_not_called()

    @patch("pet_alliance.adb.subprocess.run")
    def test_restart_dry_run_logs_without_resolve_stop_or_start(self, run):
        ctx = ScriptContext(self.device(), Path("unused"), dry_run=True)
        with self.assertLogs("pet_alliance", level="INFO") as log:
            ctx.restart_app("com.monster.mxm1old")
        self.assertIn("[预演] 重启游戏 com.monster.mxm1old", log.output[0])
        run.assert_not_called()

    @patch("pet_alliance.adb.subprocess.run")
    def test_missing_or_other_package_launcher_never_stops_game(self, run):
        for output in (b"No activity found\n", b"com.other.game/.MainActivity\n",
                       b"com.monster.mxm1old/.MainActivity;reboot\n", b""):
            with self.subTest(output=output):
                run.reset_mock()
                run.return_value = subprocess.CompletedProcess([], 0, output, b"")
                with self.assertRaisesRegex(AutomationError, "启动入口"):
                    self.device().restart_app("com.monster.mxm1old")
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][4:7], ["cmd", "package", "resolve-activity"])

    @patch("pet_alliance.adb.subprocess.run")
    def test_force_stop_failure_preserves_error_and_never_starts(self, run):
        run.side_effect = [
            subprocess.CompletedProcess([], 0, b"com.monster.mxm1old/.MainActivity\n", b""),
            subprocess.CompletedProcess([], 0, b"", b"Security exception: Permission Denial"),
        ]
        with self.assertRaisesRegex(AutomationError, "Permission Denial"):
            self.device().restart_app("com.monster.mxm1old")
        self.assertEqual(run.call_count, 2)

    @patch("pet_alliance.adb.subprocess.run")
    def test_start_checks_android_error_and_status_even_when_adb_succeeds(self, run):
        for stdout, stderr in (
            (b"Error type 3\nError: Activity class does not exist.", b""),
            (b"", b"Error: Activity not started, unable to resolve Intent"),
            (b"Starting: Intent\nStatus: timeout\n", b""),
        ):
            with self.subTest(stdout=stdout, stderr=stderr):
                run.reset_mock()
                run.side_effect = [
                    subprocess.CompletedProcess([], 0, b"com.monster.mxm1old/.MainActivity\n", b""),
                    subprocess.CompletedProcess([], 0, b"", b""),
                    subprocess.CompletedProcess([], 0, stdout, stderr),
                ]
                with self.assertRaises(AutomationError) as error:
                    self.device().restart_app("com.monster.mxm1old")
                expected = (stderr or stdout).decode().strip()
                self.assertIn(expected, str(error.exception))
                self.assertEqual(run.call_count, 3)

    @patch("pet_alliance.adb.subprocess.run")
    def test_context_real_restart_calls_device_and_propagates_failure(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, b"", b"device offline")
        ctx = ScriptContext(self.device(), Path("unused"))
        with self.assertRaisesRegex(AutomationError, "device offline"):
            ctx.restart_app("com.monster.mxm1old")
        self.assertEqual(run.call_count, 1)

    def test_local_script_discovery_and_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.py").write_text(
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n"
                "@dataclass\n"
                "class Result:\n    message: str\n"
                "def run(ctx):\n    ctx.log.append(Result('ran').message)\n",
                encoding="utf-8",
            )
            (root / "_helper.py").write_text("raise RuntimeError('do not import')", encoding="utf-8")
            self.assertEqual(list_scripts(root), ["hello"])
            ctx = ScriptContext(self.device(), root)
            ctx.log = []
            run_script("hello", root, ctx)
            self.assertEqual(ctx.log, ["ran"])
            with self.assertRaises(AutomationError):
                run_script("../hello", root, ctx)


if __name__ == "__main__":
    unittest.main()
