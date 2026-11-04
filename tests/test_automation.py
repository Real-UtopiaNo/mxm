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
