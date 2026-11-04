from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pet_alliance.adb import AutomationError, find_adb
from pet_alliance.cli import main
from pet_alliance.configuration import load_config


class PortabilityTests(unittest.TestCase):
    def test_machine_overrides_preserve_shared_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "automation.toml"
            config.write_text('[device]\nadb_path=""\nserial=""\ntimeout=30\n[daily_hero]\nmax_retries=10\n', encoding="utf-8")
            (root / "automation.local.toml").write_text("[device]\nadb_path='D:\\other install\\adb.exe'\nserial='emulator-5560'\n", encoding="utf-8")
            values = load_config(config)
            self.assertEqual(values["device"], {"adb_path": r"D:\other install\adb.exe", "serial": "emulator-5560", "timeout": 30})
            self.assertEqual(values["daily_hero"]["max_retries"], 10)
            self.assertEqual(load_config(config, include_local=False)["device"]["serial"], "")

    @patch("pet_alliance.adb._windows_adb_candidates")
    @patch("pet_alliance.adb.shutil.which", return_value=None)
    @patch.dict("pet_alliance.adb.os.environ", {}, clear=True)
    def test_discovers_adb_in_custom_install_with_spaces(self, which, candidates):
        with tempfile.TemporaryDirectory(prefix="custom LDPlayer ") as directory:
            adb = Path(directory) / "adb.exe"
            adb.touch()
            candidates.return_value = [str(adb)]
            self.assertEqual(find_adb(), str(adb.resolve()))

    @patch("pet_alliance.adb._windows_adb_candidates")
    @patch("pet_alliance.adb.shutil.which", return_value=None)
    def test_environment_path_has_priority_over_detection(self, which, candidates):
        with tempfile.TemporaryDirectory() as directory:
            adb = Path(directory) / "adb.exe"
            adb.touch()
            with patch.dict("pet_alliance.adb.os.environ", {"ADB_PATH": str(adb)}):
                self.assertEqual(find_adb(), str(adb.resolve()))
            candidates.assert_not_called()

    @patch("pet_alliance.adb.shutil.which", return_value=None)
    def test_invalid_explicit_adb_does_not_silently_select_another(self, which):
        with self.assertRaisesRegex(AutomationError, "automation.local.toml"):
            find_adb("missing-adb.exe")

    @patch("pet_alliance.cli.find_adb", return_value="adb.exe")
    @patch("pet_alliance.adb.subprocess.run")
    def test_blank_default_serial_auto_selects_online_device(self, run, find):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "automation.toml"
            config.write_text('[device]\nadb_path=""\nserial=""\n', encoding="utf-8")
            run.return_value = subprocess.CompletedProcess([], 0, b"List of devices attached\nemulator-5560\tdevice\n", b"")
            self.assertEqual(main(["--config", str(config), "--dry-run", "tap", "1", "2"]), 0)
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
