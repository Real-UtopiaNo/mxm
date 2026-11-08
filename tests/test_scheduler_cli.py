from pathlib import Path
import unittest
from unittest.mock import patch

from pet_alliance.cli import main


class SchedulerCliTests(unittest.TestCase):
    def test_opening_panel_does_not_load_config_or_connect_adb(self):
        with patch("pet_alliance.scheduler_ui.main", return_value=0) as panel, \
                patch("pet_alliance.cli.load_config") as load_config, \
                patch("pet_alliance.cli.find_adb") as find_adb, \
                patch("pet_alliance.cli.AndroidDevice") as device:
            self.assertEqual(main(["scheduler"]), 0)
            panel.assert_called_once_with([])
            load_config.assert_not_called()
            find_adb.assert_not_called()
            device.assert_not_called()

    def test_scheduler_forwards_options_in_either_command_order(self):
        options = ["--config", str(Path("alternate settings.toml")), "--adb", "D:/LD Player/adb.exe",
                   "--serial", "emulator-5570"]
        for arguments in (options + ["scheduler"], ["scheduler"] + options):
            with self.subTest(arguments=arguments), \
                    patch("pet_alliance.scheduler_ui.main", return_value=7) as panel, \
                    patch("pet_alliance.cli.find_adb") as find_adb:
                self.assertEqual(main(arguments), 7)
                panel.assert_called_once_with(options)
                find_adb.assert_not_called()

    def test_scheduler_dry_run_is_rejected_without_opening_panel(self):
        for arguments in (["--dry-run", "scheduler"], ["scheduler", "--dry-run"]):
            with self.subTest(arguments=arguments), \
                    patch("pet_alliance.scheduler_ui.main") as panel, \
                    patch("pet_alliance.cli.find_adb") as find_adb, \
                    self.assertLogs(level="ERROR") as logs:
                self.assertEqual(main(arguments), 1)
                self.assertIn("不支持 --dry-run", logs.output[0])
                panel.assert_not_called()
                find_adb.assert_not_called()


if __name__ == "__main__":
    unittest.main()
