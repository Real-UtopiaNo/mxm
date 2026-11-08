from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from pet_alliance.arena_state import persist_arena_state
from pet_alliance.cli import main


CONFIG = '''# Preserve this configuration.
[device]
adb_path = ""
[check_arena]
arena_list = ["COCO号"]
player_name = "1111"
arena_expired_time = "old"
'''
EXPIRES = datetime(2026, 11, 6, 5, 10, 2, tzinfo=timezone(timedelta(hours=8)))


class CheckArenaCliTests(unittest.TestCase):
    def run_cli(self, root, arguments):
        def check_script(name, directory, ctx):
            self.assertEqual(name, "check_arena")
            self.assertEqual(ctx.settings["check_arena"]["arena_list"], ["COCO号"])
            persist_arena_state(ctx, EXPIRES, "COCO号")

        with patch("pet_alliance.cli.PROJECT_ROOT", root), \
                patch("pet_alliance.cli.find_adb", return_value=Path("unused-adb.exe")), \
                patch("pet_alliance.cli.AndroidDevice") as device, \
                patch("pet_alliance.cli.run_script", side_effect=check_script):
            self.assertEqual(main(arguments), 0)
            device.return_value.tap.assert_not_called()

    def test_default_cli_writes_local_override_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "automation.toml"
            base.write_text(CONFIG, encoding="utf-8")
            self.run_cli(root, ["run", "check_arena"])
            self.assertEqual(base.read_text(encoding="utf-8"), CONFIG)
            data = tomllib.loads((root / "automation.local.toml").read_text(encoding="utf-8"))
            self.assertEqual(data["check_arena"]["arena_expired_time"], EXPIRES.isoformat())
            self.assertEqual(data["check_arena"]["arena_dungeon"], "COCO号")

    def test_explicit_cli_writes_chosen_file_and_ignores_local_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen = root / "chosen.toml"
            chosen.write_text(CONFIG, encoding="utf-8")
            local = root / "automation.local.toml"
            local.write_text('[check_arena]\narena_list = ["恶魔城"]\n', encoding="utf-8")
            before = local.read_bytes()
            self.run_cli(root, ["--config", str(chosen), "run", "check_arena"])
            data = tomllib.loads(chosen.read_text(encoding="utf-8"))
            self.assertEqual(data["check_arena"]["arena_expired_time"], EXPIRES.isoformat())
            self.assertEqual(data["check_arena"]["arena_dungeon"], "COCO号")
            self.assertTrue(chosen.read_text(encoding="utf-8").startswith("# Preserve"))
            self.assertEqual(local.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
