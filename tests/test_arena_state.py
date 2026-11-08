from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch

from pet_alliance.adb import AutomationError
from pet_alliance.arena_state import persist_arena_state
from pet_alliance.configuration import load_config


BEIJING = timezone(timedelta(hours=8))


class ArenaStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base = self.root / "automation.toml"
        self.base.write_text(
            '# base file\n[device]\nadb_path = \'D:\\Game\\adb.exe\'\n'
            '[check_arena]\narena_list = ["恶魔城", "COCO号"]\n'
            'arena_expired_time = "old"\n', encoding="utf-8",
        )
        self.ctx = SimpleNamespace(
            project_root=self.root, settings=load_config(self.base), dry_run=False,
            config_path=self.base, use_local_config=True,
        )
        self.expired = datetime(2026, 11, 6, 3, 40, 2, 999999, tzinfo=BEIJING)
        self.local = self.root / "automation.local.toml"

    def persist(self, **kwargs):
        return persist_arena_state(
            kwargs.get("ctx", self.ctx), kwargs.get("time", self.expired),
            kwargs.get("dungeon", "COCO号"),
        )

    def test_default_creates_local_override_and_preserves_base(self):
        original = self.base.read_bytes()
        result = self.persist()
        self.assertEqual(result, self.local)
        self.assertEqual(self.base.read_bytes(), original)
        config = load_config(self.base)
        self.assertEqual(config["check_arena"]["arena_expired_time"], "2026-11-06T03:40:02+08:00")
        self.assertEqual(config["check_arena"]["arena_dungeon"], "COCO号")
        self.assertEqual(config["check_arena"]["arena_list"], ["恶魔城", "COCO号"])
        self.assertEqual(self.ctx.settings["check_arena"], config["check_arena"])

    def test_missing_optional_context_attributes_default_to_local(self):
        del self.ctx.config_path
        del self.ctx.use_local_config
        self.assertEqual(self.persist(), self.local)

    def test_explicit_configuration_updates_same_file_and_ignores_local(self):
        self.ctx.use_local_config = False
        self.local.write_text('not valid TOML!', encoding="utf-8")
        original = self.base.read_text(encoding="utf-8")
        self.assertEqual(self.persist(), self.base)
        saved = self.base.read_text(encoding="utf-8")
        self.assertIn('# base file\n[device]\nadb_path = \'D:\\Game\\adb.exe\'\n', saved)
        self.assertIn('arena_list = ["恶魔城", "COCO号"]', saved)
        self.assertNotEqual(saved, original)
        self.assertEqual(self.local.read_text(encoding="utf-8"), 'not valid TOML!')

    def test_relative_explicit_path_is_resolved_from_project(self):
        self.ctx.config_path = Path("automation.toml")
        self.ctx.use_local_config = False
        self.assertEqual(self.persist(), self.base)

    def test_keeps_other_sections_comments_and_crlf(self):
        original = (
            '# 本机设置\r\n[device] # custom paths\r\n'
            'adb_path = \'C:\\custom\\adb.exe\' # untouched\r\n'
            '\r\n[check_arena] # arena\r\n'
            'arena_list = ["COCO号"] # priority\r\n'
            'arena_expired_time = "old"  # leave this comment\r\n'
            '\r\n[daily_hero]\r\npoll_interval = 0.8\r\n'
        )
        self.local.write_bytes(original.encode("utf-8"))
        self.persist()
        text = self.local.read_bytes().decode("utf-8")
        self.assertIn('# 本机设置\r\n[device] # custom paths\r\nadb_path = \'C:\\custom\\adb.exe\' # untouched\r\n', text)
        self.assertIn('arena_list = ["COCO号"] # priority\r\n', text)
        self.assertIn('arena_expired_time = "2026-11-06T03:40:02+08:00"  # leave this comment\r\n', text)
        self.assertIn('arena_dungeon = "COCO号"\r\n[daily_hero]', text)
        self.assertNotIn("\n", text.replace("\r\n", ""))
        self.assertEqual(tomllib.loads(text)["daily_hero"], {"poll_interval": 0.8})

    def test_repeated_save_replaces_keys_without_duplicates(self):
        self.persist()
        self.persist(time=self.expired + timedelta(hours=7), dungeon="恶魔城")
        text = self.local.read_text(encoding="utf-8")
        self.assertEqual(text.count("arena_expired_time ="), 1)
        self.assertEqual(text.count("arena_dungeon ="), 1)
        self.assertEqual(tomllib.loads(text)["check_arena"]["arena_expired_time"], "2026-11-06T10:40:02+08:00")

    def test_quoted_section_and_field_keys_are_supported(self):
        self.local.write_text(
            '["check_arena"]\n"arena_expired_time" = "old" # comment\n'
            "'arena_dungeon' = '旧副本'\n", encoding="utf-8",
        )
        self.persist()
        text = self.local.read_text(encoding="utf-8")
        self.assertIn('"arena_expired_time" = "2026-11-06T03:40:02+08:00" # comment', text)
        self.assertIn("'arena_dungeon' = \"COCO号\"", text)

    def test_multiline_values_cannot_spoof_sections_or_keys(self):
        original = (
            'description = """example:\n[check_arena]\narena_dungeon = "example"\n"""\n'
            '[check_arena]\narena_expired_time = """old\nvalue""" # preserved\n'
            'message = \'\'\'arena_expired_time = "keep"\n[other]\'\'\'\n'
            '[other]\nvalue = [\n 1, # comment\n 2\n]\n'
        )
        self.local.write_text(original, encoding="utf-8")
        self.persist()
        data = tomllib.loads(self.local.read_text(encoding="utf-8"))
        before = tomllib.loads(original)
        self.assertEqual(data["description"], before["description"])
        self.assertEqual(data["check_arena"]["message"], before["check_arena"]["message"])
        self.assertEqual(data["other"], before["other"])
        self.assertIn('"2026-11-06T03:40:02+08:00" # preserved', self.local.read_text(encoding="utf-8"))

    def test_dotted_root_keys_keep_their_namespace(self):
        self.local.write_text(
            'check_arena.arena_expired_time = "old"\n'
            'check_arena.arena_list = ["COCO号"]\n[device]\ntimeout=30\n', encoding="utf-8",
        )
        self.persist()
        data = tomllib.loads(self.local.read_text(encoding="utf-8"))
        self.assertEqual(data["check_arena"]["arena_dungeon"], "COCO号")
        self.assertEqual(data["device"], {"timeout": 30})

    def test_nested_tables_cannot_steal_added_fields(self):
        self.local.write_text(
            '[check_arena]\narena_list=["COCO号"]\n'
            '[check_arena.notes]\nmessage="keep"\n'
            '[[device.items]]\nname="first"\n', encoding="utf-8",
        )
        self.persist()
        data = tomllib.loads(self.local.read_text(encoding="utf-8"))
        self.assertEqual(data["check_arena"]["notes"], {"message": "keep"})
        self.assertEqual(data["device"], {"items": [{"name": "first"}]})
        self.assertEqual(data["check_arena"]["arena_dungeon"], "COCO号")

    def test_existing_array_table_before_arena_is_preserved(self):
        self.local.write_text('[[devices]]\nname="one"\n', encoding="utf-8")
        self.persist()
        data = tomllib.loads(self.local.read_text(encoding="utf-8"))
        self.assertEqual(data["devices"], [{"name": "one"}])
        self.assertEqual(data["check_arena"]["arena_dungeon"], "COCO号")

    def test_other_valid_toml_special_float_values_remain_unchanged(self):
        self.local.write_text('values = [nan, +nan, -nan, inf, -inf]\n', encoding="utf-8")
        self.persist()
        self.assertTrue(self.local.read_text(encoding="utf-8").startswith('values = [nan, +nan, -nan, inf, -inf]\n'))

    def test_utc_time_normalizes_to_beijing_and_rolls_date(self):
        self.persist(time=datetime(2026, 11, 5, 22, 20, tzinfo=timezone.utc))
        self.assertEqual(
            self.ctx.settings["check_arena"]["arena_expired_time"], "2026-11-06T06:20:00+08:00",
        )

    def test_dungeon_string_is_escaped_as_toml(self):
        self.persist(dungeon='副本"特殊\\名称')
        data = tomllib.loads(self.local.read_text(encoding="utf-8"))
        self.assertEqual(data["check_arena"]["arena_dungeon"], '副本"特殊\\名称')

    def test_invalid_inputs_do_not_create_file_or_update_settings(self):
        original = self.ctx.settings.copy()
        for time in (None, "2026-11-05", datetime(2026, 11, 5)):
            with self.subTest(time=time), self.assertRaises(AutomationError):
                self.persist(time=time)
        for dungeon in (None, "", " \t", 3):
            with self.subTest(dungeon=dungeon), self.assertRaises(AutomationError):
                self.persist(dungeon=dungeon)
        self.assertFalse(self.local.exists())
        self.assertEqual(self.ctx.settings, original)

    def test_dry_run_never_writes(self):
        self.ctx.dry_run = True
        with self.assertRaisesRegex(AutomationError, "预演"):
            self.persist()
        self.assertFalse(self.local.exists())

    def test_invalid_existing_toml_is_retained(self):
        for content in ('[check_arena]\narena_dungeon="one"\narena_dungeon="two"\n', '\ufeff[check_arena]\n'):
            with self.subTest(content=content):
                self.local.write_text(content, encoding="utf-8")
                before = self.local.read_bytes()
                with self.assertRaisesRegex(AutomationError, "无法保存"):
                    self.persist()
                self.assertEqual(self.local.read_bytes(), before)

    def test_invalid_base_toml_prevents_override_creation(self):
        self.base.write_text("[bad", encoding="utf-8")
        with self.assertRaisesRegex(AutomationError, "无法保存"):
            self.persist()
        self.assertFalse(self.local.exists())

    def test_inline_table_fails_without_touching_other_user_settings(self):
        content = 'check_arena = {arena_dungeon="old", arena_list=["COCO号"]}\n'
        self.local.write_text(content, encoding="utf-8")
        with self.assertRaisesRegex(AutomationError, "独立表"):
            self.persist()
        self.assertEqual(self.local.read_text(encoding="utf-8"), content)

    def test_non_table_section_fails_without_touching_file(self):
        self.local.write_text('check_arena = "bad"\n', encoding="utf-8")
        with self.assertRaisesRegex(AutomationError, "必须是一个表"):
            self.persist()
        self.assertEqual(self.local.read_text(encoding="utf-8"), 'check_arena = "bad"\n')

    def test_unreadable_target_is_reported(self):
        original = Path.read_bytes

        def deny(path):
            if path == self.local:
                raise PermissionError("access denied")
            return original(path)

        with patch.object(Path, "read_bytes", deny), self.assertRaisesRegex(AutomationError, "access denied"):
            self.persist()
        self.assertFalse(self.local.exists())

    def test_atomic_replace_failure_preserves_file_settings_and_cleans_temporary(self):
        self.local.write_text('# original\n[device]\ntimeout=30\n', encoding="utf-8")
        original = self.local.read_bytes()
        before = self.ctx.settings["check_arena"].copy()
        with patch("pet_alliance.arena_state.os.replace", side_effect=PermissionError("cannot replace")):
            with self.assertRaisesRegex(AutomationError, "cannot replace"):
                self.persist()
        self.assertEqual(self.local.read_bytes(), original)
        self.assertEqual(self.ctx.settings["check_arena"], before)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_file_sync_failure_preserves_original(self):
        self.local.write_text('# old file\n', encoding="utf-8")
        with patch("pet_alliance.arena_state.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(AutomationError, "disk full"):
                self.persist()
        self.assertEqual(self.local.read_text(encoding="utf-8"), '# old file\n')
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_concurrent_user_edit_is_not_overwritten(self):
        self.local.write_text('# old file\n', encoding="utf-8")
        original = Path.read_bytes
        calls = 0

        def edit_between_reads(path):
            nonlocal calls
            if path == self.local:
                calls += 1
                if calls == 2:
                    path.write_text('# newer user edit\n', encoding="utf-8")
            return original(path)

        with patch.object(Path, "read_bytes", edit_between_reads):
            with self.assertRaisesRegex(AutomationError, "保存期间发生变化"):
                self.persist()
        self.assertEqual(self.local.read_text(encoding="utf-8"), '# newer user edit\n')
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_explicit_config_deleted_during_run_is_not_silently_recreated(self):
        self.ctx.use_local_config = False
        self.base.unlink()
        with self.assertRaisesRegex(AutomationError, "找不到"):
            self.persist()
        self.assertFalse(self.base.exists())


if __name__ == "__main__":
    unittest.main()
