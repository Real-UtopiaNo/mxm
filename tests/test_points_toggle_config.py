from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

from pet_alliance.adb import AutomationError
from pet_alliance.configuration import load_config, persist_points_arena_enabled, points_arena_enabled
from pet_alliance.dragon_schedule import BEIJING
from pet_alliance.scheduler import SchedulerController
from pet_alliance.scheduler_state import SchedulerState


class PointsToggleConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "automation.toml"
        self.base.write_text('# base\n[daily_hero]\nmin_click_interval=1\n', encoding="utf-8")
        self.local = self.root / "automation.local.toml"
        self.now = datetime(2026, 11, 7, 8, tzinfo=BEIJING)

    def controller(self, *, default=False):
        with patch("pet_alliance.scheduler.PROJECT_ROOT", self.root):
            controller = SchedulerController(None if default else self.base, now=lambda: self.now)
        self.addCleanup(controller.close)
        return controller

    def seed_progress(self, controller):
        for task in ("daily_hero", "daily_hero_reset"):
            token = controller.state.start_task(task, self.now)
            controller.state.update_progress(task, {
                "completed": ["completed dungeon"], "victories": 5, "diamonds_spent": 50,
                "points_arena_pending": True, "completed_since_arena": 3,
            }, self.now, token)
            controller.state.finish_task(task, {"status": "completed"}, self.now, token)

    def test_default_is_true_and_only_literal_boolean_values_are_valid(self):
        self.assertIs(points_arena_enabled({}), True)
        for value in (True, False):
            self.assertIs(points_arena_enabled({"daily_hero": {"points_arena_enabled": value}}), value)
        for value in (1, 0, "true", "false", None, []):
            with self.subTest(value=value), self.assertRaises(AutomationError):
                points_arena_enabled({"daily_hero": {"points_arena_enabled": value}})

    def test_default_controller_creates_local_override_and_survives_reopen(self):
        original = self.base.read_bytes()
        controller = self.controller(default=True)
        self.assertTrue(controller.snapshot()["points_arena_enabled"])
        self.assertTrue(controller.snapshot()["can_edit_points_arena"])
        controller.set_points_arena_enabled(False)
        self.assertEqual(self.base.read_bytes(), original)
        self.assertIs(load_config(self.base)["daily_hero"]["points_arena_enabled"], False)
        controller.close()
        reopened = self.controller(default=True)
        self.assertFalse(reopened.snapshot()["points_arena_enabled"])
        reopened.set_points_arena_enabled(True)
        self.assertTrue(load_config(self.base)["daily_hero"]["points_arena_enabled"])

    def test_explicit_config_ignores_local_and_updates_only_explicit_file(self):
        self.local.write_text('invalid TOML!', encoding="utf-8")
        controller = self.controller()
        controller.set_points_arena_enabled(False)
        self.assertFalse(load_config(self.base, include_local=False)["daily_hero"]["points_arena_enabled"])
        self.assertEqual(self.local.read_text(encoding="utf-8"), 'invalid TOML!')

    def test_existing_local_override_and_reset_override_are_synchronized(self):
        self.base.write_text('[daily_hero]\npoints_arena_enabled=true\n'
                             '[daily_hero_reset]\npoints_arena_enabled=true\nreset_regions=["西部大陆"]\n',
                             encoding="utf-8")
        self.local.write_text('[daily_hero]\npoints_arena_enabled=false # user\n'
                              '[device]\ntimeout=30\n', encoding="utf-8")
        settings = persist_points_arena_enabled(self.base, True)
        self.assertTrue(settings["daily_hero"]["points_arena_enabled"])
        self.assertTrue(settings["daily_hero_reset"]["points_arena_enabled"])
        saved = load_config(self.base)
        self.assertTrue(saved["daily_hero_reset"]["points_arena_enabled"])
        self.assertEqual(saved["daily_hero_reset"]["reset_regions"], ["西部大陆"])
        self.assertIn('points_arena_enabled=true', self.base.read_text(encoding="utf-8"))
        self.assertIn('# user', self.local.read_text(encoding="utf-8"))
        persist_points_arena_enabled(self.base, False)
        self.assertFalse(load_config(self.base)["daily_hero_reset"]["points_arena_enabled"])

    def test_comments_crlf_multiline_and_other_settings_are_retained(self):
        source = ('description = """example:\r\n[daily_hero]\r\npoints_arena_enabled = false\r\n"""\r\n'
                  '["daily_hero"] # section\r\n"points_arena_enabled" = true  # switch\r\n'
                  'target_players=[\r\n "Hello", # name\r\n "Hell"\r\n]\r\n'
                  '[daily_hero_reset]\r\npoints_arena_enabled = true # legacy override\r\n'
                  'reset_regions = ["西部大陆"]\r\n')
        self.base.write_bytes(source.encode("utf-8"))
        before = tomllib.loads(source)
        settings = persist_points_arena_enabled(self.base, False, include_local=False)
        after = self.base.read_bytes().decode("utf-8")
        self.assertIn('"points_arena_enabled" = false  # switch\r\n', after)
        self.assertIn('points_arena_enabled = false # legacy override\r\n', after)
        self.assertNotIn("\n", after.replace("\r\n", ""))
        before["daily_hero"]["points_arena_enabled"] = False
        before["daily_hero_reset"]["points_arena_enabled"] = False
        self.assertEqual(tomllib.loads(after), before)
        self.assertEqual(settings, before)

    def test_dotted_keys_are_saved_without_changing_other_namespaces(self):
        self.local.write_text('daily_hero.points_arena_enabled = true\n'
                              'daily_hero.target_players = ["Hello"]\n[device]\ntimeout=30\n', encoding="utf-8")
        persist_points_arena_enabled(self.base, False)
        saved = load_config(self.base)
        self.assertFalse(saved["daily_hero"]["points_arena_enabled"])
        self.assertEqual(saved["daily_hero"]["target_players"], ["Hello"])
        self.assertEqual(saved["device"], {"timeout": 30})

    def test_running_pausing_starting_and_closed_controllers_reject_edits(self):
        controller = self.controller()
        before = self.base.read_bytes()
        for status, alive in (("running", True), ("pausing", True), ("paused", True),
                              ("starting", False), ("closed", False)):
            with self.subTest(status=status, alive=alive):
                controller._status = status
                controller._thread = Mock(is_alive=Mock(return_value=alive))
                self.assertFalse(controller.snapshot()["can_edit_points_arena"])
                with self.assertRaises(AutomationError):
                    controller.set_points_arena_enabled(False)
                self.assertTrue(controller.snapshot()["points_arena_enabled"])
                self.assertEqual(self.base.read_bytes(), before)
        controller._thread = None

    def test_invalid_setter_value_cannot_update_file_or_memory(self):
        controller = self.controller()
        before = self.base.read_bytes()
        for value in (1, "false", None):
            with self.subTest(value=value), self.assertRaises(AutomationError):
                controller.set_points_arena_enabled(value)
        self.assertEqual(self.base.read_bytes(), before)
        self.assertTrue(controller.snapshot()["points_arena_enabled"])

    def test_failed_atomic_save_preserves_config_memory_and_pending_progress(self):
        controller = self.controller()
        self.seed_progress(controller)
        before_config, before_state = self.base.read_bytes(), controller.state.snapshot(self.now)
        before_settings = deepcopy(controller._settings)
        with patch("pet_alliance.configuration.os.replace", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(AutomationError, "denied"):
                controller.set_points_arena_enabled(False)
        self.assertEqual(self.base.read_bytes(), before_config)
        self.assertEqual(controller._settings, before_settings)
        self.assertEqual(controller.state.snapshot(self.now), before_state)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_disable_clears_only_today_points_pending_and_count_preserving_completion(self):
        controller = self.controller()
        self.seed_progress(controller)
        controller.set_points_arena_enabled(False)
        for task, record in controller.snapshot()["tasks"].items():
            with self.subTest(task=task):
                self.assertTrue(record["done"])
                self.assertEqual(record["progress"]["completed"], ["completed dungeon"])
                self.assertEqual(record["progress"]["diamonds_spent"], 50)
                self.assertFalse(record["progress"]["points_arena_pending"])
                self.assertEqual(record["progress"]["completed_since_arena"], 0)
        controller.set_points_arena_enabled(True)
        self.assertFalse(controller.snapshot()["tasks"]["daily_hero"]["progress"]["points_arena_pending"])

    def test_initial_disabled_load_and_reload_clear_existing_pending_without_snapshot_writes(self):
        controller = self.controller()
        self.seed_progress(controller)
        controller.close()
        self.base.write_text('[daily_hero]\npoints_arena_enabled=false\n', encoding="utf-8")
        reopened = self.controller()
        self.assertFalse(reopened.snapshot()["tasks"]["daily_hero"]["progress"]["points_arena_pending"])
        with patch.object(reopened.state, "_save") as save:
            for _ in range(3):
                reopened.snapshot()
            save.assert_not_called()
        reopened.state.record_daily_sidework("daily_hero_reset", {
            "points_arena_pending": True, "completed_since_arena": 2, "diamonds_spent": 100}, self.now)
        reopened._reload()
        progress = reopened.snapshot()["tasks"]["daily_hero_reset"]["progress"]
        self.assertFalse(progress["points_arena_pending"])
        self.assertEqual(progress["completed_since_arena"], 0)
        self.assertEqual(progress["diamonds_spent"], 100)

    def test_state_cleanup_failure_rolls_back_config_and_in_memory_state(self):
        controller = self.controller()
        self.seed_progress(controller)
        before_config, before_state = self.base.read_bytes(), controller.state.snapshot(self.now)
        with patch.object(controller.state, "_save", side_effect=OSError("state full")):
            with self.assertRaisesRegex(AutomationError, "已恢复原配置"):
                controller.set_points_arena_enabled(False)
        self.assertEqual(self.base.read_bytes(), before_config)
        self.assertEqual(controller.state.snapshot(self.now), before_state)
        self.assertTrue(controller.snapshot()["points_arena_enabled"])

    def test_disabled_policy_ignores_stale_pending_but_due_guardian_still_runs(self):
        controller = self.controller()
        self.seed_progress(controller)
        controller.set_points_arena_enabled(False)
        # Even a legacy writer restoring pending after load must not force arena.
        controller.state.record_daily_sidework("daily_hero", {
            "points_arena_pending": True, "completed_since_arena": 3}, self.now)
        controller.state.set_arena(self.now + timedelta(hours=6), now=self.now)
        self.assertIsNone(controller._choose())
        controller.state.set_arena(self.now, now=self.now)
        self.assertEqual(controller._choose(), "check_arena")

    def test_history_is_not_modified_by_today_cleanup(self):
        state = SchedulerState(self.root / "state.json", self.now - timedelta(days=1))
        state.record_daily_sidework("daily_hero", {"points_arena_pending": True,
                                                  "completed_since_arena": 3}, self.now - timedelta(days=1))
        state.record_daily_sidework("daily_hero", {"points_arena_pending": True,
                                                  "completed_since_arena": 2}, self.now)
        before = state.snapshot(self.now)["history"]
        state.clear_points_arena_progress(self.now)
        self.assertEqual(state.snapshot(self.now)["history"], before)


if __name__ == "__main__":
    unittest.main()
