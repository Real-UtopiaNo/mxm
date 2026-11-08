from datetime import datetime, timezone
from pathlib import Path
import queue
import tempfile
import tkinter as tk
from tkinter import ttk
import time
import unittest
from unittest.mock import patch

from pet_alliance.scheduler_ui import SchedulerWindow, format_beijing, task_summary, view_state


class FakeController:
    SCRIPT_METADATA = (
        ("Daily_Hero", "英雄级副本", "检查红点副本并完成英雄级挑战"),
        ("daily_hero_reset", "英雄级副本 reset", "重置选定区域并完成英雄级挑战"),
        ("check_arena", "擂台检查", "按配置顺序检查并占领擂台"),
        ("slay_dragon", "屠龙托管", "按屠龙时段持续托管"),
        ("capture_once", "保存截图", "保存模拟器当前画面"),
    )

    def __init__(self):
        self.logs = queue.Queue()
        self.starts = self.pauses = self.closes = 0
        self.script_starts = []
        self.script_stops = []
        self.points_arena_changes = []
        self.points_arena_error = None
        self.state = {
            "status": "paused", "date": "2026-11-06", "current_task": None,
            "mode": None, "single_script": None, "worker_alive": False,
            "points_arena_enabled": True,
            "scripts": [
                {"name": name, "title": title, "description": description,
                 "status": "idle", "reason": ""}
                for name, title, description in self.SCRIPT_METADATA
            ],
            "tasks": {
                "daily_hero": {"done": True, "progress": {"completed": [1, 2]}},
                "daily_hero_reset": {"done": False, "status": "pending"},
            },
            "arena_expired_time": "2026-11-06T12:00:00+08:00",
            "next_dragon_start": "2026-11-06T18:00:00+08:00",
            "reason": "",
        }

    def start(self):
        self.starts += 1
        self.state.update(status="running", current_task="slay_dragon",
                          mode="scheduler", worker_alive=True)

    def pause(self):
        self.pauses += 1
        self.state["status"] = "pausing"

    def start_script(self, name):
        self.script_starts.append(name)
        self.state.update(status="running", current_task=name, mode="single",
                          single_script=name, worker_alive=True)
        self._script(name).update(status="running", reason="")

    def stop_script(self, name):
        self.script_stops.append(name)
        self.state["status"] = "pausing"
        self._script(name)["status"] = "pausing"

    def finish_script(self, name):
        self.state.update(status="paused", current_task=None, mode=None,
                          single_script=None, worker_alive=False)
        self._script(name)["status"] = "completed"

    def _script(self, name):
        return next(item for item in self.state["scripts"] if item["name"] == name)

    def close(self):
        self.closes += 1

    def set_points_arena_enabled(self, enabled):
        if self.points_arena_error is not None:
            raise self.points_arena_error
        self.points_arena_changes.append(enabled)
        self.state["points_arena_enabled"] = enabled

    def snapshot(self):
        return {
            **self.state,
            "can_edit_points_arena": self.state["status"] in {"paused", "error", "idle"}
            and not self.state["worker_alive"],
        }


class SchedulerViewTests(unittest.TestCase):
    def test_timestamps_are_displayed_in_beijing(self):
        self.assertEqual(format_beijing("2026-11-06T04:05:06Z"), "11月06日 12:05:06")
        self.assertEqual(format_beijing("2026-11-06T12:05:06"), "11月06日 12:05:06")
        self.assertEqual(format_beijing(datetime(2026, 11, 6, tzinfo=timezone.utc)), "11月06日 08:00:00")
        self.assertEqual(format_beijing(None), "尚未记录")
        self.assertEqual(format_beijing("未设置"), "未设置")

    def test_summary_preserves_done_and_progress(self):
        self.assertEqual(task_summary({"done": True, "status": "pending", "progress": {
            "completed_dungeons": 3, "skipped_dungeons": 2, "diamond_spent": 150,
        }}), "已完成 · 通关 3 个，跳过 2 个，消耗 150 钻石")

    def test_view_state_controls_start_pause_and_task_names(self):
        state = view_state(FakeController().snapshot())
        self.assertTrue(state["can_start"])
        self.assertFalse(state["can_pause"])
        self.assertEqual(state["daily_hero"], "已完成 · 通关 2 个")
        state = view_state({"status": "running", "current_task": "check_arena"})
        self.assertEqual(state["current_task"], "擂台检查")
        self.assertEqual(view_state({"status": "running", "current_task": "midnight_restart"})["current_task"],
                         "零点重启游戏")
        self.assertFalse(state["can_start"])
        self.assertTrue(state["can_pause"])
        state = view_state({"status": "pausing"})
        self.assertFalse(state["can_start"])
        self.assertFalse(state["can_pause"])
        state = view_state({"status": "idle", "worker_alive": True})
        self.assertFalse(state["can_start"])
        self.assertTrue(state["can_pause"])
        self.assertTrue(view_state({"status": "idle", "worker_alive": False})["can_start"])

    def test_launcher_uses_non_console_runtime_and_does_not_start_tasks(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_scheduler.ps1").read_text(encoding="utf-8")
        self.assertIn("pythonw.exe", source)
        self.assertIn("pet_alliance.scheduler_ui", source)
        self.assertIn("WindowStyle = 'Hidden'", source)
        self.assertNotIn("'run'", source)


class SchedulerWindowTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()
        self.controller = FakeController()
        self.window = SchedulerWindow(self.root, self.controller)

    def tearDown(self):
        if hasattr(self, "window") and not self.window.closed:
            # Fake workers acknowledge cancellation immediately during cleanup.
            self.controller.state["worker_alive"] = False
            if self.window._closing:
                self.window._wait_for_close()
            else:
                self.window.close()

    def _assert_all_script_starts_disabled(self):
        for name, button in self.window.start_buttons.items():
            with self.subTest(script=name):
                self.assertEqual(str(button["state"]), "disabled")

    def _points_toggles(self):
        return self.window.schedule_points_toggle, self.window.single_points_toggle

    def test_both_pages_share_points_arena_option_and_send_boolean_updates(self):
        self.assertIsInstance(self.window.points_arena_var, tk.BooleanVar)
        self.assertTrue(self.window.points_arena_var.get())
        for toggle in self._points_toggles():
            self.assertIsInstance(toggle, ttk.Checkbutton)
            self.assertEqual(str(toggle["variable"]), str(self.window.points_arena_var))
            self.assertEqual(str(toggle["state"]), "normal")
            self.assertTrue(toggle.instate(["selected"]))

        self.window.schedule_points_toggle.invoke()
        self.assertFalse(self.window.points_arena_var.get())
        self.assertFalse(self.controller.state["points_arena_enabled"])
        self.window.notebook.select(1)
        self.assertFalse(self.window.single_points_toggle.instate(["selected"]))
        self.window.single_points_toggle.invoke()
        self.assertTrue(self.window.points_arena_var.get())
        self.assertTrue(self.window.schedule_points_toggle.instate(["selected"]))
        self.assertEqual(self.controller.points_arena_changes, [False, True])
        self.assertTrue(all(type(value) is bool for value in self.controller.points_arena_changes))
        self.assertEqual(self.controller.starts, 0)
        self.assertEqual(self.controller.pauses, 0)
        self.assertEqual(self.controller.script_starts, [])
        self.assertEqual(self.controller.script_stops, [])

    def test_points_arena_option_is_restored_when_window_is_reopened(self):
        self.window.single_points_toggle.invoke()
        self.assertFalse(self.controller.state["points_arena_enabled"])
        self.window.close()
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = SchedulerWindow(self.root, self.controller)
        self.assertFalse(self.window.points_arena_var.get())
        self.assertEqual(self.controller.points_arena_changes, [False])
        for toggle in self._points_toggles():
            self.assertFalse(toggle.instate(["selected"]))

    def test_points_arena_toggles_are_locked_until_worker_has_stopped(self):
        for status, alive, expected in (
            ("paused", False, "normal"),
            ("error", False, "normal"),
            ("idle", False, "normal"),
            ("starting", False, "disabled"),
            ("running", True, "disabled"),
            ("pausing", True, "disabled"),
            ("stopping", True, "disabled"),
            ("paused", True, "disabled"),
            ("error", True, "disabled"),
            ("idle", True, "disabled"),
            ("closed", False, "disabled"),
        ):
            with self.subTest(status=status, worker_alive=alive):
                self.controller.state.update(status=status, worker_alive=alive)
                self.window.refresh()
                for toggle in self._points_toggles():
                    self.assertEqual(str(toggle["state"]), expected)
                    if expected == "disabled":
                        toggle.invoke()
                self.assertTrue(self.window.points_arena_var.get())
                self.assertEqual(self.controller.points_arena_changes, [])

    def test_points_arena_failed_update_restores_choice_and_explains_failure(self):
        self.controller.points_arena_error = OSError("配置保存失败：测试写入被拒绝")
        self.window.single_points_toggle.invoke()
        self.assertTrue(self.window.points_arena_var.get())
        self.assertTrue(self.controller.state["points_arena_enabled"])
        self.assertEqual(self.controller.points_arena_changes, [])
        for toggle in self._points_toggles():
            self.assertTrue(toggle.instate(["selected"]))
            self.assertEqual(str(toggle["state"]), "normal")
        self.assertIn("配置保存失败", self.window.reason_var.get())
        self.assertIn("测试写入被拒绝", self.window.log_text.get("1.0", "end-1c"))

    def test_points_arena_selection_does_not_change_start_or_stop_actions(self):
        self.window.schedule_points_toggle.invoke()
        self.window.start_button.invoke()
        self.assertEqual(self.controller.starts, 1)
        self.assertEqual(self.controller.points_arena_changes, [False])
        for toggle in self._points_toggles():
            self.assertEqual(str(toggle["state"]), "disabled")
        self.window.pause_button.invoke()
        self.assertEqual(self.controller.pauses, 1)
        self.controller.state.update(status="paused", current_task=None, mode=None,
                                     worker_alive=False)
        self.window.refresh()
        self.window.start_buttons["Daily_Hero"].invoke()
        self.window.stop_buttons["Daily_Hero"].invoke()
        self.assertEqual(self.controller.script_starts, ["Daily_Hero"])
        self.assertEqual(self.controller.script_stops, ["Daily_Hero"])
        self.assertEqual(self.controller.points_arena_changes, [False])
        self.assertFalse(self.window.points_arena_var.get())

    def test_single_task_page_lists_all_scripts_with_titles_and_original_names(self):
        pages = [self.window.notebook.tab(tab, "text") for tab in self.window.notebook.tabs()]
        self.assertEqual(pages, ["全天托管", "单任务运行"])
        expected = {name for name, _, _ in self.controller.SCRIPT_METADATA}
        self.assertEqual(set(self.window.start_buttons), expected)
        self.assertEqual(set(self.window.stop_buttons), expected)
        self.assertEqual(set(self.window.script_status_vars), expected)

        def visible_text(widget):
            values = []
            if "text" in widget.keys():
                values.append(str(widget.cget("text")))
            for child in widget.winfo_children():
                values.extend(visible_text(child))
            return values

        labels = "\n".join(visible_text(self.window.notebook))
        for name, title, _ in self.controller.SCRIPT_METADATA:
            with self.subTest(script=name):
                self.assertIn(name, labels)
                self.assertIn(title, labels)
                self.assertTrue(self.window.script_status_vars[name].get())
                self.assertEqual(str(self.window.start_buttons[name]["state"]), "normal")
                self.assertEqual(str(self.window.stop_buttons[name]["state"]), "disabled")
        self.assertEqual(self.controller.script_starts, [])

    def test_each_single_task_start_button_forwards_its_exact_script_name(self):
        for name, _, _ in self.controller.SCRIPT_METADATA:
            with self.subTest(script=name):
                self.window.start_buttons[name].invoke()
                self.assertEqual(self.controller.script_starts[-1], name)
                self.assertEqual(self.controller.state["single_script"], name)
                self.controller.finish_script(name)
                self.window.refresh()
        self.assertEqual(self.controller.script_starts,
                         [name for name, _, _ in self.controller.SCRIPT_METADATA])
        self.assertEqual(self.controller.starts, 0)

    def test_running_single_task_disables_other_starts_and_enables_only_its_stop(self):
        name = "check_arena"
        self.window.start_buttons[name].invoke()
        self._assert_all_script_starts_disabled()
        self.assertEqual(str(self.window.start_button["state"]), "disabled")
        for other_name, button in self.window.stop_buttons.items():
            with self.subTest(script=other_name):
                self.assertEqual(str(button["state"]), "normal" if other_name == name else "disabled")
        self.assertRegex(self.window.script_status_vars[name].get(), "运行|进行")
        self.window.start_buttons["slay_dragon"].invoke()
        self.window.start_button.invoke()
        self.assertEqual(self.controller.script_starts, [name])
        self.assertEqual(self.controller.starts, 0)

    def test_running_scheduler_disables_all_single_task_controls(self):
        self.window.start_button.invoke()
        self._assert_all_script_starts_disabled()
        for button in self.window.stop_buttons.values():
            self.assertEqual(str(button["state"]), "disabled")
        for toggle in self._points_toggles():
            self.assertEqual(str(toggle["state"]), "disabled")
        self.assertEqual(self.controller.state["mode"], "scheduler")

    def test_single_task_stop_waits_for_worker_before_releasing_start_buttons(self):
        name = "slay_dragon"
        self.window.start_buttons[name].invoke()
        self.window.stop_buttons[name].invoke()
        self.assertEqual(self.controller.script_stops, [name])
        self.assertTrue(self.controller.state["worker_alive"])
        self._assert_all_script_starts_disabled()
        self.assertEqual(str(self.window.start_button["state"]), "disabled")
        self.assertEqual(str(self.window.stop_buttons[name]["state"]), "disabled")
        # A final status alone cannot release the worker's input ownership.
        self.controller.state["status"] = "paused"
        self.controller._script(name)["status"] = "stopped"
        self.window.refresh()
        self._assert_all_script_starts_disabled()
        self.assertEqual(str(self.window.start_button["state"]), "disabled")
        self.controller.finish_script(name)
        self.window.refresh()
        self.assertEqual(str(self.window.start_buttons[name]["state"]), "normal")
        self.assertEqual(str(self.window.start_button["state"]), "normal")

    def test_completed_single_task_displays_result_and_can_be_started_again(self):
        name = "capture_once"
        self.window.start_buttons[name].invoke()
        self.controller.finish_script(name)
        self.window.refresh()
        self.assertIn("完成", self.window.script_status_vars[name].get())
        self.assertEqual(str(self.window.stop_buttons[name]["state"]), "disabled")
        self.window.start_buttons[name].invoke()
        self.assertEqual(self.controller.script_starts, [name, name])

    def test_tabs_share_one_live_read_only_log_stream(self):
        original_log = self.window.log_text
        self.controller.logs.put("全天托管：保存进度")
        self.window._drain_logs()
        self.window.notebook.select(1)
        self.window.start_buttons["check_arena"].invoke()
        self.controller.logs.put("单任务：检查擂台")
        self.window._drain_logs()
        self.window.notebook.select(0)
        self.assertIs(self.window.log_text, original_log)
        self.assertEqual(original_log.get("1.0", "end-1c"), "全天托管：保存进度\n单任务：检查擂台\n")
        self.assertEqual(str(original_log["state"]), "disabled")

    def test_close_disables_all_single_task_controls_until_worker_stops(self):
        name = "daily_hero_reset"
        self.window.start_buttons[name].invoke()
        self.window.close()
        self.assertFalse(self.window.closed)
        self._assert_all_script_starts_disabled()
        for button in self.window.stop_buttons.values():
            self.assertEqual(str(button["state"]), "disabled")
        for toggle in self._points_toggles():
            self.assertEqual(str(toggle["state"]), "disabled")
        self.assertEqual(self.controller.closes, 1)
        self.assertIsNotNone(self.window._close_after_id)
        self.controller.finish_script(name)
        self.window._wait_for_close()
        self.assertTrue(self.window.closed)

    def test_both_pages_keep_controls_logs_and_footer_visible_at_minimum_size(self):
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        footer = next(
            child for child in descendants(self.root)
            if "text" in child.keys()
            and str(child.cget("text")).startswith("暂停或停止后")
        )

        def assert_inside(widget, width, height):
            self.assertTrue(widget.winfo_ismapped(), str(widget))
            left = widget.winfo_rootx() - self.root.winfo_rootx()
            top = widget.winfo_rooty() - self.root.winfo_rooty()
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + widget.winfo_width(), width)
            self.assertLessEqual(top + widget.winfo_height(), height)

        # Map briefly off-screen so Tk computes real geometry without leaving a
        # visible test panel over the user's game or desktop.
        try:
            for width, height in ((1040, 850), (900, 760)):
                self.root.geometry(f"{width}x{height}+3000+100")
                self.root.deiconify()
                self.root.update()
                for page in (0, 1):
                    with self.subTest(size=(width, height), page=page):
                        self.window.notebook.select(page)
                        self.root.update()
                        self.assertEqual((self.root.winfo_width(), self.root.winfo_height()),
                                         (width, height))
                        assert_inside(self.window.log_text, width, height)
                        self.assertGreaterEqual(self.window.log_text.winfo_height(), 120)
                        assert_inside(footer, width, height)
                        if page == 0:
                            assert_inside(self.window.start_button, width, height)
                            assert_inside(self.window.pause_button, width, height)
                            assert_inside(self.window.schedule_points_toggle, width, height)
                        else:
                            assert_inside(self.window.single_points_toggle, width, height)
                            for name in self.window.start_buttons:
                                assert_inside(self.window._script_rows[name], width, height)
                                assert_inside(self.window.start_buttons[name], width, height)
                                assert_inside(self.window.stop_buttons[name], width, height)
        finally:
            self.root.withdraw()

    def test_construction_never_starts_game_work(self):
        self.assertEqual(self.controller.starts, 0)
        self.assertEqual(self.window.status_var.get(), "已暂停")
        self.assertIn("已完成", self.window.hero_var.get())
        self.assertEqual(str(self.window.start_button["state"]), "normal")
        self.assertEqual(str(self.window.pause_button["state"]), "disabled")

    def test_buttons_forward_once_and_update_without_worker_tk_calls(self):
        self.window.start_button.invoke()
        self.assertEqual(self.controller.starts, 1)
        self.assertEqual(self.window.task_var.get(), "当前任务：屠龙托管")
        self.window.pause_button.invoke()
        self.assertEqual(self.controller.pauses, 1)
        self.assertEqual(str(self.window.pause_button["state"]), "disabled")

    def test_logs_are_drained_on_ui_thread_and_text_is_read_only(self):
        self.controller.logs.put("第一行")
        self.controller.logs.put("第二行\n")
        self.window._drain_logs()
        self.assertEqual(self.window.log_text.get("1.0", "end-1c"), "第一行\n第二行\n")
        self.assertEqual(str(self.window.log_text["state"]), "disabled")
        self.assertTrue(self.controller.logs.empty())

    def test_log_backlog_only_displays_the_snapshot_day_and_keeps_ui_errors(self):
        self.window._append_log(["操作失败：测试错误"])
        self.controller.logs.put("2026-11-05 23:59:59 昨日任务")
        self.controller.logs.put("2026-11-06 00:00:00 今日任务\n第二行详情")
        self.controller.logs.put("2026-11-07 00:00:00 其他日期任务")
        self.window._drain_logs()
        self.assertEqual(
            self.window.log_text.get("1.0", "end-1c"),
            "操作失败：测试错误\n2026-11-06 00:00:00 今日任务\n第二行详情\n",
        )
        self.assertEqual(str(self.window.log_text["state"]), "disabled")
        self.assertTrue(self.controller.logs.empty())

    def test_midnight_drain_clears_yesterday_and_keeps_new_day_messages(self):
        self.controller.logs.put("2026-11-06 23:59:59 昨日任务")
        self.window._drain_logs()
        self.controller.state["date"] = "2026-11-07"
        self.controller.logs.put("2026-11-06 23:59:59 昨日队列积压")
        self.controller.logs.put("2026-11-07 00:00:00 零点重启")
        # Draining happens more often than the full snapshot refresh; it must
        # use the new snapshot day before choosing which queued lines to keep.
        self.window._drain_logs()
        self.assertEqual(self.window.log_text.get("1.0", "end-1c"),
                         "2026-11-07 00:00:00 零点重启\n")
        self.assertEqual(str(self.window.log_text["state"]), "disabled")
        self.assertTrue(self.controller.logs.empty())

    def test_refresh_clears_old_logs_after_day_change_without_queued_messages(self):
        self.controller.logs.put("2026-11-06 23:59:59 昨日任务")
        self.window._drain_logs()
        self.controller.state["date"] = "2026-11-07"
        self.window.refresh()
        self.assertEqual(self.window.log_text.get("1.0", "end-1c"), "")
        self.window._append_log(["操作失败：今日错误"])
        self.window.refresh()
        self.assertEqual(self.window.log_text.get("1.0", "end-1c"), "操作失败：今日错误\n")

    def test_log_day_snapshot_failure_leaves_messages_queued_until_retry(self):
        self.controller.state["date"] = "2026-11-07"
        self.controller.logs.put("2026-11-07 00:00:00 今日任务")
        with patch.object(self.controller, "snapshot", side_effect=OSError("读取失败")):
            self.window._drain_logs()
        self.assertEqual(self.controller.logs.qsize(), 1)
        self.window._drain_logs()
        self.assertEqual(self.window.log_text.get("1.0", "end-1c"),
                         "2026-11-07 00:00:00 今日任务\n")

    def test_log_batch_is_bounded_to_keep_ui_responsive(self):
        for number in range(self.window.LOG_BATCH + 2):
            self.controller.logs.put(str(number))
        self.window._drain_logs()
        self.assertEqual(self.controller.logs.qsize(), 2)

    def test_close_cancels_polling_and_requests_controller_shutdown_once(self):
        self.window.close()
        self.window.close()
        self.assertTrue(self.window.closed)
        self.assertIsNone(self.window._after_id)
        self.assertEqual(self.controller.closes, 1)

    def test_close_waits_responsively_until_worker_saves_and_stops(self):
        self.controller.state["worker_alive"] = True
        self.window.close()
        self.assertFalse(self.window.closed)
        self.assertEqual(self.window.status_var.get(), "正在停止并保存状态")
        self.assertEqual(str(self.window.start_button["state"]), "disabled")
        self.assertEqual(str(self.window.pause_button["state"]), "disabled")
        self.assertIsNone(self.window._after_id)
        self.assertIsNotNone(self.window._close_after_id)
        self.window.close()
        self.assertEqual(self.controller.closes, 1)
        self.controller.logs.put("状态已保存")
        self.controller.state["worker_alive"] = False
        deadline = time.monotonic() + 2
        while not self.window.closed and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertTrue(self.window.closed)
        self.assertIsNone(self.window._close_after_id)
        self.assertTrue(self.controller.logs.empty())

    def test_pause_does_not_claim_it_returns_to_town(self):
        self.window.start()
        self.window.pause()
        self.assertEqual(self.window.status_var.get(), "正在暂停自动操作")

    def test_real_controller_opens_paused_without_adb_and_close_releases_lease(self):
        from pet_alliance.scheduler import SchedulerController

        self.window.close()
        self.root = tk.Tk()
        self.root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "automation.toml"
            config.write_text("[scheduler]\npoll_interval = 1\n", encoding="utf-8")
            with patch("pet_alliance.scheduler.AndroidDevice") as device, patch("pet_alliance.scheduler.find_adb") as find_adb:
                controller = SchedulerController(config_path=config)
                self.window = SchedulerWindow(self.root, controller)
                self.root.update()
                snapshot = controller.snapshot()
                self.assertEqual(snapshot["status"], "paused")
                self.assertFalse(snapshot["worker_alive"])
                self.assertEqual(self.window.status_var.get(), "已暂停")
                self.assertEqual(self.window.hero_var.get(), "待完成")
                self.assertIn("回到天星镇", self.window.reason_var.get())
                self.window.close()
                reopened = SchedulerController(config_path=config)
                reopened.close()
                device.assert_not_called()
                find_adb.assert_not_called()


if __name__ == "__main__":
    unittest.main()
