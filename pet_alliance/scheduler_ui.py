"""Small desktop control panel for the game scheduler.

The controller owns all game interaction. Tk only reads immutable snapshots and
drains the controller's log queue on its own thread.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import queue
import re
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Protocol


BEIJING = timezone(timedelta(hours=8))
_LOG_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?=\s|$)")
STATUS_NAMES = {
    "paused": "已暂停",
    "starting": "正在启动",
    "running": "运行中",
    "pausing": "正在暂停自动操作",
    "error": "需要处理",
    "stopping": "正在停止",
    "closed": "已停止",
    "idle": "等待任务",
}
TASK_NAMES = {
    "check_arena": "擂台检查",
    "slay_dragon": "屠龙托管",
    "daily_hero": "英雄级副本",
    "Daily_Hero": "英雄级副本",
    "daily_hero_reset": "英雄级副本 reset",
    "capture_once": "单次截图",
    "return_to_town": "返回天星镇",
    "midnight_restart": "零点重启游戏",
    "return_town": "返回天星镇",
    "waiting": "等待下一项任务",
}
TASK_STATUS_NAMES = {
    "pending": "待完成",
    "running": "进行中",
    "completed": "已完成",
    "done": "已完成",
    "paused": "未完成，等待继续",
    "preempted": "已暂存，等待继续",
    "interrupted": "未完成，等待继续",
    "failed": "执行失败",
    "error": "执行失败",
    "limited": "本次到达数量限制",
    "stopped": "未完成，等待继续",
}
SCRIPT_STATUS_NAMES = {
    "idle": "待运行", "pending": "待运行", "starting": "正在启动",
    "running": "运行中", "pausing": "正在停止", "stopping": "正在停止",
    "paused": "已停止", "stopped": "已停止", "interrupted": "已停止",
    "completed": "已完成", "done": "已完成", "error": "执行失败",
    "failed": "执行失败", "limited": "已达到运行上限",
    "blocked": "前置条件未满足", "exhausted": "暂无可用目标",
}


class Controller(Protocol):
    logs: queue.Queue[str]

    def start(self) -> None: ...

    def pause(self) -> None: ...

    def start_script(self, name: str) -> None: ...

    def stop_script(self, name: str) -> None: ...

    def set_points_arena_enabled(self, enabled: bool) -> None: ...

    def close(self) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


def format_beijing(value: Any) -> str:
    """Show persisted ISO timestamps in the game's UTC+8 clock."""
    if not value:
        return "尚未记录"
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return str(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=BEIJING)
    return moment.astimezone(BEIJING).strftime("%m月%d日 %H:%M:%S")


def task_summary(task: dict[str, Any]) -> str:
    status = "已完成" if task.get("done") else TASK_STATUS_NAMES.get(
        str(task.get("status", "pending")), str(task.get("status", "待完成"))
    )
    progress = task.get("progress") or {}
    details: list[str] = []
    fields = (
        (("completed_dungeons", "completed", "victories"), "通关", "个"),
        (("skipped_dungeons", "skipped"), "跳过", "个"),
        (("diamond_spent", "diamonds_spent"), "消耗", "钻石"),
    )
    for keys, label, unit in fields:
        for key in keys:
            if key in progress:
                value = progress[key]
                if isinstance(value, (list, tuple)):
                    value = len(value)
                details.append(f"{label} {value} {unit}")
                break
    if progress.get("points_arena_pending") is True:
        details.append("积分赛待继续")
    return status + (" · " + "，".join(details) if details else "")


def view_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Convert controller state into display text without touching Tk."""
    status = str(snapshot.get("status", "paused"))
    tasks = snapshot.get("tasks") or {}
    current = snapshot.get("current_task")
    return {
        "status": STATUS_NAMES.get(status, status),
        "date": str(snapshot.get("date") or datetime.now(BEIJING).date()),
        "current_task": TASK_NAMES.get(str(current), str(current)) if current else "暂无任务",
        "daily_hero": task_summary(tasks.get("daily_hero") or {}),
        "daily_hero_reset": task_summary(tasks.get("daily_hero_reset") or {}),
        "arena_expired_time": format_beijing(snapshot.get("arena_expired_time")),
        "next_dragon_start": format_beijing(snapshot.get("next_dragon_start")),
        "reason": str(snapshot.get("reason") or ""),
        "can_start": status in {"paused", "error", "idle"} and not snapshot.get("worker_alive", False),
        "can_pause": status in {"starting", "running", "idle"} and snapshot.get("mode") != "single",
        "points_arena_enabled": snapshot.get("points_arena_enabled", True),
        "can_edit_points_arena": snapshot.get("can_edit_points_arena", (
            status in {"paused", "error", "idle"} and not snapshot.get("worker_alive", False)
        )),
    }


def script_view_states(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep every start button locked until the sole worker has exited."""
    status = str(snapshot.get("status", "paused"))
    busy = snapshot.get("worker_alive", False) or status in {
        "starting", "running", "pausing", "stopping",
    }
    active = snapshot.get("single_script") if snapshot.get("mode") == "single" else None
    rows = []
    for script in snapshot.get("scripts") or []:
        state = str(script.get("status", "idle"))
        rows.append({
            **script,
            "status_text": SCRIPT_STATUS_NAMES.get(state, state),
            "can_start": not busy and status != "closed",
            "can_stop": active == script["name"] and busy and status in {"starting", "running", "idle"},
        })
    return rows


class SchedulerWindow:
    POLL_MS = 150
    SNAPSHOT_POLLS = 3
    MAX_LOG_LINES = 1600
    LOG_BATCH = 250

    def __init__(self, root: tk.Tk, controller: Controller) -> None:
        self.root = root
        self.controller = controller
        self.closed = False
        self._closing = False
        self._after_id: str | None = None
        self._close_after_id: str | None = None
        self._poll_count = 0
        self._log_day: str | None = None
        self._build()
        self.refresh()
        self._schedule_poll()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        self.root.title("MXM 游戏托管")
        self.root.geometry("1040x850")
        self.root.minsize(900, 760)
        self.root.configure(background="#eef2f6")
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", foreground="#253347", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 11, "bold"), foreground="#176b57")
        style.configure("Muted.TLabel", foreground="#687689", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(16, 7))
        style.configure("TCheckbutton", background="#eef2f6", foreground="#253347",
                        font=("Microsoft YaHei UI", 10))
        style.configure("TLabelframe", background="#eef2f6")
        style.configure("TLabelframe.Label", background="#eef2f6", font=("Microsoft YaHei UI", 10, "bold"))

        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="MXM 游戏托管", style="Title.TLabel").pack(side="left")
        self.clock_var = tk.StringVar(self.root)
        ttk.Label(header, textvariable=self.clock_var, style="Muted.TLabel").pack(side="right", anchor="s")

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="x", pady=(16, 0))
        scheduler_page = ttk.Frame(self.notebook, padding=(12, 0, 12, 12))
        single_page = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(scheduler_page, text="全天托管")
        self.notebook.add(single_page, text="单任务运行")

        toolbar = ttk.Frame(scheduler_page, padding=(0, 12, 0, 12))
        toolbar.pack(fill="x")
        self.start_button = ttk.Button(toolbar, text="启动", command=self.start)
        self.start_button.pack(side="left")
        self.pause_button = ttk.Button(toolbar, text="暂停", command=self.pause)
        self.pause_button.pack(side="left", padx=(10, 18))
        self.status_var = tk.StringVar(self.root)
        ttk.Label(toolbar, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.task_var = tk.StringVar(self.root)
        ttk.Label(toolbar, textvariable=self.task_var).pack(side="right")

        self.points_arena_var = tk.BooleanVar(self.root, value=True)
        toggle_text = "英雄副本及重置：每三个副本参加一次积分赛"
        self.schedule_points_toggle = ttk.Checkbutton(
            scheduler_page, text=toggle_text, variable=self.points_arena_var,
            command=self.set_points_arena_enabled,
        )
        self.schedule_points_toggle.pack(anchor="w", pady=(0, 10))

        self.today_var = tk.StringVar(self.root)
        today = ttk.LabelFrame(scheduler_page, text="今日完成情况", padding=(16, 12))
        today.pack(fill="x")
        today.columnconfigure(1, weight=1)
        ttk.Label(today, textvariable=self.today_var, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 9))
        self.hero_var = tk.StringVar(self.root)
        self.reset_var = tk.StringVar(self.root)
        self.arena_var = tk.StringVar(self.root)
        self.dragon_var = tk.StringVar(self.root)
        for row, (label, variable) in enumerate((
            ("英雄级副本", self.hero_var),
            ("英雄级副本 reset", self.reset_var),
            ("擂台守护到期", self.arena_var),
            ("屠龙下一次开始", self.dragon_var),
        ), start=1):
            ttk.Label(today, text=label, width=21).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Label(today, textvariable=variable).grid(row=row, column=1, sticky="w", pady=4)

        ttk.Label(
            single_page, text="选择一个脚本单独运行；完成后停止。运行期间其他任务的启动按钮会暂时禁用。",
            style="Muted.TLabel", wraplength=920,
        ).pack(fill="x", pady=(0, 8))
        self.single_points_toggle = ttk.Checkbutton(
            single_page, text=toggle_text, variable=self.points_arena_var,
            command=self.set_points_arena_enabled,
        )
        self.single_points_toggle.pack(anchor="w", pady=(0, 8))
        self.script_list_frame = ttk.Frame(single_page)
        self.script_list_frame.pack(fill="x")
        self.start_buttons: dict[str, ttk.Button] = {}
        self.stop_buttons: dict[str, ttk.Button] = {}
        self.script_status_vars: dict[str, tk.StringVar] = {}
        self._script_rows: dict[str, ttk.Frame] = {}

        self.reason_var = tk.StringVar(self.root)
        ttk.Label(outer, textvariable=self.reason_var, foreground="#a14d16", wraplength=920).pack(fill="x", pady=(10, 8))

        log_frame = ttk.LabelFrame(outer, text="后台运行日志（仅今日）", padding=8)
        log_frame.pack(fill="both", expand=True)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_frame, state="disabled", wrap="word", background="#162130",
            foreground="#dbe6f1", insertbackground="#dbe6f1", relief="flat",
            font=("Microsoft YaHei UI", 9), padx=12, pady=10, height=12,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)
        ttk.Label(
            outer, text="暂停或停止后，请等状态确认再手动操作；手动操作结束后，回到天星镇再启动任务。",
            style="Muted.TLabel", wraplength=920,
        ).pack(side="bottom", before=log_frame, fill="x", pady=(12, 0))

    def _refresh_scripts(self, snapshot: dict[str, Any]) -> None:
        scripts = script_view_states(snapshot)
        names = {script["name"] for script in scripts}
        for name in set(self._script_rows) - names:
            self._script_rows.pop(name).destroy()
            self.start_buttons.pop(name)
            self.stop_buttons.pop(name)
            self.script_status_vars.pop(name)
        for index, script in enumerate(scripts):
            name = script["name"]
            if name not in self._script_rows:
                row = ttk.Frame(self.script_list_frame, padding=(0, 7))
                row.columnconfigure(0, weight=1)
                title = script.get("title") or TASK_NAMES.get(name, name)
                ttk.Label(row, text=f"{title}  ·  {name}").grid(row=0, column=0, sticky="w")
                ttk.Label(row, text=script.get("description", ""), style="Muted.TLabel",
                          wraplength=560).grid(row=1, column=0, sticky="w", pady=(2, 0))
                variable = tk.StringVar(self.root)
                ttk.Label(row, textvariable=variable, width=14).grid(row=0, column=1, rowspan=2, padx=10)
                start = ttk.Button(row, text="启动", command=lambda selected=name: self.start_script(selected))
                start.grid(row=0, column=2, rowspan=2, padx=(0, 8))
                stop = ttk.Button(row, text="停止", command=lambda selected=name: self.stop_script(selected))
                stop.grid(row=0, column=3, rowspan=2)
                self._script_rows[name] = row
                self.start_buttons[name] = start
                self.stop_buttons[name] = stop
                self.script_status_vars[name] = variable
            self._script_rows[name].grid(row=index, column=0, sticky="ew")
            self.script_list_frame.columnconfigure(0, weight=1)
            self.script_status_vars[name].set(script["status_text"])
            self.start_buttons[name].configure(state="normal" if script["can_start"] else "disabled")
            self.stop_buttons[name].configure(state="normal" if script["can_stop"] else "disabled")

    def _schedule_poll(self) -> None:
        if not self.closed and not self._closing:
            self._after_id = self.root.after(self.POLL_MS, self._poll)

    def _poll(self) -> None:
        if self.closed or self._closing:
            return
        self._after_id = None
        self._drain_logs()
        self.clock_var.set(datetime.now(BEIJING).strftime("北京时间 %Y-%m-%d %H:%M:%S"))
        self._poll_count += 1
        if self._poll_count % self.SNAPSHOT_POLLS == 0:
            self.refresh()
        self._schedule_poll()

    def _append_log(self, lines: list[str]) -> None:
        if not lines:
            return
        following = self.log_text.yview()[1] >= 0.98
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "\n".join(line.rstrip("\r\n") for line in lines) + "\n")
        count = int(self.log_text.index("end-1c").split(".")[0])
        if count > self.MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{count - self.MAX_LOG_LINES + 1}.0")
        self.log_text.configure(state="disabled")
        if following:
            self.log_text.see("end")

    def _drain_logs(self) -> None:
        # The log queue is drained more frequently than the full UI refresh.
        # Read its day first so new-day messages are not lost at midnight.
        try:
            snapshot = self.controller.snapshot()
            self._set_log_day(str(snapshot.get("date") or datetime.now(BEIJING).date()))
        except Exception:
            # A transient snapshot failure must not discard unread messages.
            return
        lines = []
        for _ in range(self.LOG_BATCH):
            try:
                line = str(self.controller.logs.get_nowait())
            except queue.Empty:
                break
            if not _LOG_TIMESTAMP.match(line) or self._log_day is None or line[:10] == self._log_day:
                lines.append(line)
        self._append_log(lines)

    def _set_log_day(self, day: str) -> None:
        if self._log_day is not None and self._log_day != day:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")
        self._log_day = day

    def refresh(self) -> None:
        if self.closed or self._closing:
            return
        try:
            snapshot = self.controller.snapshot()
            state = view_state(snapshot)
        except Exception as exc:
            self.status_var.set("状态读取失败")
            self.reason_var.set(str(exc))
            return
        self._set_log_day(state["date"])
        self.clock_var.set(datetime.now(BEIJING).strftime("北京时间 %Y-%m-%d %H:%M:%S"))
        self.status_var.set(state["status"])
        self.task_var.set(f"当前任务：{state['current_task']}")
        self.today_var.set(f"{state['date']} · 每天北京时间零点刷新")
        self.hero_var.set(state["daily_hero"])
        self.reset_var.set(state["daily_hero_reset"])
        self.arena_var.set(state["arena_expired_time"])
        self.dragon_var.set(state["next_dragon_start"])
        self.reason_var.set(state["reason"])
        self.start_button.configure(state="normal" if state["can_start"] else "disabled")
        self.pause_button.configure(state="normal" if state["can_pause"] else "disabled")
        self.points_arena_var.set(state["points_arena_enabled"])
        for toggle in (self.schedule_points_toggle, self.single_points_toggle):
            toggle.configure(state="normal" if state["can_edit_points_arena"] else "disabled")
        self._refresh_scripts(snapshot)

    def _action(self, action: Any) -> None:
        if self.closed or self._closing:
            return
        try:
            action()
        except Exception as exc:
            self.reason_var.set(str(exc))
            self._append_log([f"操作失败：{exc}"])
            return
        self.refresh()

    def start(self) -> None:
        self._action(self.controller.start)

    def pause(self) -> None:
        self._action(self.controller.pause)

    def start_script(self, name: str) -> None:
        self._action(lambda: self.controller.start_script(name))

    def stop_script(self, name: str) -> None:
        self._action(lambda: self.controller.stop_script(name))

    def set_points_arena_enabled(self) -> None:
        if self.closed or self._closing:
            return
        try:
            self.controller.set_points_arena_enabled(bool(self.points_arena_var.get()))
        except Exception as exc:
            # A failed save must leave both copies showing the saved value.
            self.refresh()
            self.reason_var.set(str(exc))
            self._append_log([f"积分赛配置保存失败：{exc}"])
            return
        self.refresh()

    def close(self) -> None:
        if self.closed or self._closing:
            return
        # Keep Tk responsive until the worker acknowledges cancellation and saves
        # its state. Neither closing nor pausing sends additional game input.
        self.controller.close()
        self._closing = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.status_var.set("正在停止并保存状态")
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="disabled")
        for button in (*self.start_buttons.values(), *self.stop_buttons.values()):
            button.configure(state="disabled")
        self.schedule_points_toggle.configure(state="disabled")
        self.single_points_toggle.configure(state="disabled")
        self._wait_for_close()

    def _wait_for_close(self) -> None:
        self._close_after_id = None
        self._drain_logs()
        try:
            worker_alive = self.controller.snapshot().get("worker_alive", False)
        except Exception as exc:
            self.reason_var.set(f"等待后台停止：{exc}")
            worker_alive = True
        if worker_alive:
            self._close_after_id = self.root.after(self.POLL_MS, self._wait_for_close)
            return
        self.closed = True
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MXM 全天托管控制面板")
    parser.add_argument("--config", type=Path, help="主配置路径")
    parser.add_argument("--adb", help="ADB 可执行文件路径")
    parser.add_argument("--serial", help="模拟器设备序列号")
    args = parser.parse_args(argv)
    root = tk.Tk()
    root.withdraw()
    try:
        # Importing the UI or opening the window never starts game interaction.
        from .scheduler import SchedulerController

        controller = SchedulerController(config_path=args.config, adb_path=args.adb, serial=args.serial)
        SchedulerWindow(root, controller)
    except Exception as exc:
        messagebox.showerror("托管面板无法启动", str(exc), parent=root)
        root.destroy()
        return 1
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
