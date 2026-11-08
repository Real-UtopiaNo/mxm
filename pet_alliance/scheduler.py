"""One cooperative worker coordinates the existing game scripts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Callable

from .adb import AndroidDevice, AutomationError, find_adb
from .configuration import load_config, persist_points_arena_enabled, points_arena_enabled
from .cooperation import TaskInterrupted
from .dragon_schedule import DEFAULT_START_TIMES, beijing_now, next_dragon_start, parse_start_times
from .runtime import ScriptContext, run_script
from .run_logs import RunLogRetention
from .scheduler_jobs import run_task
from .scheduler_navigation import SchedulerNavigator
from .scheduler_policy import DAILY_TASKS, beijing_time, choose_task, parse_timestamp
from .scheduler_state import SchedulerState
from .single_tasks import script_catalog, task_for_script
from .today_logging import TodayFileHandler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIORITY = {name: index for index, name in enumerate(("check_arena", "slay_dragon", *DAILY_TASKS))}


class _Lease:
    """OS-owned lock also survives an unclean exit without a stale PID file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        try:
            self.handle.seek(0, 2)
            if not self.handle.tell():
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise AutomationError("此目录已有调度器打开，请使用原窗口或先关闭它。") from exc

    def close(self):
        if not self.handle.closed:
            self.handle.close()


class _LogHandler(logging.Handler):
    def __init__(self, output: queue.Queue, now: Callable):
        super().__init__()
        self.output, self.now = output, now

    def emit(self, record):
        line = f"{beijing_time(self.now()):%Y-%m-%d %H:%M:%S} {record.getMessage()}"
        try:
            self.output.put_nowait(line)
        except queue.Full:
            try:
                self.output.get_nowait()
            except queue.Empty:
                pass
            self.output.put_nowait(line)


def _merge_progress(previous: dict, report: dict) -> dict:
    """Accumulate confirmed work across attempts, keeping each run's raw report."""
    result = deepcopy(previous)
    counters = {"victories", "failures", "arena_visits", "arena_battles", "arena_no_tickets",
                "diamonds_spent", "diamond_spent", "resets", "reset_count"}
    for key, value in report.items():
        if key in {"completed", "skipped", "diamond_spending"} and isinstance(value, list):
            result[key] = deepcopy(previous.get(key, [])) + deepcopy(value)
        elif key in counters and isinstance(value, (int, float)):
            result[key] = previous.get(key, 0) + value
        elif key not in {"status", "reason"}:
            result[key] = deepcopy(value)
    return result


class SchedulerController:
    def __init__(self, config_path: Path | None = None, include_local: bool = True,
                 adb_path: str | None = None, serial: str | None = None, *,
                 now: Callable[[], datetime] = beijing_now, context_factory: Callable | None = None,
                 task_runner: Callable = run_task, navigator_factory: Callable = SchedulerNavigator,
                 state_path: Path | None = None, script_runner: Callable = run_script):
        self.config_path = Path(config_path or PROJECT_ROOT / "automation.toml").resolve()
        self.include_local = include_local and config_path is None
        self.project_root = self.config_path.parent
        self._now = now
        self._adb_path, self._serial = adb_path, serial
        self._context_factory, self._task_runner = context_factory, task_runner
        self._script_runner = script_runner
        self._navigator_factory = navigator_factory
        self._lock = threading.RLock()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._closing = False
        self._current_task: str | None = None
        self._mode: str | None = None
        self._single_script: str | None = None
        self._attempt: str | None = None
        self._attempt_day: str | None = None
        self._baseline: dict = {}
        self._needs_town = True
        self._last_input = float("-inf")
        self._status, self._reason = "paused", "回到天星镇后点击启动。"
        self.logs: queue.Queue[str] = queue.Queue(maxsize=5000)
        self._settings = load_config(self.config_path, include_local=self.include_local)
        self._read_options()
        directory = self.project_root / "artifacts" / "scheduler"
        self._lease = _Lease(directory / "active.lock")
        self.log = logging.getLogger(f"pet_alliance.scheduler.{id(self)}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        try:
            self.log.addHandler(_LogHandler(self.logs, self._now))
            self._disk_log = TodayFileHandler(directory / "scheduler.log", now=self._now)
            self.log.addHandler(self._disk_log)
            chosen = state_path or self.project_root / self._settings.get("scheduler", {}).get(
                "state_file", "artifacts/scheduler/state.json")
            self.run_logs = RunLogRetention(self.project_root, now=self._now, log=self.log,
                                           protected_paths=(chosen, self.config_path))
            self.run_logs.maintain()
            self.state = SchedulerState(chosen, self._now())
            self._sync_arena_config()
            self._sync_points_arena_progress()
            self.log.info("已读取每日状态；调度器当前暂停，回到天星镇后点击启动。")
        except Exception:
            self._dispose()
            raise

    def _read_options(self):
        self._points_arena_enabled = points_arena_enabled(self._settings)
        values = self._settings.get("scheduler", {})
        self._poll = values.get("poll_interval", 1)
        self._arena_retry = values.get("arena_retry_seconds", 300)
        self._min_click = self._settings.get("daily_hero", {}).get("min_click_interval", 1)
        for name, value in (("poll_interval", self._poll), ("arena_retry_seconds", self._arena_retry),
                            ("min_click_interval", self._min_click)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise AutomationError(f"调度器 {name} 必须是有限正数。")
        self._starts = self._settings.get("slay_dragon", {}).get("start_times", DEFAULT_START_TIMES)
        parse_start_times(self._starts)

    def _sync_arena_config(self):
        value = self._settings.get("check_arena", {}).get("arena_expired_time")
        expiry = parse_timestamp(value, "check_arena.arena_expired_time")
        if expiry is not None:
            expiry = expiry.replace(microsecond=0)
        raw = self.state.snapshot(self._now())["arena"]
        # A script or manual edit may have updated the configuration while paused.
        if expiry is not None and parse_timestamp(raw["next_due"]) != expiry:
            self.state.set_arena(expiry, now=self._now())

    def _reload(self):
        with self._lock:
            self._settings = load_config(self.config_path, include_local=self.include_local)
            self._read_options()
            self._sync_arena_config()
            self._sync_points_arena_progress()

    def _sync_points_arena_progress(self):
        if not self._points_arena_enabled:
            self.state.clear_points_arena_progress(self._now())

    def set_points_arena_enabled(self, enabled: bool):
        with self._lock:
            alive = bool(self._thread and self._thread.is_alive())
            if self._closing or alive or self._status not in {"paused", "error", "idle"}:
                raise AutomationError("请先暂停并等待当前任务完全停止，再修改是否积分赛。")
            try:
                settings = persist_points_arena_enabled(
                    self.config_path, enabled, include_local=self.include_local,
                    after_save=(lambda: self.state.clear_points_arena_progress(self._now()))
                    if enabled is False else None,
                )
            except Exception as exc:
                # A failed state cleanup normally restores the original file.
                # If rollback itself failed, show the actual saved switch value.
                try:
                    actual = load_config(self.config_path, include_local=self.include_local)
                    actual_enabled = points_arena_enabled(actual)
                except Exception:
                    pass
                else:
                    self._settings, self._points_arena_enabled = actual, actual_enabled
                self._reason = str(exc)
                self.log.error("保存积分赛开关失败：%s", exc)
                raise
            self._settings = settings
            self._points_arena_enabled = enabled
            self.log.info("英雄级副本及英雄重置积分赛：%s。", "开启" if enabled else "关闭")

    def _choose(self) -> str | None:
        with self._lock:
            current = self._now()
            raw = self.state.snapshot(current)
            arena = raw["arena"]
            return choose_task(current, raw["daily"], arena["next_due"], arena["retry_after"],
                               arena["checked"], self._starts, self._points_arena_enabled)

    def snapshot(self) -> dict:
        with self._lock:
            current = self._now()
            self._disk_log.maintain()
            if not self._thread or not self._thread.is_alive():
                self.run_logs.maintain()
            raw = self.state.snapshot(current)
            worker_alive = bool(self._thread and self._thread.is_alive())
            scripts = []
            for script in self._scripts():
                record = raw.get("single_runs", {}).get(script["name"], {})
                status, reason = record.get("status", "idle"), record.get("reason", "")
                if self._mode == "single" and self._single_script == script["name"]:
                    status = "stopping" if self._pause.is_set() else self._status
                    reason = self._reason
                scripts.append({**script, "status": status, "reason": reason})
            return {"status": self._status, "reason": self._reason, "date": raw["date"],
                    "current_task": self._current_task, "tasks": raw["daily"],
                    "mode": self._mode, "single_script": self._single_script, "scripts": scripts,
                    "arena_expired_time": raw["arena"]["next_due"],
                    "next_dragon_start": next_dragon_start(current, self._starts).isoformat(),
                    "worker_alive": worker_alive,
                    "points_arena_enabled": self._points_arena_enabled,
                    "can_edit_points_arena": not self._closing and not worker_alive
                    and self._status in {"paused", "error", "idle"}}

    def _scripts_directory(self) -> Path:
        return self.project_root / self._settings.get("automation", {}).get("scripts_dir", "scripts")

    def _scripts(self) -> list[dict]:
        return script_catalog(self._scripts_directory())

    def start(self):
        with self._lock:
            if self._closing:
                raise AutomationError("窗口正在关闭。")
            if self._thread and self._thread.is_alive():
                return
            self._pause.clear()
            self._mode, self._single_script = "scheduler", None
            self._status, self._reason = "starting", "正在检查配置和模拟器。"
            self._needs_town = True
            self._thread = threading.Thread(target=self._worker, name="mxm-scheduler", daemon=False)
            self._thread.start()

    def start_script(self, name: str):
        with self._lock:
            if self._closing:
                raise AutomationError("窗口正在关闭。")
            if self._thread and self._thread.is_alive():
                raise AutomationError("已有任务运行或正在停止，请等待当前任务完全停止。")
            self._reload()
            if name not in {script["name"] for script in self._scripts()}:
                raise AutomationError(f"找不到脚本 {name}。")
            self._pause.clear()
            self._mode, self._single_script = "single", name
            self._status, self._reason = "starting", f"正在启动单任务：{name}。"
            self._thread = threading.Thread(target=self._single_worker, args=(name,),
                                            name=f"mxm-single-{name}", daemon=False)
            self._thread.start()

    def stop_script(self, name: str):
        with self._lock:
            if self._mode != "single" or self._single_script != name:
                return
            self.pause()

    def pause(self):
        self._pause.set()
        with self._lock:
            alive = self._thread and self._thread.is_alive()
            self._status = "pausing" if alive else "paused"
            self._reason = "正在停止自动输入并保存进度。" if alive else "已暂停；回到天星镇后可再次启动。"

    def close(self):
        with self._lock:
            self._closing = True
        self.pause()
        if not self._thread or not self._thread.is_alive():
            with self._lock:
                self._status = "closed"
                self._dispose()

    def _dispose(self):
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)
        self._lease.close()

    def _interruption(self, safe: bool = False):
        if self._pause.is_set():
            raise TaskInterrupted("paused", "用户暂停，已停止自动输入。")
        if self._mode == "single":
            return
        task = self._current_task
        if task in PRIORITY or task == "return_to_town":
            with self._lock:
                midnight_due = self.state.midnight_restart_due(self._now())
            if midnight_due:
                raise TaskInterrupted("preempted", "北京时间已过零点，先重启游戏再继续调度。")
        if task not in PRIORITY or task == "check_arena":
            return
        if task in DAILY_TASKS and not safe:
            return
        if task in DAILY_TASKS and self._attempt_day != beijing_time(self._now()).date().isoformat():
            raise TaskInterrupted("preempted", "北京时间已过零点，重新检查今日副本。")
        selected = self._choose()
        if task == "slay_dragon" and selected != task:
            raise TaskInterrupted("preempted", "擂台到期或屠龙时段结束，切换任务。")
        if selected in PRIORITY and PRIORITY[selected] < PRIORITY[task]:
            raise TaskInterrupted("preempted", f"更高优先级任务已就绪：{selected}。")

    def _control(self, operation: str):
        self.run_logs.maintain()
        self._interruption()
        # Wait before capturing, so a task's first click uses a fresh location
        # even when the preceding task has just clicked a moving control.
        if operation in {"capture", "restart_app"}:
            delay = self._min_click - (time.monotonic() - self._last_input)
            if delay > 0:
                self._pause.wait(delay)
                self._interruption()
        if operation in {"tap", "swipe", "key", "restart_app"}:
            self._last_input = time.monotonic()

    def _checkpoint(self, name: str, payload: dict):
        with self._lock:
            task = self._current_task
            if task not in DAILY_TASKS or self._attempt is None:
                return
            progress = _merge_progress(self._baseline, payload.get("report", {}))
            progress["checkpoint"] = name
            progress["completed_since_arena"] = payload.get("completed_since_arena", 0)
            self.state.record_progress(task, progress, self._now(), attempt_id=self._attempt)
        self._interruption(safe=name != "hero_points_pending")

    def _context(self):
        self._control("connect")
        if self._context_factory:
            ctx = self._context_factory(self)
        else:
            device_options = self._settings.get("device", {})
            device = AndroidDevice(find_adb(self._adb_path or device_options.get("adb_path")),
                                   self._serial or device_options.get("serial") or None,
                                   float(device_options.get("timeout", 30)))
            device.before_command = lambda: self._control("adb")
            device.connect()
            ctx = ScriptContext(device, self.project_root / self._settings.get("automation", {}).get(
                "screenshots_dir", "artifacts/screenshots"), settings=self._settings,
                project_root=self.project_root, config_path=self.config_path,
                use_local_config=self.include_local, run_logs=self.run_logs)
        ctx.control, ctx.checkpoint_callback, ctx.log = self._control, self._checkpoint, self.log
        return ctx

    def _restart_for_midnight(self, ctx) -> bool:
        restarted = False
        while True:
            self._control("midnight_restart_check")
            with self._lock:
                current = self._now()
                if not self.state.midnight_restart_due(current):
                    return restarted
                day = beijing_time(current).date().isoformat()
                self._current_task = "midnight_restart"
                self._status, self._reason = "running", "北京时间零点刷新，先重启游戏并登录天星镇。"
            self.log.info("%s 零点重启：先重启游戏并登录天星镇。", day)
            try:
                self._navigator_factory(ctx).restart_to_town()
                with self._lock:
                    self.state.complete_midnight_restart(day, self._now())
                    self._needs_town = False
                restarted = True
                self.log.info("%s 零点重启完成，继续按任务优先级调度。", day)
            finally:
                self._current_task = None

    def cycle(self, ctx) -> str | None:
        """Execute one task; exposed for deterministic clock/flow verification."""
        self._control("cycle")
        self._reload()
        ctx.settings = self._settings
        restarted = self._restart_for_midnight(ctx)
        # An idle period may last hours; do not reuse a town confirmation from
        # before a disconnect when a new task eventually becomes ready.
        if not restarted and (self._needs_town or self._choose() is not None):
            with self._lock:
                self._current_task = "return_to_town"
                self._reason = "确认返回天星镇后再启动任务。"
            try:
                self._navigator_factory(ctx).ensure_town()
                self._needs_town = False
            except TaskInterrupted as exc:
                if exc.status != "preempted" or not self.state.midnight_restart_due(self._now()):
                    raise
                self.log.info("回城导航已跨零点，停止旧画面操作，先重启游戏。")
            finally:
                self._current_task = None
        # Ordinary return navigation can itself cross midnight. Restart before
        # choosing a job, even when town was just reached by another route.
        self._restart_for_midnight(ctx)
        task = self._choose()
        if task is None:
            with self._lock:
                self._status, self._reason = "idle", "今日副本已完成，等待擂台到期、屠龙开场或零点刷新。"
            return None
        with self._lock:
            current = self._now()
            self._attempt = self.state.start_task(task, current)
            self._attempt_day = beijing_time(current).date().isoformat()
            self._current_task = task
            self._baseline = self.state.snapshot(current)["daily"].get(task, {}).get("progress", {})
            self._status, self._reason = "running", f"正在执行 {task}。"
        ctx.resume_data = deepcopy(self._baseline)
        with self._lock:
            hero_state = self.state.snapshot(self._now())
            ctx.hero_resume_data = deepcopy(hero_state["daily"]["daily_hero"]["progress"])
            ctx.hero_resume_date = hero_state["date"]
        self.log.info("启动任务：%s", task)
        self._needs_town = True
        report, failure = {}, None
        try:
            report = self._task_runner(task, ctx)
        except Exception as exc:
            failure = exc
            report = dict(getattr(exc, "report", None) or {})
            report.update(status=getattr(exc, "status", report.get("status", "stopped")), reason=str(exc))
            if getattr(exc, "run_dir", None):
                report["run_dir"] = exc.run_dir
        with self._lock:
            if task == "check_arena":
                # Clearing a guardian's prerequisite is real daily hero work,
                # but it does not complete the entire daily hero route.
                by_day = report.get("hero_progress_by_day") or {}
                if not by_day and report.get("hero_progress"):
                    by_day = {report["hero_progress_date"]: report["hero_progress"]}
                for day, increment in by_day.items():
                    saved = self.state.snapshot(self._now())
                    records = saved["daily"] if saved["date"] == day else saved["history"].get(day, {})
                    previous = records.get("daily_hero", {}).get("progress", {})
                    progress = _merge_progress(previous, increment)
                    self.state.record_daily_sidework("daily_hero", progress, self._now(),
                                                     day=day, attempt_id=self._attempt)
            if task in DAILY_TASKS:
                cumulative = _merge_progress(self._baseline, report)
                report = {**cumulative, **{key: report[key] for key in ("status", "reason") if key in report}}
                # Keep the latest safe every-three counter, even on an abrupt pause.
                saved = self.state.snapshot(self._now())
                records = saved["daily"] if saved["date"] == self._attempt_day else saved["history"].get(self._attempt_day, {})
                report["completed_since_arena"] = records.get(task, {}).get("progress", {}).get("completed_since_arena", 0)
            self.state.finish_task(task, report, self._now(), attempt_id=self._attempt)
            self._attempt = None
            self._current_task = None
            self._backoff_arena_report(task, report)
        self.log.info("任务结束：%s，状态：%s%s", task, report.get("status"), f"；{report['reason']}" if report.get("reason") else "")
        if isinstance(failure, TaskInterrupted):
            if failure.status == "paused":
                raise failure
            return task
        if (task == "check_arena" and report.get("status") in {"exhausted", "blocked"}
                and not report.get("returned_to_town")):
            raise failure or AutomationError("擂台检查未能确认返回天星镇，已停止调度。")
        if failure and not (task == "check_arena" and report.get("status") in {"exhausted", "blocked"} and report.get("returned_to_town")):
            raise failure
        if report.get("status") not in {"completed", "exhausted", "blocked"}:
            raise AutomationError(f"{task} 未完整完成（{report.get('status')}），请检查运行记录后再启动。")
        return task

    def _backoff_arena_report(self, task: str, report: dict):
        """Use the same persisted arena retry policy for both run modes."""
        if (task == "check_arena" and report.get("status") in {"completed", "exhausted", "blocked"}
                and (report.get("status") == "completed" or report.get("returned_to_town"))):
            current = self._now()
            due = parse_timestamp(self.state.snapshot(current)["arena"]["next_due"])
            if report.get("status") in {"exhausted", "blocked"} or due is None or due <= beijing_time(current):
                retry = self.state.backoff_arena(self._arena_retry, current)
                self.log.info("本轮暂无可占擂台，或英雄前置未通过，%s 后再检查；继续其他任务。", retry)

    def _worker(self):
        try:
            self._reload()
            ctx = self._context()
            self.log.info("调度器启动：擂台 > 屠龙 > 英雄级副本 > 英雄级副本 reset。")
            while not self._pause.is_set():
                task = self.cycle(ctx)
                if task is None:
                    self._pause.wait(min(self._poll, 5))
        except TaskInterrupted as exc:
            self.log.info("%s", exc)
        except Exception as exc:
            self.log.exception("调度器停止：%s", exc)
            with self._lock:
                self._status, self._reason = "error", str(exc)
        finally:
            with self._lock:
                self._current_task = None
                if self._status != "error":
                    self._status = "closed" if self._closing else "paused"
                    self._reason = "已停止自动输入；手动操作结束并回到天星镇后可再次启动。"
                if self._closing:
                    self._dispose()
                self._mode, self._single_script = None, None

    def _single_worker(self, name: str):
        """Run exactly one requested script, sharing the scheduler's input gate."""
        started = self._now()
        task = task_for_script(name)
        report: dict = {}
        failure = None
        try:
            with self._lock:
                self.state.record_single_script(name, {"status": "running"}, started, self._now())
            ctx = self._context()
            if task:
                self._current_task = "return_to_town"
                self._reason = "确认天星镇后启动单任务。"
                self._navigator_factory(ctx).ensure_town()
                self._control("single_task_start")
                with self._lock:
                    started = self._now()
                    saved = self.state.snapshot(started)
                    record = saved["daily"].get(task, {})
                    self._attempt = self.state.start_task(task, started, manual=True)
                    self._attempt_day = beijing_time(started).date().isoformat()
                    self._baseline = deepcopy(record.get("progress", {}))
                    ctx.resume_data = {} if record.get("done") else deepcopy(self._baseline)
                    ctx.hero_resume_data = deepcopy(saved["daily"]["daily_hero"]["progress"])
                    ctx.hero_resume_date = saved["date"]
            self._current_task = task or name
            self._status, self._reason = "running", f"正在单独运行：{name}。"
            self.log.info("单任务启动：%s", name)
            if task:
                report = self._task_runner(task, ctx)
            else:
                self._control("single_script_start")
                self._script_runner(name, self._scripts_directory(), ctx)
                report = {"status": "completed"}
        except Exception as exc:
            failure = exc
            report = dict(getattr(exc, "report", None) or {})
            report.update(status=getattr(exc, "status", report.get("status", "stopped")), reason=str(exc))
            if getattr(exc, "run_dir", None):
                report["run_dir"] = str(exc.run_dir)
        finally:
            try:
                with self._lock:
                    if task and self._attempt is not None:
                        self._finish_single_business(task, report)
                    if self._pause.is_set() and report.get("status") == "completed":
                        # A runner that returned while the user stopped must not
                        # advertise continued activity. Its confirmed report is
                        # still kept in daily state before this UI-only result.
                        report = {**report, "status": "paused"}
                    self.state.record_single_script(name, report, started, self._now())
            except Exception as exc:
                failure = exc
                self.log.exception("保存单任务结果失败：%s", exc)
            with self._lock:
                self._attempt = self._attempt_day = None
                self._current_task = None
                self._needs_town = True
                if failure and not isinstance(failure, TaskInterrupted):
                    self._status, self._reason = "error", str(failure)
                    self.log.error("单任务停止：%s；%s", name, failure)
                else:
                    self._status = "closed" if self._closing else "paused"
                    self._reason = (f"单任务 {name} 已结束；状态：{report.get('status', 'stopped')}。"
                                    "回到天星镇后可再次启动。")
                    self.log.info("%s", self._reason)
                self._mode, self._single_script = None, None
                if self._closing:
                    self._dispose()

    def _finish_single_business(self, task: str, report: dict):
        """Commit the manual run's real work, retaining an already completed day."""
        if task == "check_arena":
            by_day = report.get("hero_progress_by_day") or {}
            if not by_day and report.get("hero_progress"):
                by_day = {report["hero_progress_date"]: report["hero_progress"]}
            for day, increment in by_day.items():
                saved = self.state.snapshot(self._now())
                records = saved["daily"] if saved["date"] == day else saved["history"].get(day, {})
                progress = _merge_progress(records.get("daily_hero", {}).get("progress", {}), increment)
                self.state.record_daily_sidework("daily_hero", progress, self._now(),
                                                 day=day, attempt_id=self._attempt)
        if task in DAILY_TASKS:
            cumulative = _merge_progress(self._baseline, report)
            report.update(cumulative)
            saved = self.state.snapshot(self._now())
            records = saved["daily"] if saved["date"] == self._attempt_day else saved["history"].get(self._attempt_day, {})
            report["completed_since_arena"] = records.get(task, {}).get("progress", {}).get("completed_since_arena", 0)
        self.state.finish_task(task, report, self._now(), attempt_id=self._attempt)
        self._backoff_arena_report(task, report)
