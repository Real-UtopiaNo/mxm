"""Atomic, restart-safe daily completion records for the game scheduler."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from .adb import AutomationError
from .dragon_schedule import beijing_now
from .scheduler_policy import DAILY_TASKS, TASKS, beijing_time, parse_timestamp


def _stamp(now: datetime) -> str:
    return beijing_time(now).isoformat(timespec="seconds")


def _daily_record(day: str) -> dict:
    return {"date": day, "done": False, "status": "pending", "progress": {},
            "latest_report": None, "last_attempt": None}


def _daily_records(day: str) -> dict:
    return {task: _daily_record(day) for task in DAILY_TASKS}


def _report_dict(report: Mapping | object) -> dict:
    result = asdict(report) if is_dataclass(report) else dict(report)
    # Validate JSON compatibility before altering the in-memory state.
    return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))


class SchedulerState:
    """One process owns this store; its caller enforces the process lock.

    A completed result belongs to the Beijing date on which its attempt began.
    This prevents a run crossing midnight from completing the new day's work.
    Only a literal ``status == 'completed'`` records a finished daily task.
    """

    def __init__(self, path: Path | str, now: datetime | None = None):
        self.path = Path(path)
        current = now or beijing_now()
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text(encoding="utf-8-sig"))
                self._validate()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise AutomationError(f"调度器状态文件无法读取，已保留原文件：{self.path}；{exc}") from exc
            # Migrate before rolling the daily records over.  Their old date is
            # the baseline, so a paused scheduler still owes its midnight restart.
            changed = "midnight_restart" not in self._state
            if changed:
                self._state["midnight_restart"] = {"last_completed_date": self._state["date"]}
            changed = self._recover_running(current) or changed
        else:
            day = beijing_time(current).date().isoformat()
            self._state = {"version": 1, "date": day, "daily": _daily_records(day),
                           "history": {}, "tasks": {"slay_dragon": {"status": "pending", "progress": {},
                           "latest_report": None, "last_attempt": None}}, "arena": {"next_due": None, "retry_after": None,
                           "checked": False, "status": "pending", "last_attempt": None,
                           "last_checked": None, "latest_report": None}, "running": None,
                           "midnight_restart": {"last_completed_date": day}}
            changed = True
        if not self.refresh_day(current) and changed:
            self._save()

    def _validate(self) -> None:
        state = self._state
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("不支持的调度器状态格式")
        date.fromisoformat(state["date"])
        if not isinstance(state["history"], dict) or not isinstance(state["arena"], dict):
            raise ValueError("状态记录格式错误")
        self._validate_daily(state["daily"], state["date"])
        for day, records in state["history"].items():
            date.fromisoformat(day)
            self._validate_daily(records, day)
        arena = state["arena"]
        parse_timestamp(arena.get("next_due"), "arena.next_due")
        parse_timestamp(arena.get("retry_after"), "arena.retry_after")
        if "midnight_restart" in state:
            restart = state["midnight_restart"]
            if not isinstance(restart, dict):
                raise ValueError("零点重启记录格式错误")
            completed_day = restart.get("last_completed_date")
            if not isinstance(completed_day, str) or date.fromisoformat(completed_day).isoformat() != completed_day:
                raise ValueError("零点重启完成日期无效")
        if not isinstance(state.setdefault("tasks", {"slay_dragon": {"status": "pending", "progress": {},
                                                    "latest_report": None, "last_attempt": None}}), dict):
            raise ValueError("后台任务记录格式错误")
        single_runs = state.get("single_runs", {})
        if (not isinstance(single_runs, dict)
                or any(not isinstance(record, dict) for record in single_runs.values())):
            raise ValueError("单任务运行记录格式错误")
        running = state.get("running")
        if running is not None:
            if not isinstance(running, dict) or running.get("task") not in TASKS:
                raise ValueError("运行中任务记录格式错误")
            date.fromisoformat(running["date"])
            if not isinstance(running.get("attempt_id"), str):
                raise ValueError("运行中任务标识缺失")

    @staticmethod
    def _validate_daily(records: object, day: str) -> None:
        if not isinstance(records, dict):
            raise ValueError("每日任务记录缺失")
        for task in DAILY_TASKS:
            record = records[task]
            if not isinstance(record, dict) or record.get("date") != day or type(record.get("done")) is not bool:
                raise ValueError(f"{task} 的日期或完成标记无效")
            if record["done"] and record.get("status") != "completed":
                raise ValueError(f"{task} 的完成状态不一致")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             prefix=f".{self.path.name}.", suffix=".tmp",
                                             dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(self._state, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _records_for(self, day: str) -> dict:
        if day == self._state["date"]:
            return self._state["daily"]
        return self._state["history"].setdefault(day, _daily_records(day))

    def _recover_running(self, now: datetime) -> bool:
        changed = False
        for record in self._state.get("single_runs", {}).values():
            if record.get("status") == "running":
                record.update(status="interrupted", reason="上次窗口关闭时单任务尚未完成。",
                              finished_at=_stamp(now))
                changed = True
        running = self._state.get("running")
        if running is None:
            return changed
        task = running["task"]
        record = self._task_record(task, running["date"])
        if record is not None and not record.get("done", False):
            record["status"] = "interrupted"
            record["latest_report"] = {"status": "interrupted", "reason": "上次调度器结束时任务尚未完成。",
                                        "recovered_at": _stamp(now)}
        self._state["running"] = None
        return True

    def refresh_day(self, now: datetime | None = None) -> bool:
        day = beijing_time(now or beijing_now()).date().isoformat()
        if self._state["date"] == day:
            return False
        self._state["history"][self._state["date"]] = self._state["daily"]
        self._state["date"] = day
        self._state["daily"] = self._state["history"].pop(day, _daily_records(day))
        # Keep a month of daily evidence, including an older in-flight attempt.
        retain = set(sorted(self._state["history"])[-31:])
        if self._state["running"]:
            retain.add(self._state["running"]["date"])
        self._state["history"] = {key: value for key, value in self._state["history"].items() if key in retain}
        self._save()
        return True

    def snapshot(self, now: datetime | None = None) -> dict:
        self.refresh_day(now)
        return deepcopy(self._state)

    def daily_flags(self, now: datetime | None = None) -> dict[str, bool]:
        self.refresh_day(now)
        return {task: record["done"] for task, record in self._state["daily"].items()}

    def clear_points_arena_progress(self, now: datetime | None = None) -> bool:
        """Discard disabled points visits for today without erasing confirmed work."""
        self.refresh_day(now)
        previous = {}
        for task in DAILY_TASKS:
            record = self._state["daily"][task]
            progress = record.get("progress", {})
            if (progress.get("points_arena_pending") is True
                    or progress.get("completed_since_arena", 0) != 0):
                previous[task] = progress
                record["progress"] = {**progress, "points_arena_pending": False,
                                      "completed_since_arena": 0}
        if not previous:
            return False
        try:
            self._save()
        except BaseException:
            for task, progress in previous.items():
                self._state["daily"][task]["progress"] = progress
            raise
        return True

    def midnight_restart_due(self, now: datetime | None = None) -> bool:
        """Keep the midnight barrier independent of daily progress/UI refreshes."""
        today = beijing_time(now or beijing_now()).date().isoformat()
        return today > self._state["midnight_restart"]["last_completed_date"]

    def complete_midnight_restart(self, day: str, now: datetime | None = None) -> None:
        """Commit only after restart/login reached town, for its *start* date.

        A restart crossing another midnight must leave the newer day's barrier
        pending.  Repeated or older acknowledgements cannot move the record back.
        """
        current = now or beijing_now()
        today = beijing_time(current).date()
        try:
            parsed = date.fromisoformat(day)
            if parsed.isoformat() != day or parsed > today:
                raise ValueError("日期必须为 YYYY-MM-DD，且不能晚于当前北京时间日期")
        except (TypeError, ValueError) as exc:
            raise AutomationError(f"零点重启完成日期无效：{day!r}。") from exc
        previous = self._state["midnight_restart"]
        if day <= previous["last_completed_date"]:
            return
        self.refresh_day(current)
        self._state["midnight_restart"] = {"last_completed_date": day, "completed_at": _stamp(current)}
        try:
            self._save()
        except BaseException:
            # A failed commit must not let this process proceed through the gate.
            self._state["midnight_restart"] = previous
            raise

    def _task_record(self, task: str, day: str | None = None) -> dict:
        if task in DAILY_TASKS:
            return self._records_for(day or self._state["date"])[task]
        if task == "check_arena":
            return self._state["arena"]
        return self._state["tasks"].setdefault(task, {"status": "pending", "progress": {},
                                                    "latest_report": None, "last_attempt": None})

    def start_task(self, task: str, now: datetime | None = None, *, manual: bool = False) -> str:
        if task not in TASKS:
            raise AutomationError(f"未知的调度任务：{task}。")
        self.refresh_day(now)
        current = now or beijing_now()
        if self._state["running"] is not None:
            raise AutomationError("已有调度任务运行，不能同时启动另一任务。")
        record = self._task_record(task)
        was_done = record.get("done", False)
        if was_done and not manual:
            raise AutomationError(f"{task} 今天已经完成，不能重复调度。")
        # Re-running a completed route must not erase its confirmed completion,
        # including when the process exits before the manual attempt finishes.
        if not was_done:
            record["status"] = "running"
        record["last_attempt"] = _stamp(current)
        token = uuid4().hex
        self._state["running"] = {"task": task, "attempt_id": token, "date": self._state["date"],
                                  "started_at": _stamp(current), "manual": manual,
                                  "was_done": was_done}
        self._save()
        return token

    def _attempt(self, task: str, attempt_id: str | None) -> dict:
        running = self._state["running"]
        if running is None or running["task"] != task or (attempt_id is not None and running["attempt_id"] != attempt_id):
            raise AutomationError("调度任务结果与当前运行记录不一致，未更新状态。")
        return running

    def update_progress(self, task: str, progress: Mapping, now: datetime | None = None,
                        attempt_id: str | None = None) -> None:
        value = _report_dict(progress)
        running = self._attempt(task, attempt_id)
        self.refresh_day(now)
        self._task_record(task, running["date"])["progress"] = value
        self._save()

    def record_progress(self, task: str, progress: Mapping, now: datetime | None = None,
                        *, attempt_id: str | None = None) -> None:
        self.update_progress(task, progress, now, attempt_id)

    def record_daily_sidework(self, task: str, progress: Mapping, now: datetime | None = None,
                              *, day: str | None = None, attempt_id: str | None = None) -> None:
        """Replace accumulated daily progress produced inside another task.

        Arena prerequisites can finish individual hero dungeons without finishing
        the whole daily task. Their already-merged progress belongs to the day on
        which that work happened, independently of the enclosing arena attempt.
        """
        if task not in DAILY_TASKS:
            raise AutomationError(f"额外副本进度必须属于每日任务：{task}。")
        if not isinstance(progress, Mapping):
            raise AutomationError("额外副本进度必须是 JSON 对象。")
        value = _report_dict(progress)
        current = now or beijing_now()
        today = beijing_time(current).date()
        target_day = today.isoformat() if day is None else day
        try:
            parsed_day = date.fromisoformat(target_day)
            if parsed_day.isoformat() != target_day or parsed_day > today:
                raise ValueError("日期必须为 YYYY-MM-DD，且不能晚于当前北京时间日期")
        except (TypeError, ValueError) as exc:
            raise AutomationError(f"额外副本进度日期无效：{target_day!r}。") from exc
        if attempt_id is not None:
            running = self._state["running"]
            if running is None or running["attempt_id"] != attempt_id:
                raise AutomationError("额外副本进度与当前运行记录不一致，未更新状态。")
        self.refresh_day(current)
        self._records_for(target_day)[task]["progress"] = value
        self._save()

    def finish_task(self, task: str, report: Mapping | object, now: datetime | None = None,
                    attempt_id: str | None = None) -> dict:
        value = _report_dict(report)
        running = self._attempt(task, attempt_id)
        current = now or beijing_now()
        self.refresh_day(current)
        status = value.get("status", "stopped")
        if task in DAILY_TASKS:
            record = self._records_for(running["date"])[task]
            record["done"] = status == "completed" or bool(running.get("manual") and running.get("was_done"))
            record["status"] = "completed" if record["done"] else status
            record["latest_report"] = value
            record["progress"].update({key: deepcopy(item) for key, item in value.items()
                                       if key not in {"status", "reason"}})
            record["finished_at"] = _stamp(current)
        elif task == "check_arena":
            arena = self._state["arena"]
            arena.update(status=status, checked=True, last_checked=_stamp(current), latest_report=value)
            expiry = value.get("arena_expired_time")
            if status == "completed" and expiry:
                arena["next_due"] = _stamp(parse_timestamp(expiry, "arena_expired_time"))
                arena["retry_after"] = None
        else:
            record = self._task_record(task)
            record.update(status=status, latest_report=value, finished_at=_stamp(current))
        self._state["running"] = None
        self._save()
        return self.snapshot(current)

    def record_single_script(self, name: str, report: Mapping, started_at: datetime,
                             now: datetime | None = None) -> None:
        """Persist a script's latest manual attempt without changing daily history."""
        value = _report_dict(report)
        current = now or beijing_now()
        self.refresh_day(current)
        self._state.setdefault("single_runs", {})[name] = {
            "status": value.get("status", "stopped"), "reason": value.get("reason", ""),
            "date": beijing_time(started_at).date().isoformat(),
            "started_at": _stamp(started_at),
            "finished_at": None if value.get("status") == "running" else _stamp(current),
            "latest_report": value,
        }
        self._save()

    def set_arena(self, next_due: str | datetime | None = None,
                  retry_after: str | datetime | None = None, report: Mapping | None = None,
                  now: datetime | None = None) -> None:
        expiry = parse_timestamp(next_due, "arena.next_due")
        retry = parse_timestamp(retry_after, "arena.retry_after")
        value = _report_dict(report) if report is not None else None
        self.refresh_day(now)
        arena = self._state["arena"]
        arena["next_due"] = _stamp(expiry) if expiry is not None else None
        arena["retry_after"] = _stamp(retry) if retry is not None else None
        if value is not None:
            arena["latest_report"] = value
            arena["status"] = value.get("status", "stopped")
            arena["checked"] = True
            arena["last_checked"] = _stamp(now or beijing_now())
        self._save()

    def backoff_arena(self, seconds: float = 300, now: datetime | None = None) -> str:
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not 0 < seconds < float("inf"):
            raise AutomationError("擂台重试间隔必须是有限正数。")
        current = now or beijing_now()
        retry = beijing_time(current) + timedelta(seconds=seconds)
        arena = self._state["arena"]
        self.set_arena(arena["next_due"], retry, now=current)
        return _stamp(retry)
