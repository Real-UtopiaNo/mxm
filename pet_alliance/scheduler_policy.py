"""Pure scheduling decisions, interpreted against the Beijing game clock."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from .adb import AutomationError
from .dragon_schedule import BEIJING, DEFAULT_START_TIMES, is_dragon_active


DAILY_TASKS = ("daily_hero", "daily_hero_reset")
TASKS = ("check_arena", "slay_dragon", *DAILY_TASKS)


def beijing_time(value: datetime) -> datetime:
    """Treat a naive system-clock value as Beijing time, never host local time."""
    return value.replace(tzinfo=BEIJING) if value.tzinfo is None else value.astimezone(BEIJING)


def parse_timestamp(value: str | datetime | None, field: str = "时间") -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return beijing_time(value)
    if not isinstance(value, str):
        raise AutomationError(f"{field} 必须是带日期的 ISO 时间字符串。")
    try:
        return beijing_time(datetime.fromisoformat(value))
    except ValueError as exc:
        raise AutomationError(f"{field} 不是有效的 ISO 时间：{value!r}。") from exc


def _done(value: object, today: str) -> bool:
    if isinstance(value, Mapping):
        return value.get("done") is True and value.get("date", today) == today
    return value is True


def choose_task(
    now: datetime,
    daily_flags: Mapping[str, object],
    arena_expired_time: str | datetime | None = None,
    arena_retry_at: str | datetime | None = None,
    arena_checked: bool = False,
    start_times: list[str] | tuple[str, ...] = DEFAULT_START_TIMES,
    points_arena_enabled: bool = True,
) -> str | None:
    """Select arena > active dragon > unfinished hero > unfinished reset.

    An unknown arena deadline requires a check. A retry deadline only suppresses
    arena work; other tasks can continue during the backoff. ``arena_checked``
    describes persisted inspection state and does not turn an unknown deadline
    into an indefinite exemption from future checks.
    """
    current = beijing_time(now)
    expiry = parse_timestamp(arena_expired_time, "arena_expired_time")
    retry_at = parse_timestamp(arena_retry_at, "arena_retry_at")
    hero = daily_flags.get("daily_hero")
    progress = hero.get("progress") if isinstance(hero, Mapping) else None
    pending_points = (isinstance(progress, Mapping) and progress.get("points_arena_pending") is True
                      and hero.get("date", current.date().isoformat()) == current.date().isoformat())
    if points_arena_enabled and pending_points:
        # Resume the unfinished arena-related visit before selecting another
        # long-running job, even if a prior guardian deadline is still future.
        return "check_arena"
    arena_due = expiry is None or current >= expiry
    if arena_due and (retry_at is None or current >= retry_at):
        return "check_arena"
    if is_dragon_active(current, start_times):
        return "slay_dragon"
    today = current.date().isoformat()
    for task in DAILY_TASKS:
        if not _done(daily_flags.get(task), today):
            return task
    return None
