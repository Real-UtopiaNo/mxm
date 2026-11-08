"""Daily dragon start times, interpreted in Beijing time on every platform."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import re

from .adb import AutomationError


BEIJING = timezone(timedelta(hours=8), "Asia/Shanghai")
DEFAULT_START_TIMES = (
    "09:00", "10:00", "12:00", "13:00", "14:00",
    "15:00", "18:00", "19:00", "20:00", "21:00",
)


def parse_start_times(values: list[str] | tuple[str, ...]) -> tuple[time, ...]:
    """Validate HH:MM settings and return unique start times in daily order."""
    message = "slay_dragon.start_times 必须是非空的 HH:MM 字符串列表（例如 [\"09:00\", \"18:00\"]）。"
    if not isinstance(values, (list, tuple)) or not values:
        raise AutomationError(message)
    parsed = set()
    for value in values:
        if not isinstance(value, str) or re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is None:
            raise AutomationError(message)
        hour, minute = value.split(":")
        parsed.add(time(int(hour), int(minute)))
    return tuple(sorted(parsed))


def beijing_now() -> datetime:
    """Read the system clock as an aware Beijing datetime."""
    return datetime.now(BEIJING)


def is_dragon_active(
    now: datetime,
    start_times: list[str] | tuple[str, ...] = DEFAULT_START_TIMES,
) -> bool:
    """Check the supplied timetable's one-hour sessions, including their start.

    Each session lasts one hour, as shown in the game timetable. Its end is
    excluded; adjacent sessions therefore form one continuous active period.
    Naive inputs mean Beijing time, and a previous-day session may cross midnight.
    """
    times = parse_start_times(start_times)
    current = now.replace(tzinfo=BEIJING) if now.tzinfo is None else now.astimezone(BEIJING)
    for day in (current.date(), current.date() - timedelta(days=1)):
        for start in times:
            candidate = datetime.combine(day, start, BEIJING)
            if candidate <= current < candidate + timedelta(hours=1):
                return True
    return False


def next_dragon_start(
    now: datetime,
    start_times: list[str] | tuple[str, ...] = DEFAULT_START_TIMES,
) -> datetime:
    """Return the first start at or after now; naive inputs mean Beijing time."""
    times = parse_start_times(start_times)
    current = now.replace(tzinfo=BEIJING) if now.tzinfo is None else now.astimezone(BEIJING)
    for start in times:
        candidate = datetime.combine(current.date(), start, BEIJING)
        if candidate >= current:
            return candidate
    return datetime.combine(current.date() + timedelta(days=1), times[0], BEIJING)
