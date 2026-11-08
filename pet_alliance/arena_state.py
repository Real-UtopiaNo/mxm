"""Persist the last confirmed dungeon guardianship without rewriting user settings."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import tomllib

from .adb import AutomationError


BEIJING = timezone(timedelta(hours=8))
_SECTION = "check_arena"
_KEYS = ("arena_expired_time", "arena_dungeon")


def _statements(source: str):
    """Yield TOML statements, ignoring newlines inside strings and collections."""
    start = index = depth = 0
    quote = None
    multiline = False
    comment = False
    while index < len(source):
        character = source[index]
        if comment:
            if character == "\n":
                comment = False
            else:
                index += 1
                continue
        if quote:
            if quote == '"' and character == "\\":
                index += 2
                continue
            if character == quote:
                if multiline:
                    if source.startswith(quote * 3, index):
                        end = index + 3
                        # TOML permits one or two literal quotes before closing.
                        while end < len(source) and source[end] == quote:
                            end += 1
                        index = end
                        quote = None
                        continue
                else:
                    quote = None
            index += 1
            continue
        if character in "\"'":
            quote = character
            multiline = source.startswith(character * 3, index)
            index += 3 if multiline else 1
            continue
        if character == "#":
            comment = True
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
        elif character == "\n" and depth == 0:
            yield start, index + 1, source[start:index + 1]
            start = index + 1
        index += 1
    if start < len(source):
        yield start, len(source), source[start:]


def _path_to_value(data: dict, value) -> tuple[str, ...] | None:
    path = []
    while isinstance(data, dict) and len(data) == 1:
        key, data = next(iter(data.items()))
        path.append(key)
    return tuple(path) if data == value else None


def _assignment(statement: str) -> tuple[int, tuple[str, ...]] | None:
    quote = None
    escaped = False
    for index, character in enumerate(statement):
        if escaped:
            escaped = False
            continue
        if quote:
            if quote == '"' and character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "=":
            keys = tomllib.loads(statement[:index] + " = 0")
            path = _path_to_value(keys, 0)
            return (index, path) if path else None
        elif character == "#":
            return None
    return None


def _value_comment(value: str) -> int | None:
    """Find an inline comment outside strings; field values may be multiline."""
    quote = None
    multiline = False
    depth = index = 0
    while index < len(value):
        character = value[index]
        if quote:
            if quote == '"' and character == "\\":
                index += 2
                continue
            if character == quote:
                if multiline:
                    if value.startswith(quote * 3, index):
                        index += 3
                        while index < len(value) and value[index] == quote:
                            index += 1
                        quote = None
                        continue
                else:
                    quote = None
        elif character in "\"'":
            quote = character
            multiline = value.startswith(character * 3, index)
            if multiline:
                index += 2
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
        elif character == "#":
            if depth == 0:
                return index
            newline = value.find("\n", index)
            if newline < 0:
                return None
            index = newline
        index += 1
    return None


def _replace_value(statement: str, equals: int, value: str) -> str:
    ending = "\r\n" if statement.endswith("\r\n") else "\n" if statement.endswith("\n") else ""
    old = statement[equals + 1:]
    leading = re.match(r"[ \t]*", old).group()
    comment_at = _value_comment(old)
    comment = ""
    if comment_at is not None:
        spacing = re.search(r"[ \t]*$", old[:comment_at]).group()
        comment = (spacing or " ") + old[comment_at:].rstrip("\r\n")
    return statement[:equals + 1] + leading + value + comment + ending


def _same_values(left, right) -> bool:
    # TOML supports nan, whose ordinary equality check is always false.
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_values(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_values(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _updated_text(source: str, values: dict, *, section_name: str = _SECTION) -> str:
    """Edit selected assignments; validate that all unrelated TOML is retained."""
    data = tomllib.loads(source)
    section = data.get(section_name, {})
    if not isinstance(section, dict):
        raise AutomationError(f"配置中的 [{section_name}] 必须是一个表。")
    expected = deepcopy(data)
    expected.setdefault(section_name, {}).update(values)
    newline = "\r\n" if "\r\n" in source else "\n"
    replacements = []
    found = set()
    current = ()
    section_end = None
    first_header = len(source)
    in_section = False
    for start, end, statement in _statements(source):
        stripped = statement.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("["):
            first_header = min(first_header, start)
            if in_section:
                section_end = start
            header_data = tomllib.loads(statement + "\n__arena_state_marker__ = 0\n")
            header_path = _path_to_value(header_data, 0)
            current = header_path[:-1] if header_path else None
            in_section = current == (section_name,)
            if in_section:
                section_end = len(source)
            continue
        assignment = _assignment(statement)
        if assignment is None or current is None:
            continue
        equals, keys = assignment
        path = current + keys
        if len(path) == 2 and path[0] == section_name and path[1] in values:
            field = path[1]
            replacements.append((start, end, _replace_value(
                statement, equals, json.dumps(values[field], ensure_ascii=False),
            )))
            found.add(field)
    missing = [key for key in values if key not in found]
    if missing:
        if section_end is not None:
            insert_at = section_end
            addition = "".join(f"{key} = {json.dumps(values[key], ensure_ascii=False)}{newline}" for key in missing)
        elif section_name in data:
            # A table created with dotted keys must keep using dotted keys.
            insert_at = first_header
            addition = "".join(f"{section_name}.{key} = {json.dumps(values[key], ensure_ascii=False)}{newline}" for key in missing)
        else:
            insert_at = len(source)
            addition = f"{newline}[{section_name}]{newline}" + "".join(
                f"{key} = {json.dumps(values[key], ensure_ascii=False)}{newline}" for key in missing
            )
        if insert_at and source[insert_at - 1] != "\n":
            addition = newline + addition
        replacements.append((insert_at, insert_at, addition))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    try:
        actual = tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise AutomationError(f"无法保留现有 TOML 结构并更新配置，请将 {section_name} 写为独立表。") from exc
    if not _same_values(actual, expected):
        raise AutomationError("擂主时间配置校验失败：更新会影响其他配置，已取消保存。")
    return source


def persist_arena_state(ctx, expired_time: datetime, dungeon: str) -> Path:
    """Save a freshly confirmed guardianship; retain old state on any failure.

    Default configuration uses automation.local.toml. Explicit --config runs
    set use_local_config=False and update that exact configuration file.
    """
    if not isinstance(expired_time, datetime) or expired_time.utcoffset() is None:
        raise AutomationError("擂主到期时间必须是带时区的 datetime。")
    if not isinstance(dungeon, str) or not dungeon.strip():
        raise AutomationError("擂主副本名必须是非空字符串。")
    if getattr(ctx, "dry_run", False):
        raise AutomationError("预演模式不能保存新的擂主到期时间。")
    if not isinstance(ctx.settings, dict) or not isinstance(ctx.settings.get(_SECTION, {}), dict):
        raise AutomationError("配置中的 [check_arena] 必须是一个表。")
    try:
        timestamp = expired_time.astimezone(BEIJING).isoformat(timespec="seconds")
    except (OverflowError, ValueError) as exc:
        raise AutomationError(f"无法转换擂主到期时间为北京时间：{exc}") from exc
    values = {"arena_expired_time": timestamp, "arena_dungeon": dungeon}
    root = Path(ctx.project_root).resolve()
    configured = getattr(ctx, "config_path", None)
    path = Path(configured) if configured is not None else root / "automation.toml"
    if not path.is_absolute():
        path = root / path
    use_local = getattr(ctx, "use_local_config", True)
    config_path = path.resolve()
    if use_local:
        path = path.with_name("automation.local.toml")
    path = path.resolve()
    temporary = None
    try:
        if use_local and config_path != path:
            # The saved override must remain loadable with its actual base file.
            tomllib.loads(config_path.read_text(encoding="utf-8"))
        try:
            original = path.read_bytes()
            mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            if not use_local:
                raise AutomationError(f"找不到需要更新的配置文件：{path}")
            original = None
            mode = None
        source = original.decode("utf-8") if original is not None else ""
        result = _updated_text(source, values)
        encoded = result.encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            current = None
        if current != original:
            raise AutomationError(f"配置在保存期间发生变化，已保留现有文件，请重试：{path}")
        os.replace(temporary, path)
        temporary = None
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AutomationError(f"无法保存擂主到期时间到 {path}：{exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    ctx.settings.setdefault(_SECTION, {}).update(values)
    return path
