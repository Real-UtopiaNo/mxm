"""Shared configuration with optional, untracked machine overrides."""

import os
from pathlib import Path
import stat
import tempfile
import tomllib

from .adb import AutomationError
from .arena_state import _updated_text


def load_config(path: Path, *, include_local: bool = True) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    local = path.with_name("automation.local.toml")
    if include_local and local.is_file() and local != path:
        with local.open("rb") as stream:
            overrides = tomllib.load(stream)
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section].update(values)
            else:
                config[section] = values
    return config


def points_arena_enabled(settings: dict) -> bool:
    """The hero switch applies globally, including the reset route."""
    section = settings.get("daily_hero", {})
    if not isinstance(section, dict):
        raise AutomationError("配置中的 [daily_hero] 必须是一个表。")
    value = section.get("points_arena_enabled", True)
    if type(value) is not bool:
        raise AutomationError("daily_hero.points_arena_enabled 必须是布尔值 true 或 false。")
    return value


def _replace_config_bytes(path: Path, original: bytes | None, encoded: bytes | None) -> None:
    """Compare before committing, retaining original permissions and all bytes."""
    temporary = None
    try:
        if encoded is not None:
            mode = stat.S_IMODE(path.stat().st_mode) if original is not None else None
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if mode is not None:
                os.chmod(temporary, mode)
        current = path.read_bytes() if path.exists() else None
        if current != original:
            raise AutomationError(f"配置在保存期间发生变化，已保留现有文件，请重试：{path}")
        if encoded is None:
            path.unlink(missing_ok=True)
        else:
            os.replace(temporary, path)
            temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def persist_points_arena_enabled(config_path: Path, enabled: bool, *, include_local: bool = True,
                                 after_save=None) -> dict:
    """Preserve TOML while saving one shared hero/reset setting.

    The optional state update happens after the atomic config commit. If it fails,
    restore the original configuration without overwriting a concurrent edit.
    """
    if type(enabled) is not bool:
        raise AutomationError("是否积分赛必须是布尔值。")
    config_path = Path(config_path).resolve()
    path = config_path.with_name("automation.local.toml") if include_local else config_path
    try:
        settings = load_config(config_path, include_local=include_local)
        points_arena_enabled(settings)
        original = path.read_bytes() if path.exists() else None
        if original is None and not include_local:
            raise AutomationError(f"找不到需要更新的配置文件：{path}")
        source = original.decode("utf-8") if original is not None else ""
        updated = _updated_text(source, {"points_arena_enabled": enabled}, section_name="daily_hero")
        reset = settings.get("daily_hero_reset", {})
        if isinstance(reset, dict) and "points_arena_enabled" in reset:
            # A legacy per-reset override must not cancel the shared UI choice.
            updated = _updated_text(updated, {"points_arena_enabled": enabled},
                                    section_name="daily_hero_reset")
        encoded = updated.encode("utf-8")
        _replace_config_bytes(path, original, encoded)
        try:
            if after_save is not None:
                after_save()
        except Exception as exc:
            try:
                _replace_config_bytes(path, encoded, original)
            except Exception as rollback:
                raise AutomationError(f"积分赛配置已保存，但清理旧进度失败，恢复配置也失败：{rollback}；"
                                      f"请检查 {path}。原错误：{exc}") from exc
            raise AutomationError(f"无法清理积分赛旧进度，已恢复原配置：{exc}") from exc
        settings.setdefault("daily_hero", {})["points_arena_enabled"] = enabled
        if isinstance(reset, dict) and "points_arena_enabled" in reset:
            reset["points_arena_enabled"] = enabled
        return settings
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AutomationError(f"无法保存积分赛配置到 {path}：{exc}") from exc
