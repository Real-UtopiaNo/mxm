"""Shared configuration with optional, untracked machine overrides."""

from pathlib import Path
import tomllib


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
