"""Configuration loading with explicit command-line overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return value


def flatten_sections(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if child_key in flat:
                    raise ValueError(f"Duplicate flattened configuration key: {child_key}")
                flat[child_key] = child_value
        else:
            flat[key] = value
    return flat
