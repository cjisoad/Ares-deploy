from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_rs02_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"RS02 config must be a YAML mapping: {path}")
    return config


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid RS02 config section: {name}")
    return section
