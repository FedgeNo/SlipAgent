"""Inspection preferences share the existing private preferences file."""

from __future__ import annotations

import json
import shlex
import shutil
from typing import Any

from .config import ConfigError, _read_preferences, _preferences_path, save_private_text


def load_ui_preferences() -> dict[str, Any]:
    value = _read_preferences().get("inspection", {})
    if not isinstance(value, dict):
        raise ConfigError("Invalid inspection preferences")
    if "wrap" in value and not isinstance(value["wrap"], bool):
        raise ConfigError("Editor wrap preference must be a boolean")
    if "tab_width" in value and value["tab_width"] not in (2, 4, 8):
        raise ConfigError("Editor tab width must be 2, 4, or 8")
    if "tab_width" in value:
        value = {**value, "tab_width": int(value["tab_width"])}
    if "folder_opener" in value and not isinstance(value["folder_opener"], str):
        raise ConfigError("Folder opener preference must be text")
    return dict(value)


def save_ui_preference(name: str, value: Any) -> None:
    preferences = _read_preferences()
    inspection = preferences.setdefault("inspection", {})
    if not isinstance(inspection, dict):
        raise ConfigError("Invalid inspection preferences")
    inspection[name] = value
    save_private_text(_preferences_path(), json.dumps(preferences, ensure_ascii=False) + "\n")


def configured_folder_opener(command: str) -> tuple[str, ...] | None:
    if not command.strip():
        return None
    parts = shlex.split(command)
    if not parts:
        return None
    if any("{path}" in part and part != "{path}" for part in parts):
        raise ConfigError("Use {path} as a separate argument")
    executable = shutil.which(parts[0])
    if executable is None:
        raise ConfigError(f"Folder opener was not found: {parts[0]}")
    return (executable, *parts[1:])
