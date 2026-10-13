"""Named prompt sections rendered in one deterministic, inspectable order."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from string import Template
from typing import Any


class PromptError(RuntimeError):
    """An editable prompt cannot be loaded."""


def prompt_directory() -> Path:
    """Use the checkout's visible folder, or the installed package resources."""
    package = Path(__file__).resolve().parent
    checkout = package.parent.parent / "prompts"
    return checkout if package.parent.name == "src" and (package.parent.parent / "pyproject.toml").is_file() else package / "prompts"


def load_prompt(filename: str, **values: object) -> str:
    """Read and render a prompt with one newline on each side."""
    path = prompt_directory() / filename
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise PromptError(f"Missing prompt file: {path}. Restore this file in the prompts folder.") from None
    except OSError as exc:
        raise PromptError(f"Cannot read prompt file: {path}: {exc}") from exc
    rendered = Template(text).substitute({name: str(value) for name, value in values.items()}) if values else text
    return "\n" + rendered.strip() + "\n"


def __getattr__(name: str) -> str:
    resources = {"SYSTEM_PROMPT": "system-prompt.md", "COMPACTION_PROMPT": "background-summary-prompt.md"}
    if name not in resources:
        raise AttributeError(name)
    return load_prompt(resources[name])


def model_prompt_profile(model: str) -> dict[str, Any]:
    """Load an exact model's optional prompt sections and response settings."""
    try:
        profiles = json.loads(load_prompt("model-profiles.json"))
    except ValueError as exc:
        raise PromptError(f"Invalid model-profiles.json: {exc}") from exc
    if not isinstance(profiles, dict):
        raise PromptError("model-profiles.json must map model IDs to profiles")
    profile = profiles.get(model, {})
    if not isinstance(profile, dict) or profile.get("tool_channel") not in (None, "native", "embedded"):
        raise PromptError(f"Invalid prompt profile for {model}")
    sections = profile.get("sections", [])
    if not isinstance(sections, list) or any(not isinstance(path, str) for path in sections):
        raise PromptError(f"Prompt sections for {model} must be a list of filenames")
    if not isinstance(profile.get("xml_tool_calls", False), bool):
        raise PromptError(f"xml_tool_calls for {model} must be a boolean")
    if not isinstance(profile.get("structured_output", True), bool):
        raise PromptError(f"structured_output for {model} must be a boolean")
    if profile.get("stream_tool_identity", "delta") not in ("delta", "snapshot"):
        raise PromptError(f"stream_tool_identity for {model} must be delta or snapshot")
    if profile.get("stream_tool_indices", "index") not in ("index", "id"):
        raise PromptError(f"stream_tool_indices for {model} must be index or id")
    limits = profile.get("provider_context_limits", {})
    if not isinstance(limits, dict) or any(
            not isinstance(provider, str) or not provider.strip()
            or type(limit) is not int or limit <= 0 for provider, limit in limits.items()):
        raise PromptError(f"provider_context_limits for {model} must map provider tags to positive integer token limits")
    return profile


@dataclass(frozen=True)
class PromptSection:
    name: str
    title: str
    content: str
    order: int
    owner: str
    dynamic: bool


class PromptSections:
    """Per-request assembly; no mutable global registration survives reloads."""

    def __init__(self) -> None:
        self.sections: dict[str, PromptSection] = {}

    def add(self, name: str, title: str, content: str, order: int, *, owner: str = "agent", dynamic: bool = True) -> None:
        if name in self.sections:
            raise ValueError(f"Duplicate prompt section: {name}")
        self.sections[name] = PromptSection(name, title, content, order, owner, dynamic)

    def render(self) -> str:
        return "\n".join(f"{section.content.strip()}\n" for section in
                           sorted(self.sections.values(), key=lambda item: (item.dynamic, item.order, item.name))
                           if section.content)
