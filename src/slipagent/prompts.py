"""Named prompt sections rendered in one deterministic, inspectable order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template


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
