"""Named prompt sections rendered in one deterministic, inspectable order."""

from __future__ import annotations

from dataclasses import dataclass


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
        return "\n\n".join(f"{section.title}:\n\n{section.content}" for section in
                           sorted(self.sections.values(), key=lambda item: (item.dynamic, item.order, item.name))
                           if section.content)
