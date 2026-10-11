"""Selector-based inspection options with an isolated command input."""

from __future__ import annotations

import asyncio
from typing import Any

from prompt_toolkit.document import Document
from prompt_toolkit.filters import to_filter
from prompt_toolkit.widgets import TextArea

from .config import ConfigError
from .explorer import folder_opener
from .uipreferences import load_ui_preferences, save_ui_preference, configured_folder_opener


class OptionsPanel:
    def __init__(self, terminal: Any) -> None:
        self.terminal = terminal
        self.active = False
        self.field_future: asyncio.Future[str | None] | None = None
        self.field = TextArea(multiline=False, height=5, accept_handler=self.accept)
        try:
            self.values = load_ui_preferences()
            configured = configured_folder_opener(self.values.get("folder_opener", ""))
            if configured is not None:
                terminal._folder_opener = configured
        except (ConfigError, ValueError) as exc:
            self.values = {}
            terminal.__dict__["_notification"] = str(exc)

    def accept(self, buffer: Any) -> bool:
        if self.field_future is not None and not self.field_future.done():
            self.field_future.set_result(buffer.text)
        return True

    def cancel(self) -> None:
        if self.field_future is not None and not self.field_future.done():
            self.field_future.set_result(None)

    async def command(self) -> str | None:
        self.field.buffer.set_document(Document(self.values.get("folder_opener", "")))
        self.field_future = asyncio.get_running_loop().create_future()
        self.terminal._options_field_active = True
        self.terminal.notify("Use {path}, or the directory is appended.")
        self.terminal._focus.focus(self.field)
        self.terminal.app.invalidate()
        try:
            return await self.field_future
        finally:
            self.field_future = None
            self.terminal._options_field_active = False
            self.terminal._focus.restore_primary()
            self.terminal.app.invalidate()

    async def show(self) -> None:
        if self.active or self.terminal._menu_future is not None:
            return
        self.active = True
        try:
            while True:
                action = await self.terminal.choose("Inspection Options", [
                    ("wrap", "Editor: " + ("Disable" if self.values.get("wrap", True) else "Enable") + " Line Wrapping"),
                    ("tab_width", f"Editor Tab Width: {self.values.get('tab_width', 4)} Spaces"),
                    ("folder_opener", "Configure Folder Opener"),
                    ("default_opener", "Use Default Folder Opener"),
                    ("done", "Close Options"),
                ])
                if action in (None, "done"):
                    break
                try:
                    if action == "wrap":
                        value: Any = not self.values.get("wrap", True)
                    elif action == "tab_width":
                        choice = await self.terminal.choose("Editor Tab Width", [(str(width), f"{width} Spaces") for width in (2, 4, 8)])
                        if choice is None:
                            continue
                        value = int(choice)
                    elif action == "folder_opener":
                        value = await self.command()
                        if value is None:
                            continue
                        configured_folder_opener(value)
                    else:
                        action, value = "folder_opener", ""
                    save_ui_preference(action, value)
                    self.values[action] = value
                    if action == "wrap":
                        self.terminal._editor_input.wrap_lines = value
                        self.terminal._editor_input.window.wrap_lines = to_filter(value)
                    if action == "folder_opener":
                        self.terminal._folder_opener = configured_folder_opener(value) or folder_opener()
                    self.terminal.notify("Option saved.")
                except (OSError, ConfigError, ValueError) as exc:
                    self.terminal.notify("Could not save option.")
                    self.terminal.write(str(exc))
        finally:
            self.active = False
            self.terminal._focus.restore_primary()
            self.terminal.app.invalidate()
