"""Clickable action words in application-owned keyboard hints."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from prompt_toolkit.formatted_text import StyleAndTextTuples

from .terminalbindings import hold_model, interrupt_work, skip_model, toggle_context

if TYPE_CHECKING:
    from .terminal import TerminalUI

KEY_HINT = re.compile(
    r'(?<![\w+])(?P<key>(?:(?:Ctrl|Alt|Shift)[+-])+(?:Tab|Enter|Esc|[A-Z\\])|Esc(?:ape)?|Enter|Tab|F[0-9]{1,2})'
    r'(?P<join>(?:\s+again)?\s+to\s+|:\s*|\s*=\s*|\s+or\s+click\s*=\s*|/click\s+|'
    r'\s+(?=(?:saves|closes|refreshes|toggles|interrupts|quits)\b))'
    r'(?P<action>[^.;,)|\n]+)', re.IGNORECASE)

Action = tuple[Callable[[], object], Callable[[], bool]]


def _action(ui: TerminalUI, key: str, label: str) -> Action | None:
    key = key.casefold().replace('-', '+')
    words = label.casefold()
    if key in ('esc', 'escape'):
        if words.startswith(('interrupt', 'cancel the turn')):
            return lambda: interrupt_work(ui), lambda: getattr(ui, '_interrupt_callback', None) is not None
        if words.startswith(('stay on this model', 'keep waiting')):
            return lambda: hold_model(ui), lambda: ui.__dict__.get('_hold_callback') is not None
        if 'search' in words and ui._feed_search.active:
            return ui._feed_search.cancel, lambda: ui._feed_search.active
        if 'editor' in words and ui._editor_active:
            return lambda: ui.app.create_background_task(ui._close_editor()), lambda: ui._editor_active
        if 'viewer' in words and ui._viewer_active:
            return ui.close_viewer, lambda: ui._viewer_active
        if ui._file_picker.active or ui._menu_future is not None:
            return lambda: ui._finish_menu(None), lambda: ui._menu_future is not None
        if ui._options_field_active:
            return ui._options_panel.cancel, lambda: ui._options_field_active
        if ui._feed_search.active:
            return ui._feed_search.cancel, lambda: ui._feed_search.active
        if ui._editor_active:
            return lambda: ui.app.create_background_task(ui._close_editor()), lambda: ui._editor_active
        if ui._viewer_active:
            return ui.close_viewer, lambda: ui._viewer_active
        return lambda: interrupt_work(ui), lambda: getattr(ui, '_interrupt_callback', None) is not None
    if key == 'ctrl+n':
        return lambda: skip_model(ui), lambda: ui.__dict__.get('_skip_callback') is not None
    if key == 'shift+tab':
        return lambda: ui._cycle_panel(components=True), lambda: True
    if key == 'tab':
        return ui._cycle_panel, lambda: True
    if key == 'f3':
        return ui._feed_search.open, lambda: not ui._password and not ui._file_picker.active
    if key == 'ctrl+\\':
        return lambda: toggle_context(ui), lambda: ui.context_visible if words.startswith('close') else True
    if key == 'ctrl+s' and ui._editor_active:
        return lambda: ui.app.create_background_task(ui._save_editor()), lambda: ui._editor_active
    if key == 'enter':
        if ui._file_picker.active:
            return lambda: ui._file_picker.accept(ui._file_picker.field.buffer), lambda: ui._file_picker.active
        if ui._menu_future is not None:
            def select() -> None:
                if ui._menu_options:
                    ui._finish_menu(ui._menu_options[ui._menu_index][0])
            return select, lambda: ui._menu_future is not None and bool(ui._menu_options)
        if ui._options_field_active:
            return lambda: ui._options_panel.accept(ui._options_panel.field.buffer), lambda: ui._options_field_active
        if ui._feed_search.active:
            return lambda: ui._feed_search.accept(ui._feed_search.field.buffer), lambda: ui._feed_search.active
        return ui.input.buffer.validate_and_handle, lambda: True
    return None


def _display_key(key: str) -> str:
    names = {'ctrl': 'Ctrl', 'alt': 'Alt', 'shift': 'Shift', 'tab': 'Tab', 'enter': 'Enter',
             'esc': 'Esc', 'escape': 'Esc'}
    return ''.join(names.get(part.casefold(), part.upper()) for part in re.split(r'([+-])', key))


def link_hints(ui: TerminalUI, text: str, *, style: str = '') -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = []
    position = 0
    for match in KEY_HINT.finditer(text):
        label = match['action'].rstrip()
        action = _action(ui, match['key'], label)
        start, end = match.start('action'), match.start('action') + len(label)
        key = _display_key(match['key'])
        fragments.append((style, text[position:match.start('key')] + key + text[match.end('key'):start]))
        if action is None:
            fragments.append((style, label))
        else:
            fragments.extend(ui._action_link(label, ('key-hint', key, label), action[0],
                                              style=style, enabled=action[1]))
        position = end
    fragments.append((style, text[position:]))
    return fragments
