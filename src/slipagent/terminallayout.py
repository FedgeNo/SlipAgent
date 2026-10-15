"""Terminal presentation assembly around retained panel and input objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import ConditionalContainer, Dimension, DynamicContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.styles import Style

from .focus import FocusCoordinator
from .links import ActionControl
from .menulifecycle import MenuLifecycle
from .palette import (
    CONTENT_BACKGROUND_COLOR, FOOTER_BACKGROUND_COLOR, MENU_BACKGROUND_COLOR,
    FREE_MODEL_BACKGROUND_COLOR,
    MUTED_COLOR, USER_PROMPT_STYLE, TEXT_COLOR, LINK_COLOR, TOOL_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, ERROR_COLOR, PANEL_HEADER_BACKGROUND_COLOR,
    EDITOR_BACKGROUND_COLOR, DIFF_ADDED_BACKGROUND_COLOR, DIFF_REMOVED_BACKGROUND_COLOR,
)
from .uiwindows import TranscriptScrollbar

if TYPE_CHECKING:
    from .terminal import TerminalUI


def build_layout(ui: TerminalUI) -> Layout:
    from .terminal import FOOTER_ROWS, InputWindow, TranscriptControl, WordWrapInput

    ui.transcript.style = "class:content"
    ui.__dict__.setdefault("_diff_reference", None)
    for processor in ui.input.control.input_processors or []:
        if isinstance(processor, BeforeInput):
            processor.text = lambda: ui._prompt_label()
    ui._ensure_editor()
    ui._ensure_options()
    ui._ensure_context()
    ui._ensure_prompt()
    ui._ensure_menu()
    ui._ensure_viewer()
    ui._ensure_inspection()
    ui._ensure_focus()
    ui._ensure_queued_prompts()
    if not isinstance(ui.input.window, InputWindow):
        ui.input.window = InputWindow(ui.input.control)
        ui.input.window.style = "class:text-area class:user"
    body_height = Dimension(min=1, preferred=FOOTER_ROWS - 2, max=FOOTER_ROWS - 2)
    ui.input.window.height = body_height
    ui._menu_window.height = body_height
    ui._editor_panel.controls.height = body_height
    if not isinstance(ui._editor_panel.controls.content, ActionControl):
        ui._editor_panel.controls.content = ActionControl(ui._editor_panel.actions)
    ui._editor_panel.controls.style = "class:menu"
    ui._options_panel.field.window.height = body_height
    # Attach during layout rebuilds so existing sessions gain input wrapping.
    processors = ui.input.control.input_processors or []
    if not any(isinstance(processor, WordWrapInput) for processor in processors):
        ui.input.control.input_processors = [*processors, WordWrapInput()]
    if ui._input_label == "you › ":
        ui._input_label = "> "
    activity_row = VSplit([Window(ActionControl(lambda: ui._activity()), height=1),
                          Window(ActionControl(ui._inspect_link), height=1, width=lambda: ui._right_control_width(), align=WindowAlign.RIGHT)])
    scroll_row = VSplit([
        Window(ActionControl(lambda: ui._activity()), height=1,
               width=lambda: max(0, min((ui.output.get_size().columns - 18) // 2,
                                       ui.output.get_size().columns - 18 - ui._right_control_width()))),
        Window(ActionControl(ui._scroll_bottom_link), height=1, width=18),
        Window(ActionControl(ui._inspect_link), height=1, align=WindowAlign.RIGHT),
    ])
    footer = HSplit([
        DynamicContainer(lambda: scroll_row if ui._show_scroll_bottom() else activity_row),
        DynamicContainer(lambda: ui._editor_panel.controls if ui._editor_active and not ui._focus.prompt_active else ui.input),
        ui._bottom_row(menu=False),
    ], height=lambda: ui._footer_rows(), style="class:footer")
    menus = MenuLifecycle(ui)
    menu_footer = menus.footer(
        ActionControl(ui._menu_heading),
        ui._menu_window, lambda: ui._menu_window, ui._bottom_row(menu=True))
    setting_footer = menus.footer(
        ActionControl(lambda: ui._activity_notice([("class:menu-title", "Folder Opener")])),
        ui._options_panel.field, lambda: ui._options_panel.field, ui._bottom_row(menu=False))
    context = FocusCoordinator.panel(
        FormattedTextControl("Context — latest model request | Ctrl+\\: close"),
        ui.context_window, lambda: ui.context_window)
    # Overlay the prompt so switching between prompts of different heights
    # does not move the scroll position and select a different prompt again.
    pinned_transcript = FloatContainer(ui.transcript, floats=[Float(
        top=0, left=0, right=0, height=ui._pinned_prompt_height,
        content=Window(TranscriptControl(ui._pinned_prompt_content), style="class:user"),
    )])
    transcript = DynamicContainer(lambda: pinned_transcript if ui.output.get_size().rows > ui._footer_rows() + ui._queued_prompt_height() + 1 else ui.transcript)
    inspection_visible = Condition(lambda: ui.main_area.inspector_visible)
    ui._scrollbar = TranscriptScrollbar(lambda: ui.main_area.scroll_target())
    return Layout(HSplit([
        VSplit([DynamicContainer(lambda: ui.main_area.container(context, transcript)),
                Window(ui._scrollbar, width=1, style="class:scroll-track"),
                ConditionalContainer(ui._inspection_panel, inspection_visible)], style="class:content"),
        ConditionalContainer(VSplit([Window(TranscriptControl(ui._queued_prompt_content),
                                    height=ui._queued_prompt_height, dont_extend_height=True,
                                    style="class:user"),
                                     Window(width=1, style="class:scroll-track"),
                                     ConditionalContainer(Window(width=ui._inspection_panel.width, style="class:menu"), inspection_visible)],
                                    height=ui._queued_prompt_height),
                             Condition(lambda: ui._queued_prompt_height() > 0)),
        DynamicContainer(lambda: menu_footer if ui._menu_future is not None else setting_footer if ui._options_field_active else footer),
    ]), focused_element=ui._menu_window if ui._menu_future is not None else ui._options_panel.field if ui._options_field_active else ui._editor_input if ui._editor_active else
        ui._viewer_window if ui._viewer_active else ui.input)


def build_style() -> Style:
    return Style.from_dict({
        "": TEXT_COLOR,
        "status": MUTED_COLOR, "pulse": "bold " + WARNING_COLOR, "idle": "bold " + SUCCESS_COLOR,
        "held": "bold " + TOOL_COLOR, "user": USER_PROMPT_STYLE,
        "context-system": TOOL_COLOR, "retry": "bold " + WARNING_COLOR,
        "footer": TEXT_COLOR + " bg:" + FOOTER_BACKGROUND_COLOR,
        "content": TEXT_COLOR + " bg:" + CONTENT_BACKGROUND_COLOR,
        "editor": TEXT_COLOR + " bg:" + EDITOR_BACKGROUND_COLOR,
        "scroll-track": LINK_COLOR + " bg:" + FOOTER_BACKGROUND_COLOR,
        "menu": TEXT_COLOR + " bg:" + MENU_BACKGROUND_COLOR,
        "panel-title": "bold " + TOOL_COLOR + " bg:" + PANEL_HEADER_BACKGROUND_COLOR,
        "status-flash": "bold reverse",
        "keyboard-focus": "bold reverse",
        "menu-title": "bold " + TOOL_COLOR, "menu-selected": "reverse bold " + LINK_COLOR,
        "link": LINK_COLOR,
        "clickable": "underline", "hover": "reverse",
        "diff-number": MUTED_COLOR,
        "menu-item": MUTED_COLOR,
        "menu-free": "noreverse bg:" + FREE_MODEL_BACKGROUND_COLOR + " " + TEXT_COLOR,
        "call-running": WARNING_COLOR, "call-completed": SUCCESS_COLOR, "call-error": ERROR_COLOR,
        "diff-added": SUCCESS_COLOR + " bg:" + DIFF_ADDED_BACKGROUND_COLOR,
        "diff-removed": ERROR_COLOR + " bg:" + DIFF_REMOVED_BACKGROUND_COLOR,
        "diff-header": "bold " + TOOL_COLOR, "diff-hunk": LINK_COLOR,
        "search": "reverse " + WARNING_COLOR, "search.current": "bold reverse " + WARNING_COLOR,
        "selected-text": "reverse " + LINK_COLOR,
        "line-number": MUTED_COLOR, "line-number.current": LINK_COLOR,
    })
