"""Shared list behavior, independent of the terminal host."""

from pathlib import Path

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from slipagent.listpanels import CallsPanel, ChoicePanel, ListActions, TouchedPanel


def actions(opened):
    return ListActions(opened.append, lambda: None, lambda: True,
                       lambda key: False, lambda key: None)


def test_touched_selection_survives_insertions_and_clamps_after_removal():
    items = {"first": "read", "second": "edited"}
    panel = TouchedPanel(items, lambda: None, lambda: False, actions([]))
    panel.move(1)
    assert panel.selection == "first"
    items["third"] = "read"
    assert panel.selection == "first"
    panel.move(100)
    assert panel.selection == "first"
    del items["first"]
    assert panel.selection == "third"
    items.clear()
    panel.move(-1)
    assert panel.selection is None
    assert panel.fragments() == [("class:status", "None yet")]


def test_calls_keep_distinct_step_keys_and_mouse_opens_the_clicked_item():
    opened = []
    panel = CallsPanel({(1, "same"): {"tool": "read_file", "status": "done"},
                        (3, "same"): {"tool": "write_file", "status": "done"}}, actions(opened))
    fragments = panel.fragments()
    fragments[2][2](MouseEvent(Point(0, 1), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
    assert opened == [(1, "same")]
    assert panel.selection == (1, "same")


def test_call_statuses_have_distinct_semantic_colors():
    from slipagent.terminallayout import build_style
    from slipagent.palette import SUCCESS_COLOR, WARNING_COLOR, ERROR_COLOR

    items = {(1, status): {"tool": "read_file", "status": status}
             for status in ("running", "completed", "error")}
    panel = CallsPanel(items, actions([]))
    styles = build_style()
    for status, color in (("running", WARNING_COLOR), ("completed", SUCCESS_COLOR), ("error", ERROR_COLOR)):
        assert styles.get_attrs_for_style_str(panel.item_style((1, status))).color == color[1:]


def test_directories_need_an_opener_for_keyboard_and_mouse(tmp_path: Path):
    opened = []
    panel = TouchedPanel({".": "read"}, lambda: tmp_path, lambda: False, actions(opened))
    panel.open_selected()
    assert opened == []
    assert len(panel.fragments()[0]) == 2


def test_choices_clamp_rapid_moves_and_use_the_same_activation_path():
    opened = []
    panel = ChoicePanel(actions(opened))
    panel.options = [(str(i), f"choice {i}") for i in range(30)]
    for _ in range(40):
        panel.move(1)
    assert panel.selection == 29
    content = panel.window.content.create_content(40, 5)
    panel.window._scroll(content, 40, 5)
    visible = ["".join(part[1] for part in content.get_line(index))
               for index in range(panel.window.vertical_scroll, panel.window.vertical_scroll + 5)]
    assert visible[0].strip() == "choice 25"
    assert visible[-1] == "› choice 29"
    for _ in range(40):
        panel.move(-1)
    panel.open_selected()
    assert opened == [0]
    panel.window._scroll(panel.window.content.create_content(40, 5), 40, 5)
    assert panel.window.vertical_scroll == 0
