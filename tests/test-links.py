"""Action links share hover, activation, and explicit asynchronous ownership."""

import asyncio

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from slipagent.links import CallbackLink, LinkHover, TaskLink


def event(kind, button=MouseButton.LEFT):
    return MouseEvent(Point(0, 0), kind, button, frozenset())


def test_links_hover_without_activation_and_ignore_other_buttons():
    hovered = []
    opened = []
    link = CallbackLink("Back", "back", LinkHover(lambda: hovered[-1] if hovered else None, hovered.append),
                        lambda: opened.append(True))
    assert link.fragments()[0][1] == "Back"
    link.mouse(event(MouseEventType.MOUSE_MOVE))
    assert "class:hover" in link.fragments()[0][0]
    assert opened == []
    link.mouse(event(MouseEventType.MOUSE_UP, MouseButton.RIGHT))
    assert opened == []
    link.mouse(event(MouseEventType.MOUSE_UP))
    assert opened == [True]


def test_disabled_link_retains_label_but_cannot_activate():
    opened = []
    link = CallbackLink("Save File", "save", LinkHover(lambda: None, lambda key: None),
                        lambda: opened.append(True), enabled=lambda: False)
    link.mouse(event(MouseEventType.MOUSE_UP))
    assert opened == []
    assert link.fragments()[0][1] == "Save File"


async def test_async_link_uses_supplied_task_owner():
    tasks = []
    opened = []
    async def action():
        opened.append(True)
    def spawn(operation):
        tasks.append(asyncio.create_task(operation))
    link = TaskLink("Open", "open", LinkHover(lambda: None, lambda key: None), action, spawn)
    link.mouse(event(MouseEventType.MOUSE_MOVE))
    assert tasks == []
    link.mouse(event(MouseEventType.MOUSE_UP))
    assert len(tasks) == 1
    await tasks[0]
    assert opened == [True]
