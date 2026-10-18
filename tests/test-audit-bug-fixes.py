"""Regression coverage for file safety, focus ownership and paged editing."""

import asyncio
import io
import os
import threading

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output.vt100 import Vt100_Output

from slipagent.artifacts import ArtifactStore
from slipagent.context import CompletedStep
from slipagent.compaction import StepCompactor
from slipagent.editor import FileEditor
from slipagent.terminal import TerminalUI
from slipagent.tools.files import ReadFileTool, WriteFileTool, EditFileTool
from slipagent.types import Completion, Message
from slipagent.workspace import Workspace


@pytest.fixture
def ui():
    sink = io.StringIO()
    sizes = [Size(rows=24, columns=100)]
    output = Vt100_Output(sink, lambda: sizes[0], term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe:
        terminal = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        yield terminal, sizes
        terminal.close()


def press(terminal, key):
    with set_app(terminal.app):
        terminal.app.key_processor.feed(KeyPress(key))
        terminal.app.key_processor.process_keys()


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO")
@pytest.mark.parametrize("tool", ["read", "write", "edit"])
async def test_file_tools_reject_fifo_without_a_writer(tmp_path, tool):
    path = tmp_path / "pipe"
    os.mkfifo(path)
    store = ArtifactStore(1_000_000)
    workspace = Workspace(tmp_path)
    tools = {"read": (ReadFileTool(workspace), {"path": "pipe"}),
             "write": (WriteFileTool(workspace, store), {"path": "pipe", "content": "new"}),
             "edit": (EditFileTool(workspace, store), {"path": "pipe", "old_string": "old", "new_string": "new"})}
    selected, arguments = tools[tool]
    task = asyncio.create_task(selected.run(**arguments))
    try:
        done, _ = await asyncio.wait([task], timeout=1)
        assert done, "File tool waited for a FIFO writer"
        assert task.result().is_error
        assert path.is_fifo()
    finally:
        if not task.done():
            # Release only this fixture's FIFO if the regression returns.
            descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            os.write(descriptor, b"old")
            os.close(descriptor)
            await task
        await store.aclose()


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO")
async def test_descriptor_validation_rejects_regular_file_swapped_for_fifo(tmp_path, monkeypatch):
    path = tmp_path / "raced"
    path.write_text("old")
    original_open = os.open
    def swapped(name, flags, *args, **kwargs):
        if name == "raced":
            path.unlink()
            os.mkfifo(path)
        return original_open(name, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", swapped)
    async with asyncio.timeout(1):
        result = await ReadFileTool(Workspace(tmp_path)).run("raced")
    assert result.is_error and path.is_fifo()


@pytest.mark.slow
@pytest.mark.parametrize("columns", [40, 59, 60, 100])
async def test_existing_main_regions_are_reachable_at_every_width(ui, tmp_path, columns):
    terminal, size = ui
    size[0] = Size(rows=24, columns=columns)
    terminal.open_viewer("Result", "one\ntwo\n")
    terminal._focus.focus(terminal.input)
    terminal._cycle_panel()
    assert terminal.app.layout.current_window is terminal._viewer_window
    terminal.close_viewer()
    path = tmp_path / "file"
    path.write_text("text\n")
    await terminal._editor_panel.open(path, 1)
    for _ in range(30):
        terminal._cycle_panel()
        if terminal.app.layout.current_control is terminal.input.control:
            break
    assert terminal.app.layout.current_control is terminal.input.control


async def test_viewer_and_context_keys_leave_focused_prompt_alone(ui):
    terminal, _ = ui
    interrupted = []
    terminal.set_interrupt_handler(lambda: interrupted.append(True))
    terminal.open_viewer("Result", "one\ntwo\n")
    terminal.input.buffer.history.append_string("previous prompt")
    terminal._focus.focus(terminal.input)
    with set_app(terminal.app):
        terminal.input.buffer.load_history_if_not_yet_loaded()
        await terminal.input.buffer._load_history_task
    press(terminal, Keys.Up)
    assert terminal.input.text == "previous prompt"
    press(terminal, Keys.Escape)
    assert interrupted and terminal._viewer_active
    terminal.close_viewer()
    terminal.context_visible = True
    terminal._focus.focus(terminal.input)
    terminal.input.text = "abc"
    terminal.input.buffer.cursor_position = 3
    press(terminal, Keys.Home)
    assert terminal.input.buffer.cursor_position == 0


async def test_menus_restore_invoking_focus_and_resize_cannot_hide_focus(ui):
    terminal, size = ui
    terminal.open_viewer("Result", "one\n")
    terminal._focus.focus(terminal.input)
    menu = asyncio.create_task(terminal.choose("Example", [("ok", "OK")]))
    await asyncio.sleep(0)
    terminal._finish_menu(None)
    await menu
    assert terminal.app.layout.current_control is terminal.input.control
    terminal.toggle_inspection()
    for _ in range(30):
        terminal._cycle_panel()
        if terminal._inspection_focused():
            break
    assert terminal._inspection_focused()
    size[0] = Size(rows=24, columns=40)
    with set_app(terminal.app):
        terminal.app.before_render.fire()
        terminal.app.renderer.render(terminal.app, terminal.app.layout)
    assert terminal.app.layout.current_window is terminal._viewer_window


async def test_delayed_save_does_not_steal_a_later_focus_choice(ui, tmp_path, monkeypatch):
    terminal, _ = ui
    path = tmp_path / "file"
    path.write_text("old\n")
    await terminal._editor_panel.open(path, 1)
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = FileEditor.save
    def slow(file, **options):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)
        original(file, **options)
    monkeypatch.setattr(FileEditor, "diff", lambda file: "")
    monkeypatch.setattr(FileEditor, "save", slow)
    terminal._editor_input.text = "new\n"
    task = asyncio.create_task(terminal._editor_panel.save())
    try:
        await asyncio.wait_for(started.wait(), 10)
        terminal._focus.focus(terminal.input)
        release.set()
        assert await task
        assert terminal.app.layout.current_control is terminal.input.control
    finally:
        release.set()
        await task


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("direction", [-1, 1])
async def test_boundary_line_join_supports_document_undo_redo(ui, tmp_path, ending, direction):
    terminal, _ = ui
    path = tmp_path / "file"
    original = ending.join(f"line {n}" for n in range(70)) + ending
    path.write_bytes(original.encode())
    await terminal._editor_panel.open(path, 33 if direction < 0 else 32)
    panel = terminal._editor_panel
    if direction > 0:
        panel.input.buffer.cursor_position = len(panel.input.text) - len(ending)
    panel.delete_grapheme(direction)
    panel.flush()
    expected = original.replace("line 31" + ending + "line 32", "line 31line 32")
    assert "".join(panel.file.chunks()) == expected
    panel.change_page(1)
    panel.input.buffer.undo()
    assert "".join(panel.file.chunks()) == original
    panel.change_page(1)
    panel.input.buffer.redo()
    panel.flush()
    assert "".join(panel.file.chunks()) == expected
    assert await panel.save()
    assert path.read_bytes() == expected.encode()


async def test_undo_follows_edit_order_across_pages_and_save(ui, tmp_path):
    terminal, _ = ui
    path = tmp_path / "file"
    original = "".join(f"line {n}\n" for n in range(100))
    path.write_text(original)
    await terminal._editor_panel.open(path, 1)
    panel = terminal._editor_panel
    buffer = panel.input.buffer
    buffer.save_to_undo_stack()
    buffer.insert_text("FIRST ")
    panel.change_page(1)
    buffer.cursor_position = 0
    buffer.save_to_undo_stack()
    buffer.insert_text("SECOND ")
    assert await panel.save()
    panel.change_page(1)
    buffer.undo()
    assert panel.page == 1 and not buffer.text.startswith("SECOND")
    buffer.undo()
    assert panel.page == 0 and "".join(panel.file.chunks()) == original
    buffer.redo()
    panel.change_page(1)
    buffer.redo()
    panel.flush()
    assert "FIRST line 0" in "".join(panel.file.chunks())
    assert "SECOND line 32" in "".join(panel.file.chunks())


@pytest.mark.parametrize("cluster", ["e\u0301", "👍🏽", "👩‍💻", "🇨🇦", "界", "\r\n"])
async def test_editor_moves_and_deletes_whole_graphemes(ui, tmp_path, cluster):
    terminal, _ = ui
    path = tmp_path / "file"
    path.write_bytes((cluster + "x\n").encode())
    await terminal._editor_panel.open(path, 1)
    panel = terminal._editor_panel
    press(terminal, Keys.Right)
    assert panel.input.buffer.cursor_position == len(cluster)
    press(terminal, Keys.Backspace)
    assert panel.input.text == "x\n"
    panel.input.buffer.undo()
    press(terminal, Keys.Left)
    panel.input.buffer.cursor_position = 0
    press(terminal, Keys.Delete)
    assert panel.input.text == "x\n"


async def test_keyboard_selection_and_word_delete_keep_clusters_whole(ui, tmp_path):
    terminal, _ = ui
    path = tmp_path / "unicode"
    path.write_text("e\u0301👩‍💻x\n")
    await terminal._editor_panel.open(path, 1)
    buffer = terminal._editor_input.buffer
    press(terminal, Keys.Right)
    press(terminal, Keys.ShiftLeft)
    assert buffer.cursor_position == 0
    press(terminal, Keys.Delete)
    assert buffer.text == "👩‍💻x\n"
    buffer.undo()
    buffer.cursor_position = 5
    buffer.delete_before_cursor(1)
    assert buffer.text == "e\u0301x\n"


def test_indexed_page_boundaries_do_not_split_graphemes():
    from wcwidth import iter_graphemes
    from slipagent.inspection import IndexedText, CHUNK_SIZE
    text = ("x" * (CHUNK_SIZE - 1) + "e\u0301👩‍💻🇨🇦") * 40
    snapshot = IndexedText()
    try:
        snapshot.text.write(text)
        snapshot.finish()
        parts = [snapshot.segment(n) for n in range(len(snapshot))]
        assert "".join(parts) == text
        boundaries, position = {0}, 0
        for cluster in iter_graphemes(text):
            position += len(cluster)
            boundaries.add(position)
        position = 0
        for part in parts:
            position += len(part)
            assert position in boundaries
    finally:
        snapshot.close()


async def test_compaction_materializes_only_active_work_and_drains_deferred_steps(monkeypatch):
    from slipagent import compaction

    # Cross the ready-queue boundary three times without a large history scan.
    queue_size = 3
    monkeypatch.setattr(compaction, "COMPACTION_QUEUE_SIZE", queue_size)
    release, started = asyncio.Event(), asyncio.Event()
    active = peak = calls = materialized = 0
    class Client:
        async def chat(self, **kwargs):
            nonlocal active, peak, calls
            active += 1
            peak = max(peak, active)
            calls += 1
            started.set()
            try:
                await release.wait()
                return Completion(Message.assistant('{"summary":"Completed step.","reasoning_summary":""}'), "stub")
            finally:
                active -= 1
    original = CompletedStep.compaction_input
    def counted(step):
        nonlocal materialized
        materialized += 1
        return original(step)
    monkeypatch.setattr(CompletedStep, "compaction_input", counted)
    owner = StepCompactor(lambda usage: None, lambda error: pytest.fail(error))
    steps = [CompletedStep(n + 1, "Request", [Message.user("Request"), Message.assistant("Reply")])
             for n in range(queue_size * 3 + 1)]
    client = Client()
    try:
        for step in steps:
            owner.submit(step, client, "test", None, None, conversation=steps)
        assert materialized == 0 and len(owner.jobs) == 1
        assert len(owner.__dict__["_ready"]) == queue_size
        await asyncio.wait_for(started.wait(), 1)
        assert materialized == 1 and calls == 1
        assert any(step.compaction_status == "deferred" for step in steps)
        release.set()
        await owner.wait()
        assert peak == 1 and calls == len(steps)
        assert all(step.summary == "Completed step." for step in steps)
    finally:
        release.set()
        await owner.aclose()


async def test_reset_cancels_ready_and_deferred_compactions_without_losing_originals():
    started = asyncio.Event()
    class Client:
        async def chat(self, **kwargs):
            started.set()
            await asyncio.Event().wait()
    owner = StepCompactor(lambda usage: None, lambda error: pytest.fail(error))
    steps = [CompletedStep(n + 1, "Request", [Message.user("Request"), Message.assistant("Reply")]) for n in range(100)]
    try:
        for step in steps:
            owner.submit(step, Client(), "test", None, None, conversation=steps)
        await asyncio.wait_for(started.wait(), 1)
        owner.reset()
        await owner.wait()
        assert not owner.jobs
        assert all(step.summary is None and step.compaction_status == "cancelled" for step in steps)
        assert all(step.parts()["response"] == "Reply" for step in steps)
        assert all(getattr(step, "_compaction_request", None) is None for step in steps)
    finally:
        await owner.aclose()
