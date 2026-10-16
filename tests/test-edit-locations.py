"""Edit references identify changed lines in live and restored transcripts."""

import io

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from slipagent.agent import AgentEvent
from slipagent.artifacts import ArtifactStore
from slipagent.cli import Renderer, Style
from slipagent.terminal import TerminalUI
from slipagent.tools.base import ToolResult, current_invocation
from slipagent.tools.files import EditFileTool
from slipagent.types import Message, ToolCall


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("before,after,line", [
    ("context\nunchanged\nold\n", "context\nunchanged\nnew\n", 3),
    ("context\n", "context\nadded\n", 2),
    ("context\nremoved\nlast\n", "context\nlast\n", 2),
    ("context", "context\n", 1),
])
async def test_edit_reports_changed_line_after_unchanged_context(workspace, newline, before, after, line):
    path = workspace.root / "source.txt"
    path.write_bytes(before.replace("\n", newline).encode())
    result = await EditFileTool(workspace).invoke({
        "path": path.name, "old_string": before, "new_string": after,
    })
    assert not result.is_error, result.text()
    assert result.text().splitlines()[0] == f"Replaced 1 occurrence(s) in source.txt at line {line}."
    expected = after.replace("\n", newline) if "\n" in before else after
    assert path.read_bytes() == expected.encode()


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("artifacts", [False, True])
@pytest.mark.parametrize("arguments,line,numbered", [
    ({"old_string": "line 6\nline 7\nline 8", "new_string": "line 6\nline 7\nchanged"}, 8, True),
    ({"edits": [{"old_string": "line 8", "new_string": "changed"}]}, 8, True),
    ({"edits": [{"old_string": "line 9", "new_string": "last"},
                {"old_string": "line 4", "new_string": "first\ninserted"}]}, 4, False),
    ({"old_string": "repeat", "new_string": "changed", "replace_all": True}, 11, False),
    ({"old_string": "line 8", "new_string": "changed", "replace_all": True}, 8, True),
    ({"old_string": "line 8\n", "new_string": ""}, 8, True),
    ({"old_string": "line 8\n", "new_string": "line 8\ninserted\n"}, 9, True),
    ({"edits": [{"old_string": "line 3", "new_string": "line 3"},
                {"old_string": "line 9", "new_string": "changed"}]}, 9, False),
])
async def test_edit_links_use_completed_location(workspace, restored, artifacts, arguments, line, numbered):
    path = workspace.root / "source.txt"
    path.write_text("".join(f"line {n}\n" for n in range(1, 11)) + "repeat\nrepeat\n")
    call = ToolCall("edit", "edit_file", {"path": path.name, **arguments})
    store = ArtifactStore(1_000_000)
    store.use_directory(workspace.root / "artifacts")
    sink = io.StringIO()
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=DummyOutput(),
                        project_directory=workspace.root)
        ui.__dict__["_artifact_store"] = store
        renderer = Renderer(Style(False), sink, verbose=True)
        renderer.terminal = ui
        try:
            if not restored:
                renderer.handle(AgentEvent("tool_start", tool_call=call, step=1))
                assert not ui.__dict__.get("_file_links"), "An unfinished edit has no known location."
            token = current_invocation.set((1, call.id))
            try:
                result = await EditFileTool(workspace, store if artifacts else None).invoke(call.arguments)
            finally:
                current_invocation.reset(token)
            assert not result.is_error, result.text()
            references = list(store.records.values())
            if restored:
                await renderer.restore_transcript([
                    Message.assistant("", [call]),
                    Message.tool_result(call.id, {"tool": call.name, "status": "success",
                                                 "content": result.content, "artifacts": references}),
                ], [])
            else:
                renderer.handle(AgentEvent("tool_end", tool_call=call, step=1, result=result))
                for reference in references:
                    renderer.handle(AgentEvent("artifact", result=ToolResult.ok(reference)))
            content = ui._transcript_content(160)
            rows = ["".join(text for _, text, *_ in content.get_line(y)) for y in range(content.line_count)]
            label = f"    source.txt:{line}" if numbered else "    source.txt"
            expected = label + (f"  View Diff — Record {references[0]['record_id']}" if artifacts else "")
            assert expected in rows
            assert list(ui.__dict__["_file_links"].values()) == [(path, line)]
            assert ui._file_at(Point(x=4, y=rows.index(expected))) == (path, line)
            if not numbered:
                assert " at line " not in result.text().splitlines()[0]
        finally:
            ui.close()
            await store.aclose()


async def test_restored_edit_uses_diff_after_old_summary_points_to_context(workspace):
    path = workspace.root / "source.txt"
    path.write_text("one\ntwo\nthree\nnew\n")
    call = ToolCall("edit", "edit_file", {"path": path.name, "old_string": "one\ntwo\nthree\nold",
                                          "new_string": "one\ntwo\nthree\nnew"})
    result = ("Replaced 1 occurrence(s) in source.txt at line 1.\n"
              "--- source.txt\n+++ source.txt\n@@ -1,4 +1,4 @@\n one\n two\n three\n-old\n+new")
    sink = io.StringIO()
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=DummyOutput(),
                        project_directory=workspace.root)
        renderer = Renderer(Style(False), sink, verbose=False)
        renderer.terminal = ui
        try:
            await renderer.restore_transcript([
                Message.assistant("", [call]),
                Message.tool_result(call.id, {"status": "success", "content": result}),
            ], [])
            assert list(ui.__dict__["_file_links"].values()) == [(path, 4)]
        finally:
            ui.close()


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("failed", [False, True])
async def test_edit_without_a_known_location_never_labels_line_one(workspace, restored, failed):
    path = workspace.root / "source.txt"
    path.write_text("contents\n")
    call = ToolCall("edit", "edit_file", {"path": path.name})
    result = ToolResult("Edit failed." if failed else "Updated source.txt.", failed)
    sink = io.StringIO()
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=DummyOutput(),
                        project_directory=workspace.root)
        renderer = Renderer(Style(False), sink, verbose=True)
        renderer.terminal = ui
        try:
            if restored:
                await renderer.restore_transcript([
                    Message.assistant("", [call]),
                    Message.tool_result(call.id, {"status": "error" if failed else "success",
                                                 "content": result.content}),
                ], [])
            else:
                renderer.handle(AgentEvent("tool_start", tool_call=call, step=1))
                renderer.handle(AgentEvent("tool_end", tool_call=call, step=1, result=result))
            content = ui._transcript_content(100)
            rows = ["".join(text for _, text, *_ in content.get_line(y)) for y in range(content.line_count)]
            assert not any("source.txt:1" in row for row in rows)
            assert ("    source.txt" in rows) == (not failed)
        finally:
            ui.close()
