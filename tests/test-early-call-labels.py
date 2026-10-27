"""A description that leads a call's arguments reaches the feed before the call runs."""

import io
import json

from slipagent.api import _StreamCompletion
from slipagent.cli import Renderer, Style
from slipagent.events import AgentEvent
from slipagent.types import ToolCall


def stream(*arguments, index=0, identity=True):
    """Feed one call's argument fragments; return what the stream announced."""
    announced = []
    state = _StreamCompletion(lambda kind, chunk: announced.append((kind, json.loads(chunk))))
    first = {"index": index, "function": {"arguments": arguments[0]}}
    if identity:
        first.update(id="call-1")
        first["function"]["name"] = "read_file"
    for position, fragment in enumerate(arguments):
        entry = first if position == 0 else {"index": index, "function": {"arguments": fragment}}
        state.add(json.dumps({"choices": [{"delta": {"tool_calls": [entry]}}]}))
    return announced


def test_leading_description_is_announced_as_soon_as_its_string_closes():
    assert stream('{"call_label": "Reading the con', 'fig", "path": "a.py"}') == [
        ("tool_label", {"id": "call-1", "name": "read_file", "label": "Reading the config"})]
    early = stream('{"call_label": "Reading', ' it"')
    assert early[0][1]["label"] == "Reading it"


def test_description_after_other_arguments_or_missing_is_not_announced():
    assert stream('{"path": "a.py", "call_label": "Reading it"}') == []
    assert stream('{"path": "a.py"}') == []
    assert stream('{"call_label": ""}') == []
    assert stream('{"call_label": "  "}') == []


def test_description_is_announced_once_and_unicode_escapes_are_decoded():
    announced = stream('{"call_label": "Checking \\u00e9"', ', "path": "a.py"', '}')
    assert [item[1]["label"] for item in announced] == ["Checking é"]


def test_announcement_waits_for_the_call_identity():
    announced = []
    state = _StreamCompletion(lambda kind, chunk: announced.append(json.loads(chunk)))
    state.add(json.dumps({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": '{"call_label": "Listing files"}'}}]}}]}))
    assert announced == []
    state.add(json.dumps({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call-2", "function": {"name": "list_dir"}}]}}]}))
    assert announced == []  # Only an argument fragment triggers a check, and none remains.


def renderer():
    out = io.StringIO()
    return Renderer(Style(False), out, False), out


def call(label=True, **arguments):
    arguments = ({"call_label": "Reading the config"} if label else {}) | {"path": "a.py"} | arguments
    return ToolCall("call-1", "read_file", arguments)


def test_early_description_is_not_repeated_when_the_call_starts():
    shown, out = renderer()
    shown.handle(AgentEvent(kind="tool_label", text="Reading the config", tool_call=ToolCall("call-1", "read_file")))
    assert out.getvalue().count("Reading the config") == 1
    shown.handle(AgentEvent(kind="tool_start", step=1, tool_call=call()))
    text = out.getvalue()
    assert text.count("Reading the config") == 1
    assert 'read_file({"path": "a.py"})' in text
    assert text.index("Reading the config") < text.index("read_file(")


def test_call_without_an_early_description_is_displayed_as_before():
    early, early_out = renderer()
    plain, plain_out = renderer()
    early.handle(AgentEvent(kind="tool_start", step=1, tool_call=call()))
    plain.handle(AgentEvent(kind="tool_start", step=1, tool_call=call()))
    assert early_out.getvalue() == plain_out.getvalue()
    assert "Reading the config" in plain_out.getvalue() and 'read_file({"path": "a.py"})' in plain_out.getvalue()


def test_description_with_other_output_after_it_is_shown_in_full_with_its_call():
    shown, out = renderer()
    shown.handle(AgentEvent(kind="tool_label", text="First", tool_call=ToolCall("one", "read_file")))
    shown.handle(AgentEvent(kind="tool_label", text="Second", tool_call=ToolCall("two", "read_file")))
    shown.handle(AgentEvent(kind="tool_start", step=1, tool_call=ToolCall("one", "read_file", {"call_label": "First", "path": "a"})))
    text = out.getvalue()
    assert text.count("First") == 2 and text.count("Second") == 1
    assert text.rindex("First") < text.index("read_file(")


def test_answer_and_changed_descriptions_are_not_taken_from_the_early_label():
    shown, out = renderer()
    shown.handle(AgentEvent(kind="tool_label", text="Wrapping up", tool_call=ToolCall("call-1", "answer")))
    assert out.getvalue() == ""
    shown.handle(AgentEvent(kind="tool_label", text="Old text", tool_call=ToolCall("call-1", "read_file")))
    shown.handle(AgentEvent(kind="tool_start", step=1, tool_call=call()))
    assert "Reading the config" in out.getvalue()


def test_reset_forgets_descriptions_of_an_abandoned_attempt():
    shown, out = renderer()
    shown.handle(AgentEvent(kind="tool_label", text="Reading the config", tool_call=ToolCall("call-1", "read_file")))
    shown.handle(AgentEvent(kind="tool_label_reset"))
    shown.handle(AgentEvent(kind="tool_start", step=1, tool_call=call()))
    # The call's own heading is complete again, so nothing refers to the abandoned display.
    assert out.getvalue().count("Reading the config") == 2


def batch(count=3):
    return [ToolCall(f"call-{n}", "read_file", {"path": f"f{n}.py"}) for n in range(count)]


def test_concurrent_calls_are_each_followed_by_their_own_result():
    from slipagent.tools.base import ToolResult
    shown, out = renderer()
    calls = batch()
    for item in calls:
        shown.handle(AgentEvent(kind="tool_start", step=1, tool_call=item))
    assert "f0.py" in out.getvalue() and "f1.py" not in out.getvalue()
    for n, item in enumerate(calls):
        shown.handle(AgentEvent(kind="tool_end", step=1, tool_call=item, result=ToolResult.error(f"bad {n}")))
    text = out.getvalue()
    positions = [text.index(marker) for n in range(3) for marker in (f"f{n}.py", f"bad {n}")]
    assert positions == sorted(positions)


def test_next_call_appears_as_soon_as_the_one_before_it_finishes():
    from slipagent.tools.base import ToolResult
    shown, out = renderer()
    calls = batch(2)
    for item in calls:
        shown.handle(AgentEvent(kind="tool_start", step=1, tool_call=item))
    shown.handle(AgentEvent(kind="tool_end", step=1, tool_call=calls[0], result=ToolResult.error("bad 0")))
    assert "f1.py" in out.getvalue()
    shown.handle(AgentEvent(kind="tool_end", step=1, tool_call=calls[1], result=ToolResult.error("bad 1")))
    assert out.getvalue().count("f1.py") == 1
