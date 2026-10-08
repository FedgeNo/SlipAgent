
from slipagent.types import content_text

from slipagent.types import decode_json_content
import json
import shlex
import sys

import pytest

from slipagent.tools.output import CommandArchive, ReadCommandOutputTool
from slipagent.tools import build_default_registry
from slipagent.tools.shell import RunCommandTool
from slipagent.agent import Agent
from slipagent.workspace import Workspace
from test_agent import StubClient, completion, call


async def test_complete_output_pages_and_tail_survive_preview_truncation(tmp_path):
    archive = CommandArchive(100_000, directory=tmp_path)
    try:
        log = archive.start("command", step_id=7, call_id="call/unsafe")
        original = "start\n" + "αβ🙂\n" * 6000 + "LAST LINE\n"
        log.write("stdout", original)
        log.finish(0, False)
        tool = ReadCommandOutputTool(archive)
        content, offset = "", 0
        while True:
            page = decode_json_content((await tool.run(log_id=log.id, offset=offset, limit=1111)).content)
            content += page["content"]
            offset = page["next_offset"]
            if offset is None:
                break
        assert content == original
        tail = decode_json_content((await tool.run(log_id=log.id, tail=True, limit=10)).content)
        assert tail["content"] == original[-10:]
        records = decode_json_content((await tool.run(step_id=7, call_id="call/unsafe")).content)
        assert records["logs"][0]["log_id"] == log.id
        assert not (tmp_path / "unsafe").exists()
    finally:
        await archive.aclose()
    assert list(tmp_path.iterdir()) == []


async def test_shell_archives_middle_and_timeout_output_with_call_provenance(tmp_path):
    registry = build_default_registry(Workspace(tmp_path))
    archive = registry.services["command_archive"]
    command = shlex.join([sys.executable, "-c", "import time; print('A'*35000+'MIDDLE'+'Z'*35000, flush=True); time.sleep(10)"])
    client = StubClient([completion(tool_calls=[call("run_command", command=command, timeout=.5)]), completion("Inspected timeout")])
    agent = Agent(client, registry, "test")
    try:
        await agent.run("Run a test command")
        log = next(iter(archive.logs.values()))
        assert (log.step_id, log.call_id, log.timed_out) == (1, "call_run_command", True)
        assert log.streams["stdout"].retained_bytes == 70007
        result = await registry.invoke("read_command_output", {"log_id": log.id, "offset": 34997, "limit": 12})
        assert decode_json_content(result.content)["content"] == "AAAMIDDLEZZZ"
        observation = agent.history.steps[0].messages[-1].content
        assert log.id in observation["content"] and "Effects may be partial" in observation["content"]
        directory = archive.directory
        agent.reset()
        assert not directory.exists() and not archive.logs
    finally:
        await registry.aclose()


async def test_disk_failure_does_not_abort_or_block_the_command(tmp_path, monkeypatch):
    archive = CommandArchive(10000, directory=tmp_path)
    def fail(self):
        raise OSError("simulated disk full")
    monkeypatch.setattr(CommandArchive, "directory", property(fail))
    tool = RunCommandTool(Workspace(tmp_path), archive)
    result = await tool.run(shlex.join([sys.executable, "-c", "print('observed output')"]))
    assert not result.is_error
    assert "observed output" in content_text(result.content)
    assert "simulated disk full" in content_text(result.content) and "Lost 16 UTF-8 bytes" in content_text(result.content)
    await archive.aclose()
    assert list(tmp_path.iterdir()) == []


async def test_quota_never_evicts_prior_logs_and_reports_lost_bytes(tmp_path):
    archive = CommandArchive(10, directory=tmp_path)
    try:
        first = archive.start("first")
        first.write("stdout", "abc")
        first.finish(0, False)
        second = archive.start("second")
        second.write("stderr", "🙂🙂🙂")
        second.finish(-9, True)
        assert archive.used_bytes == 7
        assert second.streams["stderr"].lost_bytes == 8
        page = decode_json_content((await ReadCommandOutputTool(archive).run(log_id=second.id, stream="stderr")).content)
        assert page["content"] == "🙂"
        assert page["lost_bytes"] == 8 and page["timed_out"] is True
        assert "quota" in page["retention_error"]
        archive.clear()
        assert archive.used_bytes == 0
        assert (await ReadCommandOutputTool(archive).run(log_id=first.id)).is_error
    finally:
        await archive.aclose()
