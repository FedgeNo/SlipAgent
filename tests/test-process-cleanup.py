"""Process cleanup remains bounded when detached children retain output pipes."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from types import SimpleNamespace

import pytest

from slipagent.jobs import CommandJobs
from slipagent.tools import shell
from slipagent.tools.output import CommandArchive


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.01)


@pytest.fixture(params=["exited", "running"])
async def detached_process(tmp_path, monkeypatch, archive, request, caplog):
    if os.name != "posix":
        pytest.skip("Detached POSIX process groups")
    monkeypatch.setattr(shell, "CLEANUP_TIMEOUT", .05)
    child = textwrap.dedent('''
        import sys, time
        from pathlib import Path
        print('stdout before detaching', flush=True)
        print('stderr before detaching', file=sys.stderr, flush=True)
        sys.stdout.buffer.write(b'\\xe2\\x82')
        sys.stdout.buffer.flush()
        Path('ready').touch()
        deadline = time.monotonic() + 10
        while not Path('release').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        Path('exited').touch()
    ''')
    parent = ("import subprocess, sys; subprocess.Popen([sys.executable, '-u', '-c', "
              + repr(child) + "], start_new_session=True)")
    if request.param == "running":
        parent += "; import time; time.sleep(60)"
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", parent, cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    tasks = []

    def start(operation):
        task = asyncio.create_task(operation)
        tasks.append(task)
        return task

    try:
        await wait_until(lambda: (tmp_path / "ready").exists()
                         and (request.param == "running" or process.returncode is not None))
        yield SimpleNamespace(process=process, start=start, exited=tmp_path / "exited")
        assert process.returncode is not None, "The command's parent process was not reaped"
        assert not [record for record in caplog.get_records("call")
                    if record.name == "asyncio" and record.levelno >= 40]
    finally:
        (tmp_path / "release").touch()
        await wait_until((tmp_path / "exited").exists)
        if process.returncode is None:
            process.kill()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        await asyncio.wait_for(process.wait(), 3)


@pytest.fixture
async def archive(tmp_path):
    archive = CommandArchive(100_000, directory=tmp_path)
    try:
        yield archive
    finally:
        await archive.aclose()


async def finished_within_cleanup_deadline(task):
    done, _ = await asyncio.wait({task}, timeout=.6)
    assert task in done, "Cleanup waited for a detached process's pipe to close"


async def test_timeout_closes_detached_pipes_and_retains_partial_output(detached_process, archive):
    log = archive.start("detached command")
    task = detached_process.start(shell.capture_process(detached_process.process, .05, log=log))
    await finished_within_cleanup_deadline(task)
    stdout, stderr, timed_out = await task
    assert timed_out and log.finished and log.timed_out
    assert stdout == "stdout before detaching\n�"
    assert "stderr before detaching" in stderr
    assert "output may be incomplete" in stderr and "detached processes may still be running" in stderr
    assert "output may be incomplete" in log.notice()
    assert not detached_process.exited.exists()


async def test_cancellation_closes_detached_pipes_despite_repeated_cancel(detached_process, archive):
    log = archive.start("cancel detached command")
    ready = asyncio.Event()
    log.monitor_output = lambda channel, text: ready.set()
    task = detached_process.start(shell.capture_process(detached_process.process, 60, log=log))
    await asyncio.wait_for(ready.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await finished_within_cleanup_deadline(task)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert log.finished and not log.timed_out
    assert "output may be incomplete" in log.notice()
    assert log.streams["stdout"].retained_bytes >= len("stdout before detaching\n")
    assert not detached_process.exited.exists()


async def test_kill_and_drain_closes_detached_pipes(detached_process, archive):
    log = archive.start("stop detached command before capture")
    task = detached_process.start(shell.kill_and_drain(detached_process.process, log=log))
    await finished_within_cleanup_deadline(task)
    await task
    assert "output may be incomplete" in log.notice()
    assert log.streams["stdout"].retained_bytes > 0 and log.streams["stderr"].retained_bytes > 0
    assert not detached_process.exited.exists()
    log.finish(detached_process.process.returncode, False)


async def test_command_reports_detached_pipe_cutoff_without_an_archive(detached_process, workspace, monkeypatch):
    async def spawn(*args, **kwargs):
        return detached_process.process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", spawn)
    task = detached_process.start(shell.RunCommandTool(workspace).run("detached command", timeout=.05))
    await finished_within_cleanup_deadline(task)
    result = await task
    assert result.is_error and "timed out" in result.content
    assert "stdout before detaching" in result.content and "stderr before detaching" in result.content
    assert "output may be incomplete" in result.content and "detached processes may still be running" in result.content
    assert not detached_process.exited.exists()


async def test_cancellation_during_spawn_closes_detached_pipes(detached_process, archive, workspace, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        return detached_process.process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", spawn)
    task = detached_process.start(shell.RunCommandTool(workspace, archive).run("detached command"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
    finally:
        release.set()
    await finished_within_cleanup_deadline(task)
    with pytest.raises(asyncio.CancelledError):
        await task
    log = next(iter(archive.logs.values()))
    assert log.finished and "output may be incomplete" in log.notice()
    assert not detached_process.exited.exists()


async def test_background_shutdown_closes_detached_pipes(detached_process, archive, workspace, monkeypatch):
    async def spawn(*args, **kwargs):
        return detached_process.process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", spawn)
    jobs = CommandJobs()
    try:
        result = await shell.RunCommandTool(workspace, archive, jobs).run("detached command", background=True)
        log = archive.logs[result.content["log_id"]]
        await wait_until(lambda: log.streams["stdout"].retained_bytes or log.finished)
        assert log.streams["stdout"].retained_bytes, jobs.jobs[log.id].result
        task = detached_process.start(jobs.aclose())
        await finished_within_cleanup_deadline(task)
        await task
        assert not jobs.active and log.finished
        assert "output may be incomplete" in log.notice()
        assert not detached_process.exited.exists()
    finally:
        # The fixture releases the child before awaiting pending shutdown tasks.
        if jobs.active:
            detached_process.start(jobs.aclose())


async def test_capture_error_still_closes_detached_pipes(detached_process, archive, monkeypatch):
    log = archive.start("broken output callback")

    def fail_write(channel, text):
        raise OSError("output callback failed")

    monkeypatch.setattr(log, "write", fail_write)
    task = detached_process.start(shell.capture_process(detached_process.process, 60, log=log))
    await finished_within_cleanup_deadline(task)
    with pytest.raises(OSError, match="output callback failed"):
        await task
    assert log.finished
    assert not detached_process.exited.exists()
