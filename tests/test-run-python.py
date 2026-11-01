"""Python scripts use the harness interpreter with shared process ownership."""

import asyncio
import sys
from pathlib import Path

import pytest

from offline.sitecustomize import GUARD_DIRECTORY
from slipagent.tools import build_default_registry
from slipagent.tools.output import CommandArchive, ReadCommandOutputTool
from slipagent.tools.python import RunPythonTool


async def test_python_uses_harness_environment_cwd_math_and_command_logs(workspace, monkeypatch):
    archive = CommandArchive(100000)
    # The implicit workspace path must not shadow installed or standard libraries.
    (workspace.root / 'fractions.py').write_text('raise RuntimeError("workspace shadow imported")')
    monkeypatch.setenv('PYTHONPATH', GUARD_DIRECTORY)
    tool = RunPythonTool(workspace, archive)
    try:
        result = await tool.invoke({'code': "import sys, os\nfrom fractions import Fraction\n"
            "print('MATH_RESULT', Fraction(1, 3) + Fraction(1, 6))\n"
            "print('INTERPRETER', sys.executable)\nprint('PREFIX', sys.prefix)\nprint('CWD', os.getcwd())\n"
            "print('STDIN', repr(sys.stdin.read()))\n"
            "print('OFFLINE_GUARD', hasattr(sys.modules.get('sitecustomize'), 'ExternalNetworkBlocked'))"})
        assert not result.is_error
        for text in ('MATH_RESULT 1/2', 'INTERPRETER ' + sys.executable, 'PREFIX ' + sys.prefix,
                     'CWD ' + str(workspace.root), "STDIN ''", 'OFFLINE_GUARD True'):
            assert text in result.content
        [log_id] = archive.logs
        output = await ReadCommandOutputTool(archive).run(log_id=log_id)
        assert 'MATH_RESULT 1/2' in str(output.content)
        assert archive.logs[log_id].finished
    finally:
        await archive.aclose()


async def test_python_executes_long_source_without_shell_quoting_and_cleans_scripts(workspace, monkeypatch):
    scripts = []
    spawn = asyncio.create_subprocess_exec

    async def observed(*args, **kwargs):
        scripts.append(Path(args[-1]))
        return await spawn(*args, **kwargs)

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', observed)
    result = await RunPythonTool(workspace).invoke({'code': "print('LITERAL $(unused) `unused`')\n#" + 'x' * 150000})
    assert not result.is_error and 'LITERAL $(unused) `unused`' in result.content
    assert scripts and all(not script.exists() and not script.parent.exists() for script in scripts)


async def test_python_calls_are_fresh_and_return_exceptions(workspace):
    tool = RunPythonTool(workspace)
    assert not (await tool.invoke({'code': "previous_value = 42\nprint('stored')"})).is_error
    result = await tool.invoke({'code': "print('PERSISTED', 'previous_value' in globals())"})
    assert not result.is_error and 'PERSISTED False' in result.content
    failed = await tool.invoke({'code': "print('BEFORE_FAILURE')\nraise ValueError('fixture failure')"})
    assert failed.is_error
    assert 'BEFORE_FAILURE' in failed.content and 'ValueError: fixture failure' in failed.content


async def test_python_timeout_reaps_process_and_finishes_log(workspace, monkeypatch):
    children = []
    scripts = []
    spawn = asyncio.create_subprocess_exec

    async def observed(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        children.append(process)
        scripts.append(Path(args[-1]))
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', observed)
    archive = CommandArchive(100000)
    try:
        result = await RunPythonTool(workspace, archive).invoke({'code': 'import time\ntime.sleep(60)', 'timeout': .1})
        assert result.is_error and 'timed out' in result.content
        assert all(child.returncode is not None for child in children)
        assert all(log.finished and log.timed_out for log in archive.logs.values())
        assert all(not script.exists() for script in scripts)
    finally:
        await archive.aclose()


async def test_cancellation_during_python_spawn_reaps_child_before_removing_script(workspace, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    children, scripts = [], []
    spawn = asyncio.create_subprocess_exec

    async def gated(*args, **kwargs):
        child = await spawn(*args, **kwargs)
        children.append(child)
        scripts.append(Path(args[-1]))
        entered.set()
        await release.wait()
        return child

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', gated)
    task = asyncio.create_task(RunPythonTool(workspace).run('import time\ntime.sleep(60)'))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and scripts[0].exists()
    finally:
        release.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    assert all(child.returncode is not None for child in children)
    assert all(not script.exists() for script in scripts)


@pytest.mark.parametrize('arguments', [{'code': ' '}, {'code': 'x' * 262145}, {'code': 'pass', 'timeout': 601}])
async def test_invalid_python_arguments_create_no_script_or_process(workspace, monkeypatch, arguments):
    async def unexpected(*args, **kwargs):
        pytest.fail('invalid arguments started Python')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', unexpected)
    assert (await RunPythonTool(workspace).invoke(arguments)).is_error


async def test_python_tool_is_available_by_default_and_respects_allowlist(workspace):
    registry = build_default_registry(workspace)
    try:
        from slipagent.agent import Agent
        agent = Agent(None, registry, 'test')
        assert 'run_python' in {spec.name for spec in agent._visible_specs(registry.specs())}
        assert not (await registry.invoke('run_python', {'code': "print('READY_TO_CALCULATE')"})).is_error
        agent.exposed_tools = ('load_tools',)
        assert 'run_python' not in {spec.name for spec in agent._visible_specs(registry.specs())}
    finally:
        await registry.aclose()
