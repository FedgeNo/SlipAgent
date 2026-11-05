"""TTY mode drift must not turn SSH mouse reports into echoed input."""

import asyncio
import importlib
import os
import json
import shlex
import sys
from unittest.mock import Mock

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.input.defaults import create_input

from slipagent.terminal import TerminalUI

helpers = importlib.import_module('test-terminal')
display = helpers.display
wait_until = helpers.wait_until
pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX terminal modes')


@pytest.mark.parametrize('background', [False, True])
async def test_run_command_cannot_change_ui_terminal_or_consume_mouse_input(tmp_path, background):
    import termios
    import tty
    from test_cli_e2e import cli_environment

    # Only the disposable slave is inherited by the harness subprocess. Never
    # replace pytest's stdin or let the probe touch the developer's terminal.
    master, slave = os.openpty()
    process = None
    try:
        tty.setraw(slave)
        original = termios.tcgetattr(slave)
        packet = b'\x1b[<64;1;2M'
        os.write(master, packet)
        probe = '''import json, os, termios
from pathlib import Path
terminal = os.isatty(0)
if terminal:
    modes = termios.tcgetattr(0)
    modes[3] |= termios.ECHO | termios.ICANON
    termios.tcsetattr(0, termios.TCSANOW, modes)
Path('stdin-probe.json').write_text(json.dumps({'terminal': terminal,
    'input': None if terminal else os.read(0, 1).hex()}))
'''
        command = shlex.join([sys.executable, '-c', probe])
        driver = '''import asyncio, sys
from pathlib import Path
from slipagent.jobs import CommandJobs, CommandJobsTool
from slipagent.tools.output import CommandArchive
from slipagent.tools.shell import RunCommandTool
from slipagent.workspace import Workspace
async def main():
    archive, jobs = CommandArchive(100000), CommandJobs()
    try:
        result = await RunCommandTool(Workspace(Path.cwd()), archive, jobs).run(
            sys.argv[1], background=sys.argv[2] == 'True')
        assert not result.is_error, result.content
        if sys.argv[2] == 'True':
            result = await CommandJobsTool(jobs).run('wait', result.content['job_id'], timeout=5)
            assert result.content['state'] == 'completed', result.content
    finally:
        await jobs.aclose()
        await archive.aclose()
asyncio.run(main())
'''
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-c', driver, command, str(background), cwd=tmp_path,
            stdin=slave, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, env=cli_environment())
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, (stdout, stderr)
        assert termios.tcgetattr(slave) == original
        assert json.loads((tmp_path / 'stdin-probe.json').read_text()) == {'terminal': False, 'input': ''}
        os.set_blocking(slave, False)
        assert os.read(slave, len(packet)) == packet
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.communicate()
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize('drift', ['cooked', 'input_translation'])
async def test_mouse_and_interrupt_recover_after_tty_mode_drift(display, drift):
    import termios
    sink, _, output, _, snapshot = display
    master, slave = os.openpty()
    original = termios.tcgetattr(slave)
    stream = os.fdopen(os.dup(slave), 'r', encoding='utf-8')
    input = create_input(stdin=stream)
    ui = TerminalUI(lambda width: 'readout', sink, input=input, output=output)
    interrupt = Mock()
    ui.set_interrupt_handler(interrupt)
    task = asyncio.create_task(ui.run())
    try:
        await wait_until(lambda: ui.app.is_running and not termios.tcgetattr(slave)[3] & termios.ICANON)
        ui.working = True
        await helpers.render_after(ui, lambda: ui.write('\n'.join(f'output {n}' for n in range(60))))
        os.write(master, b'unfinished draft')
        await wait_until(lambda: ui.input.text == 'unfinished draft')
        await helpers.render_after(ui, lambda: None)
        before = ui.transcript.vertical_scroll
        assert before > 0
        modes = termios.tcgetattr(slave)
        if drift == 'cooked':
            modes[3] |= termios.ECHO | termios.ICANON | termios.ISIG
            ui.output.write_raw('echoed-mouse-data')
            ui.output.flush()
            assert any('echoed-mouse-data' in line for line in snapshot())
        else:
            modes[0] |= termios.IXON | termios.ICRNL
        termios.tcsetattr(slave, termios.TCSANOW, modes)
        os.write(master, b'\x1b[<64;1;2M')
        await wait_until(lambda: not termios.tcgetattr(slave)[3] & (termios.ECHO | termios.ICANON | termios.ISIG))
        await wait_until(lambda: not termios.tcgetattr(slave)[0] & (termios.IXON | termios.ICRNL))
        await wait_until(lambda: ui.transcript.vertical_scroll < before)
        await wait_until(lambda: all('echoed-mouse-data' not in line for line in snapshot()))
        assert ui.input.text == 'unfinished draft'
        assert ui._lines.empty()
        os.write(master, b'\x1b')
        await wait_until(lambda: interrupt.call_count == 1)
        assert ui.input.text == 'unfinished draft'
    finally:
        ui.close()
        await task
        assert termios.tcgetattr(slave) == original
        input.close()
        stream.close()
        os.close(master)
        os.close(slave)


async def test_deliberate_terminal_suspension_keeps_cooked_mode(display, monkeypatch):
    import termios
    sink, _, output, _, _ = display
    master, slave = os.openpty()
    stream = os.fdopen(os.dup(slave), 'r', encoding='utf-8')
    input = create_input(stdin=stream)
    ui = TerminalUI(lambda width: 'readout', sink, input=input, output=output)
    tick = asyncio.Event()
    original_refresh = ui._refresh_needed

    def refreshed(last, now):
        if ui.app._running_in_terminal:
            tick.set()
        return original_refresh(last, now)

    monkeypatch.setattr(ui, '_refresh_needed', refreshed)
    task = asyncio.create_task(ui.run())
    try:
        await wait_until(lambda: ui.app.is_running and not termios.tcgetattr(slave)[3] & termios.ICANON)
        with set_app(ui.app):
            async with in_terminal():
                await asyncio.wait_for(tick.wait(), 3)
                assert termios.tcgetattr(slave)[3] & termios.ICANON
                assert termios.tcgetattr(slave)[3] & termios.ECHO
        assert not termios.tcgetattr(slave)[3] & termios.ICANON
    finally:
        ui.close()
        await task
        input.close()
        stream.close()
        os.close(master)
        os.close(slave)
