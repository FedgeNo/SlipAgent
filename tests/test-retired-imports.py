"""Retained callbacks import from accepted source after their namespace retires."""

from importlib import import_module

import pytest

run_copy = import_module('test-runtime').run_copy
pytestmark = pytest.mark.slow  # Reload subprocesses; run with --slow.


def test_retained_callback_imports_latest_accepted_helpers(run_copy):
    run_copy('''
        import sys
        (root / 'retained_helper.py').write_text('VALUE = "first"\\n')
        (root / 'tools' / 'retained_callback.py').write_text(
            'def read():\\n    from ..retained_helper import VALUE\\n    return VALUE\\n')
        frame.request()
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        previous = frame.namespace
        callback = sys.modules[previous + '.tools.retained_callback'].read
        (root / 'retained_helper.py').write_text('VALUE = "second"\\n')
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        assert previous not in sys.modules
        assert callback() == 'second'
        (root / 'retained_helper.py').write_text('VALUE = "unaccepted disk edit"\\n')
        (root / 'cli.py').write_text('invalid syntax !!!')
        await frame.checkpoint()
        assert frame.generation == 2
        assert callback() == 'second'
    ''')


def test_retained_commands_callback_survives_multiple_reloads(run_copy):
    run_copy('''
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from slipagent.terminal import TerminalUI
        with create_pipe_input() as pipe:
            terminal = TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            renderer.terminal = terminal
            try:
                frame.request()
                await frame.checkpoint()
                assert frame.generation == 1, sink.getvalue()
                retained = terminal._choose_command
                shown = []
                async def choose(title, options):
                    shown.append((title, options))
                    return None
                terminal.choose = choose
                for generation in range(2, 5):
                    frame.request()
                    await frame.checkpoint()
                    assert frame.generation == generation, sink.getvalue()
                    await retained()
                assert len(shown) == 3
                assert all(title == 'Commands' and options for title, options in shown)
            finally:
                terminal.close()
                renderer.terminal = None
    ''')


def test_hashline_switch_and_loaded_definitions_survive_reload(run_copy):
    run_copy('''
        from slipagent.tools.hashline import HASHLINE_NAMES, set_hashline
        from slipagent.tools.discovery import tool_catalog
        tool_catalog(registry).loaded.update(HASHLINE_NAMES)
        frame.request()
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert HASHLINE_NAMES <= tool_catalog(registry).loaded
        await cli._handle_command(session, '/hashline off')
        frame.request()
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        assert not HASHLINE_NAMES & set(registry.names)
        await cli._handle_command(session, '/hashline on')
        frame.request()
        await frame.checkpoint()
        assert frame.generation == 3, sink.getvalue()
        assert HASHLINE_NAMES <= set(registry.names)
        assert HASHLINE_NAMES <= registry.builtin_names
    ''')
