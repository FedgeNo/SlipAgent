"""Feed search keeps drafts and running work separate from view navigation."""

import asyncio
import importlib
from unittest.mock import Mock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.formatted_text import fragment_list_to_text
from wcwidth import width as display_width

from slipagent.feedsearch import excerpt
from slipagent.terminal import TerminalUI

terminal_tests = importlib.import_module('test-terminal')
display = terminal_tests.display
wait_until = terminal_tests.wait_until


@pytest.mark.parametrize('text, term, columns', [
    ('left ' * 30 + 'Needle' + ' right' * 30, 'Needle', 60),
    ('漢字 ' * 30 + '👩‍💻' + ' 漢字' * 30, '👩‍💻', 30),
    ('prefix é suffix', 'é', 10),
    ('verylongmatch', 'verylongmatch', 5),
])
def test_excerpts_fit_one_row_and_highlight_the_literal_match(text, term, columns):
    start = text.index(term)
    fragments = excerpt(text, start, start + len(term), columns)
    shown = fragment_list_to_text(fragments)
    assert display_width(shown) <= columns
    highlighted = ''.join(part for style, part in fragments if 'reverse' in style)
    assert highlighted and term.startswith(highlighted)
    if display_width(text) > columns:
        assert shown.endswith('…')
    if term == 'Needle':
        position = shown.index(term)
        assert abs(position + len(term) / 2 - columns / 2) <= 2


async def test_f3_searches_while_working_and_clicking_returns_to_live_feed(display):
    sink, sizes, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'readout', sink, input=pipe, output=output)
        interrupted = Mock()
        ui.set_interrupt_handler(interrupted)
        task = asyncio.create_task(ui.run())
        try:
            ui.write('before\n' * 10)
            ui.write('left ' * 30 + '[A.B]' + ' right' * 30)
            ui.write('other\n' * 10)
            ui.write('another [a.b] result')
            ui.write('after\n' * 40)
            ui.working = True
            pipe.send_text('unfinished draft')
            await wait_until(lambda: ui.input.text == 'unfinished draft')
            pipe.send_text('\x1bOR')
            await wait_until(lambda: ui._feed_search.active)
            pipe.send_text('[a.b]')
            await wait_until(lambda: ui._feed_search.status.startswith('2 results'))
            await wait_until(lambda: '[A.B]' in snapshot()[1])
            assert not snapshot()[2].strip()
            assert '[a.b]' in snapshot()[3]
            assert snapshot()[1].rstrip().endswith('…')
            assert screen.buffer[1][snapshot()[1].index('[A.B]')].reverse
            ui.write('work continues while searching')
            assert ui.working and ui._lines.empty()
            interrupted.assert_not_called()
            column = snapshot()[1].index('[A.B]') + 1
            pipe.send_text(f'\x1b[<0;{column};2M\x1b[<0;{column};2m')
            await wait_until(lambda: not ui._feed_search.show_results)
            await wait_until(lambda: any('[A.B]' in line for line in snapshot()[:17]))
            assert not ui.transcript._following
            pipe.send_text('\x1b')
            await wait_until(lambda: not ui._feed_search.active)
            assert ui.input.text == 'unfinished draft'
            assert ui.working and ui._lines.empty()
            interrupted.assert_not_called()
        finally:
            ui.close()
            await task


async def test_search_crosses_wraps_and_retargets_results_after_resize(display):
    from prompt_toolkit.data_structures import Size
    sink, sizes, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'readout', sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            ui.write('x' * 95 + 'crossboundary' + ' rest' * 40)
            ui.write('tail\n' * 40)
            pipe.send_text('\x1bORcrossboundary')
            await wait_until(lambda: ui._feed_search.status.startswith('1 results'))
            sizes[0] = Size(rows=24, columns=50)
            screen.resize(lines=24, columns=50)
            ui.app._on_resize()
            await ui._feed_search.activate(0)
            await wait_until(lambda: not ui._feed_search.show_results)
            assert ui._feed_search.columns == 49
            assert any('crossboundary' in line for line in snapshot()[:17])
            ui.refresh()
            assert ui._feed_search.active
            pipe.send_text('\x1b')
            await wait_until(lambda: not ui._feed_search.active)
        finally:
            ui.close()
            await task


async def test_search_keyboard_activation_no_matches_and_escape_preserve_the_turn(display):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'readout', sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            ui.write('alpha target omega\n' + 'tail\n' * 40)
            pipe.send_text('\x1bORmissing')
            await wait_until(lambda: ui._feed_search.status.startswith('0 results'))
            ui._feed_search.field.text = 'target'
            await wait_until(lambda: ui._feed_search.status.startswith('1 results'))
            await terminal_tests.send_keys(ui, pipe, '\t')
            await wait_until(lambda: ui.app.layout.current_window is ui._feed_search.window)
            await terminal_tests.send_keys(ui, pipe, '\t')
            assert ui._feed_search.window.content.__dict__['_keyboard_key'] == ('feed-search', 0)
            await terminal_tests.send_keys(ui, pipe, '\r')
            await wait_until(lambda: not ui._feed_search.show_results)
            assert any('alpha target omega' in line for line in snapshot()[:17])
            assert ui._lines.empty()
            pipe.send_text('\x1b')
            await wait_until(lambda: not ui._feed_search.active)
            assert ui.input.text == '' and ui._lines.empty()
        finally:
            ui.close()
            await task
