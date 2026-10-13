"""Concurrent tool announcements cannot separate grep from its file links."""

import io

import pytest

from slipagent.cli import Renderer, Style
from slipagent.events import AgentEvent
from slipagent.tools.base import ToolResult
from slipagent.types import ToolCall


class Terminal:
    def __init__(self):
        self.lines = []
        self.running = []

    def write(self, text, **options):
        self.lines.extend(text.splitlines())

    def track_call(self, call, step):
        self.running.append((step, call.id))

    def finish_call(self, *args):
        pass

    def write_directory_reference(self, path):
        self.lines.append('Directory: ' + path)

    def write_file_reference(self, path, *, line=1):
        self.lines.append(f'File: {path}:{line}')


@pytest.mark.parametrize('failed', [False, True])
def test_grep_heading_and_results_stay_together_after_concurrent_starts(failed):
    terminal = Terminal()
    renderer = Renderer(Style(False), io.StringIO(), verbose=False)
    renderer.terminal = terminal
    grep = ToolCall('search', 'grep', {'pattern': 'needle', 'path': '.'})
    others = [ToolCall('read', 'read_file', {'path': 'unrelated.py'}),
              ToolCall('list', 'list_dir', {'path': 'src'})]
    for call in [grep, *others]:
        renderer.handle(AgentEvent(kind='tool_start', step=1, tool_call=call))
    assert terminal.running == [(1, 'search'), (1, 'read'), (1, 'list')]
    assert not any('⚙ grep(' in line for line in terminal.lines)
    result = ToolResult.error('Search failed') if failed else ToolResult.ok('a.py:12:needle\nb.py:34:needle')
    renderer.handle(AgentEvent(kind='tool_end', step=1, tool_call=grep, result=result))
    start = next(i for i, line in enumerate(terminal.lines) if '⚙ grep(' in line)
    group = terminal.lines[start:]
    assert not any('⚙ read_file(' in line or '⚙ list_dir(' in line for line in group)
    if failed:
        assert any('Search failed' in line for line in group)
        assert not any('File: a.py:' in line for line in group)
    else:
        assert group[-2:] == ['File: a.py:12', 'File: b.py:34']
