"""Read-only reviews using the ordinary agent, journal, and file viewers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING

from .tools.base import Tool, ToolResult
from .tools.blocking import run_blocking
from .tools.files import ReadFileTool, MAX_READ_BYTES
from .tools.git import GitStatusTool, GitDiffTool, GitLogTool
from .tools.navigate import ListDirTool
from .tools.search import GlobTool, GrepTool
from .viewerpanels import file_signature

if TYPE_CHECKING:
    from .agent import Agent
    from .terminal import TerminalUI
    from .workspace import Workspace

READ_TOOLS = frozenset({'read_file', 'list_dir', 'glob', 'grep', 'git_status', 'git_diff', 'git_log', 'answer'})
SEVERITIES = ('critical', 'high', 'medium', 'low')


class ReviewState:
    def __init__(self, workspace: Workspace, scope: str, session_id: str) -> None:
        self.workspace = workspace
        self.scope = scope
        self.session_id = session_id
        self.files: dict[str, Any] = {}
        self.report: dict[str, Any] | None = None

    def signature(self, path: str) -> Any:
        return file_signature(self.workspace.resolve(path).stat())

    def stale(self, path: str) -> bool:
        try:
            return path not in self.files or self.files[path] != self.signature(path)
        except (OSError, ValueError):
            return True

    def incomplete(self, reason: str) -> None:
        previous = self.report or {'summary': '', 'coverage': 'The requested scope was not fully assessed.', 'findings': []}
        self.report = {**previous, 'status': 'incomplete', 'summary': reason + ' ' + previous['summary']}

    def text(self) -> str:
        report = self.report
        if report is None:
            return 'Review incomplete: no validated report was submitted.'
        lines = [f"Review {report['status']}", f'Scope: {self.scope}', report['summary'],
                 'Coverage: ' + report['coverage']]
        if not report['findings']:
            lines.append('No defects reported in the reviewed scope.' if report['status'] == 'completed'
                         else 'No findings submitted; review coverage is incomplete.')
        for index, finding in enumerate(report['findings'], 1):
            stale = ' [file changed or source not read]' if self.stale(finding['path']) else ''
            lines += ['', f"{index}. [{finding['severity']}] {finding['title']}",
                      f"{finding['path']}:{finding['line']}{stale}", finding['evidence'],
                      'Uncertainty: ' + finding['uncertainty']]
        return '\n'.join(lines)


class ReviewReadFileTool(ReadFileTool):
    def __init__(self, state: ReviewState) -> None:
        super().__init__(state.workspace)
        self.state = state

    def _read(self, path: str, offset: int, limit: int | None) -> ToolResult:
        try:
            before = self.state.signature(path)
            result = super()._read(path, offset, limit)
            if not result.is_error and before == self.state.signature(path):
                relative = self.workspace.relative(self.workspace.resolve(path))
                self.state.files[relative] = before
            return result
        except (OSError, ValueError) as exc:
            return ToolResult.error(str(exc))


class ReviewAnswerTool(Tool):
    name = 'answer'
    description = ''
    description_prompt = 'tools/review-answer.md'
    parameters = {'type': 'object', 'properties': {
        'status': {'type': 'string', 'enum': ['completed', 'incomplete']},
        'summary': {'type': 'string', 'minLength': 1, 'maxLength': 10000},
        'coverage': {'type': 'string', 'minLength': 1, 'maxLength': 10000},
        'findings': {'type': 'array', 'maxItems': 100, 'items': {
            'type': 'object', 'properties': {
                'severity': {'type': 'string', 'enum': list(SEVERITIES)},
                'title': {'type': 'string', 'minLength': 1, 'maxLength': 500},
                'path': {'type': 'string', 'minLength': 1, 'maxLength': 4096},
                'line': {'type': 'integer', 'minimum': 1},
                'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 10000},
                'uncertainty': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
            }, 'required': ['severity', 'title', 'path', 'line', 'evidence', 'uncertainty'],
            'additionalProperties': False}},
    }, 'required': ['status', 'summary', 'coverage', 'findings'], 'additionalProperties': False}

    def __init__(self, state: ReviewState) -> None:
        self.state = state

    async def run(self, **report: Any) -> ToolResult:
        return await run_blocking(self.submit, report)

    def submit(self, report: dict[str, Any]) -> ToolResult:
        from .fileio import read_regular
        from .tools.editing import split_lines
        try:
            findings = []
            for finding in report['findings']:
                target = self.state.workspace.resolve(finding['path'])
                relative = self.state.workspace.relative(target)
                data = read_regular(target, MAX_READ_BYTES).decode('utf-8')
                if finding['line'] > len(split_lines(data)):
                    return ToolResult.error(f'Finding line is outside {relative}; inspect the source and correct it.')
                if relative not in self.state.files:
                    return ToolResult.error(f'Read {relative} with read_file before submitting a finding about it.')
                findings.append({**finding, 'path': relative})
            findings.sort(key=lambda item: SEVERITIES.index(item['severity']))
            self.state.report = {**report, 'findings': findings}
            return ToolResult.ok({'text': self.state.text(), 'review': self.state.report})
        except (OSError, ValueError, UnicodeError) as exc:
            return ToolResult.error(f'Cannot validate finding: {exc}')


@contextmanager
def review_mode(agent: Agent, scope: str) -> Iterator[ReviewState]:
    """Borrow session owners; restore every tool even on cancellation or failure."""
    registry = agent.registry
    workspace = registry.resources.workspace
    if workspace is None:
        raise ValueError('Review requires a workspace.')
    state = ReviewState(workspace, scope, agent.session_id)
    original = [registry.get(name) for name in registry.names]
    exposed = agent.exposed_tools
    registry.replace([ReviewReadFileTool(state), ListDirTool(workspace), GlobTool(workspace),
                      GrepTool(workspace), GitStatusTool(workspace), GitDiffTool(workspace),
                      GitLogTool(workspace), ReviewAnswerTool(state)])
    agent.exposed_tools = None
    catalog = registry.services.pop('tool_catalog', None)
    registry.services['active_review'] = state
    try:
        yield state
    finally:
        registry.services.pop('active_review', None)
        registry.services['last_review'] = state
        agent.exposed_tools = exposed
        registry.replace([tool for tool in original if tool is not None])
        if catalog is not None:
            registry.services['tool_catalog'] = catalog


async def show_review(terminal: TerminalUI, state: ReviewState) -> None:
    report = state.report
    if report is None or not report['findings']:
        terminal.open_viewer('Review', await run_blocking(state.text))
        return
    labels = await run_blocking(lambda: [
        (str(index), f"[{finding['severity']}] {finding['path']}:{finding['line']} — {finding['title']}"
         + (' [file changed]' if state.stale(finding['path']) else ''))
        for index, finding in enumerate(report['findings'])])
    selected = await terminal.choose('Review Findings', [('report', 'View Complete Report'), *labels])
    if selected == 'report':
        terminal.open_viewer('Review', await run_blocking(state.text))
    elif selected is not None:
        finding = report['findings'][int(selected)]
        if await run_blocking(state.stale, finding['path']):
            terminal.notify('File changed since review; the finding line may be stale.')
        target = await run_blocking(state.workspace.resolve, finding['path'])
        await terminal._read_file(str(target), line=finding['line'])
