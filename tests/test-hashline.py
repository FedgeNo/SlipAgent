"""Optional anchored edits preserve normal editing and disappear when disabled."""

import io
import sys

import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.tools import build_default_registry
from slipagent.tools.hashline import HASHLINE_NAMES, anchored_view, set_hashline
from slipagent.tools.discovery import tool_catalog
from slipagent.types import Message, content_text
from test_agent import StubClient, completion, call
from test_cli_e2e import metadata_server


@pytest.fixture
def registry(workspace):
    return build_default_registry(workspace)


async def read(registry, path='example.txt', **arguments):
    result = await registry.invoke('read_hashlines', {'path': path, **arguments})
    assert not result.is_error, result.content
    return result.content


async def edit(registry, view, edits, path='example.txt'):
    return await registry.invoke('edit_hashlines', {'path': path, 'revision': view['revision'], 'edits': edits})


@pytest.mark.parametrize('ending', ['\n', '\r\n', '\r'])
@pytest.mark.parametrize('final', [False, True])
async def test_read_replace_and_chain_preserve_line_endings(workspace, registry, ending, final):
    path = workspace.root / 'example.txt'
    before = ending.join(['same', 'same', '界 👩\u200d💻 e\u0301', 'last']) + (ending if final else '')
    path.write_bytes(before.encode())
    view = await read(registry)
    assert view['lines'][0]['anchor'] != view['lines'][1]['anchor']
    assert view['lines'][0]['anchor'].split(':')[1] == view['lines'][1]['anchor'].split(':')[1]
    result = await edit(registry, view, [
        {'op': 'replace', 'start': view['lines'][1]['anchor'], 'lines': ['changed', 'extra']},
        {'op': 'replace', 'start': view['lines'][3]['anchor'], 'lines': ['new last']},
    ])
    assert not result.is_error, result.content
    expected = ending.join(['same', 'changed', 'extra', '界 👩\u200d💻 e\u0301', 'new last']) + (ending if final else '')
    assert path.read_bytes() == expected.encode()
    fresh = result.content
    assert fresh['revision'] != view['revision']
    result = await edit(registry, fresh, [{'op': 'replace', 'start': fresh['lines'][0]['anchor'], 'lines': ['chained']}])
    assert not result.is_error
    assert path.read_bytes().startswith(('chained' + ending).encode())


@pytest.mark.parametrize('operation', ['insert_before', 'insert_after', 'replace'])
@pytest.mark.parametrize('final', [False, True])
async def test_insert_and_delete_at_file_boundaries(workspace, registry, operation, final):
    path = workspace.root / 'example.txt'
    path.write_bytes(b'first\nlast' + (b'\n' if final else b''))
    view = await read(registry)
    result = await edit(registry, view, [{'op': operation, 'start': view['lines'][1]['anchor'],
                                         'lines': [] if operation == 'replace' else ['added', 'again']}])
    assert not result.is_error, result.content
    if operation == 'replace':
        expected = b'first\n'
    elif operation == 'insert_before':
        expected = b'first\nadded\nagain\nlast' + (b'\n' if final else b'')
    else:
        expected = b'first\nlast\nadded\nagain' + (b'\n' if final else b'')
    assert path.read_bytes() == expected


@pytest.mark.parametrize('change', ['changed', 'moved', 'same-lines-different-ending'])
async def test_stale_revision_rejects_without_overwriting_external_changes(workspace, registry, change):
    path = workspace.root / 'example.txt'
    path.write_bytes(b'first\nsame\nlast\n')
    view = await read(registry)
    changed = {'changed': b'first\nsame\nexternal\n', 'moved': b'same\nfirst\nlast\n',
               'same-lines-different-ending': b'first\r\nsame\r\nlast\r\n'}[change]
    path.write_bytes(changed)
    result = await edit(registry, view, [{'op': 'replace', 'start': view['lines'][1]['anchor'], 'lines': ['bad']}])
    assert result.is_error and 'revision changed' in result.text()
    assert path.read_bytes() == changed


@pytest.mark.parametrize('problem', ['overlap', 'bad-anchor', 'duplicate-insertion', 'inside-insertion', 'copied-anchor', 'embedded-newline', 'reverse-range'])
async def test_invalid_batch_applies_nothing(workspace, registry, problem):
    path = workspace.root / 'example.txt'
    before = b'one\ntwo\nthree\n'
    path.write_bytes(before)
    view = await read(registry)
    one, two, three = [line['anchor'] for line in view['lines']]
    edits = [{'op': 'replace', 'start': one, 'lines': ['valid first edit']}]
    edits += {
        'overlap': [{'op': 'replace', 'start': one, 'end': two, 'lines': ['bad']}],
        'bad-anchor': [{'op': 'replace', 'start': '2:0000000000', 'lines': ['bad']}],
        'duplicate-insertion': [{'op': 'insert_before', 'start': two, 'lines': ['a']}, {'op': 'insert_before', 'start': two, 'lines': ['b']}],
        'inside-insertion': [{'op': 'replace', 'start': two, 'end': three, 'lines': ['a']}, {'op': 'insert_before', 'start': three, 'lines': ['b']}],
        'copied-anchor': [{'op': 'replace', 'start': two, 'lines': [two + ' source']}],
        'embedded-newline': [{'op': 'replace', 'start': two, 'lines': ['a\nb']}],
        'reverse-range': [{'op': 'replace', 'start': three, 'end': two, 'lines': ['bad']}],
    }[problem]
    result = await edit(registry, view, edits)
    assert result.is_error
    assert path.read_bytes() == before


async def test_paging_empty_binary_and_outside_paths(workspace, registry, tmp_path_factory):
    path = workspace.root / 'example.txt'
    path.write_bytes(b'a\nb\nc\n')
    view = await read(registry, offset=2, limit=1)
    assert view['total_lines'] == 3 and view['next_offset'] == 3
    assert view['lines'] == anchored_view(path.read_bytes(), 2, 1)['lines']
    assert len(view['lines']) == 1 and view['lines'][0]['text'] == 'b'
    path.write_bytes(b'')
    assert (await read(registry))['lines'] == []
    path.write_bytes(b'\xff')
    assert (await registry.invoke('read_hashlines', {'path': 'example.txt'})).is_error
    outside = tmp_path_factory.mktemp('hashline-outside') / 'file.txt'
    outside.write_text('private')
    assert not workspace.contains(outside) and not outside.is_relative_to(workspace.temporary_directory)
    assert (await registry.invoke('read_hashlines', {'path': str(outside)})).is_error
    result = await registry.invoke('edit_hashlines', {'path': str(outside), 'revision': '0' * 64,
                                                      'edits': [{'op': 'replace', 'start': '1:0000000000', 'lines': ['bad']}]})
    assert result.is_error and outside.read_text() == 'private'


async def test_noop_has_no_write_or_checkpoint(workspace, registry, monkeypatch):
    path = workspace.root / 'example.txt'
    path.write_text('same\n')
    view = await read(registry)
    def fail(*args):
        pytest.fail('No-op attempted a write')
    monkeypatch.setattr('slipagent.tools.hashline._atomic_write', fail)
    result = await edit(registry, view, [{'op': 'replace', 'start': view['lines'][0]['anchor'], 'lines': ['same']}])
    assert not result.is_error and result.content['changed'] is False


async def test_changed_during_planning_is_rejected(workspace, registry, monkeypatch):
    from slipagent.tools import hashline
    path = workspace.root / 'example.txt'
    path.write_text('before\n')
    view = await read(registry)
    original = hashline.apply_hashline_edits
    def changed(text, edits):
        result = original(text, edits)
        path.write_text('external\n')
        return result
    monkeypatch.setattr(hashline, 'apply_hashline_edits', changed)
    result = await edit(registry, view, [{'op': 'replace', 'start': view['lines'][0]['anchor'], 'lines': ['after']}])
    assert result.is_error and 'changed during edit preparation' in result.text()
    assert path.read_text() == 'external\n'


async def test_deferred_definitions_and_disabled_prompts(workspace, registry, monkeypatch):
    agent = Agent(StubClient([]), registry, 'test')
    context = await agent._queued_context(registry.specs(), 1)
    assert 'read_hashlines' in content_text(context[0].content)
    assert 'Copy anchors exactly' not in content_text(context[0].content)
    assert 'edit_hashlines' not in {spec.name for spec in agent._visible_specs(registry.specs())}
    result = await registry.invoke('load_tools', {'names': sorted(HASHLINE_NAMES)})
    assert not result.is_error
    assert result.content['loaded'] == sorted(HASHLINE_NAMES)
    assert result.content['definitions'] == []
    agent.extend([Message.assistant(None, [call('load_tools', names=sorted(HASHLINE_NAMES))]),
                  Message.tool_result('call_load_tools', {'tool': 'load_tools', 'status': 'success', 'content': result.content})])
    context = await agent._queued_context(registry.specs(), 1)
    assert 'Copy anchors exactly' in content_text(context[0].content)
    assert HASHLINE_NAMES <= {spec.name for spec in agent._visible_specs(registry.specs())}
    catalog = tool_catalog(registry)
    rebuilt = build_default_registry(workspace, services=registry.services)
    assert catalog.loaded >= HASHLINE_NAMES
    assert rebuilt.builtin_names >= HASHLINE_NAMES
    set_hashline(registry, False)
    context = await agent._queued_context(registry.specs(), 1)
    supplied = '\n'.join(content_text(message.content) for message in context)
    assert 'Copy anchors exactly' not in supplied
    assert not HASHLINE_NAMES & {spec.name for spec in agent._visible_specs(registry.specs())}
    assert not HASHLINE_NAMES & set(registry.names)
    assert (await registry.invoke('load_tools', {'names': ['edit_hashlines']})).is_error
    assert not HASHLINE_NAMES & set(build_default_registry(workspace, services=registry.services).names)


async def test_disabled_cli_and_toggle_restore_tools(tmp_path, metadata_server):
    session = await cli.build_session(cli.build_parser().parse_args(['--no-mcp', '--no-hashline', '-w', str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    try:
        assert not HASHLINE_NAMES & set(session.registry.names)
        await cli._handle_command(session, '/hashline on')
        assert HASHLINE_NAMES <= set(session.registry.names)
        assert all(tool.name in session.registry.builtin_names for tool in session.reloader._builtin_tools)
        await cli._handle_command(session, '/hashline off')
        assert not HASHLINE_NAMES & set(session.registry.names)
        await cli._handle_command(session, '/hashline')
        assert 'Hashline editing OFF' in session.renderer.stream.getvalue()
        assert 'read_file' in session.registry and 'edit_file' in session.registry
    finally:
        await cli._shutdown(session)


async def test_agent_loads_reads_edits_checks_and_rewinds(workspace, registry):
    registry.services['project_environment'].python_override = sys.executable
    path = workspace.root / 'example.py'
    original = b'value = 1\r\n'
    path.write_bytes(original)
    view = anchored_view(original)
    client = StubClient([
        completion(tool_calls=[call('load_tools', names=sorted(HASHLINE_NAMES))]),
        completion(tool_calls=[call('read_hashlines', path='example.py')]),
        completion(tool_calls=[call('edit_hashlines', path='example.py', revision=view['revision'],
                                    edits=[{'op': 'replace', 'start': view['lines'][0]['anchor'], 'lines': ['value = (']}])]),
        completion('The syntax check reported an error.'),
    ])
    agent = Agent(client, registry, 'test')
    try:
        await agent.run('Make the requested change')
        await agent.wait_for_compaction()
        assert len(client.calls) == 4
        assert path.read_bytes() == b'value = (\r\n'
        assert 'Python syntax: FAILED' in content_text(client.calls[-1]['messages'][-1].content)
        store = agent._checkpoints()
        assert len(store.listing()) == 1
        store.rewind(store.listing()[0]['id'])
        assert path.read_bytes() == original
    finally:
        await registry.aclose()


async def test_agent_rejects_entire_batch_before_definition_delivery(workspace, registry):
    path = workspace.root / 'example.txt'
    path.write_text('original')
    client = StubClient([
        completion(tool_calls=[call('load_tools', names=sorted(HASHLINE_NAMES)),
                               call('edit_file', path='example.txt', old_string='original', new_string='bad'),
                               call('read_hashlines', path='example.txt')]),
        completion('Stopped after the rejected batch.'),
    ])
    agent = Agent(client, registry, 'test')
    try:
        await agent.run('Try a premature tool call')
        await agent.wait_for_compaction()
        assert path.read_text() == 'original'
        assert not agent._checkpoints().listing()
        outcomes = [message.content for message in agent.messages if message.role == 'tool']
        assert len(outcomes) == 3
        assert all(outcome['status'] == 'error' for outcome in outcomes)
    finally:
        await registry.aclose()
