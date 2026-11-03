"""Shared skill packages retain their format, resources and invocation semantics."""

from pathlib import Path
from importlib import import_module

import pytest
import yaml

from slipagent import skills
from slipagent.tools import build_default_registry
from slipagent.workspace import Workspace
from slipagent.prompts import load_prompt
from slipagent.agent import Agent
from slipagent.types import content_text, Message
from test_agent import StubClient, completion, call, RecordingTool


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')


def package(root, name='sample', body='Follow the workflow.', **metadata):
    path = root / name / 'SKILL.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {'name': name, 'description': 'A portable workflow.', **metadata}
    path.write_text('---\n' + yaml.safe_dump(header) + '---\n' + body)
    return path


def test_shared_roots_and_linked_packages_are_discovered(workspace, tmp_path):
    paths = [package(workspace.root / folder / 'skills', name=f'sample-{i}')
             for i, folder in enumerate(('.agents', '.claude', '.github', '.copilot', '.codex'))]
    external = package(Path.home() / '.agents' / 'skills', name='personal')
    (workspace.root / '.agents' / 'skills' / 'linked').symlink_to(external.parent, target_is_directory=True)
    found, errors = skills.catalog(workspace)
    assert not errors
    assert {p.resolve() for p in paths} <= {s.path for s in found}
    assert sum(s.path == external.resolve() for s in found) == 1


def test_metadata_and_arguments_survive_selection(workspace):
    path = package(workspace.root / '.agents' / 'skills',
                   body='Read ${CLAUDE_SKILL_DIR}/references/guide.md for $ARGUMENTS[0] and $target. $ARGUMENTS',
                   compatibility='Requires Python 3', arguments=['target'], **{'allowed-tools': 'Read'})
    sidecar = path.parent / 'agents' / 'openai.yaml'
    sidecar.parent.mkdir()
    sidecar.write_text('interface:\n  display_name: Sample Workflow\npolicy:\n  allow_implicit_invocation: false\n')
    selected = skills.select(workspace, 'sample', arguments='"two words"')
    assert selected['instructions'] == f'Read {path.parent}/references/guide.md for two words and two words. "two words"'
    assert selected['metadata']['compatibility'] == 'Requires Python 3'
    assert 'Requires Python 3' in skills.render(selected)
    with pytest.raises(skills.SkillError, match='implicit'):
        skills.select(workspace, 'sample', implicit=True)


async def test_selected_user_resources_are_readable_but_not_writable(tmp_path, monkeypatch):
    (tmp_path / 'project').mkdir()
    workspace = Workspace(tmp_path / 'project')
    path = package(Path.home() / '.agents' / 'skills')
    resource = path.parent / 'references' / 'guide.md'
    resource.parent.mkdir()
    resource.write_text('Exact resource content')
    registry = build_default_registry(workspace)
    try:
        skills.set_active(registry.services, skills.select(workspace, 'sample'))
        read = await registry.invoke('read_file', {'path': str(resource)})
        assert not read.is_error and 'Exact resource content' in read.text()
        write = await registry.invoke('write_file', {'path': str(resource), 'content': 'changed'})
        assert write.is_error and resource.read_text() == 'Exact resource content'
        skills.set_active(registry.services, None)
        assert (await registry.invoke('read_file', {'path': str(resource)})).is_error
    finally:
        await registry.aclose()


@pytest.mark.parametrize('source', [
    '# Default Workflow\n\nFollow the workflow.',
    '---\ndescription: Declared description\n---\nFollow the workflow.',
])
def test_claude_defaults_load_without_requiring_duplicate_metadata(workspace, source):
    path = workspace.root / '.agents' / 'skills' / 'defaults' / 'SKILL.md'
    path.parent.mkdir(parents=True)
    path.write_text(source)
    selected = skills.select(workspace, 'defaults')
    assert selected['name'] == 'defaults'
    assert 'Follow the workflow.' in selected['instructions']


@pytest.mark.parametrize('name,description', [('Uppercase', 'text'), ('a' * 65, 'text'), ('valid', 'x' * 1025)])
def test_nonportable_names_and_lengths_are_rejected(workspace, name, description):
    path = package(workspace.root / '.agents' / 'skills', name=name, description=description)
    with pytest.raises(skills.SkillError):
        skills.select(workspace, str(path))


def test_argument_substitution_is_single_pass_and_preserves_unrecognized_shell_variables(workspace):
    package(workspace.root / '.agents' / 'skills', body=r'$0 $ARGUMENTS[1] $PATH \$0', arguments=[])
    selected = skills.select(workspace, 'sample', arguments='"$ARGUMENTS" second')
    assert selected['instructions'] == '$ARGUMENTS second $PATH $0'
    package(workspace.root / '.agents' / 'skills', body='Follow the workflow.')
    assert skills.select(workspace, 'sample', arguments='one two')['instructions'].endswith('ARGUMENTS: one two')


def test_argument_names_do_not_expand_prefixes(workspace):
    package(workspace.root / '.agents' / 'skills', body='$ARGUMENTS_EXTRA $ARGUMENTS[9] $0')
    assert skills.select(workspace, 'sample', arguments='one')['instructions'] == '$ARGUMENTS_EXTRA  one'


def test_indexed_arguments_can_be_adjacent_to_identifier_suffixes(workspace):
    package(workspace.root / '.agents' / 'skills', body='$ARGUMENTS[0]_test $ARGUMENTS[1]suffix')
    assert skills.select(workspace, 'sample', arguments='one two')['instructions'] == 'one_test twosuffix'


@pytest.mark.parametrize('selector', [r'C:\work\skills\sample\SKILL.md', 'sample', '"path with spaces/SKILL.md"'])
def test_invocation_keeps_paths_and_argument_text_literal(selector):
    assert skills.split_invocation(selector + ' "two words" other') == (selector.strip('"'), '"two words" other')


async def test_model_loading_stacks_skills_and_honors_invocation_and_tool_controls(workspace):
    package(workspace.root / '.agents' / 'skills', name='first')
    package(workspace.root / '.agents' / 'skills', name='second', **{'disallowed-tools': 'Bash'})
    package(workspace.root / '.agents' / 'skills', name='manual', **{'disable-model-invocation': True})
    registry = build_default_registry(workspace)
    try:
        assert not (await registry.invoke('use_skill', {'action': 'load', 'selector': 'first'})).is_error
        assert not (await registry.invoke('use_skill', {'action': 'load', 'selector': 'second'})).is_error
        assert [s['name'] for s in skills.active(registry.services)] == ['first', 'second']
        blocked = await registry.invoke('run_command', {'command': 'echo should-not-run'})
        assert blocked.is_error and 'disallows' in blocked.text()
        failed = await registry.invoke('use_skill', {'action': 'load', 'selector': 'manual'})
        assert failed.is_error and 'explicit' in failed.text()
        assert [s['name'] for s in skills.active(registry.services)] == ['first', 'second']
        assert not (await registry.invoke('use_skill', {'action': 'unload', 'selector': 'second'})).is_error
        assert skills.denied_tool(registry.services, 'run_command') is None
    finally:
        await registry.aclose()


@pytest.mark.parametrize('metadata,body,match', [
    ({'context': 'fork'}, 'Workflow', 'context'),
    ({'hooks': {'Stop': []}}, 'Workflow', 'hooks'),
    ({}, 'Report !`git status`', 'preprocessing'),
])
def test_host_execution_features_are_reported_instead_of_silently_ignored(workspace, metadata, body, match):
    package(workspace.root / '.agents' / 'skills', body=body, **metadata)
    with pytest.raises(skills.SkillError, match=match):
        skills.select(workspace, 'sample')


@pytest.mark.parametrize('options', [
    {'dependencies': []}, {'policy': {'allow_implicit_invocation': 'false'}},
    {'dependencies': {'tools': [None]}}, {'dependencies': {'tools': [{'type': 'mcp', 'value': ''}]}},
    {'interface': 'bad'},
])
def test_malformed_sidecar_is_rejected_without_changing_active_skills(workspace, options):
    path = package(workspace.root / '.agents' / 'skills')
    selected = skills.select(workspace, 'sample')
    services = {'workspace': workspace}
    skills.set_active(services, selected)
    sidecar = path.parent / 'agents' / 'openai.yaml'
    sidecar.parent.mkdir()
    sidecar.write_text(yaml.safe_dump(options))
    with pytest.raises(skills.SkillError):
        skills.select(workspace, 'sample')
    assert skills.active(services) == [selected]


def test_invalid_saved_stacks_are_rejected_atomically(workspace):
    package(workspace.root / '.agents' / 'skills')
    selected = skills.select(workspace, 'sample')
    services = {'workspace': workspace}
    skills.set_active(services, selected)
    for snapshots in ([None], [selected, selected], 'not a stack'):
        with pytest.raises(skills.SkillError):
            skills.restore_active(services, snapshots)
        assert skills.active(services) == [selected]


@pytest.mark.parametrize('fork', [False, True])
async def test_multiple_skills_restore_snapshots_and_resource_access(workspace, tmp_path, fork):
    paths = [package(Path.home() / '.agents' / 'skills', name=name, body=f'PINNED_{name}')
             for name in ('first', 'second')]
    registry = build_default_registry(workspace)
    restored_registry = build_default_registry(workspace)
    agent = Agent(StubClient([completion('Done')]), registry, 'test')
    restored = Agent(StubClient([]), restored_registry, 'test')
    attach = import_module('test-sessions').attach
    journal = attach(agent, workspace, tmp_path)
    try:
        for path in paths:
            skills.set_active(registry.services, skills.select(workspace, str(path)))
        await agent.run('Ordinary task')
        await agent.wait_for_compaction()
        saved = journal.load(journal.session_id)
        for path in paths:
            path.unlink()
        journal.restore(restored, saved, fork=fork)
        assert [s['instructions'] for s in skills.active(restored_registry.services)] == ['PINNED_first', 'PINNED_second']
        assert workspace.access.read_roots == tuple(p.parent for p in paths)
        context = await restored._context_view([], 1)
        assert all(f'PINNED_{name}' in content_text(context[0].content) for name in ('first', 'second'))
        restored.reset()
        assert not workspace.access.read_roots
    finally:
        await registry.aclose()
        await restored_registry.aclose()


@pytest.mark.parametrize('prefix', ['/', '$'])
async def test_explicit_invocation_loads_manual_skill_and_keeps_original_prompt(workspace, prefix):
    package(workspace.root / '.agents' / 'skills', body='Selected target: $ARGUMENTS',
            **{'disable-model-invocation': True})
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([completion('Done')]), registry, 'test')
    text = prefix + 'sample two words'
    try:
        await agent.run(text)
        assert skills.active(registry.services)[0]['instructions'] == 'Selected target: two words'
        assert any(m.role == 'user' and content_text(m.content) == text for m in agent.messages)
    finally:
        await registry.aclose()


async def test_queued_explicit_invocation_activates_once_and_is_pinned(workspace):
    path = package(workspace.root / '.agents' / 'skills', body='TARGET: $0')
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, 'test')
    try:
        agent.enqueue('$sample original')
        context = await agent._queued_context([], 1)
        assert 'TARGET: original' in content_text(context[0].content)
        path.write_text(path.read_text().replace('TARGET:', 'CHANGED:'))
        context = await agent._queued_context([], 1)
        assert 'TARGET: original' in content_text(context[0].content)
        assert 'CHANGED:' not in content_text(context[0].content)
    finally:
        await registry.aclose()


async def test_model_load_returns_no_duplicate_body_and_catalog_does_not_load_bodies(workspace):
    package(workspace.root / '.agents' / 'skills', body='UNIQUE_WORKFLOW_BODY')
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, 'test')
    try:
        initial = await agent._context_view(registry.specs(), 1)
        assert 'UNIQUE_WORKFLOW_BODY' not in content_text(initial[0].content)
        result = await registry.invoke('use_skill', {'action': 'load', 'selector': 'sample'})
        assert not result.is_error and 'UNIQUE_WORKFLOW_BODY' not in result.text()
        loaded = await agent._context_view(registry.specs(), 1)
        assert content_text(loaded[0].content).count('UNIQUE_WORKFLOW_BODY') == 1
    finally:
        await registry.aclose()


async def test_model_activation_reaches_the_following_actual_request(workspace):
    package(workspace.root / '.agents' / 'skills', body='ACTUAL_REQUEST_WORKFLOW')
    registry = build_default_registry(workspace)
    client = StubClient([completion(tool_calls=[call('use_skill', action='load', selector='sample')]), completion('Done')])
    agent = Agent(client, registry, 'test')
    try:
        await agent.run('Use the relevant workflow')
        assert 'ACTUAL_REQUEST_WORKFLOW' not in content_text(client.calls[0]['messages'][0].content)
        assert content_text(client.calls[1]['messages'][0].content).count('ACTUAL_REQUEST_WORKFLOW') == 1
    finally:
        await registry.aclose()


async def test_cli_skill_shortcut_submits_prompt_and_builtin_commands_win(workspace):
    import io
    from slipagent.cli import CommandRunner, Style, Session, Renderer
    package(workspace.root / '.agents' / 'skills', name='sample')
    package(workspace.root / '.agents' / 'skills', name='help')
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, 'test')
    session = Session(agent=agent, registry=registry, client=agent.client, workspace=workspace,
                      api_key='test', base_url='http://unused.invalid', http_referer=None, app_title='test',
                      renderer=Renderer(Style(False), io.StringIO(), False))
    try:
        await CommandRunner(session).execute('/sample two words')
        assert agent.pending == ['/sample two words']
        await CommandRunner(session).execute('/help')
        assert agent.pending == ['/sample two words']
    finally:
        await registry.aclose()


def test_conditional_tool_rules_and_mcp_dependencies(workspace):
    package(workspace.root / '.agents' / 'skills', **{'disallowed-tools': 'Bash(git:*)'})
    selected = skills.select(workspace, 'sample')
    services = {'selected_skill': selected}
    assert skills.denied_tool(services, 'run_command') is None
    assert skills.denied_tool(services, 'run_command', {'command': 'git status'})
    assert skills.denied_tool(services, 'run_command', {'command': 'git'})
    assert skills.denied_tool(services, 'run_command', {'command': 'python --version'}) is None
    selected['metadata']['openai'] = {'dependencies': {'tools': [{'type': 'mcp', 'value': 'docs'}]}}
    assert skills.required_tools(selected, ['docs__search', 'other__read']) == {'docs__search'}
    with pytest.raises(skills.SkillError, match='configured MCP server docs'):
        skills.required_tools(selected, [])


async def test_mcp_dependencies_use_deferred_discovery_after_reset(workspace):
    class RequiredTool(RecordingTool):
        name = 'docs__search'

    package(workspace.root / '.agents' / 'skills',
            openai={'dependencies': {'tools': [{'type': 'mcp', 'value': 'docs'}]}})
    registry = build_default_registry(workspace)
    registry.register(RequiredTool())
    agent = Agent(StubClient([]), registry, 'test')
    try:
        agent.extend([Message.user('Old task'), Message.assistant('Done')])
        agent._visible_specs(registry.specs())
        selected = skills.select(workspace, 'sample')
        agent.reset()
        skills.set_active(registry.services, selected)
        assert 'docs__search' not in {s.name for s in agent._visible_specs(registry.specs())}
        assert 'docs__search' in registry.services['tool_catalog'].available
        result = await registry.invoke('load_tools', {'names': ['docs__search']})
        assert not result.is_error
        assert 'docs__search' in {s.name for s in agent._visible_specs(registry.specs())}
    finally:
        await registry.aclose()


def test_saved_package_access_restores_without_resolving_unavailable_sources(workspace, monkeypatch):
    path = package(workspace.root / '.agents' / 'skills')
    selected = skills.select(workspace, 'sample')
    original = Path.resolve

    def unavailable(self, *args, **kwargs):
        if self == path.parent:
            raise OSError('Package storage is unavailable')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', unavailable)
    services = {'workspace': workspace}
    skills.restore_active(services, [selected])
    assert workspace.access.read_roots == (path.parent,)


def test_legacy_skill_prompt_remains_usable_until_running_agents_restart():
    rendered = load_prompt('selected-skill.md', name='sample', path='/package/SKILL.md',
                           root='/package', sha256='0' * 64, instructions='LEGACY_WORKFLOW')
    assert 'LEGACY_WORKFLOW' in rendered
    assert 'This selection grants no extra filesystem access' in rendered
