"""Explicit skill selection preserves instructions without executing packages."""

import copy
import io
from importlib import import_module
from types import SimpleNamespace

import pytest

from slipagent.agent import Agent
from slipagent.cli import Style
from slipagent.commands import CommandContext, SkillCommand, SkillsCommand
from slipagent.skills import SkillError, catalog, select, validate_snapshot
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, content_text
from test_agent import StubClient, completion
attach = import_module('test-sessions').attach


@pytest.fixture(autouse=True)
def isolated_bundled_skills(monkeypatch, tmp_path):
    monkeypatch.setattr('slipagent.skills.prompt_directory', lambda: tmp_path / 'prompts')


def package(root, folder='release', name='release', *, enabled=True, body='Check the release carefully.'):
    path = root / folder / 'SKILL.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'---\nname: {name}\ndescription: >\n  Release workflow\n  for this project.\nenabled: {str(enabled).lower()}\n---\n{body}\n')
    return path


def roots(workspace, monkeypatch, tmp_path):
    user = tmp_path / 'user-data'
    monkeypatch.setenv('SLIPAGENT_STATE_DIR', str(user))
    return workspace.root / '.slipagent' / 'skills', user / 'skills'


def test_catalog_reads_metadata_only_and_reports_invalid_packages(workspace, monkeypatch, tmp_path):
    project, user = roots(workspace, monkeypatch, tmp_path)
    path = package(project)
    path.write_bytes(path.read_bytes() + b'\xff')
    disabled = package(user, 'disabled', 'disabled', enabled=False)
    bad = project / 'bad' / 'SKILL.md'
    bad.parent.mkdir()
    bad.write_text('no front matter')
    packages, errors = catalog(workspace)
    assert {skill.name for skill in packages} == {'release', 'disabled'}
    assert next(skill.description for skill in packages if skill.name == 'release') == 'Release workflow for this project.'
    assert len(errors) == 1 and str(bad) in errors[0]
    with pytest.raises(SkillError, match='disabled'):
        select(workspace, str(disabled))
    with pytest.raises(UnicodeError):
        select(workspace, 'release')


def test_project_precedence_duplicate_names_paths_and_no_script_execution(workspace, monkeypatch, tmp_path):
    project, user = roots(workspace, monkeypatch, tmp_path)
    personal = package(user, body='Personal workflow.')
    local = package(project, body='Read resources/notes.md; run scripts/release.py only when authorized.')
    marker = local.parent / 'executed'
    script = local.parent / 'scripts' / 'release.py'
    script.parent.mkdir()
    script.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    snapshot = select(workspace, 'release')
    assert snapshot['path'] == str(local)
    assert snapshot['root'] == str(local.parent)
    assert not marker.exists()
    assert select(workspace, str(personal))['instructions'] == 'Personal workflow.'
    other = package(project, 'other', 'release')
    with pytest.raises(SkillError, match='Ambiguous'):
        select(workspace, 'release')
    assert select(workspace, str(other.parent))['path'] == str(other)
    local.unlink()
    assert snapshot['instructions'].startswith('Read resources/')
    with pytest.raises((SkillError, OSError)):
        select(workspace, str(local))


@pytest.mark.parametrize('content', [
    '---\nname: wrong\ndescription: text\nenabled: maybe\n---\nbody',
    '---\nname: wrong\n---\nbody',
    '---\nname: wrong\ndescription: text\n---\n',
    '---\nname: ' + 'a' * 17000,
])
def test_invalid_metadata_and_empty_instructions_are_rejected(workspace, monkeypatch, tmp_path, content):
    project, _ = roots(workspace, monkeypatch, tmp_path)
    path = package(project)
    path.write_text(content)
    with pytest.raises(SkillError):
        select(workspace, str(path))


async def test_skill_commands_pin_refresh_and_clear_without_changing_selection_on_error(workspace, monkeypatch, tmp_path):
    project, _ = roots(workspace, monkeypatch, tmp_path)
    path = package(project, folder='with spaces')
    registry = ToolRegistry()
    out = io.StringIO()
    from slipagent.sessionstate import Session
    agent = Agent(StubClient([]), registry, 'test')
    session = Session(agent=agent, registry=registry, client=agent.client, workspace=workspace,
                      api_key='test', base_url='http://unused.invalid', http_referer=None, app_title='test')
    context = CommandContext(session, Style(False), out)
    try:
        await SkillsCommand().run(context, '')
        assert 'release' in out.getvalue() and str(path) in out.getvalue()
        await SkillCommand().run(context, f'"{path}"')
        first = copy.deepcopy(registry.services['selected_skill'])
        path.write_text(path.read_text().replace('carefully', 'again'))
        await SkillCommand().run(context, '')
        assert registry.services['selected_skill'] == first
        await SkillCommand().run(context, 'missing')
        assert registry.services['selected_skill'] == first
        await SkillCommand().run(context, 'release')
        assert registry.services['selected_skill']['sha256'] != first['sha256']
        await SkillCommand().run(context, 'off')
        assert 'selected_skill' not in registry.services
    finally:
        await registry.aclose()


async def test_selected_skill_survives_history_reduction_and_custom_prompt(workspace, monkeypatch, tmp_path):
    project, _ = roots(workspace, monkeypatch, tmp_path)
    path = package(project, body='ALWAYS_KEEP_SELECTED_WORKFLOW')
    registry = ToolRegistry(services={'workspace': workspace, 'selected_skill': select(workspace, 'release')})
    agent = Agent(StubClient([]), registry, 'test', system_prompt='Custom base instructions.', context_tokens=16000)
    try:
        for index in range(8):
            agent.extend([Message.user(f'Old request {index} ' + 'old history ' * 5000), Message.assistant('Old response')])
        for step in agent.history.steps:
            step.summary = 'Short completed history.'
        agent.messages.append(Message.user('Current request'))
        path.unlink()
        context = await agent._context_view([], 1)
        system = content_text(context[0].content)
        assert 'ALWAYS_KEEP_SELECTED_WORKFLOW' in system
        assert 'Custom base instructions.' in system
        assert str(path.parent) in system
        assert 'Short completed history.' in '\n'.join(content_text(message.content) for message in context)
        agent.reset()
        assert 'selected_skill' not in registry.services
    finally:
        await registry.aclose()


@pytest.mark.parametrize('fork', [False, True])
async def test_skill_snapshot_round_trips_without_reopening_package(workspace, monkeypatch, tmp_path, fork):
    project, _ = roots(workspace, monkeypatch, tmp_path)
    path = package(project)
    snapshot = select(workspace, 'release')
    registry = ToolRegistry(services={'selected_skill': snapshot})
    agent = Agent(StubClient([completion('Done')]), registry, 'test')
    journal = attach(agent, workspace, tmp_path)
    restored = Agent(StubClient([]), ToolRegistry(), 'test')
    try:
        await agent.run('Do the release')
        await agent.wait_for_compaction()
        saved = journal.load(journal.session_id)
        path.unlink()
        journal.restore(restored, saved, fork=fork)
        assert restored.registry.services['selected_skill'] == snapshot
        assert restored.registry.services['selected_skill'] is not snapshot
        restored.reset()
        assert 'selected_skill' not in restored.registry.services
    finally:
        await registry.aclose()
        await restored.registry.aclose()


def test_invalid_saved_skill_is_rejected():
    validate_snapshot(None)
    with pytest.raises(SkillError, match='Invalid saved'):
        validate_snapshot({'name': 'incomplete'})


def test_all_bundled_workflows_are_listed_and_explicitly_selectable(workspace, monkeypatch, tmp_path):
    from slipagent.prompts import prompt_directory
    roots(workspace, monkeypatch, tmp_path)
    monkeypatch.setattr('slipagent.skills.prompt_directory', prompt_directory)
    packages, errors = catalog(workspace)
    expected = {'fix-bug', 'implement-feature', 'review-changes', 'prepare-commit', 'investigate-failure'}
    assert not errors
    assert {skill.name for skill in packages if skill.scope == 'bundled'} == expected
    for name in expected:
        snapshot = select(workspace, name)
        validate_snapshot(snapshot)
        assert snapshot['instructions'].strip()
        assert snapshot['name'] == name
        assert snapshot['root'] == str(prompt_directory() / 'skills' / name)


def test_project_and_user_packages_override_bundled_names(workspace, monkeypatch, tmp_path):
    project, user = roots(workspace, monkeypatch, tmp_path)
    bundled = tmp_path / 'prompts' / 'skills'
    package(bundled, name='release', body='Bundled workflow.')
    assert select(workspace, 'release')['instructions'] == 'Bundled workflow.'
    personal = package(user, body='User workflow.')
    assert select(workspace, 'release')['path'] == str(personal)
    local = package(project, body='Project workflow.')
    assert select(workspace, 'release')['path'] == str(local)
    package(project, enabled=False)
    with pytest.raises(SkillError, match='disabled'):
        select(workspace, 'release')
    assert select(workspace, str(personal))['instructions'] == 'User workflow.'


def test_installed_layout_discovers_bundled_skill_resources(workspace, monkeypatch, tmp_path):
    from slipagent import prompts
    roots(workspace, monkeypatch, tmp_path)
    package_root = tmp_path / 'site-packages' / 'slipagent'
    path = package(package_root / 'prompts' / 'skills', name='installed-workflow')
    monkeypatch.setattr(prompts, '__file__', str(package_root / 'prompts.py'))
    monkeypatch.setattr('slipagent.skills.prompt_directory', prompts.prompt_directory)
    assert select(workspace, 'installed-workflow')['path'] == str(path)


async def test_restoring_unselected_session_clears_active_skill(workspace, monkeypatch, tmp_path):
    project, _ = roots(workspace, monkeypatch, tmp_path)
    package(project)
    agent = Agent(StubClient([completion('Done')]), ToolRegistry(), 'test')
    journal = attach(agent, workspace, tmp_path)
    restored = Agent(StubClient([]), ToolRegistry(services={'selected_skill': select(workspace, 'release')}), 'test')
    try:
        await agent.run('Ordinary task')
        await agent.wait_for_compaction()
        journal.restore(restored, journal.load(journal.session_id))
        assert 'selected_skill' not in restored.registry.services
    finally:
        await agent.registry.aclose()
        await restored.registry.aclose()


def test_catalog_does_not_follow_linked_packages_or_instruction_files(workspace, monkeypatch, tmp_path):
    project, user = roots(workspace, monkeypatch, tmp_path)
    external = package(user)
    project.mkdir(parents=True)
    (project / 'linked-package').symlink_to(external.parent, target_is_directory=True)
    folder = project / 'linked-file'
    folder.mkdir()
    (folder / 'SKILL.md').symlink_to(external)
    packages, errors = catalog(workspace)
    assert [skill.scope for skill in packages] == ['user']
    assert len(errors) == 1 and 'linked-file' in errors[0]
