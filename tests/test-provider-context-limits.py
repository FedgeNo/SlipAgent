"""Local context caps follow the exact model and eligible OpenRouter routes."""

import json
import shutil

import pytest

from slipagent.agent import Agent
from slipagent.capabilities import ModelCapabilities, RequestProfile
from slipagent.context import ConversationHistory
from slipagent.prompts import PromptError, model_prompt_profile
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall, content_text
from test_agent import StubClient

MODEL = 'dots-studio/dots-3-note-preview:free'


def capabilities(*tags, context=512000):
    return ModelCapabilities({}, [{'tag': tag, 'context_length': context, 'supported_parameters': ['tools']} for tag in tags])


@pytest.mark.parametrize('model,tags,manual,expected', [
    (MODEL, ['atlas-cloud/fp8'], None, 256000),
    (MODEL, ['atlas-cloud'], None, 256000),
    (MODEL, ['atlas-cloud/fp16'], None, 256000),
    (MODEL, ['atlas-cloud/fp8', 'another-provider'], None, 256000),
    (MODEL, ['another-provider'], None, 512000),
    ('dots-studio/dots-3-note-preview', ['atlas-cloud/fp8'], None, 512000),
    ('another/model', ['atlas-cloud/fp8'], None, 512000),
    (MODEL, ['atlas-cloud/fp8'], 128000, 128000),
])
async def test_cap_applies_only_to_the_selected_model_and_eligible_provider(monkeypatch, model, tags, manual, expected):
    seen = []

    async def view(self, messages, specs, **options):
        seen.append(options['context_length'])
        return [Message.user('context')]

    monkeypatch.setattr(ConversationHistory, 'view', view)
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, model, context_tokens=manual)
    profile = capabilities(*tags)
    try:
        await agent._context_view([], 1, capabilities=profile)
        assert seen == [expected]
        assert profile.context_length == 512000
    finally:
        await registry.aclose()


async def test_other_api_providers_are_not_given_an_openrouter_cap(monkeypatch):
    seen = []

    async def view(self, messages, specs, **options):
        seen.append(options['context_length'])
        return [Message.user('context')]

    monkeypatch.setattr(ConversationHistory, 'view', view)
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, MODEL, context_steps=50)
    try:
        await agent._context_view([], 1, capabilities=RequestProfile(parameters={'tools'}, context_length=512000))
        assert seen == [512000]
    finally:
        await registry.aclose()


async def test_profile_edits_affect_the_next_request_with_cached_capabilities(tmp_path, monkeypatch):
    import slipagent.prompts as prompts
    directory = tmp_path / 'prompts'
    shutil.copytree(prompts.prompt_directory(), directory)
    monkeypatch.setattr(prompts, 'prompt_directory', lambda: directory)
    path = directory / 'model-profiles.json'
    seen = []

    async def view(self, messages, specs, **options):
        seen.append(options['context_length'])
        return [Message.user('context')]

    monkeypatch.setattr(ConversationHistory, 'view', view)
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, MODEL)
    profile = capabilities('atlas-cloud/fp8')
    try:
        for limits in [{'atlas-cloud': 192000, 'atlas-cloud/fp8': 224000}, {'atlas-cloud': 128000}, {}]:
            path.write_text(json.dumps({MODEL: {'provider_context_limits': limits}}))
            await agent._context_view([], 1, capabilities=profile)
        assert seen == [224000, 128000, 512000]
    finally:
        await registry.aclose()


@pytest.mark.parametrize('limits', [None, [], 256000, {'atlas-cloud': True}, {'atlas-cloud': 0},
                                   {'atlas-cloud': -1}, {'atlas-cloud': '256000'}, {'': 256000}])
def test_invalid_provider_caps_are_reported(tmp_path, monkeypatch, limits):
    import slipagent.prompts as prompts
    directory = tmp_path / 'prompts'
    shutil.copytree(prompts.prompt_directory(), directory)
    monkeypatch.setattr(prompts, 'prompt_directory', lambda: directory)
    (directory / 'model-profiles.json').write_text(json.dumps({MODEL: {'provider_context_limits': limits}}))
    with pytest.raises(PromptError, match='provider_context_limits'):
        model_prompt_profile(MODEL)


async def test_large_read_history_is_reduced_while_archived_originals_survive():
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, MODEL, context_steps=50)
    agent.messages = [Message.system('Pinned guidance.')]
    payload = 'source_code = value + 1; ' * 7000
    for step in range(24):
        agent.messages.extend([Message.user('Read more.'),
                               Message.assistant('Reading.', [ToolCall(str(step), 'read_file', {'path': f'file-{step}.py'})]),
                               Message.tool_result(str(step), payload)])
    agent.history.sync(agent.messages)
    for step in agent.history.steps:
        step.summary = f'Read file {step.id}.'
        step.results_summarized = True
        step.observed = True
    originals = [step.full_text() for step in agent.history.steps]
    agent.messages.append(Message.user('Continue reading.'))
    try:
        uncapped = await agent._context_view([], 25, capabilities=capabilities('another-provider'))
        capped = await agent._context_view([], 25, capabilities=capabilities('atlas-cloud/fp8'))
        def largest(messages):
            return max(len(content_text(m.content).encode()) for m in messages)
        assert largest(uncapped) > 1048576
        assert largest(capped) < 1048576
        assert [step.full_text() for step in agent.history.steps] == originals
        assert 'Continue reading.' in '\n'.join(content_text(m.content) for m in capped)
    finally:
        await registry.aclose()


@pytest.mark.parametrize('tag,capped', [('atlas-cloud/fp8', True), ('another-provider', False)])
async def test_background_compaction_respects_the_same_provider_cap(tag, capped):
    from slipagent.compaction import StepCompactor
    history = ConversationHistory()
    history.sync([Message.user('Read files.'),
                  Message.assistant('Reading.', [ToolCall('one', 'read_file', {'path': 'file.py'})]),
                  Message.tool_result('one', 'x' * 300000)])
    step = history.steps[0]
    original = step.full_text()
    source = step.compaction_input()
    source['history'] = ['Earlier context ' + 'x' * 20000 for _ in range(25)]
    source['history_omitted'] = 0
    client = StubClient([])
    compactor = StepCompactor(lambda _: None, lambda _: None)
    await compactor._compact_once(step, source, client, MODEL, capabilities(tag), None, 0)
    assert step.compaction_status == 'complete'
    assert bool(source['history_omitted']) == capped
    assert step.full_text() == original
