"""Published reasoning controls and exact-route fallbacks verified on 2026-11-07."""

from __future__ import annotations

import copy
from typing import Any

from .capabilities import ModelCapabilities
from .data_text import MessageSections, RenderedContent


# Provider-specific controls take precedence over successful generic enum probes.
NVIDIA_OPTIONS: dict[str, dict[str, Any]] = {
    'minimaxai/minimax-m3.1-flash': {'reasoning_effort': 'max'},
    'meta/muse-glimmer-30b': {'reasoning_effort': 'max'},
    'mistralai/mistral-large-4-1t-a52b': {'reasoning_effort': 'high'},
    'openai/gpt-oss-20b': {'reasoning_effort': 'high'},
    'deepseek-ai/deepseek-v4.1-flash': {'reasoning_effort': 100},
    'z-ai/glm-5.3': {'reasoning_effort': 'max'},
    'nvidia/nemotron-3-ultra-550b-a55b': {
        'reasoning_effort': 'high', 'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_budget': -1},
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning': {'chat_template_kwargs': {'enable_thinking': True}},
    'nvidia/nemotron-3.5-super-vl-120b-a12b': {'chat_template_kwargs': {'enable_thinking': True}},
    'nvidia/nemotron-3.5-lightning-30b-a3b': {
        'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_budget': -1},
    # No published effort ladder was found for these routes; max completed a live probe.
    'stepfun-ai/step-5-preview': {'reasoning_effort': 'max'},
    'poolside/laguna-xs-2.1': {'reasoning_effort': 'max'},
    'nvidia/nemotron-3.5-bolt-e4b': {'reasoning_effort': 'max'},
}

GATEWAY_TESTED_MAX = {
    'openrouter': {
        'liquid/lfm-3-1.7b:free', 'liquid/lfm-2.5-2.6b:free', 'recursive/flint-v1.0:free',
        'xingchen-agi/xing4.0-29b-a4b:free', 'nvidia/nemotron-3.5-super-vl:free',
        'nvidia/nemotron-3.5-lightning:free', 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
        'dots-studio/dots-3-note-preview:free', 'cohere/north-mini-code:free',
    },
}

MISTRAL_HIGH = {
    'mistral-small-2603', 'mistral-small-latest', 'mistral-large-4', 'mistral-large-4-0',
    'mistral-medium-3-5', 'mistral-medium-3.5', 'mistral-medium-latest',
    'labs-leanstral-1-5', 'labs-leanstral-1-5-1',
}


def claude_model(model: str) -> bool:
    return model.removeprefix('~').startswith(('anthropic/claude-', 'claude-'))


def gateway_reasoning(provider: str, model: str, profile: ModelCapabilities) -> None:
    """Fill only missing effort metadata; published endpoint choices remain authoritative."""
    if (model in GATEWAY_TESTED_MAX.get(provider, set()) and profile.reasoning is not None
            and 'effort' not in profile.reasoning):
        profile.reasoning = {'effort': 'max', 'exclude': False}


def reasoning_options(provider: str, model: str) -> dict[str, Any]:
    if provider == 'nvidia':
        return copy.deepcopy(NVIDIA_OPTIONS.get(model, {}))
    if provider == 'groq':
        if model in {'openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'qwen/qwen3.8-27b'}:
            return {'reasoning_effort': 'high'}
        if model == 'meta-models/muse-spark-open':
            return {'reasoning_effort': 'xhigh'}
    if provider == 'mistral':
        if model in MISTRAL_HIGH:
            return {'reasoning_effort': 'high'}
        if model in {'zai-glm-5-3', 'zai-glm-5-3-flash'}:
            return {'reasoning_effort': 'max'}
    if provider == 'zai':
        # Every listed GLM model accepts the published on/off thinking switch.
        return {'thinking': {'type': 'enabled'}}
    if provider == 'google':
        config: dict[str, Any] = {'include_thoughts': True}
        if model == 'gemini-2.5-pro':
            config['thinking_budget'] = 32768
        elif model in {'gemini-2.5-flash', 'gemini-2.5-flash-lite'}:
            config['thinking_budget'] = 24576
        elif model.startswith('gemini-3') or model in {
                'gemini-flash-latest', 'gemini-flash-lite-latest', 'gemini-pro-latest'}:
            config['thinking_level'] = 'high'
        else:
            return {}
        return {'extra_body': {'google': {'thinking_config': config}}}
    return {}


def enable_system_thinking(payload: dict[str, Any]) -> None:
    """Gemma's published thinking switch belongs in the outgoing system text."""
    for message in payload['messages']:
        if message['role'] == 'system':
            message['content'] = RenderedContent(MessageSections(['<|think|>', message['content']]))
            return
    payload['messages'].insert(0, {'role': 'system', 'content': '<|think|>'})
