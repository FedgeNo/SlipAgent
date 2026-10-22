"""Groq discovery, capabilities, and request wire compatibility."""

import httpx
import pytest

from slipagent.api import APIConfigError
from slipagent.groq import GroqClient
from slipagent.modelrouting import eligible_free
from slipagent.providers import key_provider
from slipagent.types import Message, ToolSpec


def catalog():
    chat = {"output_modalities": ["text"], "context_window": 131072, "max_completion_tokens": 65536,
            "supported_sampling_parameters": ["temperature", "top_p", "max_tokens"]}
    return {"data": [
        {"id": "openai/gpt-oss-120b", "active": True, "supported_features": ["tools", "json_mode"],
         "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}, **chat},
        {"id": "allam-2-7b", "active": True, "supported_features": ["json_mode"], "pricing": None, **chat},
        {"id": "plain-text", "active": True, "supported_features": [], "pricing": {"prompt": "0.000001", "completion": "0.000002"}, **chat},
        {"id": "retired", "active": False, "supported_features": ["tools"], **chat},
        {"id": "whisper-large-v3", "active": True, "output_modalities": ["transcription"], "supported_features": None},
        {"id": "prompt-guard", "active": True, "output_modalities": ["text"], "supported_features": None, "context_window": 512},
    ]}


async def test_catalog_capabilities_and_request():
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=catalog())
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}]})

    async with GroqClient("dummy", transport=httpx.MockTransport(handle)) as client:
        models = await client.list_models()
        assert {m.id for m in models} == {"openai/gpt-oss-120b", "plain-text", "allam-2-7b"}
        assert all(m.provider == "groq" for m in models)
        assert {m.id: m.pricing["prompt"] for m in models} == {"openai/gpt-oss-120b": "0.00000015", "plain-text": "0.000001", "allam-2-7b": "0"}
        assert [m.id for m in models if eligible_free(m)] == ["allam-2-7b"]
        profile = await client.model_capabilities("openai/gpt-oss-120b")
        assert profile.native_tools and profile.context_length == 131072
        assert not (await client.model_capabilities("plain-text")).native_tools
        for rejected in ("retired", "whisper-large-v3", "prompt-guard"):
            with pytest.raises(APIConfigError, match="active chat model"):
                await client.model_capabilities(rejected)
        await client.chat(model="openai/gpt-oss-120b", messages=[Message.user("Inspect")],
                          tools=[ToolSpec("inspect", "Inspect", {})], session_id="test")
    body = requests[-1].read()
    assert b"session_id" not in body and requests[-1].url.path == "/openai/v1/chat/completions"


def test_groq_keys_are_recognised_by_prefix():
    assert key_provider("gsk_abcdef123456") == "groq"
    assert key_provider("gsk_") is None


def test_allam_is_hidden_from_coding_models():
    from slipagent.model_catalog import coding_models
    from slipagent.types import ModelInfo
    assert not coding_models([ModelInfo("allam-2-7b", provider="groq")])
