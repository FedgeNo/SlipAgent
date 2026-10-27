"""Hosted NVIDIA limits and parameters reviewed against its API reference."""

from .capabilities import RequestProfile

# Context limits come from the NVIDIA model cards; output limits and native
# tool parameters come from docs.api.nvidia.com/nim/reference/<model>-infer.
# A missing output cap means the reference leaves it unspecified.
MODEL_LIMITS: dict[str, tuple[int, int | None]] = {
    "deepseek-ai/deepseek-v4.1-flash": (1_000_000, 1_048_576),
    "stepfun-ai/step-5-preview": (1_000_000, 1_048_576),
    "z-ai/glm-5.3": (1_000_000, None),
    "z-ai/glm-5.3-flash": (1_000_000, None),
    "moonshotai/kimi-k2.6": (262_144, 65_536),
    "nvidia/nemotron-3.5-super-vl-120b-a12b": (1_000_000, 131_072),
    "nvidia/nemotron-3-ultra-550b-a55b": (1_000_000, 32_768),
    "nvidia/nemotron-3-super-120b-a12b": (1_000_000, 32_768),
    "nvidia/nemotron-3.5-lightning-30b-a3b": (1_000_000, 32_768),
    "nvidia/nemotron-nano-3-30b-a3b": (1_000_000, 32_768),
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": (256_000, 65_536),
    "meta/muse-glimmer-30b": (131_072, 131_072),
    "google/gemma-4-31b-it": (256_000, 32_768),
    "google/diffusiongemma-26b-a4b-it": (250_000, 32_768),
    "poolside/laguna-xs-2.1": (262_144, 16_384),
    "openai/gpt-oss-20b": (128_000, 4_096),
    # NVIDIA documents only MiniMax M3. The hosted Flash endpoint rejects requests
    # over 262,144 tokens with a 400 context_length error, not OpenRouter's 1M.
    "minimaxai/minimax-m3.1-flash": (262_144, None),
    # The API reference is unpublished; the model card states 64K input and output.
    "nvidia/nemotron-3.5-bolt-e4b": (65_536, 65_536),
}

NATIVE_TOOLS = {
    "stepfun-ai/step-5-preview", "moonshotai/kimi-k2.6", "meta/muse-glimmer-30b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b",
    "openai/gpt-oss-20b", "minimaxai/minimax-m3.1-flash", "nvidia/nemotron-3.5-bolt-e4b",
}

# These hosted schemas accept user/assistant messages with embedded tool calls.
USER_INSTRUCTIONS = {"google/gemma-4-31b-it", "google/diffusiongemma-26b-a4b-it"}


def builtin_profile(model: str) -> RequestProfile | None:
    limits = MODEL_LIMITS.get(model)
    if limits is None:
        return None
    parameters = {"temperature", "max_tokens"}
    if model in NATIVE_TOOLS:
        parameters.update({"tools", "tool_choice"})
    return RequestProfile(parameters=parameters, context_length=limits[0], max_completion_tokens=limits[1])
