"""Documented sampling defaults, filtered by the selected endpoint's capabilities."""

from .capabilities import RequestProfile


def sampling_defaults(model: str, profile: RequestProfile | None) -> dict[str, float]:
    if profile is None:
        return {}
    name = model.lower().removesuffix(':free')
    defaults: dict[str, float] = {}
    # These are the original Qwen3 hybrid checkpoints, not later Coder/Thinking releases.
    if name in {f"qwen/qwen3-{size}" for size in
                ("4b", "8b", "14b", "32b", "30b-a3b", "235b-a22b")}:
        defaults = {"top_p": 0.95, "top_k": 20.0, "min_p": 0.0}
    elif name in {"nvidia/nemotron-3.5-lightning", "nvidia/nemotron-3.5-lightning-30b-a3b"}:
        defaults = {"top_p": 0.95}
    return {key: value for key, value in defaults.items() if key in profile.parameters}
