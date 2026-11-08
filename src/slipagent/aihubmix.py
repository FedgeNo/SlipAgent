"""AIHubMix's OpenAI-compatible gateway and its public model catalog."""

from __future__ import annotations

import copy
from typing import Any

from .api import APIClient, APIConfigError, APIResponseError
from .capabilities import RequestProfile
from .types import ModelInfo

# Catalog vendors as the namespaces used by model scores.
VENDORS = {"zhipu": "z-ai", "moonshot": "moonshotai", "minimax": "minimax", "nvidia": "nvidia",
           "google": "google", "xiaomi": "xiaomi", "cohere": "cohere", "alibaba": "qwen",
           "dots-studio": "dots-studio", "liquid": "liquid", "xingchen": "xingchen-agi"}
# Free IDs the catalog lists without a vendor, or under a vendor prefix of their own.
IDENTITIES = {"flint-v1.0": "recursive/flint-v1.0", "mimo-v2-flash": "xiaomi/mimo-v2-flash",
              "crouton-alpha": "stealth/crouton-alpha", "xiaomi-mimo-v2.6-pro": "xiaomi/mimo-v2.6-pro",
              "xiaomi-mimo-v2.6-flash": "xiaomi/mimo-v2.6-flash", "intern-s2": "internlm/intern-s2",
              "agents-a1": "internscience/agents-a1", "k2.6-code-preview": "moonshotai/k2.6-code-preview"}
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
TOOL_FEATURES = {"tools", "tool_calling", "function_calling"}


def _positive(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _price(value: Any) -> str | None:
    """Per-token text from the catalog's USD-per-million price."""
    if type(value) not in (int, float) or value < 0:
        return None
    return "0" if value == 0 else f"{value / 1_000_000:.12g}"


class AIHubMixClient(APIClient):
    """AIHubMix gateway access; free routes are the zero-priced `-free` model IDs."""

    provider = "aihubmix"
    default_base_url = "https://aihubmix.com/v1"
    default_model = "coding-glm-5.3-free"
    key_env = "AIHUBMIX_API_KEY"
    model_env = "AIHUBMIX_MODEL"
    base_url_env = "AIHUBMIX_BASE_URL"

    @property
    def catalog_url(self) -> str:
        # The public catalog with prices and limits sits beside the /v1 inference path.
        return self.base_url.removesuffix("/v1") + "/api/v1/models"

    @staticmethod
    def features(entry: dict[str, Any]) -> set[str]:
        value = entry.get("features")
        return {item.strip() for item in value.split(",")} if isinstance(value, str) else set()

    @staticmethod
    def chat_model(entry: dict[str, Any]) -> bool:
        types = entry.get("types")
        outputs = entry.get("output_modalities")
        return (isinstance(entry.get("model_id"), str) and entry["model_id"].strip() != ""
                and isinstance(types, str) and "llm" in types.split(",")
                and isinstance(outputs, str) and "text" in outputs.split(",")
                and entry.get("retire_stage", "active") == "active"
                and _positive(entry.get("context_length")) is not None)

    @staticmethod
    def pricing(entry: dict[str, Any]) -> dict[str, str]:
        raw = entry.get("pricing")
        if not isinstance(raw, dict):
            return {}
        prompt, completion = _price(raw.get("input")), _price(raw.get("output"))
        if prompt is None or completion is None:
            return {}
        # AIHubMix serves free routes under separate `-free` IDs; the same model
        # without the suffix bills. A zero price on any other ID is not trusted.
        if (prompt, completion) == ("0", "0") and not entry["model_id"].endswith("-free"):
            return {}
        return {"prompt": prompt, "completion": completion}

    @staticmethod
    def score_identity(entry: dict[str, Any]) -> str:
        model = entry["model_id"].removesuffix("-free").removeprefix("coding-")
        if model in IDENTITIES:
            return IDENTITIES[model]
        vendor = VENDORS.get(str(entry.get("vendor")))
        return f"{vendor}/{model}" if vendor else model

    async def list_models(self) -> list[ModelInfo]:
        raw = await self._request("GET", self.catalog_url, params={"type": "llm"})
        data = raw.get("data")
        if not isinstance(data, list) or any(not isinstance(entry, dict) for entry in data):
            raise APIResponseError("AIHubMix returned a malformed model catalog.")
        from .modelscores import model_score
        models = []
        self._model_properties = {}
        for entry in data:
            if not self.chat_model(entry):
                continue
            model_id = entry["model_id"]
            self._model_properties[model_id] = entry
            models.append(ModelInfo(model_id, name=entry.get("model_name"), provider=self.provider,
                                    context_length=entry["context_length"], pricing=self.pricing(entry),
                                    score=model_score(self.provider, self.score_identity(entry))))
        return models

    async def model_capabilities(self, model: str, *, refresh: bool = False,
                                 store: bool = True) -> RequestProfile:
        if not hasattr(self, "_capabilities"):
            self._capabilities = {}
        cached = self._capabilities.get(model)
        if cached is not None and not refresh:
            return copy.deepcopy(cached)
        if not hasattr(self, "_model_properties") or refresh:
            await self.list_models()
        entry = self._model_properties.get(model)
        if entry is None:
            raise APIConfigError(f"AIHubMix does not list an active chat model {model!r}.")
        features = self.features(entry)
        parameters = {"max_tokens", "temperature"}
        if entry.get("tool_call") is True or features & TOOL_FEATURES:
            parameters.update({"tools", "tool_choice"})
        structured = "structured_outputs" in features
        if structured:
            parameters.add("response_format")
        profile = RequestProfile(parameters=parameters, context_length=entry["context_length"],
                                 max_completion_tokens=_positive(entry.get("max_output")),
                                 format="json_schema" if structured else None)
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def reasoning_effort(self, model: str) -> str | None:
        """The highest effort the catalog publishes for this model, if any."""
        entry = getattr(self, "_model_properties", {}).get(model, {})
        published = [value for option in entry.get("reasoning_options") or []
                     if isinstance(option, dict) and option.get("type") == "effort"
                     for value in option.get("values") or [] if value in EFFORTS]
        return max(published, key=EFFORTS.index) if published else None

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
        effort = self.reasoning_effort(payload["model"])
        if effort is not None and effort != "none":
            payload.setdefault("reasoning_effort", effort)
