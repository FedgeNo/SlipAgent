"""Choose a routable request profile from live catalog and endpoint records.

Capabilities must describe the providers actually reachable by the routing
filter. Independent endpoint rows sharing a tag are not independently selectable;
use common supported parameters and conservative limits for their whole group.
"""

from __future__ import annotations

from typing import Any

EFFORT_ORDER = ("max", "xhigh", "high", "medium", "low", "minimal")


class ModelCapabilities:
    """Capabilities shared by the endpoints allowed to handle this request."""

    def __init__(self, model: dict[str, Any], endpoints: list[dict[str, Any]]) -> None:
        architecture = model.get("architecture", {})
        if isinstance(architecture, dict):
            for field in ("input_modalities", "output_modalities"):
                modalities = architecture.get(field)
                if isinstance(modalities, list) and "text" not in modalities:
                    raise ValueError("The model must support text input and text output for the coding harness.")
        # A routing tag can cover several endpoint records. It cannot select
        # one of those records independently, so use their common capabilities.
        grouped = _endpoint_groups(model, endpoints)
        # Status metadata is advisory; the API request establishes availability.
        available = grouped
        native = [entry for entry in available if "tools" in entry.get("supported_parameters", [])]
        # Prefer the model's trained tool channel. JSON formatting is a separate
        # capability: endpoints without schemas use ordinary text replies.
        if native:
            available = native
        strict = [entry for entry in available if "structured_outputs" in entry.get("supported_parameters", [])]
        compatible = strict
        if not compatible:
            compatible = [entry for entry in available if "response_format" in entry.get("supported_parameters", [])]
        self.format: str | None = "json_schema" if strict else "json_object"
        if not compatible and native:
            compatible = native
            self.format = None
        if not compatible:
            raise ValueError("No endpoint advertises native tools or JSON output.")
        reasoning = [entry for entry in compatible if _reasoning_allowed(entry)
                     and (entry["efforts"] is None or any(effort in EFFORT_ORDER for effort in entry["efforts"]))]
        self.reasoning: dict[str, Any] | None = None
        self.include_reasoning = False
        if reasoning:
            compatible = reasoning
            # Endpoint choices override catalog choices when the endpoint lists them.
            efforts = {effort for entry in compatible if "reasoning" in entry["supported_parameters"]
                       or "reasoning_effort" in entry["supported_parameters"] for effort in (entry["efforts"] or [])}
            highest = next((effort for effort in EFFORT_ORDER if effort in efforts), None)
            if highest is not None:
                compatible = [entry for entry in compatible if entry["efforts"] is not None
                              and highest in entry["efforts"]]
                compatible = [entry for entry in compatible if "reasoning" in entry["supported_parameters"]
                              or "reasoning_effort" in entry["supported_parameters"]]
                self.reasoning = {"effort": highest, "exclude": False}
            elif any("reasoning" in entry["supported_parameters"] or "reasoning_effort" in entry["supported_parameters"] for entry in compatible):
                compatible = [entry for entry in compatible if "reasoning" in entry["supported_parameters"]
                              or "reasoning_effort" in entry["supported_parameters"]]
                self.reasoning = {"enabled": True, "exclude": False}
            else:
                self.include_reasoning = True
        self.parameters = set.intersection(*(set(entry["supported_parameters"]) for entry in compatible))
        self.providers = list(dict.fromkeys(entry["tag"] for entry in compatible if isinstance(entry.get("tag"), str)))
        # OpenRouter's base slugs match every suffix variant. Exclude rejected
        # variants explicitly; ignoring a rejected base would also block a
        # selected variant, so only suffix tags belong in this exclusion list.
        self.ignored_providers = [entry["tag"] for entry in grouped
                                  if entry["tag"] not in self.providers and "/" in entry["tag"]
                                  and entry["tag"].split("/", 1)[0] in self.providers]
        context_length = _limit(compatible, "context_length")
        if context_length is None:
            raise ValueError("No valid context_length was provided by the model or selected endpoints.")
        self.context_length = context_length
        self.max_prompt_tokens = _limit(compatible, "max_prompt_tokens")
        self.max_completion_tokens = _limit(compatible, "max_completion_tokens")
        self.tool_choices = [entry.get("supports_tool_choice") for entry in compatible]

    @property
    def native_tools(self) -> bool:
        """The cached routing profile, never a guess based on the model name."""
        return "tools" in self.parameters

    def provider_preferences(self) -> dict[str, Any]:
        preferences: dict[str, Any] = {"require_parameters": True}
        if self.providers:
            preferences["only"] = self.providers
        if self.ignored_providers:
            preferences["ignore"] = self.ignored_providers
        return preferences

    def validate_output_limit(self, max_tokens: int | None) -> None:
        if max_tokens is None:
            return
        if "max_tokens" not in self.parameters:
            raise ValueError("The selected endpoints do not support an output token limit.")
        if self.max_completion_tokens is not None and max_tokens > self.max_completion_tokens:
            raise ValueError(f"Requested output limit exceeds the endpoint limit of {self.max_completion_tokens:,} tokens.")


def _reasoning_allowed(entry: dict[str, Any]) -> bool:
    return bool({"reasoning", "reasoning_effort", "include_reasoning"}.intersection(entry.get("supported_parameters", [])))


def _efforts(entry: dict[str, Any], model: dict[str, Any]) -> list[str] | None:
    details = entry.get("reasoning")
    if not isinstance(details, dict) or "supported_efforts" not in details:
        details = model.get("reasoning", {})
    if not isinstance(details, dict) or "supported_efforts" not in details:
        return None
    choices = details["supported_efforts"]
    if not isinstance(choices, list) or any(not isinstance(value, str) for value in choices):
        raise ValueError("reasoning.supported_efforts must be an array of choices when provided.")
    if any(value not in (*EFFORT_ORDER, "none") for value in choices):
        raise ValueError("The API advertised an unrecognized reasoning effort; cannot safely rank its choices.")
    return choices


def _limit(entries: list[dict[str, Any]], name: str) -> int | None:
    values = [entry.get(name) for entry in entries]
    if any(value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0) for value in values):
        raise ValueError(f"{name} must be a positive integer when provided.")
    limits = [value for value in values if isinstance(value, int)]
    return min(limits) if limits else None


def _endpoint_groups(model: dict[str, Any], endpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in endpoints:
        tag = entry.get("tag")
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("An endpoint lacks a routing tag; its capabilities cannot be targeted safely.")
        groups.setdefault(tag, []).append(entry)
    result = []
    for tag, entries in groups.items():
        parameters = set.intersection(*(set(entry.get("supported_parameters", [])) for entry in entries))
        choices = [_efforts(entry, model) for entry in entries]
        shared: list[str] | None = None
        if any(choice is not None for choice in choices):
            shared = sorted(set.intersection(*(set(choice or []) for choice in choices)))
        group: dict[str, Any] = {"tag": tag, "supported_parameters": parameters, "efforts": shared}
        for field in ("context_length", "max_prompt_tokens", "max_completion_tokens"):
            values = []
            for entry in entries:
                value = entry.get(field)
                if value is None:
                    value = model.get(field)
                if field == "context_length" and value is None:
                    raise ValueError(f"No context_length was supplied for endpoint {tag} or its model.")
                values.append({field: value})
            group[field] = _limit(values, field)
        tool_choices = [entry.get("supports_tool_choice") for entry in entries]
        explicit = [value for value in tool_choices if isinstance(value, dict)]
        if explicit:
            keys = set().union(*(value.keys() for value in explicit))
            group["supports_tool_choice"] = {key: all(value.get(key, False) for value in explicit) for key in keys}
        result.append(group)
    return result
