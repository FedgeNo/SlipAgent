"""Configuration resolution.

Model precedence: explicit CLI flag > real environment variable > saved choice
> `.env` file > built-in default. `.env` never shadows a variable the shell
already exports, so `OPENROUTER_MODEL=... slipagent` still wins. Only
SlipAgent's own `.env` is read; project `.env` files belong to the project.
"""

from __future__ import annotations

import os
import json
import math
import re
import sys
import tempfile
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Bound working-model requests, including retries, while allowing long tasks.
DEFAULT_MAX_STEPS = 200
DEFAULT_CONTEXT_STEPS = 50


def save_private_text(path: Path, content: str) -> None:
    """Replace a configuration file atomically with owner-only permissions."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_dotenv_value(path: Path, name: str, value: str) -> None:
    """Set `name=value` in a `.env` file, preserving comments and order."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or "\n" in value or "\r" in value:
        raise ConfigError("dotenv names must be identifiers and values must occupy one line")
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []

    replaced = False
    updated: list[str] = []
    for line in existing:
        key = line.strip().removeprefix("export ").partition("=")[0].strip()
        if key == name:
            updated.append(f"{name}={value}")
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        updated.append(f"{name}={value}")

    save_private_text(path, "\n".join(updated) + "\n")


def dotenv_values(path: Path) -> dict[str, str]:
    """The KEY=VALUE pairs a `.env` file would supply, without touching the environment."""
    values: dict[str, str] = {}
    if path.is_file():
        _apply_dotenv(path, values)
    return values


def remove_dotenv_value(path: Path, name: str) -> bool:
    """Drop `name` from a `.env` file, preserving comments and order; True when it was present."""
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines
            if line.strip().removeprefix("export ").partition("=")[0].strip() != name]
    if len(kept) == len(lines):
        return False
    save_private_text(path, "\n".join(kept) + ("\n" if kept else ""))
    return True


def dotenv_path() -> Path:
    """The `.env` SlipAgent would load for this run, found without reading it."""
    found = find_dotenv()
    if found is not None:
        return found
    return data_directory() / ".env"


class ConfigError(Exception):
    """Raised when the harness cannot be configured from the environment."""


def data_directory() -> Path:
    """Shared persistent data root, separate from installed application files."""
    override = os.environ.get("SLIPAGENT_STATE_DIR")
    if override:
        return Path(override).expanduser().absolute()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        if not base.is_absolute():
            raise ConfigError("LOCALAPPDATA must be an absolute path")
        return base / "SlipAgent"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SlipAgent" / "data"
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    if not base.is_absolute():
        base = Path.home() / ".local" / "share"
    return base / "slipagent"


def _preferences_path() -> Path:
    return data_directory() / "preferences.json"


def _read_preferences() -> dict[str, object]:
    path = _preferences_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not read model preferences at {path}: {exc}") from exc
    if not isinstance(value, dict) or ("model" in value and
            (not isinstance(value["model"], str) or not value["model"].strip())):
        raise ConfigError(f"invalid model preferences at {path}")
    return value


def save_model_choice(model: str, provider: str = "openrouter") -> None:
    """Remember an explicitly selected model across projects and launches."""
    preferences = _read_preferences()
    preferences["provider"] = provider
    if provider == "openrouter":
        preferences["model"] = model
    models = preferences.setdefault("models", {})
    if not isinstance(models, dict):
        raise ConfigError("invalid provider model preferences")
    models[provider] = model
    save_private_text(_preferences_path(), json.dumps(preferences) + "\n")


def configured_provider(provider: str | None = None, *, environ: Mapping[str, str] | None = None,
                        preferences: dict[str, object] | None = None) -> str:
    """Resolve the preferred active provider without requiring its credential."""
    from .providers import active_providers
    env = os.environ if environ is None else environ
    saved = _read_preferences() if preferences is None else preferences
    active = active_providers(env)
    remembered = str(saved.get("provider") or "openrouter")
    default = remembered if remembered in active else next(iter(active), remembered)
    return provider or env.get("SLIPAGENT_PROVIDER") or default


def load_dotenv(environ: dict[str, str] | None = None) -> Path | None:
    """Load KEY=VALUE pairs from SlipAgent's own `.env` into `environ`.

    Reads private user data, then an older installation's `.env`. Project
    `.env` files belong to the project and are never read for configuration.
    Existing environment variables always win.

    Returns the file that was loaded, or None if there was nothing to load.
    Set `SLIPAGENT_NO_DOTENV=1` to skip `.env` entirely (useful in CI and when
    you want the process environment to be the only source of config).
    """
    env = os.environ if environ is None else environ
    if _dotenv_disabled(env):
        return None

    candidate = find_dotenv()
    if candidate is None:
        return None
    try:
        added = _apply_dotenv(candidate, env)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not read {candidate}: {exc}") from exc
    if env is os.environ:
        _DOTENV_NAMES.update(added)
    return candidate


# Variables this process added to its environment from SlipAgent's `.env`.
# They were never in the user's shell, so child processes do not receive them.
_DOTENV_NAMES: set[str] = set()
# Variables the harness itself placed in the environment during this run, such
# as a key pasted at the first-run prompt. Child processes do not receive them.
_SESSION_NAMES: set[str] = set()


def hide_from_commands(name: str) -> None:
    """Keep a variable the harness set in `os.environ` out of child processes."""
    _SESSION_NAMES.add(name)


def environment_source(name: str) -> str | None:
    """Where this process's value of `name` came from: shell, dotenv, session, or None when unset."""
    if name not in os.environ:
        return None
    if name in _DOTENV_NAMES:
        return "dotenv"
    if name in _SESSION_NAMES:
        return "session"
    return "shell"


def forget_environment(name: str) -> str | None:
    """Remove `name` from this process's environment; returns where it had come from."""
    source = environment_source(name)
    os.environ.pop(name, None)
    _DOTENV_NAMES.discard(name)
    _SESSION_NAMES.discard(name)
    return source


def command_environment() -> dict[str, str]:
    """The process environment as the user's shell had it, without `.env` or harness-set values."""
    return {key: value for key, value in os.environ.items()
            if key not in _DOTENV_NAMES and key not in _SESSION_NAMES}


def _dotenv_disabled(env: Mapping[str, str]) -> bool:
    return env.get("SLIPAGENT_NO_DOTENV", "").strip().lower() in {"1", "true", "yes"}


def find_dotenv() -> Path | None:
    """Locate the `.env` that would be loaded, without applying it.

    Searches private user data, then package ancestors for compatibility with
    existing installation settings. With `SLIPAGENT_NO_DOTENV`, only the data
    directory is considered, so saving a value never reaches an installation file.
    """
    here = Path(__file__).resolve().parent
    legacy = () if _dotenv_disabled(os.environ) else (here, *here.parents)
    for directory in (data_directory(), *legacy):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def ignored_project_settings(start: Path) -> tuple[Path, list[str]] | None:
    """Report SlipAgent settings in the nearest project `.env`; values are not read."""
    from .providers import PROVIDERS, provider_class

    names = {"EXA_API_KEY", "OPENROUTER_REFERER", "OPENROUTER_TITLE"}
    for provider in PROVIDERS:
        implementation = provider_class(provider)
        names.update((implementation.key_env, implementation.model_env, implementation.base_url_env))
    loaded = find_dotenv()
    for directory in (start.resolve(), *start.resolve().parents):
        candidate = directory / ".env"
        if not candidate.is_file():
            continue
        if candidate == loaded:
            return None
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None
        found = []
        for line in lines:
            key = line.strip().removeprefix("export ").partition("=")[0].strip()
            if "=" in line and (key in names or key.startswith("SLIPAGENT_")) and key not in found:
                found.append(key)
        return (candidate, found) if found else None
    return None


def _apply_dotenv(path: Path, env: MutableMapping[str, str]) -> list[str]:
    """Merge `path` into `env` without overwriting what is already set."""
    added = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()

        key, separator, value = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        if key and key not in env:
            env[key] = value
            added.append(key)
    return added


@dataclass(slots=True)
class Config:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = field(default=DEFAULT_BASE_URL)
    http_referer: str | None = None
    app_title: str = "SlipAgent"
    workspace: Path = field(default_factory=lambda: Path.cwd())
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float | None = None
    max_tokens: int | None = None
    context_steps: int = DEFAULT_CONTEXT_STEPS
    overthinking: bool = True
    provider: str = "openrouter"

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigError("max_steps must be at least 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ConfigError("max_tokens must be at least 1")
        if self.context_steps < 1:
            raise ConfigError("context_steps must be at least 1")
        if self.temperature is not None and (not math.isfinite(self.temperature) or self.temperature < 0):
            raise ConfigError("temperature must be a finite non-negative number")

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        workspace: str | Path | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        context_steps: int | None = None,
        overthinking: bool = True,
        environ: dict[str, str] | None = None,
    ) -> Config:
        env = os.environ if environ is None else environ
        shell_models = dict(env)
        if environ is None:
            load_dotenv()
        from .providers import provider_class
        from .api import APIConfigError
        preferences = _read_preferences()
        selected = configured_provider(provider, environ=env, preferences=preferences)
        try:
            implementation = provider_class(selected)
        except APIConfigError as exc:
            raise ConfigError(str(exc)) from exc
        models = preferences.get("models", {})
        if not isinstance(models, dict):
            raise ConfigError("invalid provider model preferences")
        saved_model = models.get(selected) or (preferences.get("model") if selected == "openrouter" else None)
        explicit_model = model or shell_models.get(implementation.model_env)
        resolved_key = api_key or env.get(implementation.key_env, "")
        if not resolved_key.strip():
            raise ConfigError(
                f"No {selected} API key found. Set {implementation.key_env} "
                "(see docs/configuration.md) or pass --api-key."
            )

        root = Path(workspace or env.get("SLIPAGENT_WORKSPACE", ".")).expanduser()

        return cls(
            api_key=resolved_key.strip(),
            model=explicit_model or str(saved_model or env.get(implementation.model_env) or implementation.default_model),
            base_url=base_url or env.get(implementation.base_url_env) or implementation.default_base_url,
            provider=selected,
            http_referer=env.get("OPENROUTER_REFERER") or None,
            app_title=env.get("OPENROUTER_TITLE") or "SlipAgent",
            workspace=root,
            max_steps=max_steps if max_steps is not None else DEFAULT_MAX_STEPS,
            temperature=temperature,
            max_tokens=max_tokens,
            context_steps=context_steps if context_steps is not None else DEFAULT_CONTEXT_STEPS,
            overthinking=overthinking,
        )

    def with_overrides(self, **changes: object) -> Config:
        return replace(self, **changes)  # type: ignore[arg-type]
