"""Configuration resolution.

Precedence: explicit CLI flag > real environment variable > `.env` file >
built-in default. `.env` is never allowed to shadow a variable the shell
already exports, so `OPENROUTER_MODEL=... slipagent` still wins.
"""

from __future__ import annotations

import os
import math
import re
import tempfile
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Deliberately high: the loop should finish real work, not stop because a task
# happened to be long. The cap only exists to stop a model that is going in
# circles.
DEFAULT_MAX_STEPS = 200
DEFAULT_CONTEXT_POSTS = 50
DEFAULT_CONTEXT_TOKENS = 200_000


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


def dotenv_path(start: Path | None = None) -> Path:
    """The `.env` SlipAgent would load for this run, found without reading it."""
    found = find_dotenv(start)
    if found is not None:
        return found
    return Path(__file__).resolve().parents[2] / ".env"


class ConfigError(Exception):
    """Raised when the harness cannot be configured from the environment."""


def load_dotenv(start: Path | None = None, environ: dict[str, str] | None = None) -> Path | None:
    """Load KEY=VALUE pairs from the nearest `.env` into `environ`.

    Searches the directory chain above `start` first, then the directory chain
    above this package, so a single `.env` next to the install keeps working
    when you run the harness against any project directory. Existing variables
    always win, which keeps this safe to call on every run.

    Returns the file that was loaded, or None if there was nothing to load.
    Set `SLIPAGENT_NO_DOTENV=1` to skip `.env` entirely (useful in CI and when
    you want the process environment to be the only source of config).
    """
    env = os.environ if environ is None else environ
    if _dotenv_disabled(env):
        return None

    candidate = find_dotenv(start)
    if candidate is None:
        return None
    try:
        _apply_dotenv(candidate, env)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not read {candidate}: {exc}") from exc
    return candidate


def _dotenv_disabled(env: Mapping[str, str]) -> bool:
    return env.get("SLIPAGENT_NO_DOTENV", "").strip().lower() in {"1", "true", "yes"}


def find_dotenv(start: Path | None = None) -> Path | None:
    """Locate the `.env` that would be loaded, without applying it.

    Searches the directory chain above `start` first, then the chain above this
    package, so a single `.env` next to the install keeps working when the
    harness runs against any project directory.
    """
    here = Path(__file__).resolve().parent
    start_dir = (start or Path.cwd()).resolve()

    seen: set[Path] = set()
    for chain_root in (start_dir, here):
        for directory in (chain_root, *chain_root.parents):
            if directory in seen:
                continue
            seen.add(directory)
            candidate = directory / ".env"
            if candidate.is_file():
                return candidate
    return None


def _apply_dotenv(path: Path, env: MutableMapping[str, str]) -> None:
    """Merge `path` into `env` without overwriting what is already set."""
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


@dataclass(slots=True)
class Config:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = field(default=DEFAULT_BASE_URL)
    http_referer: str | None = None
    app_title: str = "slipagent"
    workspace: Path = field(default_factory=lambda: Path.cwd())
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float | None = None
    max_tokens: int | None = None
    context_posts: int = DEFAULT_CONTEXT_POSTS
    context_tokens: int = DEFAULT_CONTEXT_TOKENS

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigError("max_steps must be at least 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ConfigError("max_tokens must be at least 1")
        if self.context_posts < 1 or self.context_tokens < 1:
            raise ConfigError("context_posts and context_tokens must be at least 1")
        if self.temperature is not None and (not math.isfinite(self.temperature) or self.temperature < 0):
            raise ConfigError("temperature must be a finite non-negative number")

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        workspace: str | Path | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        context_posts: int | None = None,
        context_tokens: int | None = None,
        environ: dict[str, str] | None = None,
    ) -> Config:
        env = os.environ if environ is None else environ
        if environ is None:
            load_dotenv()

        resolved_key = api_key or env.get("OPENROUTER_API_KEY", "")
        if not resolved_key.strip():
            raise ConfigError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY "
                "(see README.md) or pass --api-key."
            )

        root = Path(workspace or env.get("SLIPAGENT_WORKSPACE", ".")).expanduser()

        return cls(
            api_key=resolved_key.strip(),
            model=model or env.get("OPENROUTER_MODEL") or DEFAULT_MODEL,
            base_url=base_url or env.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
            http_referer=env.get("OPENROUTER_REFERER") or None,
            app_title=env.get("OPENROUTER_TITLE") or "slipagent",
            workspace=root,
            max_steps=max_steps if max_steps is not None else DEFAULT_MAX_STEPS,
            temperature=temperature,
            max_tokens=max_tokens,
            context_posts=context_posts if context_posts is not None else DEFAULT_CONTEXT_POSTS,
            context_tokens=context_tokens if context_tokens is not None else DEFAULT_CONTEXT_TOKENS,
        )

    def with_overrides(self, **changes: object) -> Config:
        return replace(self, **changes)  # type: ignore[arg-type]
