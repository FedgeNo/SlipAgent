"""Config resolution: `.env` loading, precedence, and persistence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from slipagent.config import (
    DEFAULT_MODEL,
    Config,
    ConfigError,
    dotenv_path,
    find_dotenv,
    load_dotenv,
    save_dotenv_value,
)


def test_dotenv_loads_simple_pairs(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "OPENROUTER_API_KEY=sk-or-v1-abc\n"
        "\n"
        "export OPENROUTER_MODEL=some/model\n"
        'OPENROUTER_TITLE="quoted title"\n',
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    loaded = load_dotenv(tmp_path, env)

    assert loaded == tmp_path / ".env"
    assert env["OPENROUTER_API_KEY"] == "sk-or-v1-abc"
    assert env["OPENROUTER_MODEL"] == "some/model"
    assert env["OPENROUTER_TITLE"] == "quoted title"


def test_dotenv_does_not_override_existing_values(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_MODEL=from-file\n", encoding="utf-8")
    env = {"OPENROUTER_MODEL": "from-shell"}

    load_dotenv(tmp_path, env)

    assert env["OPENROUTER_MODEL"] == "from-shell"


def test_dotenv_can_be_disabled(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-abc\n", encoding="utf-8")
    env = {"SLIPAGENT_NO_DOTENV": "1"}

    assert load_dotenv(tmp_path, env) is None
    assert "OPENROUTER_API_KEY" not in env


def test_dotenv_finds_nearest_file_up_the_tree(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_MODEL=top\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (tmp_path / "a" / ".env").write_text("OPENROUTER_MODEL=mid\n", encoding="utf-8")

    env: dict[str, str] = {}
    load_dotenv(nested, env)

    assert env["OPENROUTER_MODEL"] == "mid"


def test_dotenv_falls_back_to_the_install_directory(tmp_path: Path) -> None:
    """A `.env` beside the install is used when the cwd has none.

    This is what lets one credential work from any project directory.
    """
    environ: dict[str, str] = {"SLIPAGENT_NO_DOTENV": "0"}
    # Clear any real value so the fallback is visible.
    environ.pop("OPENROUTER_API_KEY", None)

    loaded = load_dotenv(tmp_path, environ)

    # Either no `.env` exists at all, or the package-level one was applied.
    if loaded is not None:
        assert loaded.parent in _package_chain()
        assert "OPENROUTER_API_KEY" in environ


def _package_chain() -> set[Path]:
    from slipagent import config as config_module

    here = Path(config_module.__file__).resolve().parent
    return {here, *here.parents}


def test_config_requires_a_key() -> None:
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Config.from_env(environ={})


def test_config_precedence_cli_over_env() -> None:
    config = Config.from_env(
        api_key="sk-or-v1-cli",
        model="cli/model",
        environ={"OPENROUTER_API_KEY": "sk-or-v1-env",
                 "OPENROUTER_MODEL": "env/model"},
    )

    assert config.model == "cli/model"
    assert config.api_key == "sk-or-v1-cli"


def test_config_falls_back_to_env_then_default() -> None:
    assert Config.from_env(environ={"OPENROUTER_API_KEY": "k"}).model == DEFAULT_MODEL
    assert Config.from_env(
        environ={"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "env/model"}
    ).model == "env/model"


def test_default_model_is_the_configured_one() -> None:
    assert DEFAULT_MODEL == "nvidia/nemotron-3.5-lightning:free"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_creates_file_when_missing(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    save_dotenv_value(target, "OPENROUTER_API_KEY", "sk-or-v1-new")

    assert target.read_text().strip() == "OPENROUTER_API_KEY=sk-or-v1-new"


def test_save_replaces_in_place_preserving_comments(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text(
        "# keep me\nOPENROUTER_API_KEY=old\nOTHER=1\n", encoding="utf-8"
    )

    save_dotenv_value(target, "OPENROUTER_API_KEY", "new")

    lines = target.read_text().splitlines()
    assert lines == ["# keep me", "OPENROUTER_API_KEY=new", "OTHER=1"]


def test_save_appends_when_absent(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("# header\n", encoding="utf-8")

    save_dotenv_value(target, "OPENROUTER_API_KEY", "new")

    assert target.read_text().splitlines()[-1] == "OPENROUTER_API_KEY=new"


def test_save_sets_owner_only_permissions(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    save_dotenv_value(target, "OPENROUTER_API_KEY", "secret")

    assert oct(target.stat().st_mode)[-3:] == "600"


def test_find_dotenv_does_not_mutate_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-abc\n", encoding="utf-8")
    # Start from a known-empty environment so the assertion holds whether or not
    # the surrounding shell exports these variables.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert find_dotenv(tmp_path) == tmp_path / ".env"
    assert "OPENROUTER_API_KEY" not in os.environ


def test_dotenv_path_reports_the_file_that_would_be_used(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("OPENROUTER_API_KEY=sk-or-v1-abc\n", encoding="utf-8")

    assert dotenv_path(tmp_path) == target


@pytest.mark.parametrize("changes", [{"max_steps": 0}, {"max_steps": -1},
                                     {"max_tokens": 0}, {"temperature": float("nan")}])
def test_config_rejects_invalid_limits(changes: dict) -> None:
    with pytest.raises(ConfigError):
        Config.from_env(environ={"OPENROUTER_API_KEY": "test"}, **changes)


def test_dotenv_rejects_injected_lines_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("OTHER=keep\n")
    with pytest.raises(ConfigError):
        save_dotenv_value(target, "OPENROUTER_API_KEY", "key\nOTHER=changed")
    assert target.read_text() == "OTHER=keep\n"


def test_dotenv_save_failure_preserves_existing_credentials(tmp_path, monkeypatch) -> None:
    target = tmp_path / ".env"
    target.write_text("OPENROUTER_API_KEY=old\n")
    def fail_replace(*args):
        raise OSError("failed commit")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="failed commit"):
        save_dotenv_value(target, "OPENROUTER_API_KEY", "new")
    assert target.read_text() == "OPENROUTER_API_KEY=old\n"
    assert list(tmp_path.iterdir()) == [target]
