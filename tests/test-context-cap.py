"""The working-context cap accepts k and M suffixes and never exceeds the model's limit."""

import json

import pytest

from slipagent.config import ConfigError, context_tokens_preference, parse_token_count
from test_cli_e2e import metadata_server, project_dir, run_repl_commands


@pytest.mark.parametrize("text,count", [
    ("300000", 300_000), ("300k", 300_000), ("345K", 345_000), ("1.5M", 1_500_000),
    ("1m", 1_000_000), (" 2.5k ", 2_500),
])
def test_token_counts_accept_k_and_m_suffixes(text, count):
    assert parse_token_count(text) == count


@pytest.mark.parametrize("text", ["", "k", "1e5", "-3k", "5g", "1,000"])
def test_token_counts_reject_other_spellings(text):
    with pytest.raises(ValueError):
        parse_token_count(text)


@pytest.mark.parametrize("saved,expected", [(None, None), (300_000, 300_000), ("1.5M", 1_500_000)])
def test_preference_sets_the_default_cap(tmp_path, monkeypatch, saved, expected):
    monkeypatch.setenv("SLIPAGENT_STATE_DIR", str(tmp_path))
    if saved is not None:
        (tmp_path / "preferences.json").write_text(json.dumps({"contextTokens": saved}))
    assert context_tokens_preference() == expected


@pytest.mark.parametrize("saved", ["lots", 100, True, 1.5, ["300k"]])
def test_malformed_preference_is_reported(tmp_path, monkeypatch, saved):
    monkeypatch.setenv("SLIPAGENT_STATE_DIR", str(tmp_path))
    (tmp_path / "preferences.json").write_text(json.dumps({"contextTokens": saved}))
    with pytest.raises(ConfigError, match="contextTokens"):
        context_tokens_preference()


def test_context_command_sets_shows_and_removes_the_cap(project_dir, metadata_server):
    proc = run_repl_commands(project_dir, [
        "/context", "/context 300k", "/context", "/context 12", "/context lots", "/context reset",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "context cap: none" in proc.stderr
    assert "context cap: 300,000 tokens" in proc.stderr
    assert "at least 4,096 tokens" in proc.stderr
    assert "not a token count" in proc.stderr
    assert proc.stderr.rstrip().count("context cap: none") == 2


def test_cap_above_the_model_limit_reports_that_the_limit_applies(project_dir, metadata_server):
    proc = run_repl_commands(project_dir, ["/context 900M"])
    assert proc.returncode == 0, proc.stderr
    assert "the model's limit applies" in proc.stderr


async def test_cap_is_saved_with_the_session_and_restored_on_resume(tmp_path, metadata_server):
    from slipagent import cli
    from slipagent.agent import Agent
    from slipagent.tools.base import ToolRegistry
    from test_agent import StubClient, completion
    session = await cli.Launcher(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)])).build_session()
    journal = session.registry.services["session_journal"]
    try:
        await cli.CommandRunner(session).handle("/context 300k")
        session.agent.client = StubClient([completion("Answer")])
        await session.agent.run("First prompt")
        await session.agent.wait_for_compaction()
        assert journal.load(journal.session_id)["state"]["context_tokens"] == 300_000

        await cli.CommandRunner(session).handle("/context reset")
        assert session.agent.context_tokens is None
        assert journal.load(journal.session_id)["state"]["context_tokens"] is None

        await cli.CommandRunner(session).handle("/context 1.5M")
        restored = Agent(StubClient([completion("Later")]), ToolRegistry(), "test")
        journal.restore(restored, journal.load(journal.session_id))
        assert restored.context_tokens == 1_500_000
        await cli.CommandRunner(session).handle("/context default")
        journal.restore(restored, journal.load(journal.session_id))
        assert restored.context_tokens is None
    finally:
        await cli.Repl(session).shutdown()
