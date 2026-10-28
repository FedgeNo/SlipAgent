"""Startup warns about limited provider capacity on every launch."""

import pytest

from slipagent import cli
from slipagent.providers import PROVIDERS, provider_class
from test_cli_e2e import metadata_server


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("deferred", [False, True])
async def test_startup_provider_warning(tmp_path, monkeypatch, capsys, metadata_server, count, deferred):
    for provider in PROVIDERS:
        if provider != "openrouter":
            monkeypatch.delenv(provider_class(provider).key_env, raising=False)
    for provider in [name for name in PROVIDERS if name != "openrouter"][:count - 1]:
        monkeypatch.setenv(provider_class(provider).key_env, "test")
    launcher = cli.Launcher(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    for _ in range(2):
        session = await launcher.build_session(defer_startup=deferred)
        try:
            if deferred:
                await session.extensions.pop("startup")()
            output = capsys.readouterr().err
            assert ("Configure at least 3 providers" in output) == (count < 3)
            if count < 3:
                assert f"Only {count} API provider" in output
                assert "rate limiting and degraded performance" in output
        finally:
            await cli.Repl(session).shutdown()
