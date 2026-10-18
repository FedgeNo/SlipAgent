# Development

[Back to the README](../README.md)

## Development

```bash
.venv/bin/python -m pytest          # suite without slow tests, up to four parallel workers
.venv/bin/python -m pytest --slow   # everything, including tests marked slow
.venv/bin/python -m pytest -n 0     # serial execution for debugging
.venv/bin/python -m mypy            # strict type checking
```

The suite covers workspace confinement, tools, context, the model loop, the API
client, terminal behavior, and live reloads. End-to-end tests drive the installed
CLI against a local stub OpenRouter server. Reload tests edit disposable package
copies in isolated interpreters; Git tests stage and commit only in temporary
repositories. Tests use dummy credentials and block external DNS and socket
connections, including in Python CLI subprocesses. Local stub servers are
allowed; the real `.env` and model endpoints are not used. Install the `[dev]`
extra to include `pytest-xdist`; use `--durations=20` to inspect slow tests.

To watch the harness work without spending tokens, run the demo: it serves a
scripted fake API in-process and points the real CLI at it, including a
deliberate sandbox escape and an unknown-tool call so you can see the error
paths.

```bash
.venv/bin/python scripts/demo_run.py          # one-shot, verbose
.venv/bin/python scripts/demo_run.py --repl   # drive the REPL
```

`scripts/mcp_drive.py` does the same for the `/mcp` command, connecting a
scripted stub server to a local fake API. `scripts/repl_render_demo.py` also uses
a local fake API and prints a captured plain-text transcript. Its piped input
does not exercise interactive redraws or paced typing.
These three demos use disposable workspaces and need no real credentials.
`scripts/repl_drive.py` is a separate **live** manual driver: it uses the current
workspace/configuration and can make real API requests and tool changes.
