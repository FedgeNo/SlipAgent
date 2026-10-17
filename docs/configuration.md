# Configuration

[Back to the README](../README.md)

## Configure

Successful `--model` and `/model` selections remember the provider and its model separately in `preferences.json` under `SLIPAGENT_STATE_DIR` (default platform data directory; Linux: `$XDG_DATA_HOME/slipagent` or `~/.local/share/slipagent`). Initial provider selection uses `--provider`, then `SLIPAGENT_PROVIDER`, the saved provider if its key is present, and finally the first provider with a key (OpenRouter takes precedence). Initial model selection uses CLI flags first, then the selected provider's exported model variable, its remembered model, `.env`, and its built-in default. Without explicit `--model` or `--provider` flags, optimal mode replaces this initial choice with the highest-scoring eligible free model during startup.

The selected provider requires its own API key. In addition to `.env`,
you can supply it through the process environment:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `SLIPAGENT_PROVIDER` | `openrouter` | Provider module, unless a previous selection is saved. |
| `NVIDIA_API_KEY` | — | Required for NVIDIA inference. |
| `NVIDIA_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA model ID. |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA endpoint override. |
| `KILO_API_KEY` | — | Required for Kilo inference and catalog activation. |
| `KILO_MODEL` | `kilo-auto/free` | Kilo model ID. |
| `KILO_BASE_URL` | `https://api.kilo.ai/api/gateway` | Kilo endpoint override. |
| `MISTRAL_API_KEY` | — | Required for Mistral inference and model discovery. |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral model ID. |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai/v1` | Mistral endpoint override. |
| `GEMINI_API_KEY` | — | Required for Gemini inference and model discovery. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model ID. |
| `GEMINI_BILLING_TIER` | `free` | `paid` bypasses upload pacing for billed Gemini requests; always-free Gemma requests remain paced. |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | Gemini endpoint override. |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter inference. |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Default model slug. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Point at a gateway or mock. |
| `EXA_API_KEY` | — | Optional. Enables the `web_search` tool. |
| `OPENROUTER_REFERER` | SlipAgent GitHub repository URL | Override the app-attribution URL. |
| `OPENROUTER_TITLE` | `SlipAgent` | Override the app-attribution name. |
| `OPENROUTER_USER_AGENT` | `SlipAgent/<version>` | Override the HTTP client identity at startup. |
| `SLIPAGENT_WORKSPACE` | `.` | Default project directory. |
| `SLIPAGENT_STATE_DIR` | Platform data directory | Shared root for sessions, records, preferences, and global MCP settings. |
| `SLIPAGENT_NO_DOTENV` | — | Set to `1` to ignore `.env` entirely. |

OpenRouter requests identify SlipAgent using its repository URL, app title, `X-OpenRouter-Categories: cli-agent`, and a versioned `User-Agent: SlipAgent/<version>`. Other providers receive the versioned user agent. The referer and title overrides above customize OpenRouter attribution.

`.env` is loaded automatically from the platform data directory, or from an
older installation's ancestors when the data directory has none. Project `.env`
files are never read. If one contains SlipAgent settings (provider keys, model or
base-URL variables, `EXA_API_KEY`, or `SLIPAGENT_*`), startup prints a notice
naming the file and variables, without their values. Real environment
variables win over `.env`;
explicit CLI overrides win over both. Values loaded from SlipAgent's `.env` are not passed to commands, checks, Git, search, language servers, or project Python probes, which see the environment your shell provided. Harness-saved credentials use `0600`
permissions. Set `SLIPAGENT_NO_DOTENV=1` in CI to use only the process
environment and CLI configuration. Keep `.env` and credential-bearing MCP
configuration private; `.env` is ignored by Git and global MCP settings live outside the checkout.

### Project Python Environment and Log Storage

Project-specific settings live in `.slipagent/project.json`, relative to the
selected workspace. Create that directory and file when explicit settings are
needed; neither is required for other projects. The supported settings are:

```json
{
  "python": ".venv/bin/python",
  "log_quota_bytes": 1073741824,
  "python_syntax": true,
  "checks": []
}
```

`python` is an interpreter path, not a shell command. Relative paths resolve
against the workspace; absolute paths allow an intentionally shared environment.
On Windows, use a path such as `.venv/Scripts/python.exe`. A startup override wins:

```bash
slipagent --workspace /absolute/path/to/myproject --python .venv/bin/python
```

Without an explicit selection, SlipAgent inspects `.venv` and `venv` in the
workspace. One candidate is probed with isolated Python and a five-second timeout;
it must report a virtual environment. Multiple candidates require an explicit
choice. Missing, malformed, or broken configured interpreters report an error;
the harness does not silently substitute the global interpreter. Non-Python
projects can operate without a selected Python environment.

For Python work that needs an environment, the agent can call
`create_python_environment`. It defaults to a workspace `.venv`, using
SlipAgent's Python to create it, and validates and selects it for the session.
The agent receives guidance to check project instructions, environment managers,
and Python version requirements first. The tool installs no project dependencies.
Repeated calls return the selected interpreter unchanged; existing complete
environments are reused, and unrelated or incomplete directories are never
overwritten. Failed or interrupted creation leaves partial files for inspection.
Supply `path` for a nested project and `python` for another installed interpreter.
Custom paths need an explicit project setting for selection after restart;
the default root `.venv` is rediscovered automatically.

Every model request receives the selection status, interpreter/bin paths, Python
version, prefixes, pip/pytest/ensurepip availability, and commands suited to the
selected environment. The install template uses that interpreter's pip when
present, otherwise an available uv executable targeting that exact venv with
`--python`. If neither installer is available, `install_command` is null; a
venv with ensurepip receives a separate `bootstrap_command`. A missing pytest
produces a null `test_command`. Project instructions determine actual test
arguments and dependencies. Shell PATH remains inherited, so a bare `python` or
`pip` is still not a reliable way to select this environment. The harness creates
no environment and installs no packages automatically. The model is instructed
to obtain an environment choice before dependency installation when none is set.

Interpreter settings and discovery are refreshed before model requests. A probe
is cached until the executable, venv configuration, selection, or import-location
metadata changes. Installing or removing packages therefore refreshes the
reported module availability on the next request.
`--python` continues to override project-file edits for that process. The API-key
`.env` comes from the data directory or an older installation, never the project;
it is separate from workspace-specific `.slipagent/project.json` resolution.

`log_quota_bytes` sets a **session-wide disk quota**, default **1 GiB**, read at
session startup. It must be a positive integer. Logs use owner-only files saved
with the session; they survive model switches, `/task new`, component reloads,
and exit. `/new` starts a new session without deleting saved originals.
With `--no-session`, logs use a temporary directory removed on reset or normal
shutdown. The quota never silently increases and earlier logs are never evicted.

After each complete batch containing successful `write_file` or `edit_file`
calls, Python syntax is checked with the selected project interpreter without
executing/importing the edited code. No selected interpreter means an explicit
skipped check. Set `python_syntax` to false to disable it. Additional checks are
also run automatically when their tools are installed:

| Edited files | Automatic check |
| --- | --- |
| `.js`, `.mjs`, `.cjs` | `node --check`, one file at a time |
| `.php`, `.phtml` | `php -l`, one file at a time |
| `.ts`, `.tsx`, `.mts`, `.cts` | `tsc --noEmit`, once per nearest `tsconfig.json`, or on edited files when no configuration exists |

Missing tools are reported as skipped; nothing is installed automatically.
TypeScript prefers `node_modules/.bin/tsc` in the configuration directory (or
edited file's directory without a configuration) and its ancestors within the
workspace, then `tsc` on PATH. Its checks include type
checking and may report existing errors elsewhere in the same project. They
emit no compiled files and discard incremental build metadata. TypeScript
checks have a 120-second timeout; JavaScript and PHP checks allow 10 seconds per
file. These checks do not execute the edited source.

Custom checks use explicit argument arrays, for example:

```json
{"name":"Ruff", "argv":["{python}","-m","ruff","check","{files}"],
 "extensions":[".py"], "timeout":10}
```

Put these objects in `checks` (at most eight). The example requires Ruff already
installed in the selected environment. `{python}` expands to that interpreter
and `{files}` to separate absolute paths of successfully edited files, filtered
by `extensions`; an empty filter accepts all changed files. Placeholders must
occupy whole arguments. Commands run from the workspace without a shell;
timeouts default to 10 seconds and may range from 0.1 to 120. Output is bounded
and archived. Results enter the last tool observation before summarization.
These checks supplement the agent's explicit tests; a syntax pass proves only
that the checked files parse. Shell/MCP edits are not automatically tracked.

## MCP

Saved connections live in `mcp.json` in the shared platform data directory
(see Saved Sessions), or under `SLIPAGENT_STATE_DIR` when overridden. Project
`.mcp.json` files are no longer loaded. To migrate an old configuration, copy
its server entries into the global file, preserving any existing entries.
Servers still start in the active workspace unless their `cwd` is configured.

SlipAgent is an [MCP](https://modelcontextprotocol.io) client, so tools from
external servers join the same registry as the built-ins. Add one for the
session, or save it globally to `mcp.json` so later runs in every project pick it up:

```text
/mcp add my-server python /path/to/server.py
/mcp save my-server python /path/to/server.py
/mcp
```

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

Remote tools are namespaced `server__tool` so two servers can both expose
`search` without colliding. Names that providers would reject (characters other
than letters, digits, `_` and `-`, or more than 64 characters) are sanitized and
given a stable short hash suffix; calls still reach the server under its original
tool name. The model's argument validation is applied to the
JSON Schema the server advertises. The REPL connects every globally configured server
on startup; a server that fails to start is reported and skipped rather than
blocking the session.

Only the stdio transport and MCP tools are supported; resources and prompts
are not surfaced. Existing MCP connections survive component reloads.
Connected server instructions enter model context with their server scope from the first request, even while their tool definitions are deferred. Servers excluded by an explicit tool allowlist do not contribute guidance.
Structured results and error flags are retained. A server starts in the selected
workspace unless `cwd` is configured; relative values resolve against that workspace.
Tool calls wait up to 300 seconds by default; set a server's `timeout` (seconds) in
`mcp.json` to change it. Startup and tool listing keep a 30-second limit.
MCP processes receive a minimal environment, as in the reference MCP SDKs: `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER` and `TMPDIR` (or the Windows equivalents), plus locale (`LANG`, `LANGUAGE`, `LC_*`) and proxy variables. Add anything else a server needs, including API keys, under its `env` in `mcp.json`.
Project Python selection does not rewrite their command or PATH; use an explicit
interpreter in the server's `command` when required.
