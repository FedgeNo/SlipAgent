# SlipAgent

**A terminal coding agent designed for free models with large context windows on
[OpenRouter](https://openrouter.ai), [NVIDIA](https://build.nvidia.com), [Kilo](https://kilo.ai/gateway), [Mistral](https://console.mistral.ai), and [Google Gemini](https://ai.google.dev), with a harness you can modify while it runs.**

SlipAgent gives a model tools to read, search, edit, test, and commit code in a
project directory. It keeps asking the model what to do next until the task is
finished, you stop it, or the step limit is reached. By default it picks the
best-scoring free model available to your API keys and moves to the next one
when a model fails.

- **Free-model workflow:** automatic selection of the best free model and quick
  switching.
- **Resumable work:** sessions are saved as you go; resume, fork, or rewind
  file edits later. Interrupted tools are never replayed automatically.
- **Coding tools:** file editing, search, shell commands, Git, optional web
  search, semantic code search, and more tools through MCP.
- **Human inspection:** clickable file references, a built-in editor, colored
  diffs, saved tool-call records, and a Touched/Calls sidebar.
- **Live development:** the installed harness is an editable checkout, and the
  running process loads compatible changes to its own code without restarting.

## Quick Start

Requires **Python 3.11+** and Git. From a download or clone of this repository:

```bash
python3 install.py        # Windows: py -3 install.py
```

The installer creates a private environment and puts `slipagent` on your PATH;
open a new terminal if it asks you to. Then, in your project directory:

```bash
slipagent                          # interactive session
slipagent "add tests for parser"   # run one task and exit
```

The first launch without an API key asks for one, lists providers that offer
free models, and offers to save it. You can add keys for more providers later
with `/key`; more providers mean more free models to fall back on.

To update later, run `slipagent --update`. It downloads the latest version and
runs its installer, which warns before replacing any files you have edited.
[Installation details](docs/installation.md) cover data locations, Windows,
and running from a development checkout.

## Choosing Models

Startup selects the highest-ranked free model and, if a model fails or stays
silent, falls back to the next one. After 10 seconds without output the status
line counts down; press **Ctrl+N** to skip to the next model right away.

`/model` gathers every model choice in one menu, each line showing its
subcommand:

| Command | Effect |
| --- | --- |
| `/model` | Menu of the options below |
| `/model <provider>` | Choose from that provider's models, free ones first |
| `/model <words>` | Choose from models matching every word |
| `/model <model>` | Switch to an exact model and remember it |
| `/model free` | Choose from free models |
| `/model optimal` / `next` / `reset` | Automatic selection, skip the current model, or start again from the best |

See [Free Models, Optimal Mode, and Providers](docs/models.md) for scores,
provider setup, and limits.

## Everyday Use

Type a task and press Enter. While the agent works you can keep typing; your
messages queue and are sent at the next step. **Esc** interrupts immediately;
`/stop` lets the current step finish first.

| Command | Effect |
| --- | --- |
| `/help [command]`, `/menu` | Help, or browse every command |
| `/settings` | Change a setting from a menu that shows current values |
| `/status` | Show the model, workspace, interpreter, and settings |
| `/new` | Start a fresh conversation |
| `/task new` | Treat your next prompt as a new task in this conversation |
| `/resume`, `/sessions`, `/fork` | Return to, list, or copy saved sessions |
| `/rewind` | Undo the agent's file edits back to a chosen point |
| `/copy`, `/view` | Copy the latest reply; open saved outputs and diffs |
| `/key`, `/cost` | Manage API keys; show token usage |
| `/exit` | Quit (Ctrl+D also works) |

| Key | Effect |
| --- | --- |
| Esc | Interrupt the agent, or close a menu |
| Ctrl+N | Skip a silent model (optimal mode) |
| Ctrl+O | Show or hide the Touched/Calls sidebar |
| Ctrl+\\ | Show the exact context sent to the model |
| Shift+Tab | Move between the feed, sidebar, and links |
| F2 / F3 / F4 | Markdown source view / copy the latest reply / inspection options |

The [Terminal Guide](docs/terminal.md) covers every command, the sidebar,
viewers, the editor, queueing, and Danger Mode.

## Project Setup

- `/init` creates an `AGENTS.md` for project notes; SlipAgent loads it with
  other project instructions on every request.
- `.slipagent/project.json` can select the project's Python interpreter and add
  checks that run after edits. See [Configuration](docs/configuration.md).
- MCP servers are added with `/mcp add` (this session) or `/mcp save`
  (all sessions).
- Language servers for `navigate_code` are configured in your user settings,
  described under [Tools](docs/tools.md#optional-language-servers).

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Installer, updates, data locations, development checkouts |
| [Models](docs/models.md) | Free models, optimal mode, providers |
| [Configuration](docs/configuration.md) | Environment variables, `.env`, project settings, MCP |
| [Terminal Guide](docs/terminal.md) | Commands, layout, inspection, editor, queueing |
| [Sessions](docs/sessions.md) | Saved sessions, resume, fork, rewind, delete |
| [Tools](docs/tools.md) | Built-in tools, background commands, language servers |
| [Automation](docs/automation.md) | Headless runs, JSON events, repeated trials |
| [Model Context](docs/model-context.md) | Context view, response protocol, reasoning, history |
| [Self-Editing](docs/self-editing.md) | Changing SlipAgent while it runs |
| [Source Overview](docs/source-overview.md), [Development](docs/development.md) | Module map, tests, contributing |
| [Semantic Search](docs/semantic-search.md), [Model Scores](docs/model-scores.md) | Search internals and score calibration |

## Troubleshooting

| Symptom | Where to Check |
| --- | --- |
| `slipagent` is unavailable after installation | Open a new terminal; check the printed launcher directory and user PATH. Use the installer's `--dry-run` to inspect destinations. |
| Edits to the harness have no effect | Confirm you edited the installed `source/` checkout. Inspect `/generations` and reload messages; stable-contract changes need a restart. |
| Agent pauses when a file opens | Editable files hold agent work; save or discard edits and close the editor. Read-only output/diff viewers allow work to continue. |
| Directory names are not clickable | Check desktop-session/opener availability or configure a folder opener through Options. |
| A model or provider is absent | Check that provider's credentials and `/model <provider>`. Catalog errors and unsupported capability profiles are reported separately. |
| The agent misreads a result or repeats an action | Inspect the outgoing context with Ctrl+\ and the exact attempt with `/requests`; use the saved call/output viewers to compare observations. |
| Old sessions or MCP servers disappeared after a move | Check the active platform data directory and `SLIPAGENT_STATE_DIR`. Project `.mcp.json` is not loaded; global connections use the data directory's `mcp.json`. |
| An interrupted tool may have changed a file | Inspect the file and saved results before retrying. Resume never replays uncertain tools automatically. |
| Updating warns about local edits | Review every listed path and preserve the changes you want before confirming. A missing hash baseline is not proof of an unmodified installation. |

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) gives humans and agents an end-to-end change
workflow. [docs/architecture.md](docs/architecture.md) describes lifecycle,
context ordering, validation, source recovery, storage, and reload contracts.
Read applicable local project instructions, including `AGENTS.md`, first.
Keep behavior in
reloadable components, preserve persistent state layouts, and use relative
imports so a candidate generation stays coherent. Add behavior tests, run the
suite and strict mypy, and test reload behavior against temporary source copies.
Most changes can be developed and applied in the same live session. Changes to
the stable frame or shared contracts need a deliberate restart.
