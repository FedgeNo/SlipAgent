# SlipAgent

**A terminal coding agent designed for free models with large context windows on
[OpenRouter](https://openrouter.ai), with a harness you can modify while it runs.**

SlipAgent gives a model tools to read, search, edit, test, and commit code in a
project directory. It keeps asking the model what to do next until the task is
finished, you stop it, or the step limit is reached.

The harness is part of the workspace it can work on. Run it from an editable
checkout and the agent can improve its own tools, prompts, commands, context
handling, and terminal interface. A contributor can edit those same files in
their editor. The running process loads compatible changes without losing the
conversation or restarting the terminal.

- **Free-model workflow:** browse zero-cost models, switch models during a
  session, and see the remaining free-call quota when OpenRouter reports it.
- **Live development:** a stable runtime applies component changes between
  complete model/tool batches and keeps the previous version if a reload fails.
- **Retrievable history:** recent work stays in context; numbered originals and
  model-written summaries remain available through `recall_history`.
- **Resumable work:** private session journals preserve originals, summaries,
  task state, usage, and command logs; interrupted tools are never replayed automatically.
- **Coding tools:** file editing, search, shell commands, Git operations, and
  optional web search, with additional tools through MCP.
- **A responsive terminal:** wrapping output, mouse-wheel scrollback, a fixed
  prompt/status area, queued follow-ups, and `/stop` at step boundaries.

[Quick Start](#quick-start) · [Free Models](#using-free-models) ·
[Architecture](#architecture) · [Live Development](#editing-the-running-harness) ·
[Usage](#use) · [Tools](#tools) · [Development](#development) ·
[Contributor Guide](CONTRIBUTING.md) · [Internal Contracts](docs/architecture.md)

## Quick Start

Requires **Python 3.11+**, an OpenRouter API key, and Git for the Git tools.

Terminology: a message is one user, assistant, system, or tool message. A step is one model response plus any requested tool batch and its results. A run processes a user request through steps to a final answer with no tool calls. A session is the persistent conversation containing runs. Model context carries separate history-step records, rather than one combined JSON object per run.
Clone or download this repository, then run the installer from its root:

```bash
python3 install.py
```

On Windows, use `py -3 install.py`. No administrator privileges are needed. The installer copies the application source, creates a private virtual environment, installs SlipAgent in editable mode, and places its launcher on your user PATH. It uses `uv` if available, otherwise Python's `venv` and pip. Add `--dev` to include the project's testing and development dependencies.

The same script also updates an existing installation. Download the desired version and run `python3 install.py` from that download (Windows: `py -3 install.py`). It replaces installed application files and refreshes dependencies and the launcher; it does not download updates or run Git commands. Running inside the installed checkout only refreshes dependencies and the launcher. Unchanged installations update without warnings or confirmation. If application files have been modified, added, or deleted, the installer lists every affected path, warns that updating may lose local edits, and asks you to back up changes before confirming with `y` or `yes`. Enter, EOF, or Ctrl+C cancels. A missing or invalid baseline also requires confirmation and lists files to inspect. Changes from separate checkouts are not merged. First installations and `--dry-run` need no confirmation.

The installer compares SHA-256 hashes against `installed-file-hashes.json` in the installation directory, falling back to the checkout's `file-hashes.json` when available. Successful installations or updates from another checkout record the installed file hashes. Running inside the installed checkout retains the existing baseline rather than treating local edits as an unmodified release. Git internals, secrets, virtual environments, and caches are excluded from comparisons; the hash manifest itself is excluded to avoid self-reference.

| OS | Installation directory | PATH launcher |
| --- | --- | --- |
| Linux | `$XDG_DATA_HOME/slipagent`, default `~/.local/share/slipagent` | `~/.local/bin/slipagent` |
| macOS | `~/Library/Application Support/SlipAgent` | `~/.local/bin/slipagent` |
| Windows | `%LOCALAPPDATA%\SlipAgent` | `%LOCALAPPDATA%\SlipAgent\bin\slipagent.exe` |

Linux and macOS use a symlink. Windows also uses a symlink when permitted, with an executable launcher fallback that does not require elevation. Missing PATH entries are added to bash, zsh, sh, ksh, or fish configuration on Linux/macOS, or to the Windows user PATH. Open a new terminal when PATH changes; other shells receive instructions to add the directory manually. Use `--dry-run` to inspect the destinations, or `--install-dir`, `--bin-dir`, and `--python` to override them.

The installer replaces recognized SlipAgent launchers and saves a backup; it refuses unrelated commands or nonempty unmanaged installation directories. The permanent working checkout lives at `source` inside the installation directory, with its own `.venv`. Initial copies include application files, uncommitted edits, and Git metadata, but exclude existing virtual environments, caches, and secrets. Updates from another checkout replace the application directories and supplied root files while preserving installed Git metadata, the virtual environment, and configuration. Removed application directory contents are removed during updates too. Existing Git history is retained, so new downloaded files appear as working-tree changes. A failed update can leave changed source files or dependencies; the installer does not roll these back. Older release directories are retained during migration. On Linux/macOS, `current` points to `source`; `installation.json` records the source, interpreter, and launcher on all platforms. Edit the permanent checkout for live changes and normal Git development.

An existing checkout `.env` is copied to the private installation directory on the first install, with owner-only permissions on Linux/macOS. Subsequent installs preserve that configuration. The installer does not move or delete the original checkout or project sessions.

For development directly in this checkout instead of a managed installation, use the existing project environment or create one if none exists:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Create an ignored `.env` file containing `OPENROUTER_API_KEY=your-key`. Start with
NVIDIA Nemotron 3 Ultra (free), the default model, selected explicitly:

```bash
slipagent --model nvidia/nemotron-3-ultra-550b-a55b:free
```

For another project or a single task:

```bash
slipagent --workspace /absolute/path/to/myproject --model nvidia/nemotron-3-ultra-550b-a55b:free
slipagent --model nvidia/nemotron-3-ultra-550b-a55b:free -p "add type hints to src/parser.py and run the tests"
```

The editable install (`-e`) is essential for working on the running harness:
Python loads the source from its editable installation, so edits in that source tree reach the source
watcher. The `[dev]` extra installs the test and type-checking tools; use
`python -m pip install -e .` if you only need the application.

Reuse an existing project environment instead of recreating it. If that venv
has no pip, `uv pip install --python .venv/bin/python -e ".[dev]"` targets it
explicitly without needing pip inside it. See the
[environment setup and bootstrap alternatives](CONTRIBUTING.md#start-with-the-correct-environment).

## Using Free Models

**Choose a model explicitly.** We recommend free coding models with a **1M-token
context window**, such as NVIDIA Nemotron 3.5 Lightning and NVIDIA Nemotron 3 Super.
We do not recommend free-router: automatic selection can route to models
unsuitable for coding, including classifiers.
A large context window gives the agent room for project instructions, source,
tool results, and a long-running conversation as it develops a project or its
own harness.

| Recommended Model | OpenRouter Model ID |
| --- | --- |
| [NVIDIA Nemotron 3 Ultra (free)](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free) (default) | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| [NVIDIA Nemotron 3 Super (free)](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free) | `nvidia/nemotron-3-super-120b-a12b:free` |

Provider context limits and free availability can change. Check `/models` for
live catalog information; selection also fetches endpoint properties and uses
their more conservative limits. A model's advertised maximum does not guarantee
that every free endpoint offers the same window.

Use `/models free` to select from zero-cost catalog entries. Use Up/Down and
Enter to select, or Escape to cancel. `/models <text>` filters by model ID;
`/model` opens the full catalog and `/model <slug>` selects directly. Successful
selections are remembered for future launches. Paid OpenRouter models also work.
Without an interactive terminal, model searches and partial-name suggestions use ASCII tables formatted with
`tabulate`, showing model IDs, context limits, and input/output prices. Long IDs
wrap inside their cells; narrow terminals display each model's fields vertically.

SlipAgent is designed around free OpenRouter inference. Model requests are a
limited resource, so the prompt encourages batching predictable tool calls into
one response. Up to four independent file/search reads run concurrently. Writes,
shell commands, state updates, and unclassified MCP tools form ordered barriers;
all observations return in the model's declared order. A failed
tool contributes an error result instead of discarding the rest of the batch.
The model can explain its work alongside those calls.

Free inference still has availability and request limits; see
[OpenRouter's current limits](https://openrouter.ai/docs/api_reference/limits).
SlipAgent refreshes key information at startup and every 15 minutes, including
while idle or working. The footer shows the free-call count when supplied by
OpenRouter; `/key show` displays key information and `/cost` shows session token
usage and reported spend.

## Architecture

SlipAgent separates long-lived state from replaceable behavior. The stable
frame watches Python source; reloadable components drive the coding loop and
presentation. It talks directly to OpenRouter using `httpx`, and the terminal
uses `prompt-toolkit`.

```text
src/slipagent/
├── runtime.py        stable entry point, watcher, transactional reloads
├── cli.py            session, commands, one-shot mode, event rendering
├── agent.py          model-step logic and harness system prompt
├── batching.py       bounded read concurrency and ordered observations
├── lifecycle.py      stable cancellation-safe resource ownership
├── prompts.py        named, ordered prompt sections
├── diagnostics.py    exact request snapshots and failed-attempt records
├── jobs.py           session-owned background commands and job controls
├── lsp.py            optional configured language-server navigation
├── context.py        rolling context, summaries, original-step retrieval
├── records.py        separate JSON input records for history and current input
├── budget.py         measured input-token calibration
├── sessions.py       append-only session journals and safe resume
├── repomap.py        bounded, cached repository orientation
├── symbols.py        offline syntax-tree outlines and symbol references
├── checkpoints.py    durable file-edit backups and guarded rewind
├── checks.py         checks after complete edit batches
├── progress.py       unchanged-batch cycles and streamed-repetition detection
├── activity.py       throttled command-output callbacks
├── protocol.py       ordinary replies, optional schemas, call normalization
├── compaction.py     isolated background summaries of completed steps
├── task.py           retained user prompt, active goal, constraints, source references
├── environment.py    project settings and Python interpreter validation
├── openrouter.py     async API client, retries, model/key information
├── capabilities.py   endpoint selection, native tools, JSON/reasoning, limits
├── terminal.py       prompt, transcript, layout, key bindings, styles
├── palette.py        shared terminal colors and ANSI foreground codes
├── transcript.py     file-backed transcript and incremental word wrapping
├── instructions.py   project guidance discovery before model activity
├── config.py         configuration resolution
├── workspace.py      symlink-aware path boundary
├── types.py          shared messages, tool calls, usage, wire types
├── mcp.py            persistent stdio MCP connections
└── tools/
    ├── base.py       tool contract, argument validation, registry
    ├── blocking.py   owned worker threads for synchronous filesystem operations
    ├── files.py      read_file, write_file, edit_file
    ├── editing.py    atomic replacement planning, diagnostics, bounded diffs
    ├── search.py     grep, glob
    ├── grep.py       isolated, killable regex worker using a source snapshot
    ├── navigate.py   list_dir
    ├── shell.py      run_command
    ├── output.py     session log quota, storage, and read_command_output
    ├── git.py        confined status, diff, log, staging, commits
    └── web.py        web_search, fetch_page
```

The agent loop emits events and never prints. The CLI decides how to display
them, which lets terminal behavior change without rewriting the model loop.
Tools return `ToolResult` values, including errors, so the model can correct a
failed operation and continue. Built-in and namespaced MCP tools use the same
registry and tool-call path.

The code keeps these boundaries explicit so the agent can edit the machinery
it uses. Changing a tool implementation need not replace the conversation;
changing a renderer need not reconnect MCP servers or recreate the input
buffer. The same separation lets tests exercise the loop with a stub client
and exercise presentation with a simulated terminal.

## Editing the Running Harness

From an editable checkout, start a session with the harness itself as the
workspace:

```bash
slipagent --workspace . --model nvidia/nemotron-3-ultra-550b-a55b:free
```

You can then ask it, for example:

```text
> Improve the /tools command to group tools by purpose. Read AGENTS.md first, preserve live-reload contracts, and run the relevant checks.
```

The agent edits the same source files that a contributor would edit manually.
Live reload is enabled by default. Save a compatible change under
`src/slipagent/` and it applies automatically, usually within a second while
idle. During work, the current model response and its complete tool batch finish
before a new version is applied. One-shot runs also check between batches.
Use `/reload` to force another attempt or `--no-reload` to disable reloading.

### How a Reload Works

1. **Snapshot the source.** The watcher observes source bytes twice to avoid
   most partially written editor saves. At a checkpoint, the frame loads a
   complete candidate generation into a fresh import namespace.
2. **Validate it together.** Relative component imports share that generation.
   The frame checks required APIs, class/state layouts, and tool-name collisions
   before replacing live behavior. New helper modules load with the candidate.
3. **Apply at a safe boundary.** Existing classes retain their identity while
   their methods change. Built-in tools are rebuilt, the registry is refreshed,
   and the terminal rebuilds its presentation around the existing application.
   The next model step gets the current prompt and tool definitions.
4. **Preserve the session.** Conversation originals and summaries, usage,
   retained user prompts, command logs, selected project interpreter, queued
   prompts, model selection, API client/key, quota, and MCP connections
   remain alive. The terminal keeps its draft, input history, transcript, and
   scroll position. Retired built-in tool clients are closed.
5. **Keep working if a reload fails.** Syntax/import errors, missing APIs,
   incompatible layouts, or a failed terminal rebuild reject the candidate and
   leave the previous generation active, with a visible error. Fix the source
   and save again, or use `/reload` to retry.

This is why the loader uses a whole generation rather than independently
reloading whichever file changed: related components must agree about the
version they are running. Preserving class identities means existing session
objects and exception handlers continue to work. Waiting for a batch boundary
keeps foreground tool batches on one implementation. Session-owned background
jobs keep their process and log owners across compatible reloads. Rejected
reloads preserve running behavior; source edits remain on disk
for you or the agent to correct.

### What Can Change Without a Restart?

| Area | Live changes |
| --- | --- |
| Agent behavior | Model-step logic and the harness system prompt in `agent.py`. |
| Commands and output | CLI command handlers, helpers, and renderer methods. |
| Context | History handling, summarization, and retrieval behavior. |
| API behavior | OpenRouter client methods, using the existing client. |
| Terminal | Layout builders, key bindings, styles, and presentation methods. |
| Tools | Built-in implementations, added tools, and the default registry builder. |
| Guidance discovery | Root and visited nested guidance refresh before working requests. |

The persistent frame and shared contracts require a restart:
`runtime.py`, `config.py`, `types.py`, `workspace.py`, `tools/base.py`,
`mcp.py`, `lifecycle.py`, and the package's root `__init__.py`. Changing startup/lifecycle setup
or a constructor does not reinitialize existing persistent objects. Class
removals and changes to inheritance, slots, or dataclass field layouts also
require a restart. Use `session.extensions` for additional persistent component
state instead of adding fields to a live session object.

Component imports should define behavior without starting background tasks or
acquiring resources, because candidates execute before acceptance. Terminal
callbacks should dispatch through current object methods rather than holding
old bound methods. A reload validates compatibility; it does not prove the new
code is correct. Run tests and strict type checking before treating a change
as finished. Read any local `AGENTS.md` for project guidance and the
[contributor guide](CONTRIBUTING.md) and [internal contracts](docs/architecture.md)
for the implementation workflow, state ownership, and failure behavior.

## Configure

Successful `--model` and `/model` selections are remembered across projects in `preferences.json` under `SLIPAGENT_STATE_DIR` (default `~/.SlipAgent`). Model selection uses CLI flags first, then exported `OPENROUTER_MODEL`, the remembered choice, `.env`, and the built-in Nemotron Ultra default.

The only required setting is your OpenRouter API key. In addition to `.env`,
you can supply it through the process environment:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | **Required.** Your API key. |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Default model slug. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Point at a gateway or mock. |
| `EXA_API_KEY` | — | Optional. Enables the `web_search` tool. |
| `OPENROUTER_REFERER` | SlipAgent GitHub repository URL | Override the app-attribution URL. |
| `OPENROUTER_TITLE` | `SlipAgent` | Override the app-attribution name. |
| `OPENROUTER_USER_AGENT` | `SlipAgent/<version>` | Override the HTTP client identity at startup. |
| `SLIPAGENT_WORKSPACE` | `.` | Default project directory. |
| `SLIPAGENT_STATE_DIR` | Expanded user-home `.SlipAgent` directory | Root for saved sessions, titles, request diagnostics, and command logs. |
| `SLIPAGENT_NO_DOTENV` | — | Set to `1` to ignore `.env` entirely. |

API requests identify SlipAgent using its repository URL, app title, `X-OpenRouter-Categories: cli-agent`, and a versioned `User-Agent: SlipAgent/<version>`. The referer and title overrides above preserve custom app attribution.

`.env` is loaded automatically, searched upward from the current directory and
then from the install directory. Real environment variables win over `.env`;
explicit CLI overrides win over both. Harness-saved credentials use `0600`
permissions. Set `SLIPAGENT_NO_DOTENV=1` in CI to use only the process
environment and CLI configuration. Keep `.env` and credential-bearing MCP
configuration private; both `.env` and `.mcp.json` are ignored by Git.

### Project Python Environment and Log Storage

Project-specific settings live in `.slipagent/project.json`, relative to the
selected workspace. Create that directory and file when explicit settings are
needed; neither is required for other projects. The supported settings are:

```json
{
  "python": ".venv/bin/python",
  "log_quota_bytes": 104857600,
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
`.env` search starts from the launcher directory and installation directory;
it is separate from workspace-specific `.slipagent/project.json` resolution.

`log_quota_bytes` sets a **session-wide disk quota**, default **100 MiB**, read at
session startup. It must be a positive integer. Logs use owner-only files saved
with the session; they survive model switches, `/task new`, component reloads,
and exit. `/reset` starts a new session without deleting saved originals.
With `--no-session`, logs use a temporary directory removed on reset or normal
shutdown. The quota never silently increases and earlier logs are never evicted.

After each complete batch containing successful `write_file` or `edit_file`
calls, Python syntax is checked with the selected project interpreter without
executing/importing the edited code. No selected interpreter means an explicit
skipped check. Set `python_syntax` to false to disable it. Additional checks are
explicit argument arrays, for example:

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

### Saved Sessions

Sessions are saved by default under the fully expanded user-home `.SlipAgent`
directory. Each project gets a directory named `<md5>-<sha1>`, computed from
the same expanded absolute workspace path. Symlink spelling and case are
preserved; `project.json` retains the original path. `/sessions` prints the
absolute storage path and saved IDs. `SLIPAGENT_STATE_DIR` overrides the storage
root. Journals and logs may contain private project text; configuration API
keys are not included in their metadata.

Startup creates no conversation journal until the first task prompt is entered.
Help, settings, session listing, and renaming before that prompt do not add an
empty session. Resuming instead creates the restored conversation's journal.

Use `/resume` to choose a saved session by title and date. During work it queues
until the current response and tool batch finish. Up/Down
moves through the list, scrolling at either edge; Enter restores the selection
and Escape cancels. The current session is marked in the list.
You can also use `/resume <id>` or `/resume latest`, or launch with
`slipagent --resume latest`. Resume reuses the saved journal, restores numbered
originals, summaries, task state, pending input, usage and command logs, and
keeps the currently selected model/settings. Project/system instructions are
refreshed from the running installation. Type `continue` when ready; resume
does not start model requests or replay tools by itself. A started tool with
no saved result is marked as having an unknown outcome; an unstarted tool is
marked as not run. Inspect uncertain effects before retrying.

Use `/fork` to copy the current saved conversation and continue in the new
session. Its history, summaries, task state, queued input, usage, title, and
command logs carry over. Further work changes the copy; the original remains
available through `/resume`. Forking performs no model requests or tool replay.

`/rewind` opens a chooser of file-edit checkpoints. The preview lists affected
files and a bounded restoration diff; Enter confirms restoration and Escape cancels. Selecting a checkpoint
restores files to before that edit batch and undoes later checkpointed batches.
The conversation stays intact. Rewind checks every file's contents and permissions
against the recorded edit before restoring anything; a conflicting user or external
edit blocks restoration. Newly created files are removed and original bytes and
permissions are restored. Shell commands, MCP actions, and external edits are
outside checkpoint coverage. Active background commands stop before restoration.
A queued rewind leaves the agent idle until the user submits another prompt.
Use `/rewind list` to inspect available checkpoints without changing files.

Checkpoints cover built-in `write_file` and `edit_file` calls made by the agent.
Their private, deduplicated backups live beside the session journal in its
`<session-id>-checkpoints` directory. They survive resume and copy into forks;
forks share the project files, so rewind also affects those shared files.
Deleting a session removes its checkpoints. With `--no-session`, checkpoints
last only for the current conversation and are discarded on reset or exit.

`/delete` permanently removes the current session's journal, saved title,
command logs, request diagnostics, and file checkpoints. An interactive confirmation dialog opens:
Enter confirms the selected deletion action; Escape or selecting Cancel leaves
the session intact. Other saved sessions and project files are preserved.
After deletion, the start screen returns; the next task prompt creates a new
session. Deletion requires an interactive terminal. Both `/fork` and `/delete`
require a current saved session and queue until the active batch finishes.

Journals append and flush each message/state change and tool-start marker.
A torn final entry can be recovered with an explicit note; malformed complete
entries reject resume without changing the original. Pending summaries from a
closed process are marked interrupted and their originals remain available.
Saved sessions are retained until you delete their files; the log quota is per
session, not a global journal-retention limit. `--no-session` opts out of durable
storage. Existing processes acquire session persistence on their next launch.

The project settings file accepts `python`, `log_quota_bytes`, `python_syntax`,
`checks`, and the optional `language_servers` map described below. It is limited
to 16 KiB. `.slipagent/` is ignored in this repository; add it to another project's
ignore rules if its settings should remain local.

## Use

One-shot mode runs a task, prints the final answer, and exits:

```bash
slipagent --model nvidia/nemotron-3-ultra-550b-a55b:free "add type hints to src/parser.py and run the tests"
slipagent --model nvidia/nemotron-3-ultra-550b-a55b:free -p "explain what this repo does"
```

Interactive REPL — a multi-step session with conversation state:

```bash
slipagent --model nvidia/nemotron-3-ultra-550b-a55b:free
slipagent --workspace /absolute/path/to/myproject --model nvidia/nemotron-3-ultra-550b-a55b:free
```

In the REPL:

| Command | Effect |
| --- | --- |
| `/help` | Show command help |
| `/menu` | Open the command menu in the input area; Up/Down move, Enter selects, Escape closes |
| `/danger` or `/danger on` | Disable workspace path confinement; queues while working |
| `/danger off` | Restore workspace path confinement; queues while working |
| `/danger status` | Show the current access mode |
| `/overthinking on\|off` | Enable or disable thoughts in the latest 25 steps; enabled by default; queues while working |
| `/tools` | List available tools |
| `/model` | Select a model with Up/Down and Enter; Escape cancels |
| `/model <slug>` | Switch model and remember the choice for future launches |
| `/models [filter]` | Select from the catalog with context and pricing; optionally filter by name |
| `/models free` | Select from zero-cost models |
| `/temperature` | Show the effective temperature and whether the selected model supports changing it |
| `/temperature <value>` | Set the session temperature from 0 to 2; unsupported changes report an error |
| `/key show`, `/key status` | Show the masked current key, spend limit, and reported free quota |
| `/key` | Enter a different key on a terminal (hidden input, verified before use) |
| `/key <sk-or-v1-...>` | Use a specific key for this session |
| `/cost` | Session token usage and spend |
| `/task` | Show the active goal, constraints, facts, pending work, next steps, and source steps |
| `/task new` | Make the next prompt a new task, retaining previous history and command logs |
| `/rename <name>` | Save this conversation's name and set its terminal title to `{name} \| SlipAgent` |
| `/sessions` | List saved sessions for this project's path |
| `/resume` | Choose a saved session with Up/Down and Enter; Escape cancels (interactive terminal; queues while working) |
| `/resume <id>` or `/resume latest` | Restore a saved conversation and its scrolling transcript at the end; queues while working; no tools are replayed |
| `/fork` | Copy the current saved session and continue in the copy; preserve the original; queues while working |
| `/delete` | Confirm permanent deletion of the current session and all its logs; return to the start screen; Escape cancels; queues while working |
| `/rewind` | Choose a file-edit checkpoint and confirm restoration of that batch and all later batches; Escape cancels; queues while working |
| `/rewind list` | List available file-edit checkpoints without restoring files |
| `/requests [attempt]` | List the latest 20 request attempts, or inspect one exact request and its outcome |
| `/mcp` | Show MCP servers, their status, and the tools they contribute |
| `/mcp add <name> <cmd> [args…]` | Connect a server over stdio for this session only |
| `/mcp save <name> <cmd> [args…]` | Same, and persist it to `.mcp.json` |
| `/mcp remove <name>` | Disconnect a server and drop it from the config |
| `/reset` | Start a fresh conversation/task/usage/log index; retain saved sessions and current guidance/model/settings |
| `/init` | Create a project-notes scaffold in `AGENTS.md` if missing and load project guidance into this session |
| `/config-show` | Show the active harness interpreter, model, workspace, danger mode, and Overthinking Mode without exposing credentials |
| `/reload` | Reload components now, or after the current model/tool batch |
| `/generations` | Show the current component reload generation |
| `/stop` | Finish the current model response and its tool batch, then stop before the next request |
| `/exit`, `/quit` | Exit the session; cancel retry waits and let an active response/tool batch finish without another request |

`/key` validates against OpenRouter's `GET /key` before accepting anything, and
never writes to `.env` without an explicit `y` at the prompt.

Conversation titles default to `{full working directory} | SlipAgent`.
`/rename Fix login` changes the saved conversation and terminal title to
`Fix login | SlipAgent`, including while the agent works. `/sessions` shows saved
titles, and `/resume` or `--resume` restores them. In the REPL, restoring also
loads the original conversation into the scrolling area and starts at the end.
Prompts, replies, saved thoughts, tool calls/results, and queued input remain
available by scrolling up; restoring does not execute tools or call the model.
Names are private metadata
beside the saved journal in the project's hashed directory under the expanded
user-home `.SlipAgent` directory (or `SLIPAGENT_STATE_DIR`), outside the checkout.
`/reset` starts a new conversation with the default title. With `--no-session`,
renaming lasts only for the current conversation in the running process.

On an interactive terminal, six fixed rows sit at the bottom of the screen:
a pulsing activity indicator, three rows for wrapping input, an empty line,
and the readout `<cwd>  │  model: <slug>  │  free: <n>`. Another empty row
separates this area from the scrolling transcript. Output and redraws preserve
partially typed input. Input and output wrap at word boundaries to the terminal
width and reflow when you resize it. Wrapping preserves the submitted prompt's
original text. The mouse wheel scrolls the transcript, preserving
your draft and prompt history. Scrolling back pauses automatic scrolling until you
return to the bottom. The CWD is the workspace where tools run; long paths
and model names shorten to fit narrow terminals.

The nearest preceding green line starting with `>` pins to the top as you scroll.
Scrolling up into an older task shows its prompt line; scrolling down restores
the newer one. Submitting another task or queued follow-up releases the header
until the new prompt reaches the top or you scroll again. The pinned prompt uses
one line, shortening at a word boundary with `…` when needed and adjusting on resize. The
complete original remains in scrollback. The context view keeps its own layout.

The interactive transcript keeps original output and indexed display rows in
private temporary files for the session. Streaming formats new text and the
unfinished wrapping tail; ordinary redraws fetch visible rows from a bounded
cache instead of scanning all earlier output. Repaints are limited to 30 per
second. All scrollback remains available, and resizing reflows the original
text. These display files are removed when the terminal closes; they are not
persistent project memory or saved conversation archives.

Before the first model request, SlipAgent reads `CLAUDE.md`, `AGENTS.md`,
`AGENTS.override.md`, `.cursorrules`, `.clinerules`, `.windsurfrules`, and
`.github/copilot-instructions.md` from the workspace root when present. It also
loads rules from `.cursor/rules/**/*.mdc`, `.claude/rules/**/*.md`,
`.github/instructions/**/*.instructions.md`, `.clinerules/`, and
`.windsurf/rules/**/*.md`. Root guidance refreshes before each working request.
File, directory, and search tools discover guidance in the accessed path's
ancestors. Visited scopes stay available across steps and reset; changed and
removed instructions are reflected in the next request. Each scope is labelled;
the model must still obey a rule file's own path/glob restrictions.
An edit governed by unseen or changed instructions is blocked until a request
has presented them to the model. Reading and editing a new scope in the same
batch therefore cannot bypass its rules. Unreadable guidance reports its path.
Shell commands and external tools are not parsed for affected file paths, so the
model must discover applicable instructions before using them to change files.

The startup screen lists the available commands and basic usage. The free-call
count is fetched from OpenRouter at startup and refreshed every 15 minutes,
including while idle or working.

`/menu` replaces the six-row input footer with a heading, a scrolling list of
commands, and `↑/↓ = move | Enter = select | Esc = back` on the bottom line.
The transcript stays visible above it. Escape closes the menu without running
anything, preserving your draft and scroll position. Selecting an item runs its
usual slash command, including the same restrictions while the agent is working.
Commands that need arguments are still entered at the normal prompt.

State-changing commands queue in order while the agent works. They apply after
the current response and complete tool batch finish. Settings changes then resume
the task without resetting its request limit. `/stop` prevents automatic
continuation; reset, resume, and `/task new` replace the task instead of restarting
the previous work. Slash commands are never sent to the model as user messages.

The working indicator shows `Working (esc to interrupt)`. Press Escape outside a chooser to interrupt the active step immediately. It
cancels model requests, active tools, background commands, and pending summaries;
queued settings are discarded and no follow-up step starts automatically.
Completed tool results remain stored, while interrupted or unstarted calls are
recorded explicitly. Queued user messages remain available for later continuation.
An atomic file operation already running finishes safely before cleanup settles;
the terminal stays responsive. Escape inside a chooser only closes that chooser
and leaves agent work running. `/stop` retains its gentler step-boundary behavior.

Live reload scans source files and prepares candidate imports in a worker thread,
so keyboard input remains responsive during preparation. It applies the validated
generation on the main loop at a safe boundary. Restart after installing this
change to enable the threaded reload frame.

`/danger` enables access outside the workspace for built-in file, search,
navigation, and Git tools. `/danger off` restores confinement; `/danger status`
reports the mode. During work, changes queue until the current response and
complete tool batch finish, keeping that batch on one access policy.
While active, the bottom readout ends with
`| Danger Mode` in red. The current mode is supplied to the model on every request.
The working directory and relative-path base stay the same. OS permissions and
the tools' validation, atomic writes, and resource limits still apply.

For an unattended one-shot task, use `slipagent --danger -p "your task"`.
The flag enables the same mode at startup without a confirmation prompt.
The setting lasts for the current process, including `/reset`, `/resume`, and
component reloads; saved sessions do not enable it in a future launch. A new
process starts confined unless `--danger` is supplied.

Press **Ctrl+backslash (`Ctrl+\`)** to toggle the context view. It shows the
latest outgoing working request, with every system, user, assistant, and tool
message in its sent order. System prompts appear in yellow (`#FFFF00`); tool
definitions, the response schema, and other request fields are also shown.
The view updates for each request, including retries, using the selected full
or compressed history actually sent. Scroll with the mouse wheel, arrow keys,
Page Up/Down, or Home/End. The footer, draft, and transcript scroll position
remain available. Isolated background summaries do not replace this view.
Before the first working request, the view indicates that none has
been sent.

Selected historical steps are supplied as a JSON array inside the system prompt,
between large `BEGIN CONVERSATION HISTORY DATA` and `END CONVERSATION HISTORY DATA`
dividers. The separate user message contains the current-step JSON object.
`prompts/system-history.txt` explains how to use the history as reference evidence,
trace recent work, distinguish completed actions from remaining needs, and retrieve
missing outcomes. Historical text does not acquire system-instruction authority
through this placement. No prose headings are inserted into original prompts,
replies, or tool content. This experiment changes request placement, while keeping
history selection, compression, original retrieval, and prompt retention intact.
`record_type` distinguishes `history_step` from `current_step`; `representation`
distinguishes `full`, `compressed`, and `excerpt`. Full records have `user_prompt`
(an array, preserving multiple queued messages), `agent_response`, `tool_calls`,
and `tool_results`. Calls retain their names, argument objects and IDs; results
retain the matching IDs, status and content. Compressed records contain
their summary and metadata. Every bundle includes a `step_id` metadata field
for retrieving the original with `recall_history`; it is separate from message
text. Excerpts carry omission counts and retrieval instructions in
separate fields, without inserting descriptions into original text.
The current step number and task-source IDs also stay in the system prompt.
The current-step object identifies
continuation requests and retains the active prompt when history lacks a full
copy. JSON fields and escaping count toward the context budget. This input format
is separate from the selected model's response contract, which stays consistent
between calls. Native tool calls remain available for responses from models that
support them; the harness does not ask models to echo its history-record format.
Echoes of former harness response labels are removed from ordinary reply text
and from assistant replies reused as context, without rewriting saved originals.
Quoted or fenced examples, user input, tool arguments and file content remain intact.

The prompt stays live while the agent works. All waiting prompts join the next
model request together, after the current tool batch finishes. Each has a queued
label that disappears from the interactive transcript when its request is sent.
If context preparation fails or `/stop` prevents that request, the label stays.
Blank lines separate submitted prompts from the surrounding transcript.
Each model reply starts with a blank line, including replies preceded only by
tool output. Its following tool calls and results stay together until the next
model reply; each command response is another unit.
The model can explain its work alongside batched tool calls.
Shell output streams while the command runs, with updates limited to ten per
second. A bounded display buffer may abbreviate rapid output; the separate
command archive retains the decoded streams within its quota. Terminal control
characters in streamed command output are escaped. Successful streamed output
is not printed a second time at command completion.

Three identical tool batches with unchanged results produce explicit recovery
guidance in the next model request. A fourth stops the run with history intact.
Different calls/results or new user input reset this detection. Intentional
external polling can use `run_command` with `poll=true`; command-log polling and
task bookkeeping do not trigger the guard.

Readable reasoning supplied through a separate provider field streams under a
gray `Thinking:` label. Each response's reasoning is concatenated into one
string and saved with that step's uncompressed originals. Existing history is
not rewritten. **Overthinking Mode** is enabled by default: the latest 25 completed
steps include supplied thoughts as a `reasoning` string in each step's JSON input
record. Steps without thoughts gain no extra field. Use `/overthinking off` to
disable this mode, or `/overthinking on` to enable it. Thoughts count
toward the context budget; omitted steps and budget-limited excerpts omit them.
Opaque provider reasoning metadata is never included. Reasoning remains excluded
from background compression requests and available through
explicit `recall_history` retrieval; streaming thoughts still appear in the terminal.
The harness selects the highest reasoning effort
explicitly listed by the API. When reasoning is supported without listed effort
choices, it enables reasoning; when unsupported, it sends no reasoning setting.
Reply text and tool calls are buffered until call validation completes. Models
that expose no separate reasoning use the working indicator while generating.
The terminal displays the reply text defined by the selected response contract, followed by tool activity. Background summaries
stay out of the transcript. One-shot mode keeps the final answer on stdout,
with streamed thoughts and tool activity on stderr.

### Response Protocol

All built-in model prompts and tool guidance are in the top-level [`prompts/`](prompts/README.md) folder for direct human editing. SlipAgent reads prompt files when building requests and tool definitions, including with live reload disabled. Start with `prompts/system-prompt.txt` for general behavior or `prompts/background-summary-prompt.txt` for compression. The folder's README explains the other files and runtime placeholders. Keep each paragraph on one line and preserve placeholder names. The installer copies this folder; wheels bundle it inside the package. Restart once after installing this implementation change; subsequent prompt text edits take effect when used.

Runtime guidance distinguishes exploratory questions from authorized implementation and keeps work within the user's requested scope. Background summaries prioritize user requests, corrections, and boundaries over agent plans, preserve source attribution, and distinguish completed or rejected actions from unfinished requested work. Historical reasoning and optional agent suggestions do not authorize additional work.

At startup and every `/model <slug>` selection, including reselection of the
current model, SlipAgent fetches fresh model and endpoint properties from
OpenRouter. Subsequent calls use the cached properties until another selection.
There is no list of model-specific exceptions. A failed or incompatible
selection reports an error and keeps the current model and conversation.
Startup model-check failures display a warning and leave the interface available, so users can select another model with `/models` or `/model <slug>`. A model that fails its startup check is not saved as a new preference.

Temperature defaults to **1.0** for endpoints that advertise support. Use
`/temperature` to inspect it and `/temperature 0.1` to change it for subsequent
steps. Unsupported changes display an error and preserve the existing setting;
requests to unsupported endpoints omit temperature and use the provider default.
`--temperature` overrides the default at startup. The session setting carries
across model switches and applies wherever the selected endpoints support it.

SlipAgent uses [OpenRouter's native tool-call format](https://openrouter.ai/docs/guides/features/tool-calling)
when the selected endpoints advertise `tools`. Tool definitions go in the API's
`tools` parameter; calls arrive in `message.tool_calls`, separately from the
response text. The harness preserves each call ID and stores its result with
the matching `call_id` in that step's `tool_results` array. The next request
supplies the completed step as a JSON input record, including each result's
tool name, success/error status, and content. Native calls remain the model's
response format; the JSON records describe completed history. SlipAgent does
not force a `tool_choice` setting.

The selected response/tool format stays consistent between calls, including
retries, until another model selection. Endpoint metadata, including reasoning
support and provider limits, remains cached between selections.
Selection prefers native-capable endpoints;
models without them can use the JSON call format below if they support JSON
output. Their tool definitions enter the system prompt, and their history uses
the same JSON input records, without unsupported native tool parameters.

**Native-tool models without schema support use plain reply text.** They receive
no `response_format` parameter and need no JSON content record. JSON-only models
receive `response_format: {"type":"json_object"}` when schemas are unavailable.
Every step requires a nonempty reply, including steps requesting tools.
Endpoints advertising `structured_outputs`
receive a small [strict schema](https://openrouter.ai/docs/guides/features/structured-outputs):
`response` contains the plain terminal reply. Models using embedded calls also
include `tool_calls`, an ordered array, empty for a final answer. Neither mode
requires compressed fields or a task record in every response.

For example, a schema-capable native-tool response uses this content:

```json
{"response": "I will read the file."}
```

A model without schema support simply writes `I will read the file.`. In both
cases, the accompanying native `message.tool_calls` value can be:

```json
[
  {
    "id": "read-1",
    "type": "function",
    "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
  }
]
```

When native calls are unavailable, every response must be exactly one JSON
object containing `response` and `tool_calls`, including final answers with an
empty call array. Each call contains exactly `id`, `name`, and `arguments`;
arguments must be a JSON-encoded object string. IDs must be nonempty and unique.
No surrounding prose, fences, tagged calls, aliases, or extra fields are accepted.

Each request advertises only its selected response contract. Native responses
use API `message.tool_calls` exclusively and nonempty reply text, or exactly
`{"response":"..."}` when a strict response schema is supplied. Embedded JSON
responses cannot use API calls. The parser never merges call carriers or switches
modes based on the response. Format violations reject the entire batch before
text is displayed or tools execute. Duplicate keys, malformed arguments,
invalid Unicode, non-finite numbers, duplicate IDs, and truncated batches are
also rejected. Plain prose and quoted examples in native reply text never run tools.
Three consecutive invalid responses stop the run; retries count against the
working-request limit. Separate thoughts already streamed remain visible.

Interrupted streams and transient HTTP failures allow up to three transport
retries with bounded backoff. Every working attempt counts against `--max-steps`;
there is no hidden second retry loop multiplying that limit. Transport retries
remain silent until exhausted, and `/stop` interrupts their waits. Each attempt starts fresh:
partial tool arguments and rejected replies never execute or enter accepted history.
Authentication, credit, and permanent request failures remain errors.

`/requests` lists the latest 20 attempts; `/requests 3` displays attempt 3's
final request body, outcome, and bounded response/error excerpt. Request JSON
is deduplicated and gzip-compressed in private files beside the session journal,
without authentication headers. These diagnostics are never supplied as history
or sent for compaction. They have a separate **32 MiB per-session quota**; reaching
it preserves existing files and displays a logging warning. Reset/resume starts
a new diagnostic directory; earlier files remain beside their original journal.
`--no-session` uses temporary files removed on reset/exit. Treat request logs as
private project data, since they contain the same text sent to the model.

Requests set `provider.require_parameters` and restrict routing to compatible
endpoint providers. Context budgets use their actual context and prompt limits.
Unsupported reasoning and temperature settings are omitted; requested output
limits are checked. A model without available native-tool or JSON-capable
endpoints is rejected. Metadata cannot guarantee that every advertised parameter
combination works at every provider.

`/stop` lets the current response and every tool call in that response finish,
keeps their results in the conversation, and prevents another working model request. Background summarization of that
completed step still runs.
Managed background commands also continue until completion, their execution
timeout, or an explicit stop. Their completion cannot restart the agent.
Queued input does not restart a stopped run automatically. Enter a follow-up
or `continue` to resume. Ctrl-C retains its default interrupt behavior;
Ctrl-D, `/exit`, or its alias `/quit` exits the session. Pipes and `TERM=dumb`
use plain output.

### Active-Task Memory

The model retains nothing between API requests unless SlipAgent sends it again.
The harness retains the current user prompt independently of compression and
supplies it verbatim on every working request until new user input replaces it.
`current_prompt_step` identifies the call that originally received that prompt,
so the model can retrieve that call's uncompressed record with `recall_history`.
An existing full copy in the selected context satisfies this requirement; the
harness adds a copy only when needed and counts it against the endpoint budget.
It does not classify new input as a separate idea or an amendment.

Original user prompts and their source step IDs remain available independently of history compression. `/task` displays these sources and the current original prompt. `/task new` starts a new source boundary with the next prompt while preserving history; `/reset` clears prompt retention and history. Older session task-record snapshots are ignored during resume.

The step limit caps working model requests per user run. Background summary
requests are separate and their reported usage contributes to session totals. Reaching it preserves the
conversation for continuation and exits one-shot mode with status 1.

### Rolling Conversation Context

Each accepted model response forms one numbered step: its active user prompt,
response, requested tool calls, and complete tool results. Those four originals
are stored separately, alongside the response's reasoning as one string when
available. Recent steps present the full prompt, response, calls, and results,
with supplied reasoning included for the latest 25 steps in Overthinking Mode.
Older steps use their whole-step summary only when its
estimated token cost is smaller than the original; otherwise they retain their
full prompt, response, tool calls, and results. The comparison includes JSON fields,
escaped content, and message overhead. Ties retain originals.
A step never includes both its full original and its summary in the same request.

Background compression requests have a 20-minute overall timeout instead of the normal client-side HTTP timeout. Timed-out summaries are marked failed; original steps remain available through `recall_history`. Normal agent requests retain their configured timeout. Session reset and shutdown still cancel pending summaries.

Each working request identifies the user request as the overall goal for that run, possibly issued multiple steps ago, and includes its exact wording in request-only system guidance. Steps after tool batches explicitly explain that control returned automatically without a new user instruction: assess results against that goal and return an answer without tools when it is fulfilled. This guidance is assembled for the outgoing request only; it is not saved in conversation history or background summaries. Historical requests provide background rather than independently requesting more work.

The default recent window is **50 model calls**, with a minimum target of **5**
even if `--context-steps` is set lower. There is no separate history token cap:
history uses the selected model's live context allowance after reserving space
for instructions, tools, output, and estimation headroom. At most the **100 newest
older records** accompany the full window. If the request is too large, the oldest
of those records are omitted first, then the oldest calls in the full window.
Even the five-call minimum can shrink to whatever fits the model allowance.

Selection starts fresh on every request. As a large call ages out and its smaller
summary takes its place, previously omitted older history can return within the
100-record limit. Original records remain retrievable even when omitted. Project
instructions and the retained user prompt stay pinned outside this rolling window.

**One separate background request starts after each complete tool batch returns**,
or immediately after a reply without tools. It receives only that step's prompt,
response, calls, and results, plus instructions for summarization. It receives no
conversation thread, previous summaries, project instructions, task record, or reasoning.
Tool arguments and structured results remain dictionaries or lists internally;
the complete summary input is serialized once for the model. Complete history
retrievals return objects or arrays, while partial character pages return text
fragments for reconstruction with `next_offset`.
The summarizer uses the model and endpoint profile selected for that step and
returns a concise plain-text summary, limited to 6,000 characters. Short steps
should get short summaries. The working model continues without waiting.

Unseen tool results reach the next working request even if their summary has
already finished. An oversized new batch uses JSON excerpt records with separate
omission metadata and retrieval instructions as a last resort. The latest request
stays verbatim.
Neither excerpts nor summaries overwrite any captured originals.

Pending or failed summaries are labelled in context with a retrieval reference.
Compaction failures show a warning and preserve the originals; they do not
regenerate or repeat the agent's work. Normal compaction is silent. `/reset`
cancels outstanding jobs and prevents late results from entering the new
conversation. Session shutdown cancels remaining jobs; one-shot mode waits for
its summaries before shutting down. These additional requests use the same API
key and contribute to quota use, session tokens, and reported cost.

The model's `recall_history` tool searches captured originals or reads a step
by ID, with character pagination for large records. List/search pages include
`total_matches`, the number of matching steps across all pages (zero when none
match). Follow `next_offset` until null; offsets and limits remain character
counts for both listings and individual records. Select `section="prompt"`,
`"response"`, `"reasoning"`, `"tool_calls"`, or `"tool_results"` to retrieve one original part;
`sections=["prompt", "tool_results"]` selects several together. Omit the selector
for the full step. `call_id` selects one observation, or a call when paired with
`section="tool_calls"`. `section="user"` returns the original new user messages
associated with that step, for task-source recovery. Shell and Git observations show the
beginning and end of long output, up to 30,000 characters per stream. The full
decoded streams are retained separately within the session log quota and are
retrieved with `read_command_output`. Timeout results preserve partial output and
state that effects may be partial. Both history versions are journaled for
resume by default; `--no-session` keeps them only in memory until reset/exit.
The terminal transcript continues to show the full conversation.

Token counts use a UTF-8 size estimate rather than a model-specific tokenizer.
Reported prompt usage calibrates that estimate using the average measured-to-estimated ratio from the latest five working requests, plus a 10% safety margin. The multiplier can rise or fall as measurements change; the configured context and output limits remain unchanged. Calibration resets when the model or endpoint
profile changes; background-summary usage does not calibrate working context.
Explicit provider context overflow triggers up to two retries with smaller
history budgets, counted against the step limit. An identical, unshrinkable
request is not resent. Originals remain available throughout.
The reduced budget persists for that model/profile so the next call does not
immediately restore the rejected context size.
Startup and model selection fetch live catalog and endpoint capabilities,
require a valid context length, and check that the current instructions, tool
definitions, and reserved output fit before accepting the model. The harness
uses the selected endpoints' conservative limits to reduce the budget for
smaller models. The request reserves output tokens and a 15% estimation margin;
the remaining space accommodates instructions, tools, and selected history.
Older records are omitted before the recent full window is reduced. Both stored
versions remain intact, and a summary is used only when it saves tokens.
It reports a limit if mandatory input and instructions cannot fit even after
older history is omitted and oversized observations use excerpts.

A bounded repository map supplies file paths and declaration outlines for Python,
JavaScript, TypeScript/TSX, PHP, Go, Rust, C/C++, Java, C#, Ruby, and other supported
grammars. Python uses its built-in AST; other languages use bundled Tree-sitter
grammars with no parser downloads during a run. Ranking combines words in recent
user input with a graph of symbol references and Python imports. Symbol matches
are orientation hints, not compiler-verified bindings. Cached outlines refresh
when file metadata changes, and syntax errors label the outline as partial.
The map excludes dependencies, hidden directories, symlinks, and Git-ignored
files without changing file-tool visibility. Limits on scanning, source size,
and output keep it partial by design: use `glob`, `grep`, and `read_file` for
details. It counts against the request budget and is dropped if needed to fit
the current input. The map does not authorize edits or replace source reads.

Useful flags:

| Flag | Effect |
| --- | --- |
| `-h, --help` | Show all command-line options and examples. |
| `-p, --prompt` | Supply a one-shot task instead of positional task text. |
| `-m, --model` | Select and remember a model slug for future launches. |
| `-w, --workspace` | Project root and default path boundary. |
| `--danger` | Disable workspace path confinement at startup, including unattended one-shot tasks; no confirmation prompt. |
| `--max-steps` | Cap working model requests per run, including response retries (default 200); background summaries are separate. |
| `--context-steps` | Target recent model calls supplied in full (default 50, minimum 5 when context permits). |
| `--python` | Explicit project interpreter, overriding project-file settings and discovery. |
| `--temperature`, `--max-tokens` | Sampling controls. |
| `-v, --verbose` | Show full tool output and per-step token usage. |
| `--list-models` | Print OpenRouter's model catalog. |
| `--api-key` | Override the API key for this process; an environment variable avoids putting it in shell history. |
| `--base-url` | Override the API base URL, for example for a local test gateway. |
| `--no-color` | Disable ANSI color output. |
| `--no-mcp` | Skip MCP servers configured in `.mcp.json`. |
| `--no-reload` | Disable automatic source reloads for this process. |
| `--resume [id]` | Restore a saved project session; omitted ID means latest. |
| `--no-session` | Disable durable conversation and command-log storage. |
| `--mcp` | Connect MCP servers even in one-shot mode. |

In one-shot mode the final answer goes to **stdout** and progress goes to
**stderr**, so the answer can be piped to another program. Model output is
rendered Markdown in the terminal and readable plain text when redirected.
Use `--markdown` to retain Markdown source in stdout or start the REPL in source
view. LaTeX is not rendered; equations use plain text.

Assistant replies support CommonMark, tables, strikethrough, and task checkboxes.
Headings share one bright, bold style without added spacing; body and bold text
share a foreground color. Language-tagged code uses Pygments lexer tokens for
syntax colors, with a theme-specific background and no language label. Wrapped
code has a continuation marker that is excluded from copied text. Narrow tables
become header/value records. Links display `label (destination)`.

Choose `--theme dark|light|monochrome|ironbow`, or `/theme NAME` in the REPL.
`/markdown [source|rendered]` toggles the view. `/copy` copies the latest reply,
automatically choosing exact code when it contains one code block; source view
copies Markdown. `/copy text`, `/copy markdown`, and `/copy code [number]`
choose explicitly. F2 toggles the view, F3 copies, and dragging over text
copies the selection without wrap decorations. Clipboard access uses an
installed local utility or the terminal's OSC 52 support.

The renderer updates incoming Markdown chunks as they arrive. Structured model
responses remain buffered until validated; rendering does not bypass that
validation boundary.

## MCP

SlipAgent is an [MCP](https://modelcontextprotocol.io) client, so tools from
external servers join the same registry as the built-ins. Add one for the
session, or save it to `.mcp.json` so every later run picks it up:

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
`search` without colliding, and the model's argument validation is applied to the
JSON Schema the server advertises. The REPL connects everything in `.mcp.json`
on startup; a server that fails to start is reported and skipped rather than
blocking the session.

Only the stdio transport and MCP tools are supported; resources and prompts
are not surfaced. Existing MCP connections survive component reloads.
Connected server instructions enter model context with their server scope;
structured results and error flags are retained. A server starts in the selected
workspace unless `cwd` is configured; relative values resolve against that workspace.
MCP processes inherit the harness environment plus configured `env` overrides.
Project Python selection does not rewrite their command or PATH; use an explicit
interpreter in the server's `command` when required.

## Tools

The model gets these built-in tools:

| Tool | Purpose |
| --- | --- |
| `read_file` | Read a file with numbered lines, with offset/limit paging. |
| `write_file` | Create or overwrite a file, making parent directories. |
| `edit_file` | Replace one exact string or an `edits` array against one original file. All matches must validate before an atomic write; returns a bounded diff and helpful match diagnostics. |
| `grep` | Regex search across file contents, with an optional glob filter. |
| `glob` | Find files and directories by name pattern. |
| `list_dir` | List one directory, directories first. |
| `run_command` | Run a shell command and capture output; `background=true` returns a managed job ID. |
| `command_jobs` | List, inspect, wait for, or stop managed background commands. |
| `read_command_output` | Page retained stdout/stderr or read its tail by log ID; list logs by step/call ID. |
| `git_status` | Show the branch and short repository status. |
| `git_diff` | Show unstaged changes, or staged changes with `staged=true`; optionally select literal `paths`. |
| `git_log` | Show recent commit hashes and subjects (`limit=10`, maximum 100). |
| `git_add` | Stage explicit workspace-relative `paths`, including deletions; `["."]` stages all files. |
| `git_commit` | Commit already staged changes with a nonempty `message`. |
| `web_search` | Search the web via Exa. Returns titles, URLs, and snippets. |
| `fetch_page` | Fetch a URL and return readable text (scripts stripped). |
| `recall_history` | Search history or retrieve any original parts of a numbered step, with pagination. |
| `navigate_code` (configured projects only) | Definitions, references, implementations, and hover via a local language server. |

### Background Commands

Pass `{"command":"your test command","background":true,"timeout":120}` to
`run_command`. It returns a `job_id` and `log_id`. Use `command_jobs` with
`{"action":"wait","job_id":"ID_FROM_RESULT","timeout":10}` to wait briefly,
`action="status"` to inspect, or `action="stop"` to kill the process group.
`action="list"` lists jobs in pages of 50; follow `next_offset`.
Use `read_command_output` with the log ID for live pages or tails.

Waits are capped at 30 seconds and never cancel the command. Execution retains
the normal 120-second default and 600-second maximum timeout. At most four jobs
run concurrently; the 128 most recent job handles remain addressable. Older
completed jobs' logs and original tool records remain retrievable. Completion
notices are delivered once while idle or at an agent boundary and do not spend a model call.
They do not by themselves establish that tests passed: inspect exit status and output.

Compatible reloads preserve jobs and logs. `/reset`, `/resume`, `/fork`, `/delete`, and shutdown stop
and drain active jobs before replacing or closing their output archive. Saved
sessions retain output, but do not reattach to or relaunch prior processes.
After a crash, inspect recorded output and running processes before rerunning work.

### Optional Language Servers

Configure an already installed **stdio** language server in
`.slipagent/project.json`, for example:

```json
{
  "language_servers": {
    "python": {
      "command": ["/absolute/path/to/pyright-langserver", "--stdio"],
      "extensions": [".py", ".pyi"],
      "language_id": "python",
      "initialization_options": {},
      "settings": {}
    }
  }
}
```

The command is an argument array, never a shell string. Replace the executable
with the installed server's actual path; the harness installs nothing. Enable
the tool at startup or run `/reload` after first adding configuration (restart
when live reload is disabled). Servers start lazily on the first navigation call.
Changes to an existing server's command/settings retire its previous process;
the next query starts the replacement. Removing configuration retires its process
at the next working step; `/reload` also refreshes tool availability.

`navigate_code` takes `operation` (`definition`, `references`, `implementation`,
or `hover`), `path`, `line`, and `column`; supply `server` if several match the
file extension. Input lines/columns are **1-based**, with input columns counting
Unicode characters. Locations explicitly return `column_utf16`, a 1-based UTF-16
column, as used by the protocol. ASCII positions are identical; a non-BMP
character occupies two UTF-16 units. Locations page at 100 items using `offset`;
hover text is capped at 16,000 characters. Source files must be UTF-8 and at
most 2 MB. Queries time out after 30 seconds, with bounded startup/shutdown and
pipe writes. Unsupported operations and missing executables return tool errors.

Each query supplies current file contents and reports workspace file changes
since the prior query; generated/dependency trees follow the normal search
exclusions. Results outside the workspace are counted and omitted. Server
processes have the same OS permissions as shell/MCP processes. Navigation does
not apply edits requested by a server. Ordinary search/read tools remain useful
when the language server cannot resolve a symbol.

### Retrieving Command Output

Every shell/Git subprocess in the default registry gets a unique log ID. The tool
observation reports that ID and concrete retrieval arguments. A tool that runs
several Git subprocesses can produce several logs; list with `step_id` and
`call_id` to find them all. Call IDs are unique within a step, not the whole session.

```json
{"log_id": "ID_FROM_RESULT", "stream": "stdout", "offset": 0, "limit": 8000}
```

Pass that object to `read_command_output`, then use the returned `next_offset`
for the next page. Stream offsets count **UTF-8 bytes**; limits count **characters**
and are capped at 16,000. Use `stream="stderr"` for errors or `tail=true` for the
last `limit` characters. Invalid process bytes are decoded with replacement before
storage; this is a text log, not a binary artifact archive.

Omit `log_id` to list logs. For listings, `offset` is a record index and `limit`
is a record count, capped at 50; commands are shown as bounded previews. Each log
includes its step/call IDs, exit status, timeout status, retained byte counts,
lost byte counts, and any retention error. Log IDs never become user-supplied paths.

When disk quota or storage fails, output pipes continue draining and bounded
previews still reach the model. The affected log stream retains its existing
prefix and reports all later bytes as lost; it never joins disconnected pieces
into an apparently complete log. Lost bytes cannot be recalled. Redirect output
to a project file when the task requires an artifact retained beyond this session.
MCP server stderr is connection diagnostics and does not use this command archive.

`fetch_page` needs no API key and accepts only public HTTP(S) destinations,
including every redirect. `web_search` needs `EXA_API_KEY`; without one it
returns a clear setup error rather than failing silently.

Git tools accept an optional workspace-relative `repo` directory (default `.`)
and `timeout` in seconds (default 120, maximum 600). Git must be installed.
By default, the worktree and Git metadata must stay inside the workspace; parent
repositories, external worktree metadata, and escaping symlinks are rejected.
Danger mode allows those external paths. Bare repositories are not supported.
File paths are literal, so wildcards and Git pathspec magic cannot broaden a selection.
Inherited Git environment redirects are ignored. Hooks, signing, external diff
helpers, filesystem monitoring helpers, and automatic maintenance are disabled.
Implicit fetching of missing objects is disabled, so these tools stay local.
Staging rejects active clean/process filters instead of running external programs
or bypassing their transformations. Git failures, including missing identity,
index locks, and nothing to commit, are returned to the model as tool errors.

Two design choices are worth calling out.

**`edit_file` refuses to guess.** `old_string` must appear in the file and
appear exactly once, unless `replace_all` is set. This forces the model to have
actually read the file rather than reconstructing it from memory, and steps a
mis-targeted edit into a recoverable error message instead of silent
corruption.

**Filesystem tool paths are confined to the workspace by default.** Paths resolve through a symlink-aware check that
rejects anything landing outside the workspace root, so `../../.ssh/config`
is rejected by file, navigation, and search tools unless `/danger` or `--danger`
is active. `run_command` and MCP servers
run with the harness process's permissions and can access paths outside the
workspace. The workspace check is not an operating-system sandbox.

## Development

```bash
.venv/bin/python -m pytest          # full suite, up to four parallel workers
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
a local fake API and shows the REPL's spacing,
mid-step input, and the prompt line. Run it in a terminal: the prompt draws
itself, so a captured pipe shows the lines without the redraws.
These three demos use disposable workspaces and need no real credentials.
`scripts/repl_drive.py` is a separate **live** manual driver: it uses the current
workspace/configuration and can make real API requests and tool changes.

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
