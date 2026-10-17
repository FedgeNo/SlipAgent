# SlipAgent

**A terminal coding agent designed for free models with large context windows on
[OpenRouter](https://openrouter.ai), [NVIDIA](https://build.nvidia.com), [Kilo](https://kilo.ai/gateway), [Mistral](https://console.mistral.ai), and [Google Gemini](https://ai.google.dev), with a harness you can modify while it runs.**

SlipAgent gives a model tools to read, search, edit, test, and commit code in a
project directory. It keeps asking the model what to do next until the task is
finished, you stop it, or the step limit is reached.

The installed harness is an editable checkout. Make it the project workspace,
or enable Danger Mode for explicitly requested self-editing, and the agent can
improve its own tools, prompts, commands, context handling, and terminal
interface. A contributor can edit those same files in their editor. The running process loads compatible changes without losing the
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
- **Human inspection and editing:** clickable file and directory references, a
  built-in editor, colored diffs, saved tool-call records, and a Touched/Calls
  sidebar. Saved user edits return as bounded notifications with retrievable diffs.
- **Mouse and keyboard navigation:** clickable menus, wheel scrolling, a
  selection context menu, and keyboard traversal with Tab and Shift+Tab.
- **Focused tool definitions:** seven core tools are fully described initially;
  the model loads other definitions as needed from the available-tool catalog.
- **A responsive terminal:** wrapping output, a fixed prompt/status area,
  queued follow-ups, optional next-step links, and `/stop` at step boundaries.

[Quick Start](#quick-start) · [Free Models](#using-free-models) ·
[Installation and Data](#installation-and-data-locations) ·
[Terminal Guide](#terminal-layout-and-navigation) ·
[Architecture](#architecture) · [Self-Editing](#editing-the-running-harness) ·
[Usage](#use) · [Tools](#tools) · [Development](#development) ·
[Automation](#headless-automation-and-evaluations) ·
[Contributor Guide](CONTRIBUTING.md) · [Internal Contracts](docs/architecture.md) ·
[Harness Editing Guide](docs/harness-development.md) · [Editable Prompts](prompts/README.md)

## Quick Start

Requires **Python 3.11+**, an API key for your selected provider, and Git for the Git tools.

Terminology: a message is one user, assistant, system, or tool message. A step is one model response plus any requested tool batch and its results. A run processes a user request until a reply without tools, a successful answer-only batch, a stop, or a limit. A session is the conversation containing runs, saved by default. Model context carries one JSON object per history step.
Clone or download this repository, then run the installer from its root:

```bash
python3 install.py
```

On Windows, use `py -3 install.py`. Installations are per-user and require no administrator privileges. The installer copies the application source, creates a private virtual environment, installs SlipAgent in editable mode, and places its launcher on your user PATH. It uses `uv` if available, otherwise Python's `venv` and pip. Add `--dev` to include the project's testing and development dependencies.

The same script also updates an existing installation. Download the desired version and run `python3 install.py` from that download (Windows: `py -3 install.py`). It replaces installed application files and refreshes dependencies and the launcher; it does not download updates or run Git commands. Running inside the installed checkout only refreshes dependencies and the launcher. Unchanged installations update without warnings or confirmation. If application files have been modified, added, or deleted, the installer lists every affected path, warns that updating may lose local edits, and asks you to back up changes before confirming with `y` or `yes`. Enter, EOF, or Ctrl+C cancels. A missing or invalid baseline also requires confirmation and lists files to inspect. Changes from separate checkouts are not merged. First installations and `--dry-run` need no confirmation.

The installer compares SHA-256 hashes against `installed-file-hashes.json` in the installation directory, falling back to the checkout's `file-hashes.json` when available. Successful installations or updates from another checkout record the installed file hashes. Running inside the installed checkout retains the existing baseline rather than treating local edits as an unmodified release. Git internals, secrets, virtual environments, and caches are excluded from comparisons; the hash manifest itself is excluded to avoid self-reference.

| OS | Installation directory | PATH launcher |
| --- | --- | --- |
| Linux | `~/.local/lib/slipagent` | `~/.local/bin/slipagent` |
| macOS | `~/Library/Application Support/SlipAgent` | `~/.local/bin/slipagent` |
| Windows | `%LOCALAPPDATA%\Programs\SlipAgent` | Installation directory + `\bin\slipagent.exe` |

Linux and macOS use a symlink. Windows also uses a symlink when permitted, with an executable launcher fallback that does not require elevation. Persistent PATH entries are added to bash, zsh, sh, ksh, or fish configuration on Linux/macOS, or to the Windows user PATH. Open a new terminal when PATH changes; other shells receive instructions to add the directory manually. Use `--dry-run` to inspect the destinations, or `--install-dir`, `--bin-dir`, and `--python` to override them.

The installer replaces recognized SlipAgent launchers and saves a backup; it refuses unrelated commands or nonempty unmanaged installation directories. The permanent working checkout lives at `source` inside the installation directory, with its own `.venv`. Initial copies include application files, uncommitted edits, and Git metadata, but exclude existing virtual environments, caches, and secrets. Updates from another checkout replace the application directories and supplied root files while preserving installed Git metadata, the virtual environment, and configuration. Removed application directory contents are removed during updates too. Existing Git history is retained, so new downloaded files appear as working-tree changes. Failed updates restore the previous application and dependencies as described below. Older release directories are retained during migration. On Linux/macOS, `current` points to `source`; `installation.json` records the source, interpreter, and launcher on all platforms. Edit the permanent checkout for live changes and normal Git development.

An existing checkout `.env` is copied to the private user data directory on the first install, with owner-only file permissions on Linux/macOS. Subsequent installs preserve that configuration. SlipAgent reads only its own user-data `.env`; old installation `.env` files remain a compatibility fallback. Project `.env` files belong to their projects and are never read. The installer does not move or delete the original checkout or project sessions.

### Installation and Data Locations

Application files and private user data have different owners and lifetimes:

| Location | Contents |
| --- | --- |
| Installation directory + `source/` | Editable source, prompts, documentation, Git metadata when supplied, and the harness `.venv` |
| Installation directory | Managed-install marker, installation metadata, launcher backups, installed hash baseline, and the POSIX `current` link |
| Platform data directory | Saved sessions, command output, inspection artifacts, preferences, global `mcp.json`, and saved `.env` credentials |
| Project workspace | Your application's files and optional `.slipagent/project.json` |

| OS | Default Private Data Directory |
| --- | --- |
| Linux | `$XDG_DATA_HOME/slipagent`, or `~/.local/share/slipagent` |
| macOS | `~/Library/Application Support/SlipAgent/data` |
| Windows | `%LOCALAPPDATA%\SlipAgent` |

`SLIPAGENT_STATE_DIR` overrides the data location. Changing it does not move
existing data or relocate the program. On Linux, the default editable checkout
is `~/.local/lib/slipagent/source`; the executable on PATH points into that
checkout's virtual environment. Installing from a downloaded archive does not
create Git history if the download has no `.git` directory.

Managed updates back up source, the virtual environment, activation metadata and
the launcher before changing them. A failed copy, dependency installation,
verification or activation restores those components. Backup failures stop the
update before replacement. If rollback itself fails, the installer retains the
backup directory and reports its location. Private user configuration is preserved.

### Running From a Development Checkout

For development directly in this checkout instead of a managed installation, use the existing project environment or create one if none exists:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Create an ignored `.env` file containing `OPENROUTER_API_KEY=your-key`. Run
`slipagent` to select the highest-ranked eligible free model automatically, or
select NVIDIA Nemotron 3 Ultra (free) explicitly:

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

**Optimal mode is enabled by default.** It selects from known, ranked free
coding models with configured API keys. General-purpose routers such
as `openrouter/free` and `kilo-auto/free` remain manual choices because their
underlying model changes.
A large context window gives the agent room for project instructions, source,
tool results, and a long-running conversation as it develops a project or its
own harness.

| Example Free Model | OpenRouter Model ID |
| --- | --- |
| [NVIDIA Nemotron 3 Ultra (free)](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free) | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| [NVIDIA Nemotron 3 Super (free)](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free) | `nvidia/nemotron-3-super-120b-a12b:free` |

### Optimal Mode

Ordinary startup selects the highest-ranked eligible free model. Explicit
`--model` or `--provider` launch options start in manual mode. `/optimal on`
enables automatic selection again. Model metadata stores one **estimated text-coding
score from 0 to 100; higher is better**. Published benchmark results are calibrated
onto a common scale using models tested on multiple benchmarks, including paid
reference models. Visual results are excluded. These synthetic scores guide
selection; they are not measured pass percentages or guarantees of results.
The same model/version shares its score across providers, and model lists display
it. The complete table is in [`src/slipagent/modelscores.py`](src/slipagent/modelscores.py).
See [scoring and updates](docs/model-scores.md) for sources, validation limits, and
the offline tool that scores a new model using saved calibration without
recalculating existing entries.
Provider-specific IDs, including Step 5's free promotional route and NVIDIA's
longer Nemotron IDs, share the underlying model's score. A score does not make a
paid endpoint free: automatic selection still checks that endpoint's pricing.

Working requests keep their ordinary retries for transient failures. An
unusable model, permanent request failure, or exhausted recovery moves coding
to the next highest-scoring eligible model. The conversation, original task,
failed request records, and completed tool results stay intact. Tool/test failures remain
feedback for the current model. Each automatic selection is announced.
Connection failures retain the current model after its ordinary retries are
exhausted. A broken network connection does not exclude a model or provider;
retry when connectivity recovers. HTTP errors such as 429 and 503 still count
as provider failures after retry exhaustion.
A working request that produces no output within two minutes is cancelled and
moves directly to the next eligible model. This deadline runs from request start,
including connection and upload time but not upload pacing waits; keepalives and empty stream markers do not
extend it. Once reasoning, answer text, or tool-call content arrives, streaming
continues under the ordinary transport timeout. In manual mode, the deadline
reports an error and retains the selected model. Background compression retains
its separate 20-minute timeout.

The replacement remains selected until another failure, manual selection, or
reset. Both `/optimal on` and `/optimal reset` clear failed coding/compression
routes and retry from the highest score. `/optimal next` excludes the current
model until reset and selects the next ranked one; with optimal off, it starts
optimal like `/optimal on`. `/optimal off` keeps the current model
under manual control; `/optimal` reports the state. Successful manual model or
provider selection through `/model`, `/models`, or `/provider` also disables
optimal mode. During active work, `/optimal` changes apply at the next safe step
boundary in input order, without stopping the task. Opening or cancelling a
picker leaves the mode unchanged. Optimal
mode is session-only; ordinary saved model preferences remain unchanged by
automatic selection and remain available when launching with `--provider`.

When every eligible model fails, coding stops with an explanation. Another
prompt cannot reuse an excluded model: use `/optimal reset`, update credentials,
or select a model manually. Failure exclusions are session-local and clear on
a new conversation; a credential change clears that provider's exclusions.
An authentication failure excludes that provider for both coding and compression;
model-specific failures are tracked separately for those two uses.

New free catalog models without a score remain manually selectable. SlipAgent
announces their release and asks you to update SlipAgent for updated scores.
Scores are refreshed with software updates, without model calls or automatic
benchmark collection at runtime. The October 16 review scores 68 models, including
Reflection Beam, Xing 4.0, Zamba2, and the Llama 3.2 11B/90B Vision models.
Zamba2 uses a calibrated MT-Bench coding rating; the Llamas use their preserved
text-model results. SEA-LION v1 7B Instruct remains unscored because no comparable
code-correctness result was found. Known evidence gaps do not trigger new-release notices.
With `GEMINI_BILLING_TIER=paid`, Google's billed Gemini IDs are excluded from automatic
free selection; the always-free Gemma 4 models remain eligible.

Provider context limits and free availability can change. Check `/models` for
live catalog information; selection also fetches endpoint properties and uses
their more conservative limits. A model's advertised maximum does not guarantee
that every free endpoint offers the same window.
Automatic routing refreshes provider catalogs after five minutes. Removed models
keep their stored scores but are excluded from coding and compression until
they return. A failed catalog connection retains that provider's last known
inventory; a successful empty catalog removes its models from selection.

Model listings hide a curated blacklist of dedicated non-coding models, maintained
with reasons in `src/slipagent/model_catalog.py`. New model IDs remain visible by
default; missing tool-call or structured-output metadata does not exclude them.
This applies to `/models`, the `/model` picker, and `--list-models`.
Searches match every whitespace-separated word across the model ID, name, and
provider, ignoring case and word order (for example, `/models ultra nemotron`).

Use `/models free` to select from zero-cost catalog entries. Use Up/Down and
Enter to select, or Escape to cancel. `/models <text>` searches model IDs, names, and providers across active provider catalogs. Results include the provider; a provider is active only when its API key is configured. `/model` opens the full catalog; `/model <provider>::<model>` selects a specific provider and model. A bare model ID also works when it is unambiguous. Successful selections are remembered for future launches. Paid OpenRouter models also work.
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

## API Providers

OpenRouter remains the default. For NVIDIA, add `NVIDIA_API_KEY=your-key` to the ignored `.env`, then run:

```bash
slipagent --provider nvidia
```

NVIDIA defaults to `nvidia/nemotron-3-ultra-550b-a55b`, without OpenRouter's `:free` suffix. No local GPU is needed. Inside an existing session:

```text
/models nemotron
/model nvidia::nvidia/nemotron-3-ultra-550b-a55b
```

Search reads only providers with configured API keys, including keys supplied for the current session. Providers without keys are neither queried nor listed. Duplicate model IDs remain separate choices. A provider's catalog failure is reported while results from the others remain available. Selection checks credentials are present, loads capabilities, and validates the context budget before replacing the current client. Same-provider changes let existing summaries continue; provider changes cancel and drain pending summaries before closing their client. Originals remain available. Authentication is established by an authenticated API request, not by catalog access.

For Kilo Gateway, add `KILO_API_KEY=your-key` to the ignored `.env`, then run
`slipagent --provider kilo`. Its default is `kilo-auto/free`, Kilo's automatic
free-model selector. `/models free` searches the active catalogs;
`/model kilo::<model-id>` selects an exact Kilo model. SlipAgent fetches Kilo's
`/models` endpoint for model IDs, prices, context/output limits, tool support,
and reasoning options. Advertised reasoning efforts use the highest supported
level; otherwise reasoning is enabled when advertised. Catalog entries and free
availability can change. Missing models are rejected without substituting a paid
model. Kilo does not supply account quotas through this integration.

For Mistral, add `MISTRAL_API_KEY=your-key` to your local `.env`, then run
`slipagent --provider mistral`. The default is `mistral-small-latest`;
`/models mistral` lists available chat models and
`/model mistral::<model-id>` selects one. Mistral's authenticated catalog supplies
context limits, aliases, and native tool support. Leanstral 1.5.1 is labelled free;
other prices remain unknown because the catalog does not identify your account's free allowance. Free-mode access
and quotas are controlled by Mistral. Its ranked free models participate in
background compression. `/provider` switches only to defaults advertised
as free, so select Mistral models explicitly with `/model`, for example
`/model mistral::labs-leanstral-1-5-1` when available in your account's catalog.

For Google Gemini, add `GEMINI_API_KEY=your-key` to your local `.env`, then run
`slipagent --provider google`. The default is `gemini-3.8-flash`.
`/models google` lists compatible text models with Google's input and output
limits. Tool calls use SlipAgent's embedded JSON protocol so compressed history
does not depend on Google's opaque native function-call signatures.
Google's free Gemini tier and always-free Gemma 4 models participate in
background compression when eligible.
SlipAgent requests thought summaries from Google's thinking models and displays
them in the reasoning stream, using the model's default thinking level.
Leading `<thought>…</thought>` blocks are separated from response content before
tool-call JSON parsing, including when their tags span streaming chunks.
Google free-tier model uploads are paced at 200,000 estimated input tokens per
minute, including retried uploads. Paid-only models upload at full speed.
Set `GEMINI_BILLING_TIER=paid` for a billed Google account to upload billed models
at full speed; always-free Gemma requests retain pacing. The default is `free`, since the catalog does not report account
billing status. Concurrent paced uploads through the same client share the
allowance. Responses continue streaming at the provider's speed. Paced models
limit each prompt to the 200,000-token allowance, and pacing waits do not count
toward the first-output deadline.

Model lists put free models first, ordered by descending estimated coding score,
with unscored entries afterward. Ties use provider and model ID for stable ordering.
The interactive picker uses a green background
and a matching `= free` legend on the right. `/models free` includes models with
a free tier; these retain a `free tier` label because account billing and quotas
still apply. Unknown prices are shown as `-`.
Search matches every word against the displayed model row; for example,
`/models google free` shows Google's free-tier models.

NVIDIA's hosted trial catalog is free, including models developed by other
companies. Its catalog omits tool support and context limits, so
[`nvidiaprofiles.py`](src/slipagent/nvidiaprofiles.py) supplies reviewed endpoint
profiles for Step 5, DeepSeek V4.1 Flash, GLM 5.3/Flash, Kimi K2.6, current
Nemotron models, Muse Glimmer, Gemma 4, DiffusionGemma, Laguna XS 2.1, and
GPT-OSS 20B. Native tools are used where the endpoint documents them; other
profiles use embedded tool calls. Gemma endpoints that accept only user and
assistant roles receive the system instructions at the start of the first user
message, streamed from the same source files. Remaining NVIDIA entries require
an explicit capability profile before use; automatic selection skips those
whose capabilities cannot be established. Free availability and quotas remain
provider-controlled.

Python callers select the same modules programmatically:

```python
from slipagent import Config, Message, create_client

config = Config.from_env(provider="nvidia")
async with create_client(config.provider, api_key=config.api_key,
                         reasoning_effort="medium", reasoning_budget=2048) as client:
    await client.model_capabilities(config.model)
    reply = await client.chat(model=config.model,
                              messages=[Message.user("Explain this algorithm.")],
                              max_tokens=4096)
```

Pass that client to `Agent` to use the normal coding loop. `NvidiaClient(profiles={model_id: RequestProfile(...)})` supports additional explicitly verified models. Its Ultra reasoning options are `none`, `medium`, and `high` (default); `reasoning_budget` accepts -1 or 0–32,768 tokens. See [the provider contract](docs/architecture.md#api-provider-modules) for extending the registry. Restart existing SlipAgent processes after changing the shared provider/state contracts.

## Architecture

SlipAgent separates long-lived state from replaceable behavior. The stable
frame watches Python source; reloadable components drive the coding loop and
presentation. Provider modules share an OpenAI-compatible `httpx` transport, and the terminal
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
├── records.py        structured history and current-input records
├── data_text.py      JSON input and instruction-data rendering at the API boundary
├── budget.py         measured input-token calibration
├── sessions.py       append-only session journals and safe resume
├── repomap.py        bounded, cached repository orientation
├── symbols.py        offline syntax-tree outlines and symbol references
├── checkpoints.py    durable file-edit backups and guarded rewind
├── checks.py         checks after complete edit batches
├── progress.py       unchanged-batch cycles and streamed-repetition detection
├── activity.py       throttled command-output callbacks
├── protocol.py       ordinary replies, optional schemas, call normalization
├── streaming.py      provisional plain-text and incremental JSON reply decoding
├── compaction.py     isolated background summaries of completed steps
├── task.py           retained original user prompts and source references
├── inference.py      capability-filtered sampling defaults
├── model_catalog.py  curated non-coding model exclusions
├── markdown.py       Markdown rendering and copy-text mappings
├── clipboard.py      local clipboard utilities and OSC 52 encoding
├── environment.py    project settings and Python interpreter validation
├── api.py            shared transport, streaming, parsing, and retry policy
├── providers.py      provider registry and client factory
├── openrouter.py     OpenRouter routing, capabilities, and key information
├── nvidia.py         NVIDIA direct API, model profiles, and reasoning controls
├── kilo.py           Kilo Gateway and live catalog capabilities
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
    ├── answer.py     user-visible answers and answer-only completion
    ├── planning.py   bounded working-plan updates
    ├── blocking.py   owned worker threads for synchronous filesystem operations
    ├── files.py      read_file, write_file, edit_file
    ├── editing.py    atomic replacement planning, diagnostics, bounded diffs
    ├── search.py     grep, glob
    ├── grep.py       isolated, killable regex worker using a source snapshot
    ├── navigate.py   list_dir
    ├── shell.py      run_command
    ├── output.py     session log quota, storage, and read_command_output
    ├── git.py        confined status, diff, log, staging, commits
    └── web.py        web_search, fetch_page, http_fetch
```

The tree above shows the central modules. Smaller components now own the
request pipeline and interactive inspection features:

| Responsibility | Modules in `src/slipagent/` |
| --- | --- |
| Request assembly and recovery | `contextbuilder.py`, `requests.py`, `toolbatch.py`, `contextlegacy.py` |
| Headless JSON Lines output | `automation.py` |
| Session state and owned resources | `sessionstate.py`, `resources.py`, `work.py`, `events.py` |
| Slash-command behavior and help | `commands.py` |
| Saved inspection records | `artifacts.py`, `inspection.py` |
| File-backed editing and coordination | `editor.py`, `editorcontrol.py`, `editorui.py`, `useredits.py` |
| Read-only text and diff viewers | `viewers.py`, `viewerpanels.py` |
| Reusable lists, links, focus, and layout | `listpanels.py`, `links.py`, `focus.py`, `uiwindows.py` |
| Inspection configuration and folder opening | `options.py`, `uipreferences.py`, `explorer.py` |
| Inferred shell file references | `shellpaths.py` |
| Deferred tool definitions | `tools/discovery.py` |

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

The managed installer creates an editable installation. Change the files in
its permanent `source/` checkout, not a separate download, to affect that running
installation. You can also use an independently installed editable checkout.

For Linux's default installation, start with the harness as the workspace:

```bash
cd ~/.local/lib/slipagent/source
slipagent --workspace .
```

Or, from another editable checkout:

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

### Self-Editing From Another Project

Danger Mode adds a **SlipAgent Installation** section to each newly built model
request, identifying the running checkout (or the package directory for a
non-checkout installation). That section is absent when Danger Mode is off.
The location comes from the loaded code, not the current project directory.

Enable `/danger on`, then explicitly ask for the harness change you want.
Relative tool paths still refer to the project workspace; the agent must use
the supplied installation location for harness files. `/danger off` removes
this section from subsequent requests and restores built-in path confinement.
Previously delivered information can remain in retained request diagnostics.
Making the harness itself the workspace also permits editing it without Danger
Mode, but does not add the Danger Mode installation section.

### Keeping a Recovery Point

Self-editing changes real installed files. Compatible changes can reload while
stable contracts require a restart. Saving a file is not proof that the running
process accepted it: inspect `/generations`, reload messages, or the context view.
A rejected generation leaves the previous behavior running and the edited files
on disk for correction.

Before substantial edits, inspect `git status` and `git diff` in the installed
checkout and commit the working state you want to recover. Git can recover
committed tracked files; it does not preserve uncommitted changes or ignored
credentials automatically. `/rewind` covers checkpointed built-in file edits,
not arbitrary shell, MCP, or manual changes. Back up private configuration
separately when changing it. Reinstalling from another checkout may replace
self-edits; review the installer's changed-file report before confirming.

### Prompt Changes

`prompts/` contains Markdown instructions and JSON schema/profile resources.
Prompt files are read when requests are built, so saved prompt changes affect
newly assembled requests; they do not change an in-flight request or rewrite
history. Model-specific sections are selected through `model-profiles.json`.
Tool descriptions and schemas belong to their tool implementations.

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
| Agent behavior | Model-step logic in `agent.py`; instructions in `prompts/`. |
| Commands and output | CLI command handlers, helpers, and renderer methods. |
| Context | History handling, summarization, and retrieval behavior. |
| API behavior | Shared transport and provider methods, using existing clients. |
| Terminal | Layout builders, key bindings, styles, and presentation methods. |
| Tools | Built-in implementations, added tools, and the default registry builder. |
| Guidance discovery | Root and visited nested guidance refresh before working requests. |

The persistent frame and shared contracts require a restart:
`runtime.py`, `config.py`, `types.py`, `data_text.py`, `workspace.py`, `tools/base.py`,
`mcp.py`, `lifecycle.py`, `resources.py`, `work.py`, `sources.py`, the `semantic/`
package, and the package's root `__init__.py`. Changing startup/lifecycle setup
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
and exit. `/reset` starts a new session without deleting saved originals.
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

### Saved Sessions

Sessions, transcripts, command logs, artifacts, and preferences share the platform
data directory with global MCP settings. On Linux this is `$XDG_DATA_HOME/slipagent`,
defaulting to `~/.local/share/slipagent`; on macOS it is
`~/Library/Application Support/SlipAgent/data`; on Windows it is
`%LOCALAPPDATA%/SlipAgent`. Relative XDG paths are ignored. Each project gets a directory named `<md5>-<sha1>`, computed from
the same expanded absolute workspace path. Symlink spelling and case are
preserved; `project.json` retains the original path. `/sessions` prints the
absolute storage path and saved IDs. `SLIPAGENT_STATE_DIR` overrides the storage
root. Journals and logs may contain private project text; configuration API
keys are not included in their metadata.

Existing installations using `~/.SlipAgent` must move their saved data into the
new data directory before restarting to retain their session list and preferences.
Preserve existing destination files when merging; do not overwrite conflicts.

Startup creates no conversation journal until the first task prompt is entered.
Help, settings, session listing, and renaming before that prompt do not add an
empty session. Resume reopens the selected conversation's journal.

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
Copy, move, delete, and directory creation are tracked in the inspection panel
but are not undone by `/rewind`. These tools use exact paths, reject symlinks,
and report completed changes even when an operation fails partway through.
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
and `checks`. It is limited to 16 KiB. Language servers are configured in your
user settings instead (see below); a `language_servers` key in project settings
is ignored with a startup notice. `.slipagent/` is ignored in this repository; add it to another project's
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
| `/help [command]` | Show all help or one command's help; `/help init` and `/help /init` are equivalent |
| `/theme <name>` | Choose dark, light, monochrome, or ironbow colors |
| `/markdown [source\|rendered]` | Toggle or select Markdown source/rendered view |
| `/copy [text\|markdown\|code [number]]` | Copy the latest reply or a code block |
| `/menu` | Browse all commands and their arguments; selecting one inserts it into the input |
| `/view [record] [line]` | Choose a saved output/diff record or open its numeric ID at an optional line |
| `/danger` or `/danger on` | Disable workspace path confinement; queues while working |
| `/danger off` | Restore workspace path confinement; queues while working |
| `/danger status` | Show the current access mode |
| `/overthinking [on\|off]` | Enable recent archived thoughts, up to 25 steps, with `/overthinking` or `/overthinking on`; disable with `/overthinking off`. Enabled by default; queues while working. |
| `/planning [on\|off]` | Enable planning with `/planning` or `/planning on`; disable with `/planning off`. Enabled by default; queues while working. Disabling preserves the saved plan. |
| `/hashline [on\|off]` | Show or change optional anchored editing for this session; enabled by default, queues while working. Start disabled with `--no-hashline`. |
| `/tools` | List available tools |
| `/model` | Select a model with Up/Down and Enter; Escape cancels |
| `/model <slug>` | Switch model and remember the choice for future launches |
| `/models [filter]` | Select from the catalog with context and pricing; optionally filter by name |
| `/models free` | Select from zero-cost models |
| `/optimal [on\|off\|next\|reset]` | Show automatic free-model routing; on/reset clears failures and selects the highest score; next skips the current model; off keeps the current model under manual control |
| `/provider <provider>` | Select and remember a free default: OpenRouter/NVIDIA Nemotron 3 Ultra, or Kilo Auto Free; refuse missing models or nonzero/unknown pricing |
| `/temperature` | Show the explicit temperature or provider default, and whether the selected model supports changing it |
| `/temperature <value>` | Set the working-model temperature from 0 to 2; unsupported changes report an error |
| `/key show`, `/key status` | Show the masked current key, spend limit, and reported free quota |
| `/key` | Enter a key with hidden input and detect its provider |
| `/key <key>` | Detect the provider, install its key for this session, and offer to save it |
| `/key <provider> <key>` | Install a key with an explicit provider when its format is ambiguous |
| `/cost` | Session token usage and spend |
| `/task` | Show retained original user prompts and their source steps |
| `/queue` | Inspect pending user messages and choose one to view or discard |
| `/queue <number>` | View a pending message in full |
| `/queue discard <number>` | Discard one unsent message, retaining accepted history and the active task |
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
| `/mcp save <name> <cmd> [args…]` | Same, and persist it globally to `mcp.json` |
| `/mcp remove <name>` | Disconnect a server and drop it from the config |
| `/reset` | Start a fresh conversation/task/usage/log index; retain saved sessions and current guidance/model/settings |
| `/init` | Create a project-notes scaffold in `AGENTS.md` if missing and load project guidance into this session |
| `/config-show` | Show the active harness interpreter, model, workspace, danger mode, and Overthinking Mode without exposing credentials |
| `/reload` | Reload components now, or after the current model/tool batch |
| `/generations` | Show the current component reload generation |
| `/stop` | Finish the current model response and its tool batch, then stop before the next request |
| `/exit`, `/quit` | Exit the session; cancel retry waits and let an active response/tool batch finish without another request |

`/key <key>` recognizes OpenRouter (`sk-or-v1-`), NVIDIA (`nvapi-`),
Mistral (`mstrl_`), Google (`AIza` or `AQ.`), and Kilo JWTs identifying a Kilo
user. Detection happens locally. An unrecognized format asks for the provider;
you can also use `/key <provider> <key>` with `openrouter`, `nvidia`, `mistral`,
`google`, or `kilo`. A detected key must agree with an explicitly named provider.

The key becomes available to model selection, optimal fallback, and compression.
Adding another provider keeps the current model selected. Replacing the active
provider's key rebuilds its connection. OpenRouter verifies through `GET /key`;
Google and Mistral use their authenticated model catalogs. NVIDIA and Kilo report
that authentication will be checked on inference. Failed verification preserves
existing credentials. Saving for future runs still requires `y` at the prompt
and updates only the detected provider's entry in the data-directory `.env`.

Both forms supplying a key are omitted from input history, queued display text,
model input, and session journals.

Conversation titles default to `{full working directory} | SlipAgent`.
`/rename Fix login` changes the saved conversation and terminal title to
`Fix login | SlipAgent`, including while the agent works. `/sessions` shows saved
titles, and `/resume` or `--resume` restores them. In the REPL, restoring also
loads the original conversation into the scrolling area and starts at the end.
Prompts, replies, saved thoughts, tool calls/results, and queued input remain
available by scrolling up; restoring does not execute tools or call the model.
Names are private metadata
beside the saved journal in the project's hashed directory under the expanded
platform data directory (or `SLIPAGENT_STATE_DIR`), outside the checkout.
`/reset` starts a new conversation with the default title. With `--no-session`,
renaming lasts only for the current conversation in the running process.

### Terminal Layout and Navigation

Most inspection and menu actions have both mouse and keyboard paths. Type only
when entering prompts, command arguments, configuration values, or file edits.

| Action | Mouse | Keyboard |
| --- | --- | --- |
| Browse commands | Commands link, then click an item | `/menu`, arrows, Enter |
| Show/hide inspection panel | Inspect link | Ctrl+O |
| Focus regions and links | Click the region or link | Shift+Tab selects the next UI component; Tab also selects feed links outside the editor |
| Activate a focused link | Click the link | Enter |
| Scroll feed output | Mouse wheel | Arrows, Page Up/Down, Home/End with feed focus; Page Up/Down also work from the prompt |
| Open a touched item or saved call | Click its list entry | Focus its list, arrows, Enter |
| Browse a selector | Wheel by one item; click to select | Up/Down, Enter; Esc to cancel |
| Save or close an editor | Save File or Close Text Editor | Ctrl+S or Esc |
| Open inspection options | Options link | F4 |
| Inspect the actual model request | — | Ctrl+\ |
| Return to newest feed output | Scroll to Bottom link | Scroll to the end |
| Copy or reuse selected feed text | Select, right-click, choose an action | `/copy` for the latest reply |

On an interactive terminal, seven fixed rows sit at the bottom of the screen:
a pulsing activity indicator, five rows for wrapping input,
and the readout `<cwd>  │  Model: <slug>  │  Free: <n>`. Background colors distinguish this area from the scrolling transcript. Output and redraws preserve
partially typed input. The controls have a dark blue background, menus use
dark purple, and the prompt retains its dark green background. Input and output wrap at word boundaries to the
terminal width and reflow when you resize it. Wrapping preserves the submitted prompt's
original text. The mouse wheel scrolls overflowing input when hovered and scrolls
the transcript over the output area, preserving
your draft and prompt history. Scrolling back pauses automatic scrolling until you
return to the bottom. The CWD is the workspace where tools run; long paths
and model names shorten to fit narrow terminals.

The feed scrollbar uses pixel-accurate mouse positions when the terminal supports
SGR pixel mouse reporting (mode 1016) and reports its character-cell dimensions.
Dragging can then select positions between the scrollbar's character rows. Text
still renders in whole rows. Terminals without that support retain cell-based
scrollbar input.

Short notices share the activity area beside **Ready** and remain visible while
browsing selectors or the context view. Changed event notices flash three times,
then stay steady for the remainder of their 30-second lifetime, unless replaced
or cleared earlier. After expiry, rotating tips and the Commands link return.
Only the notice text flashes; activity
indicators, the Commands link, and rotating tips do not flash. Overflowing status
messages and notices pause briefly, then scroll horizontally in a loop so the
complete text can be read. Text that fits stays still. When no notice is
active, this area rotates tips for available controls. Folder-launch failures also
remain in the transcript.

### Inspection Sidebar

Click **Inspect [Ctrl+O]** or press **Ctrl+O** to show or hide the inspection
panel on the right side of the feed. It occupies one quarter of the terminal
width, with **Touched** above **Calls**. This list includes inspected
directories, which open in the file explorer when one is available.
Click a file to edit it, or a call
to inspect its decoded arguments and result as labelled fields with literal text
and line breaks. The mouse wheel scrolls the section under
the pointer. **Tab** moves forward through the prompt, active
feed/editor/viewer/context, links, lists, and footer controls. Within the feed,
Tab starts at the last link and moves upward, scrolling each selection into view.
Clicking or refocusing the feed resets traversal to start at the last link again.
**Shift+Tab** selects the next UI component, skipping individual feed links.
Links receive a visible focus highlight; **Enter** activates them. Tabbing shows
the tip “Shift+Tab to select UI component.” without flashing.
Use the **Inspect** link or **Ctrl+O** to reveal hidden sidebar lists.
Use **Up/Down** to select an entry and **Enter** to open it.
Inside the code editor, **Tab** inserts indentation; **Shift+Tab** moves focus
to the next control. Terminals narrower than 60 columns
temporarily hide the panel and skip its lists during traversal.
Hidden panels cannot retain keyboard focus. Viewer and context navigation keys
apply when that region owns focus; prompt editing keeps its normal shortcuts.
Clicking a focusable area focuses it. Prompt clicks place the cursor on the first
click; clicking a list title focuses that list for arrow-key navigation.
The feed retains at most two wrapped layouts and prepares the alternate Inspect
width in the background, so opening and closing the panel reuses formatted text.
Closing a menu restores its invoking region if it remains visible. A completed
background save preserves any focus choice made while it ran.

### Read-Only Output and Diff Viewers

Saved command output and edit diffs have clickable **View Output** or **View
Diff** references. `/view` lists records; `/view NUMBER [LINE]` opens one at a
specific line. These viewers are read-only: arrows, Page Up/Down, Home/End, and
the mouse wheel navigate; **Esc** closes the viewer without interrupting agent
work. File and inspection-record viewers use private file-backed snapshots and
disk-backed line indexes, decoding JSON records in bounded chunks. Very long
lines use bounded display segments; source line links still target logical lines.
An open file that changes, is replaced, or disappears on disk keeps its snapshot
and offers **Reload [Ctrl+R]**. Reload failure leaves the snapshot available.
Notifications and your input draft remain visible. Diffs use distinct
colors for headers, hunks, additions, and deletions, with old/new line numbers.
The feed supplies links to saved edit diffs. No editing or saving is enabled in
these viewers.

### Built-In Text Editor

File links open the built-in editor at the linked line. A completed single
`edit_file` replacement links to its first changed line, skipping unchanged
matching context. Multiple replacements show just the filename and open at the
first change; **View Diff** shows all changes. An active agent finishes
its current tool batch first. While the editor is open, the agent makes no new
working-model requests or tool calls, including read-only tools. Existing managed
background commands and history compression continue independently. Agent work
resumes when the editor closes. Queuing the editor flashes the notification
**Opening editor after this turn**, with a **Cancel Opening Editor** link beside
it. Cancelling releases the queued editor hold so
the agent continues working. Read-only viewers open immediately while the
agent continues working. **Ctrl+S** saves; **Esc** or **Close Text Editor**
closes, showing **Save**, **Don't Save**, and **Cancel** in the bottom menu when
there are unsaved changes. Clicking text places the cursor there; clicking right of a line places
it at that line's end, and clicking below the content places it at the file's end.
Editor action controls replace the prompt area while editing; the same prompt
buffer, text, cursor, and scrolling return when the editor closes.
Files containing unsupported control characters open read-only, with an explanation in the header and status. Editing and saving are disabled; tabs and normal line endings remain editable. Binary files remain unsupported.
The editor's top line shows the full file path, truncating its middle when needed.
Help, reload controls, and external-change warnings appear in the bottom area.
Clicking another touched file opens it immediately when the current file is
saved, or offers the same unsaved-change menu first. The agent stays held
throughout the switch.
Files are backed by private snapshots, with only
one bounded section in the editing buffer and neighboring sections in the view.
Arrow keys and the mouse wheel cross section boundaries automatically; each
wheel event moves the cursor three displayed lines; arrows move one.
At a screen edge, the text scrolls while the cursor stays at that edge.
**Page Up/Down** moves one line less
than the editor height in either direction, retaining visible overlap.
**Ctrl+Home/End** moves to the start/end of the document. The viewport follows
the cursor, and section changes preserve edits. The title spinner stops while
the editor holds the agent.
Untouched content and line endings are preserved. External disk changes offer
Reload and require an explicit choice before overwriting. Editor reload retains
the cursor's logical line, clamping to the last line if the file became shorter.
Undo/Redo spans storage pages, navigation, wrapping and saves. Reloading or
closing the document starts a new undo history. Undo checkpoints and changed
page versions stay in private temporary storage. Backspace and Delete operate
across page boundaries, and cursor movement, selections and deletion preserve
complete Unicode graphemes without normalizing file contents.

Git, when available,
supplies the saved user-edit diff. Its bounded notification accompanies model
input separately from user messages and tool results and survives session resume.
Notifications share a context budget; remaining edits wait for later requests.
Each save has a revision, including reversions. Only an accepted response retires
the exact revision it received into that turn's history. Failed requests leave
notifications pending. Full archived diffs remain available through `read_artifact`.
Without Git, Python's
`difflib` generates the diff; that fallback holds both files' lines in memory.

### Inspection Options

Click **Options** or press **F4** for editor wrapping, tab width, and folder-opener
settings. Folder-opener configuration uses the footer input area. These settings
share the existing private preferences file across projects.
A configured folder opener works without a desktop session: enter an executable
and arguments, optionally with a separate `{path}` argument. The directory is
passed literally, without a shell; when no placeholder is given, it is appended.
External editors are not offered.

When the feed is scrolled up, **Scroll to Bottom** appears in the center of the
row above the prompt. Click it to resume following the latest output.

### Durable Inspection Records

Completed command logs and inspection views share immutable UTF-8 payload files
within each session. Version-two inspection records reference those payloads;
log byte offsets, inspection character offsets, record IDs, and history links
remain unchanged. Identical payloads within a session share one stored file.
Live commands still append to their own streams until completion.

Older JSON inspection records remain readable. On resume or fork, command-linked
legacy records migrate only after their decoded output, errors, and provenance
match the retained logs. New payloads and the replacement index are made durable
before the journal records their paths; redundant copies are removed afterward.
Mismatches and storage failures retain the original copies and report deferred
consolidation in log metadata. Forks carry their own payload files and remain
readable after the parent is deleted. This migration does not rewrite conversation
history or combine the two existing quota limits: command retention and inspection
admission still account for their logical bytes independently. Physical disk use
is smaller than the sum of those accounting totals.

With session persistence, inspection records live beside the session journal in
its `-artifacts` directory. Microsecond timestamp filenames contain JSON with
`command`, `output`, `error`, `call_id`, `step_id`, and `kind`. Command stdout and
stderr stay separate; stderr is not itself proof of failure. The viewer shows
the command, one blank line, then output. Numeric references accompany full and
compressed history, survive resume/fork, and can be retrieved with
`read_artifact` without granting arbitrary access outside the workspace.
The tool returns one field (`output`, `error`, or `command`) in character pages:
4,000 characters by default, capped at 12,000. Follow `next_offset` until `eof`.
Listing defaults to 20 records, capped at 100, with offsets measured in records.
Command metadata is shortened to 500 characters; select `command` to retrieve it
in full through pages. Paging scans and validates the stored record with bounded
memory. If saving a diff fails, its notification reports that the archive is
unavailable; a truncated notification remains pending.
Inspection storage has the configured command-log quota independently of
command retention accounting. Saving a record does not alter raw-log byte
offsets. Without session persistence, inspection records are temporary.

Open a saved diff and select **Comment on Diff** to attach it to your prompt.
Your existing draft stays in place; `[Diff …]` beside the prompt identifies the
attachment. Enter sends an ordinary user turn, queued normally while the agent
works, with the saved diff's record, step, and call references attached. For
example, write “Undo this” or “Escape it properly.” Comments apply to the whole
diff. **Remove Diff Reference** detaches it without deleting your draft.
Opening another diff does not change an attachment unless you select its comment
action. Sending the comment clears the draft attachment; slash commands and blank
input leave it in place. Clearing the feed also clears an unsent attachment.

While a referenced turn is current, the system prompt identifies its particular
saved diff as the target. The agent can retrieve the full diff with `read_artifact`
and should inspect current files before changing or undoing an older edit. The
reference follows normal queue, history, and session-resume storage. Later
ordinary turns remove the special system instructions.

### Directory Links and Selected Text

When a native file explorer is available, the CWD readout is an underlined link
and tool calls with an existing directory `path` show a link to that directory.
Click a directory link to open it. Drag to select feed text, then right-click
to choose **Copy Text** or **Use Selection in Prompt**. Copied selections omit
display-only wrapping and retain actual line breaks. The latter appends the
selection after a newline without submitting or replacing your draft. Right-click
a selected directory reference for **Copy Absolute Path** and, inside the project,
**Copy Relative Path**. Right-clicking without a selection shows a short tip.

The CWD link opens the project directory. Linux requires a desktop session and
`xdg-open` or `gio`; macOS uses `open`, and Windows uses its default folder
handler. Without an available opener, paths remain plain text. A failed launch
reports the failure and disables directory links for that session.

### Suggested Next Steps

The model may offer up to three **Suggested Next Steps** after a finished reply.
Each link displays its entire prompt. Clicking it while idle posts that exact
prompt to the feed and starts a new run, leaving your input draft untouched.
Suggestions are optional response data, not tools or sidebar call records.

### Feed Scrolling and Prompt Pinning

The nearest preceding green line starting with `>` pins to the top as you scroll.
Scrolling up into an older task shows its prompt line; scrolling down restores
the newer one. Submitting another task or queued follow-up keeps the preceding
prompt pinned until the new prompt reaches the top. The pinned prompt uses
one line, shortening at a word boundary with `…` when needed and adjusting on resize. The
complete original remains in scrollback. The context view keeps its own layout.

The interactive transcript keeps original output and indexed display rows in
private temporary files for the session. Streaming formats new text and the
unfinished wrapping tail; ordinary redraws fetch visible rows from a bounded
cache instead of scanning all earlier output. Repaints are limited to 30 per
second. All scrollback remains available, and resizing reflows the original
text. These display files are removed when the terminal closes; they are not
persistent project memory or saved conversation archives.

### Project Instructions

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

### Command Menu

The underlined **Commands** link beside **Ready**, or `/menu`, replaces the
seven-row input footer with a heading, a five-row scrolling list of all commands with their
help syntax and descriptions, and `↑/↓ or wheel = move | Enter or click = select | Esc = back` on
the bottom line, with shorter wording on narrow terminals.
Use the mouse wheel or arrow keys to move one item at a time, and click an item or press Enter to
select it. At the edge, the list scrolls just far enough to show the whole selected
item when it fits. Items taller than the list show their beginning. The transcript
stays visible above it. Escape closes the menu without running
anything, preserving your draft and scroll position. Selecting an item saves the
current draft in input history and inserts the command into the input without
running it. Add any arguments and press Enter to submit it. The Commands link
appears only while ready. Clickable controls are underlined and invert their
colors on hover.

### Queued Input and Stopping

At startup, the interface shows Working while welcome lines appear one at a time.
Progress then reports repository mapping, model catalogs and metadata, quota,
Python discovery, session history, and each MCP server's connection result.
Independent work runs concurrently, with four worker threads and at most four MCP
connections starting together. Tasks and settings entered during startup queue
until setup finishes, with their previews visible immediately above the controls.
Esc, `/exit`, `/quit`, or Ctrl-D cancels startup and exits.

Commands and prompts share submission order while the agent works. Commands apply
after the current response and complete tool batch finish, and after earlier prompts
reach a model request. Later prompts use the resulting settings. Settings changes then resume
the task without resetting its request limit. `/stop` and exit controls remain
immediate control actions. `/stop` holds previously queued input until fresh input
arrives; reset, resume, and `/task new` replace the task instead of restarting
the previous work. Slash commands are never sent to the model as user messages.

The working indicator shows a moon-phase spinner beside `Working (esc to interrupt)`.
The spinner uses eight emoji frames of two terminal columns each and also appears
in the terminal title. Press Escape outside a chooser to interrupt the active step immediately. It
cancels model requests, active tools, background commands, and pending summaries;
queued settings are discarded. Once cancellation finishes, queued user messages
start the next request; an empty queue leaves the agent idle.
Completed tool results remain stored, while interrupted or unstarted calls are
recorded explicitly.
An atomic file operation already running finishes safely before cleanup settles;
the terminal stays responsive. Escape inside a chooser only closes that chooser
and leaves agent work running. `/stop` retains its gentler step-boundary behavior.

Live reload scans source files and prepares candidate imports in a worker thread,
so keyboard input remains responsive during preparation. It applies the validated
generation on the main loop at a safe boundary. Restart after installing this
change to enable the threaded reload frame.

### Danger Mode

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

### Context View and Model Input

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

System instructions, tool definitions, and current-step metadata appear in the
system prompt. Native tools also use the API tool field. A single user message
begins with JSON metadata containing these fields, followed by literal tool-result blocks:

| Field | Content |
| --- | --- |
| `user_prompts` | New messages in chronological order, or `[]` without new input; later messages supersede conflicting earlier instructions |
| `retained_user_prompts` | Full ordered array of user messages received before tool use began; supplied only after tools when no new user messages are present |
| `history` | Selected completed steps, oldest to newest |
| `errors` | Harness errors and response corrections not attached to a call |
| `user_edits` (when present) | Saved user file edits, with revision IDs, bounded diff prefixes and artifact references |
| `user_edits_omitted` (when present) | Pending saved-edit notifications deferred by the shared budget |

Saved-edit guidance is included only when the request contains pending edit
notifications or selected history with saved edits. Requests without them omit
both the fields and that guidance.

Each full history record contains `user_prompts`, `agent_response`, and `tool_calls`. `user_prompts` lists only the messages the user sent in that step.
Each call retains its `call_id`, `tool_name`, and argument object, with its raw
outcome reference under `result` for success or `error` for failure. Legacy observations
without a recorded status use `unclassified_result`. Labels belong to keys;
no classification prefixes are inserted into result contents. Each reference identifies a numbered block, its character count, and its `text` or `json` format. Textual results are copied verbatim, even when they resemble JSON. Explicitly structured results use pretty-printed JSON blocks.

Records remain raw objects and lists internally. At the message boundary the
user-input metadata is serialized and exact result blocks are appended without extra escaping or indentation. The HTTP client then encodes the
provider's required transport envelope; native tool-call arguments also use the
JSON string required by that protocol. Quotes, backslashes, and newlines round-trip
without changing the values or allowing text to create sibling input fields.
Metadata and length-labelled blocks provide structural separation, not a guarantee against prompt injection.
`prompts/system-history.md` explains that historical text and tool observations
are evidence, not system instructions. Request assembly does not archive the
input envelope or system guidance as conversation turns.

`representation` distinguishes `full`, `compressed`, and `excerpt` history records.
Full records also include available `compressed_summary` analysis; compressed
records contain summaries and metadata instead of originals. Every record has a
`step_id` for retrieving originals with `recall_history`. Excerpts use
`result_excerpt`, `error_excerpt`, or `unclassified_result_excerpt` on calls,
with separate response excerpts, omission counts, and retrieval guidance.
Task-source IDs and the current step's `is_tool_result_response` appear in system
metadata. Original retrieval and prompt retention remain independent of history
selection. All rendered fields, text, and message overhead count toward the context budget. This input format
is separate from the selected model's response contract, which stays consistent
between calls. Native tool calls remain available for responses from models that
support them; the harness does not ask models to echo its history-record format.
Echoes of former harness response labels are removed from ordinary reply text
and from assistant replies reused as context, without rewriting saved originals.
Quoted or fenced examples, user input, tool arguments and file content remain intact.

The prompt stays live while the agent works. Consecutive waiting prompts join the next
model request together, after the current tool batch finishes. Their previews
stay just above the bottom controls until their request is sent. Each
preview occupies one line within the main pane, stopping at the visible
inspection sidebar and adding an ellipsis when it is too long. Each queued prompt
or command has a Cancel link at the right, separated from the preview by a space:
click it, or focus it and press Enter, to
remove that specific item. API keys are hidden in command previews. On submission,
the full prompts enter the feed in arrival order before that request's output.
If context preparation fails or `/stop` prevents that request, the previews stay.
Use `/queue` to inspect pending messages and select one to view in full or discard.
In plain output it lists the complete messages; `/queue <number>` views one and
`/queue discard <number>` removes one. Numbers refer to the current queue order.
The interactive chooser retains each message's identity while open: if that
message has already been submitted, discarding it changes nothing. Duplicate
messages can be discarded individually. Discards are saved with the session and
preserve accepted history, command logs, and the current task. `/task new` remains
blocked while any unsent user input is pending.
Blank lines separate submitted prompts from the surrounding transcript.
Each model reply starts with a blank line, including replies preceded only by
tool output. Its following tool calls and results stay together until the next
model reply; each command response is another unit.
The model can explain its work alongside batched tool calls.
The interactive feed shows command activity and saved-output links instead of
printing stdout and stderr inline. Open the output viewer to inspect them;
the model still receives command results. Saved edit diffs also stay in the viewer,
with their existing View Diff links beside the file links. One-shot progress uses stderr, and
command archives retain decoded output within their quota.

### Repeated Actions Without Progress

Three repetitions of an unchanged tool batch or cycle of up to six batches
produce model-only recovery guidance with the exact calls. A fourth repetition
stops the run with history intact. New user input resets detection. Intentional
external polling can use `run_command` with `poll=true`; command-log reads and
job-control calls are exempt.

### Reasoning and Overthinking

Readable reasoning supplied through a separate provider field streams under a
gray `Thinking:` label. Each response's reasoning is concatenated into one
string and saved with that step's uncompressed originals. Existing history is
not rewritten. **Overthinking Mode** is enabled by default: the latest 25 completed
steps include supplied thoughts as a `reasoning` string in each step's input
record, subject to `--reasoning-history-steps`. Steps without thoughts gain no extra field. Use `/overthinking off` to
disable this mode, or `/overthinking` or `/overthinking on` to enable it. Thoughts count
toward the context budget; omitted steps and budget-limited excerpts omit them.
Opaque provider reasoning metadata is never included. Background compression returns
separate `summary` and `reasoning_summary` fields. It receives the completed step's
thoughts and up to 25 preceding steps as context, but compresses only that completed
step. Overthinking uses filtered thoughts when available, otherwise the originals.
An empty filtered result stays empty. Original thoughts remain available through
`recall_history`; streaming thoughts still appear in the terminal.
For OpenRouter, the harness selects the highest reasoning effort
explicitly listed by the API. When reasoning is supported without listed effort
choices, it enables reasoning; when unsupported, it sends no reasoning setting.
Reply text streams as it arrives, including text inside an unfinished JSON
`response` string. JSON escapes are decoded across chunk boundaries; tool
arguments and bookkeeping stay out of previews. These previews are provisional:
failed or rejected attempts are marked in the transcript. Tool calls still wait
for complete response validation, and only accepted responses enter history.
The terminal displays the reply text defined by the selected response contract, followed by tool activity. Background summaries
stay out of the transcript. One-shot mode keeps the final answer on stdout,
with streamed reply previews, thoughts, and tool activity on stderr.

### Response Protocol

All built-in model prompts and tool guidance are in the top-level [`prompts/`](prompts/README.md) folder for direct human editing. The `.md` templates contain Markdown headings, lists, tables, and fenced examples; static headings and response-schema descriptions live in these resources rather than Python. Only imported history retains the large opening and closing warning banners. SlipAgent rereads templates when building requests and tool definitions, including with live reload disabled. Start with `prompts/system-prompt.md` for general behavior or `prompts/background-summary-prompt.md` for compression. Preserve placeholder names and each prompt's output contract; Markdown instructions can still require JSON replies or plain-text summaries. The installer copies this folder, and wheels bundle it inside the package. See its README for the resource map and formatting conventions.

For model-specific instructions, map the exact model ID to Markdown files in `prompts/model-profiles.json`. Profiles can also select embedded tool calls, XML call compatibility, and structured reply formatting. Set `structured_output` to `false` to omit the API reply-format constraint for one model; set it to `true` or omit it to follow endpoint capabilities. Dots3 Note Preview has this setting disabled for both its standard and `:free` IDs, keeping native tools enabled. GLM-5.3 has an embedded-call profile. See [Model Profiles](prompts/README.md#model-profiles) for the settings.

Runtime guidance distinguishes exploratory questions from authorized implementation and keeps work within the user's requested scope. Background summaries prioritize user requests, corrections, and boundaries over agent plans, preserve source attribution, and distinguish completed or rejected actions from unfinished requested work. Historical reasoning and optional agent suggestions do not authorize additional work.

At startup and every model selection, including reselection, SlipAgent refreshes the selected provider's capabilities. OpenRouter uses live endpoint metadata; NVIDIA uses explicit model profiles because its catalog omits those fields. Subsequent calls reuse the profile until another selection. A failed or incompatible selection reports an error and keeps the current provider, model, and conversation.
Startup model-check failures display a warning and leave the interface available, so users can select another model with `/models` or `/model <slug>`. A model that fails its startup check is not saved as a new preference.

Temperature is omitted unless explicitly set with `/temperature <value>` or
`--temperature`, leaving each provider to choose its default. `/temperature`
shows the explicit value or `provider default`. An explicit session setting
carries across model switches and applies wherever the selected endpoints
support it. Unsupported changes report an error and preserve that setting;
requests to unsupported endpoints omit it. Individual-turn and grouped-history
compression always omit temperature, including when the working model has an override.

Supported top-p/top-k/min-p defaults remain model-specific for original Qwen3
hybrid checkpoints and Nemotron 3.5 Lightning; explicit request values take precedence.

SlipAgent uses [OpenRouter's native tool-call format](https://openrouter.ai/docs/guides/features/tool-calling)
when the selected endpoints advertise `tools`. Tool definitions go in the API's
`tools` parameter; calls arrive in `message.tool_calls`, separately from the
response text. The harness preserves each call ID and stores its result with
the matching `call_id` in that step's `tool_results` array. The next request
supplies the completed step in the user-input metadata, with each literal result block referenced
by its call under `result` or `error`. Native calls remain the model's
response format; the input records describe completed history. SlipAgent does
not force a `tool_choice` setting.

Endpoint metadata, including reasoning support and provider limits, remains
cached between selections. Model-profile formatting overrides are reread when
building working requests.
Selection prefers native-capable endpoints;
models without them can use the JSON call format below if they support JSON
output. Their tool definitions enter the system prompt, and their history uses
the same JSON input records, without unsupported native tool parameters.

**Native-tool models without schema support use plain reply text.** They receive
no `response_format` parameter and need no JSON content record. Unless disabled
in the model profile, JSON-only models receive
`response_format: {"type":"json_object"}` when schemas are unavailable.
Reply text is encouraged on tool turns but may be empty when calls are present.
Unless disabled in the model profile, endpoints advertising `structured_outputs`
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

When native calls are unavailable, requests ask for a JSON object containing
`response` and `tool_calls`. Response parsing independently detects plain replies,
JSON envelopes (including complete JSON code fences), legacy summary envelopes,
tagged JSON calls, and Qwen3-Coder's `<tool_call><function=...><parameter=...>`
format, plus `<tool_call><answer>text</answer></tool_call>`. Complete tagged JSON
may omit its outer closing tag. Answer carriers also tolerate omitted trailing
tags; executable Qwen calls require their parameter and function closing tags.
These forms require a provider-completed response;
incomplete JSON arguments and provider-reported truncation still reject the batch.
API `tool_calls`, legacy `function_call`, and `tool_use` content blocks
are also recognized. No per-model response-format record is needed.

Calls may arrive separately, inside answer text, or through both channels in the
same response. Distinct calls are combined in API-carrier order followed by text
calls; mirrored calls execute once, while intentional repetitions within a batch
are preserved. Conflicting calls sharing an ID reject the complete response.
Qwen parameters use the advertised tool schema for types and preserve string
content without XML entity decoding. Streamed calls wait for the complete response.
Malformed arguments, duplicate keys, invalid Unicode, non-finite numbers, duplicate
IDs within a carrier, and truncated batches reject the complete response before
execution. Quoted and fenced examples within explanatory prose remain text.
Request-side API capabilities still determine which parameters can be sent.
Three consecutive invalid responses stop the run; retries count against the
working-request limit. Streamed thoughts and provisional reply text remain visible;
failed reply previews are marked as unaccepted.

Interrupted streams, overloads inside streamed/JSON responses, and transient HTTP failures receive three silent retries. If failures continue, SlipAgent displays the error and a countdown while retrying with increasing delays. It gives up before scheduling an interval longer than **60 seconds**; a server `Retry-After` beyond that limit also stops recovery. The minute is an interval limit, not a total request deadline. Permanent authentication, credit, and request errors are not retried. Every working attempt counts against `--max-steps`, with no nested retry loop. `/stop` and Esc interrupt waits. Each attempt starts fresh:
partial tool arguments and rejected replies never execute or enter history as accepted actions.
Authentication, credit, and permanent request failures remain errors.

Every failed working request is preserved as a numbered history turn, including
provider/transport errors, explicit context overflows, rejected responses,
repetition stops and cancellations. Its `request_failure` record preserves the
error, a bounded rejected/partial-response excerpt, the diagnostic attempt number,
and consecutive/session failure counts. It explicitly marks that no tools from
the attempt executed. Failures receive a deterministic summary rather than an
extra model call. Full excerpts remain retrievable through `recall_history`.
Retry requests rebuild their context with those failed turns and `request_retry`
metadata identifying the failure count and turn IDs. Guidance tells the model
to honor instructions to stop or not retry rather than treat each attempt as the
first. The request budget still counts every attempt; recovery waits remain
cancellable. Failed-turn records and counts survive resume and fork, and reset
clears them. Raw request/response diagnostic files remain separate.

`/requests` lists the latest 20 attempts; `/requests 3` displays attempt 3's
final request body, outcome, and complete response/error details. Complete provider
responses are captured before parsing, with streaming text and reasoning joined
into continuous fields rather than individual fragments. Provider metadata and
tool-call fields are retained, including partial responses on failure. Request JSON
is deduplicated and gzip-compressed in private files beside the session journal,
without request authentication headers. Each attempt also captures its raw HTTP
response body in `NNNNNNNN-stream.bin.gz`, with status and response headers in
the attempt metadata. This preserves keepalives, empty events, malformed bytes,
and original line endings before HTTP content decoding or stream parsing. Capture
writes incrementally; cancellation or disconnect retains the received prefix.
The readable `/requests` view continues to show the assembled response and names
its raw file. These diagnostics are never supplied as history
or sent for compaction. Diagnostics use a **32 MiB rolling target**: oldest attempts
are removed as new ones are saved. The newest attempt keeps its complete request
and responses, including raw bytes, even if it alone exceeds the target. Shared request snapshots remain
until their last retained attempt is removed. Conversation history is unaffected.
Storage failures display a warning and preserve existing records. Reset starts a
new directory. Resuming continues the saved attempt numbering and rolling retention.
`--no-session` uses temporary files removed on reset/exit. Treat request logs as
private project data, since they contain the same text sent to the model.

OpenRouter requests set `provider.require_parameters` and restrict routing to compatible
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
Input queued before `/stop` stays pending. A new prompt or queued command entered
afterward resumes work, including while the current step is finishing. A new
prompt also starts work from idle; no special resume phrase is required.
SlipAgent defines no Ctrl-C or SIGINT key binding.
Ctrl-D, `/exit`, or its alias `/quit` exits the session. Pipes and `TERM=dumb`
use plain output.

### Active-Task Memory

The model retains nothing between API requests unless SlipAgent sends it again.
New input arrives in one chronological `user_prompts` array. Later messages
supersede earlier conflicting instructions; queued messages are not relegated
to a secondary field. With no new input, `user_prompts` is empty and
`retained_user_prompts` carries the full original array for the active request,
after tool use and on any other continuation. A new set of messages leaves the
retained array empty.

The harness stores original source messages independently of compression.
`current_prompt_step` identifies the step that received them, and the model can
retrieve its originals with `recall_history`. Retained messages count toward the
request budget. Tool-continuation guidance explains why control returned and
asks the model to assess the results against those requests.

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
Each full step also includes its available `compressed_summary` in the same JSON
object, alongside the original fields. Compressed-only records use that same key.
Both originals and their accompanying analysis count toward the context budget;
the representation comparison above uses the original before adding its summary.

Each background compression attempt has a 20-minute timeout instead of the normal HTTP timeout. Single-turn and multi-turn compression keep retrying rate limits, transient provider errors, and transport failures until recovery or cancellation. They share the working request retry schedule, then continue with waits of up to 60 seconds, honoring longer provider `Retry-After` cooldowns. Every retry selects the next available provider. Invalid replies and permanent failures use a bounded retry budget, expanded when needed to try the ranked candidates. Retry waits and additional attempts can extend the job beyond 20 minutes. Exhaustion preserves original steps for `recall_history`. Working requests retain their configured timeout and retry limit; reset and shutdown cancel pending summaries.

Forking, resuming, deleting a session, and finishing a one-shot run give all pending
compression up to two minutes to finish, including retries. At that deadline,
SlipAgent cancels and drains unfinished jobs before continuing. Original turns
remain available unless the session itself is deleted.

Each working request identifies the source step of the active requests in system guidance; their exact text belongs in the structured user-input object. Steps after tool batches explicitly explain that control returned automatically without a new user instruction: assess results against that goal and return an answer without tools when it is fulfilled. This guidance is assembled for the outgoing request only; it is not saved in conversation history or background summaries. Historical requests provide background rather than independently requesting more work.

The newest **one third** of retained individual posts targets full originals,
rounded up; the older **two thirds** use economical summaries. Missing summaries
retain originals rather than failure notices when space permits. History uses
the model's context allowance, subject to `--context-tokens`, after reserving
instructions, tools, output and estimation headroom. The oldest individual posts
leave the view as needed, recalculating the split at each window size. There is
no fixed 100-summary cap. `--context-steps` remains accepted for compatibility
but no longer controls selection. Newly returned tool results reach the working
model intact, or as explicit excerpts if even the latest batch cannot fit.

Selection starts fresh on every request, so omitted records can return as large
posts age into summaries. Originals remain retrievable even when omitted.
Project instructions and the retained user prompt stay pinned outside the window.

Older summaries can also become **one historical narrative across many turns**.
A separate background call requires at least **20 completed turns since the last
merge attempt**, at least **20 eligible older summaries**, and at least **12,000
estimated tokens** of source summaries and metadata. Small ordinary turns can
therefore go much longer between merges. Only one merge job runs at a time. Its
retries belong to that job; after final failure, at least 20 turns must pass from
its initial attempt before another job can start. Per-step summarization
continues independently as described below.

Each narrative is built from the original individual summaries, never a previous
merged narrative. It combines related work rather than recapping every step.
The prompt requires source turn citations, and the harness preserves the full
source turn list, call IDs, tool names and artifact references separately.
`recall_history` can retrieve exact turns or search the full archive when a turn
number is unknown. Narratives accompany the rolling window without duplicating
their individual source summaries. Merge results and attempt spacing persist
across resume and fork; reset cancels pending work. Invalid or non-shrinking
narratives leave the original summaries intact.

**One separate background request starts after each complete tool batch returns**,
or immediately after a reply without tools. It receives that step's prompt,
response, calls, results and thoughts, plus up to 25 preceding completed steps
as contextual history. Its instructions restrict compression to the current step.
Oldest contextual records are omitted if needed to fit; the current step remains complete.
Tool arguments and structured results remain dictionaries or lists internally;
the complete summary input is serialized once for the model. Complete history
retrievals return complete objects or arrays without character paging.
**Both individual-turn and grouped-history compression round-robin through
the best available ranked free model from each eligible provider.** Every job
and retry uses one shared rotation, in both normal and optimal modes. Providers
participate when their own API key and a ranked free model are available:
OpenRouter, NVIDIA, Kilo, Mistral, and Google. Each attempt uses its destination's
own capabilities and context/output limits.

A free working model's provider is reserved for coding when alternatives exist.
A paid working model's provider can also supply a separate free compression
model. Unknown working-model pricing keeps its provider reserved as well.
If only one provider is available, it handles both coding and compression;
SlipAgent encourages adding provider keys to improve availability.

Any compression failure sends the retry to another provider when possible.
Transient failures retain the model's eligibility. Permanent/model-format
failures exclude that compression model, so its provider contributes the next
ranked model on a later rotation. A request too large for one model tries another
without disabling that model for smaller jobs. Failed candidates are reused only
when no unfailed option remains. If no eligible free route exists, originals
remain available and SlipAgent reports why compression is unavailable.
Background clients are separate from the working client and close with the session.

The summarizer returns a JSON object with separate factual `summary` and filtered
`reasoning_summary` strings, each limited to 6,000 characters. Short steps
should get short summaries. The working model continues without waiting.
One worker runs first-pass summaries, with at most 32 ready references. Further
steps wait on their retained originals and enter the ready queue as it drains.
Only the active request copies its payload; deferred summaries select their
compression provider when they run. An occasional historical merge can run alongside it.

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
its summaries before shutting down. These additional requests use their selected
compression provider's key and contribute to its quota use, session tokens, and
reported cost.

The model's `recall_history` tool searches captured originals or reads a step
by ID, returning the complete selected object without character limits or paging.
List/search results include `total_matches`, the number of matching steps (zero
when none match). Select `section="prompt"`,
`"response"`, `"reasoning"`, `"tool_calls"`, or `"tool_results"` to retrieve one original part;
`sections=["prompt", "tool_results"]` selects several together. Omit the selector
for the full step. `call_id` selects one observation, or a call when paired with
`section="tool_calls"`. `section="user"` returns the original new user messages
associated with that step, for task-source recovery. Shell and Git observations show the
full output up to 32,000,000 characters per stream, then the beginning and end
for larger output. The full
decoded streams are retained separately within the session log quota and are
retrieved with `read_command_output`. Timeout results preserve partial output and
state that effects may be partial. Both history versions are journaled for
resume by default; `--no-session` uses temporary files removed on reset/exit.
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

When the map does not show where behavior lives, the model can load
`semantic_search` through `load_tools`. It returns five complete ranked excerpts
by default, with relative paths and line numbers. Short queries of five or six
meaningful words work best. Ranking combines the local quantized BGE encoder
with SQLite keyword matches and filename context; scores are relative rankings,
not confidence estimates. Use ordinary search when results are weak, and read
current source before editing an indexed excerpt.

Semantic search starts only on its first call. It reuses locally cached model
assets or downloads missing assets, builds a persistent index under the private
user-data directory's `semantic/` folder, and watches changes for the rest of the
session. Source and queries stay local. Updates coalesce save events, hash affected
files, and reconcile metadata every 30 seconds; queued changes flush before a
lookup. Reopening a session reuses unchanged vectors. Dependencies, hidden paths,
symlinks, Git-ignored files, and the harness's data directory are excluded.
The first build can take time on large projects. CPU inference uses one chunk
per call and the physical core count, with a two-thread fallback. No model or
indexing worker is loaded just to display the tool catalog.

Source files are hashed into one temporary snapshot beside the index, then read
in bounded blocks and chunked incrementally. Long lines are split into complete
consecutive excerpts; no source is discarded to meet a file-size cutoff. The
snapshot is removed when processing finishes or fails. Source/chunk buffers stay
bounded independently of file size; the stored vector index still grows with the
number of chunks.

The [semantic retrieval benchmarks](benchmarks/semantic/README.md) retain the
complete evaluation data and reproduction commands.

Useful flags:

| Flag | Effect |
| --- | --- |
| `-h, --help` | Show all command-line options and examples. |
| `-p, --prompt` | Supply a one-shot task instead of positional task text. |
| `-m, --model` | Select and remember a model slug; start in manual mode. |
| `-w, --workspace` | Project root and default path boundary. |
| `--danger` | Disable workspace path confinement at startup, including unattended one-shot tasks; no confirmation prompt. |
| `--max-steps` | Cap working model requests per run, including response retries (default 200); background summaries are separate. |
| `--context-steps` | Deprecated compatibility option; selection now targets the newest third of retained posts in full. |
| `--context-tokens` | Optional total working-context cap, including output reserve; at least 4096, bounded by endpoint limits. Archived originals are preserved. |
| `--reasoning-history-steps` | Archived thoughts from 0–25 recent steps (default 25); 0 supplies outcomes without archived thoughts. Does not disable current-turn reasoning. |
| `--raw-reasoning-history` | Use original thoughts instead of filtered thoughts for controlled comparisons. Compression still runs. |
| `--planning` / `--no-planning` | Planning is enabled by default. The latest successful plan is injected on subsequent turns and recovered from journaled results on resume. Use `--no-planning` to start with it disabled. |
| `--tools` | Explicit comma-separated tool allowlist, useful for focused tasks or large MCP installations. History retrieval and enabled planning remain available. Unlisted calls are rejected. |
| `--python` | Explicit project interpreter, overriding project-file settings and discovery. |
| `--temperature`, `--max-tokens` | Explicit temperature and output limits; temperature applies only to working-model requests. |
| `-v, --verbose` | Show full tool output and per-step token usage. |
| `--provider` | Select `openrouter`, `nvidia`, `kilo`, `mistral`, or `google`; start in manual mode. |
| `--list-models` | Print catalogs for providers with configured API keys and qualified model selectors. |
| `--api-key` | Override the API key for this process; an environment variable avoids putting it in shell history. |
| `--base-url` | Override the API base URL, for example for a local test gateway. |
| `--no-color` | Disable ANSI color output. |
| `--no-mcp` | Skip globally configured MCP servers. |
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
The default feed is white on black. Teal marks user input and success, violet
marks tools and panel titles, cyan marks navigation, amber marks activity and
warnings, and coral marks errors. Menus, the footer, and the editor use distinct
dark backgrounds. Reasoning stays grey, with a subtle background on inline code.
`/markdown [source|rendered]` toggles the view. `/copy` copies the latest reply,
automatically choosing exact code when it contains one code block; source view
copies Markdown. `/copy text`, `/copy markdown`, and `/copy code [number]`
choose explicitly. F2 toggles the view and F3 copies the latest reply. Dragging
selects feed text; right-click the selection to copy it without wrap decorations.
Clipboard access uses an installed local utility or the terminal's OSC 52 support.

The renderer updates incoming Markdown chunks as they arrive, including provisional
text decoded from JSON response strings. Tool calls remain buffered until the
complete response is validated; previews do not authorize tool execution.

## Headless Automation and Evaluations

One-shot runs can stream machine-readable JSON Lines instead of rendered answers:

```bash
slipagent --json-events --prompt-file task.txt --workspace /path/to/trial --no-reload > events.jsonl
```

`--prompt-file PATH` reads a UTF-8 prompt, preserving its line endings and trailing
whitespace. Use `--prompt-file -` to read UTF-8 from stdin. It cannot be combined
with `-p` or positional prompt text. `--json-events` also works with the existing
`-p` and positional forms; it requires a one-shot task and cannot list models or
start the interactive terminal. Without this flag, normal output is unchanged.

Each stdout line is one complete JSON object, flushed immediately. Human progress
and diagnostics remain on stderr. Records share `schema_version` (currently 1),
`record_type`, a sequential `sequence`, and `elapsed_seconds`:

| Record Type | Contents |
| --- | --- |
| `run_start` | Selected model/provider, workspace, and session ID |
| `event` | Agent event `kind` and step, text, structured tool call/result, or usage as applicable |
| `run_end` | Completion status, exit code, exact answer, error if any, working request count, and session usage |

Context events carry the outgoing request as a `request` **object**, rather than
a JSON string inside JSON. Tool arguments and result objects retain their data
types. Match tool observations by their step and call ID. Per-token reasoning and
reply fragments and incremental command-output chunks are omitted; completed
reply/tool events, request snapshots, and the existing saved sessions retain the
corresponding observations. This avoids enormous fragment-by-fragment logs.

`run_end.status` is `completed`, `step_limit`, `stopped`, or `error`. Completed
runs exit 0; execution errors, step limits, and early stops exit 1. Setup failures
exit 2 and emit an error `run_end` with `phase="setup"`; no `run_start` is required
in that case. Argument or prompt-file errors use argparse's stderr diagnostics
and exit 2 without starting a run. A crash or externally terminated process may
leave no final record: treat a missing `run_end` as incomplete.

**Completed means the harness finished, not that its answer or code was correct.**
Tool errors stay labelled in their event even if the model recovers. Grade the
resulting files and behavior independently. Session usage includes background
compression; `requests` counts working requests, including retries.

Automation mode does not remember its selected model in your preferences. Session
persistence, providers, tool execution, context assembly, and compaction use the
same path as normal one-shot runs. MCP remains opt-in with `--mcp`. Use
`--no-reload` when comparing a fixed code version and a separate
`SLIPAGENT_STATE_DIR` for trial records. These records contain private project
text and complete model requests; keep them out of public commits.

### Repeated Trials

Use a fresh workspace and state directory per trial. The following Python example
copies a fixture, invokes the installed harness three times, and retains results
without parsing terminal text. It uses your normal configured provider and may
consume API quota or credits:

```python
import json
import os
from pathlib import Path
import shutil
import subprocess

fixture = Path("fixtures/coding-task").resolve()
prompt = Path("task.txt").read_text(encoding="utf-8")
records = Path("agents/trials").resolve()
records.mkdir(parents=True, exist_ok=False)
for number in range(1, 4):
    trial = records / str(number)
    workspace = trial / "workspace"
    shutil.copytree(fixture, workspace)
    # Set provider credentials in the environment for isolated state directories.
    environment = {**os.environ, "SLIPAGENT_STATE_DIR": str(trial / "state")}
    with (trial / "events.jsonl").open("w", encoding="utf-8") as events, \
         (trial / "stderr.txt").open("w", encoding="utf-8") as errors:
        process = subprocess.run(
            ["slipagent", "--json-events", "--no-reload", "--no-mcp",
             "--workspace", str(workspace), "--max-steps", "20", "-p", prompt],
            stdout=events, stderr=errors, env=environment,
        )
    outcome = {"process_exit_code": process.returncode}
    (trial / "process.json").write_text(json.dumps(outcome, indent=2) + "\n")
    # Run independent behavior checks against workspace here.
```

Use small fixture directories with ordinary files, not a working installation,
symlinks to shared writable files, or a copied virtual environment. Preserve the
source version, prompt, model ID, sampling settings, grading command, and limits
with your evaluation records. Compare several trials per condition; keep
infrastructure failures separate from incorrect model behavior.

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

## Tools

The model initially gets complete definitions for `read_file`, `write_file`,
`list_dir`, `grep`, `edit_file`, `run_command`, and `load_tools`. Other tools appear
in a compact names-and-purpose catalog. The model calls `load_tools` with selected
names to obtain their full descriptions and argument schemas; those definitions
join subsequent system prompts and API tool lists for the conversation.
Only tools whose full definitions are supplied in the current request can run.
If a response calls an unavailable tool, the entire batch returns errors without
executing any calls. Load deferred tools in a separate response before using them.
System definitions separate core and loaded tools, keyed by name in indented
JSON. Harness-generated input JSON also uses indentation; file contents and
command-output strings retain their original text.
Repeated loads are safe. New conversations clear loaded definitions; resumed
conversations restore tools loaded or called in their saved history. Explicit
tool allowlists and current MCP availability still apply.

The complete built-in tool set is:

| Tool | Purpose |
| --- | --- |
| `load_tools` | Load full definitions for selected catalog names without executing them. |
| `answer` | Display `text`; end the run when called without other tools, or continue after a mixed tool batch. Also accepts `<tool_call><answer>text</answer></tool_call>`. |
| `read_file` | Read a whole UTF-8 file up to 32 MB with numbered lines; optional offset/limit ranges, without a line cap. Model context limits still apply. |
| `write_file` | Create or overwrite a file, making parent directories. |
| `copy_path` | Copy a file or directory tree to a new exact destination, including binary files. |
| `move_path` | Move or rename a file or directory tree on the same filesystem. |
| `delete_path` | Delete a file or empty directory; nonempty directories require `recursive=true`. |
| `create_directory` | Create a directory, optionally including missing parents. |
| `create_python_environment` | Create and select a Python venv when none is selected; repeated calls reuse the selected interpreter and never recreate existing directories. |
| `edit_file` | Replace one exact string or an `edits` array against one original file. All matches must validate before an atomic write; returns a bounded diff and helpful match diagnostics. |
| `read_hashlines` (hashline enabled) | Read a bounded page of UTF-8 lines with copyable anchors and a complete file revision. |
| `edit_hashlines` (hashline enabled) | Replace/delete anchored ranges or insert lines; validates the revision and every edit before one atomic write, with normal checkpoints and batch checks. |
| `grep` | Regex search across file contents, with an optional glob filter and a 120-second timeout. |
| `semantic_search` | Deferred local search by behavior or purpose when the repository map is insufficient; returns complete indexed excerpts with paths and line numbers. |
| `glob` | Find files and directories by name pattern. |
| `list_dir` | List one directory, directories first. |
| `run_command` | Run a shell command and capture output; `background=true` returns a managed job ID. |
| `command_jobs` | List, inspect, wait for, or stop managed background commands. |
| `monitor_command` | Watch background output or completion; deliver results and their initialization turn directly in request prompts. |
| `read_command_output` | Page retained stdout/stderr or read its tail by log ID; list logs by step/call ID. |
| `read_artifact` | Read a bounded character page of a saved inspection record by numeric `record_id`; select `field`, `offset`, and `limit`. Omit the ID for a paginated record list. |
| `git_status` | Show the branch and short repository status. |
| `git_diff` | Show unstaged changes, or staged changes with `staged=true`; optionally select literal `paths`. |
| `git_log` | Show recent commit hashes and subjects (`limit=10`, maximum 100,000). |
| `git_add` | Stage explicit workspace-relative `paths`, including deletions; `["."]` stages all files. |
| `git_commit` | Commit already staged changes with a nonempty `message`. |
| `web_search` | Search the web via Exa. Returns titles, URLs, and provider-supplied page text. |
| `fetch_page` | Fetch a URL and return readable text (scripts stripped). |
| `http_fetch` | Send HTTP requests with methods, headers, query parameters, and text or JSON bodies; return status, headers, and raw body. |
| `recall_history` | Search history or retrieve a complete original step or selected fields, without character limits or paging. |
| `update_plan` (planning enabled) | Save an objective and 1–8 stages, with at most one in progress; the latest successful plan is supplied with its last-updated step. |
| `navigate_code` (configured projects only) | Definitions, references, implementations, and hover via a local language server. |

### Optional Anchored Editing

The deferred `read_hashlines` and `edit_hashlines` tools provide an alternative to
exact-string editing. Load their definitions in a separate response, then copy
the revision and line anchors returned by `read_hashlines`; the model does not
calculate hashes. Each edit uses original-file anchors, including when a batch
inserts or deletes earlier lines. Any changed file revision or invalid/overlapping
edit rejects the whole edit call. Reread after external changes; moved blocks are
not matched approximately. Successful edits return fresh anchors near the change.

Use `/hashline off` or `--no-hashline` to remove both tools, their catalog entries,
and their instructions. `/hashline on` enables them again for the current session.
Their full definitions are delivered with the next request, rather than duplicated
in historical `load_tools` results, so disabled feature guidance does not persist
there. `read_file` and `edit_file` remain available in either mode.

### Data Size Limits

Tools return large results directly; request assembly applies the selected
model's context budget. Explicitly requested ranges still select only those ranges.

| Data | Limit |
| --- | --- |
| Text files read or searched; navigation source files | 32 MB per file |
| Built-in file reads and exact-edit input | 32 MB; regular UTF-8 files only |
| Command stdout/stderr, saved-output reads, edit diffs | 32 million characters per stream or result |
| History recall | Complete selected object, without character paging |
| Grep | 100,000 matches with complete lines |
| Glob and directory listings | 100,000 entries |
| Git log | Up to 100,000 commits; default 10 |
| Web fetch | 32 MB downloaded; 32 million returned characters |
| Web search | Up to 100 results, subject to provider limits; returned text is not clipped locally |
| Navigation results and hover text | Complete results within the transport limit |
| MCP and language-server messages | 256 MB per incoming protocol message |
| Saved command logs | 1 GiB per session by default; explicit project settings take precedence |

Repository orientation accepts source files up to 32 MB and up to 1,000
declarations per file. Its automatic overview remains bounded by approximately
3% of the context allowance and 128,000 characters; it does not replace source
reads. Compressed summaries and diagnostic/UI previews retain their separate
limits; they do not cap file-tool results or complete history retrieval.

### Background Commands

Pass `{"command":"your test command","background":true,"timeout":120}` to
`run_command`. It returns a `job_id` and `log_id`. Use `command_jobs` with
`{"action":"wait","job_id":"ID_FROM_RESULT","timeout":10}` to wait briefly,
`action="status"` to inspect, or `action="stop"` to kill the process group.
`action="list"` lists all jobs from the optional `offset` onward.
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

### Command Event Monitoring

The optional `monitor_command` tool watches existing managed background jobs.
The model receives setup instructions when no monitors are initialized: load the
tool with `load_tools`, start a background command, then initialize a watch using
its returned job ID. For example,
`{"action":"start","job_id":"ID_FROM_RESULT","pattern":"ready","timeout":120}`
watches a case-sensitive literal substring; omit `pattern` to watch completion.
`stream` selects `stdout`, `stderr`, or `both` (default). Patterns contain 1–256
characters and match across chunks, but never across streams. Initialization
replays up to the last 65,536 retained bytes per stream, reporting earlier omissions.

Active watches appear in an additional prompt section. When ready, another
section delivers the result itself, identifies the initialization turn, and
includes the outcome, exit status, latest matching excerpt, counts, and up to
1,000 observed characters from each output stream. Omitted text and storage loss
are explicit. The model need not spend a call retrieving the supplied result;
full retained logs remain available if it needs more detail. Results also become
saved inspection artifacts, subject to the existing archive quota.

The default watch completes on the first match. `repeat=true` watches repeated
matches until the job ends or the watch expires. Intermediate notices are limited
to one per five seconds per monitor, with suppressed counts; raw logs remain
independent of notification suppression. Completion is always announced.
`action="wait"` waits up to 30 seconds without stopping anything. Watches default
to 30 seconds and can last up to 600; they do not extend the command's timeout.
`action="cancel"` ends only the watch. `action="acknowledge"` removes a ready
result from live reminders without deleting logs or saved artifacts. At most
eight active or unacknowledged watches are retained; capacity errors never evict
unacknowledged results. `action="list"` returns the retained watches.

Monitoring never starts a model request or resumes a finished agent. Results
arrive in its next normal request. Compatible reloads and `/stop` preserve watches;
session replacement and shutdown clear them after stopping the jobs. Restarting
does not recreate subscriptions. Monitor instructions are omitted when the tool
is unavailable under the current tool restrictions.

### Optional Language Servers

Configure an already installed **stdio** language server in
`language-servers.json` in the SlipAgent data directory (for example
`~/.local/share/slipagent/language-servers.json` on Linux). Repositories cannot
name commands for SlipAgent to start, so server commands are only read from this
user-level file:

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
character occupies two UTF-16 units. Returns all locations from `offset` onward
and complete hover text. Source files must be UTF-8 and at most 32 MB.
Queries time out after 30 seconds, with bounded startup/shutdown and
pipe writes. Unsupported operations and missing executables return tool errors.

Each query supplies current file contents and reports workspace file changes
since the prior query; generated/dependency trees follow the normal search
exclusions. Results outside the workspace and configured system temp directory
are counted and omitted unless danger mode permits them. Server
processes have the same OS permissions as shell/MCP processes. Navigation does
not apply edits requested by a server. Ordinary search/read tools remain useful
when the language server cannot resolve a symbol.

### Retrieving Command Output

Every shell/Git subprocess in the default registry gets a unique log ID. The tool
observation reports that ID and concrete retrieval arguments. A tool that runs
several Git subprocesses can produce several logs; list with `step_id` and
`call_id` to find them all. Call IDs are unique within a step, not the whole session.

```json
{"log_id": "ID_FROM_RESULT", "stream": "stdout"}
```

Pass that object to `read_command_output`, then use the returned `next_offset`
for the next page. Stream offsets count **UTF-8 bytes**; limits count **characters**
and default to a maximum of 32,000,000. Use `stream="stderr"` for errors or `tail=true` for the
last `limit` characters. Invalid process bytes are decoded with replacement before
storage; this is a text log, not a binary artifact archive.

Omit `log_id` to list logs. For listings, `offset` is a record index and `limit`
is a record count; commands are shown in full. Each log
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

`http_fetch` uses the same public HTTP(S) destination checks, preserves raw
response text, and returns binary data as `body_base64` when it cannot be decoded.
It returns redirects without following them and preserves error response bodies.
Downloads are limited to 32 MB; the request timeout defaults to 120 seconds
(maximum 600). It uses the existing HTTP client and does not require cURL.

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
appear exactly once, unless `replace_all` is set. LF targets also match CRLF or
CR line endings. A mismatch returns an error; matching alone does not prove
the model read the file or that its replacement is correct.

**Filesystem tool paths allow the workspace and system temporary directory by default.**
The system temp path is selected by Python's `tempfile.gettempdir()` from the
system configuration and supplied in the model prompt. Relative paths remain
anchored to the project; use absolute paths for temporary files.
Paths resolve through a symlink-aware check that
rejects anything landing outside these directories, so `../../.ssh/config`
is rejected by file, navigation, and search tools unless `/danger` or `--danger`
is active. `run_command` and MCP servers
run with the harness process's permissions and can access paths outside the
workspace. The workspace check is not an operating-system sandbox.

## Shell File Tracking

In the interactive feed, commands show their status and a link to inspect saved
output. Stdout and stderr stay in the output viewer and model history instead of
being printed into the feed. One-shot command-line output remains available.

The inspection panel tracks literal shell file references as **inferred** reads,
writes, or deletions. It recognizes common file and search commands, output
options, wrappers, redirections, and literal working-directory changes.
The call viewer includes the analysis and any incomplete-tracking reasons.
Variables, globs, substitutions, unsupported options, scripts, and arbitrary
programs can hide additional effects. These references are neither an execution
trace nor confirmation that a file changed. Tracking does not restrict shell access.

## Troubleshooting

| Symptom | Where to Check |
| --- | --- |
| `slipagent` is unavailable after installation | Open a new terminal; check the printed launcher directory and user PATH. Use the installer's `--dry-run` to inspect destinations. |
| Edits to the harness have no effect | Confirm you edited the installed `source/` checkout. Inspect `/generations` and reload messages; stable-contract changes need a restart. |
| Agent pauses when a file opens | Editable files hold agent work; save or discard edits and close the editor. Read-only output/diff viewers allow work to continue. |
| Directory names are not clickable | Check desktop-session/opener availability or configure a folder opener through Options. |
| A model or provider is absent | Check that provider's credentials and `/models` filters. Catalog errors and unsupported capability profiles are reported separately. |
| The agent misreads a result or repeats an action | Inspect the outgoing context with Ctrl+\ and the exact attempt with `/requests`; use the saved call/output viewers to compare observations. |
| Old sessions or MCP servers disappeared after a move | Check the active platform data directory and `SLIPAGENT_STATE_DIR`. Project `.mcp.json` is not loaded; global connections use the data directory's `mcp.json`. |
| An interrupted tool may have changed a file | Inspect the file and saved results before retrying. Resume never replays uncertain tools automatically. |
| Updating warns about local edits | Review every listed path and preserve the changes you want before confirming. A missing hash baseline is not proof of an unmodified installation. |

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
a local fake API and prints a captured plain-text transcript. Its piped input
does not exercise interactive redraws or paced typing.
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
