# Editing And Extending SlipAgent

SlipAgent is an editable Python application. Its installed `source/` checkout is
the code the launcher uses, so changes there affect that installation. Humans
and coding agents use the same files, tests, and reload rules. Read the repository's
[`AGENTS.md`](../AGENTS.md), [contributor workflow](../CONTRIBUTING.md), and
[internal contracts](architecture.md) before changing the harness.

> [!WARNING]
> **A plain `pytest` run does not run every test.** Tests marked `slow` (the
> real-CLI end-to-end tests in `tests/test_cli_e2e.py`, the reload subprocess
> tests in `tests/test-retired-imports.py`, and a few long terminal, layout, and
> indexing checks) are left out by default because they take too long to run on
> every change. Run `.venv/bin/python -m pytest --slow` before committing changes
> to the CLI, terminal, transcript, reload, or session code, and before a release.

## Find The Running Checkout

A managed development install uses `python3 install.py --dev`, or
`py -3 install.py --dev` on Windows. The installer records the checkout and Python
interpreter in `installation.json`; Linux/macOS also provide a `current` shortcut.
Use the permanent checkout and its existing virtual environment, rather than
editing an unrelated download or installing into system Python. The
[installation guide](installation.md#installation-and-data-locations) lists locations.

To ask SlipAgent to edit itself, make that checkout its workspace. Danger Mode
also permits explicitly requested access from another workspace; it lifts file
confinement, not the need for a user request. Running the harness from its own
venv does not select a different project's interpreter. Project environment
selection belongs to `environment.py`.

## Choose The Owner

| Behavior | Owning Files |
| --- | --- |
| Tool execution, validation, and result contracts | `tools/base.py`, the tool implementation, `tools/__init__.py` |
| Tool instructions and argument explanations | `prompts/tools/*.md`, `prompts/tools/*-parameters.json` |
| Deferred discovery | `tools/discovery.py`, `prompts/tool-discovery.md`, request assembly in `agent.py` |
| General operating instructions | `prompts/system-prompt.md` and the resource map in `prompts/README.md` |
| Response formats and provider calls | `protocol.py`, `api.py`, provider and capability modules |
| History, original requests, and summaries | `context.py`, `records.py`, `task.py`, `compaction.py` |
| Terminal input, scrolling, and rendering | `terminal.py`, `transcript.py`, renderer in `cli.py` |
| Slash-command behavior | `commands.py`, CLI command dispatch |
| Repository orientation | `repomap.py`, `symbols.py` |
| Optional local semantic retrieval | `semantic/`, `tools/semantic.py`; see [its contracts](semantic-search.md) |
| Session resources, cancellation, and live reload | `resources.py`, `lifecycle.py`, `runtime.py` |
| Installer and release baselines | `install.py`, `pyproject.toml`, package `__init__.py`, `file-hashes.json` |

Python paths above are relative to `src/slipagent/`. Read callers and regression
tests as well as the owner. Keep development instructions in repository docs;
general model instructions must work in unrelated workspaces without this file.

## Change A Tool Completely

Every built-in tool supplies a description and a JSON argument schema. Dedicated
Markdown files document behavior, side effects, returned data, limits, and
recovery. JSON description mappings explain individual schema fields. See the
[editable prompt guide](../prompts/README.md) and [tool reference](tools.md#tools).
The Python schema remains authoritative for argument names, types, and validation;
editing a description alone does not change executable behavior.

For an existing tool, update its implementation, schema, descriptions, examples,
and affected tests together. `Tool.spec` reads editable descriptions on access.
Parameter mappings use schema paths such as `properties/query`; those paths must
exist in the Python schema. Templates preserve required placeholders and valid
JSON. Missing prompt resources and invalid templates are errors.

To add a tool:

1. Implement a `Tool` subclass with a stable `name`, argument `parameters`, and an
   asynchronous `run` returning `ToolResult.ok(...)` or `ToolResult.error(...)`.
   Ordinary failures become model-visible errors; cancellation propagates.
2. Add its documentation under `prompts/tools/`, and set `description_prompt` and
   any `parameter_prompts` mapping on the class. Put the purpose in the first prose
   line: the compact discovery catalog uses that line.
3. Register it in `build_default_registry`. `builtin_names` establishes ownership;
   module location does not. Agent-owned tools such as history retrieval are
   attached separately by their existing owners.
4. Leave uncommon tools deferred. Only names in `CORE_TOOLS` receive full initial
   definitions. The model sees other purposes in the catalog and calls
   `load_tools` before using them. Loading a definition should not run the tool,
   download a model, start a process, or mutate the workspace.
5. Test arguments, success, failure, side effects, and actual outgoing request
   definitions for both native and embedded tool-call profiles when affected.
   Include lifecycle and reload tests when the tool retains resources.

Only full definitions delivered in the current request authorize execution.
Calling an unavailable or unloaded tool rejects the whole response's tool batch,
including otherwise valid calls. A discovery preview, old model knowledge, or a
definition loaded midway through that batch does not authorize it. Allowlists
still apply after loading. Loaded definitions persist within the conversation
and restore through ordinary saved history.

Do not block the terminal event loop with local synchronous work. Use
`run_blocking` for owned thread work, or the existing subprocess ownership
machinery. Confine model-supplied paths through `Workspace.resolve`. Only declare
concurrency for audited argument sets that have no shared mutation; file writes,
external actions, and index updates remain exclusive.

## Own Persistent State

Retained resources belong to `ToolRegistry.services` or session extensions,
not globals or individual replacement tool instances. Reload candidates borrow
the live service map and must not close its owners. Resource imports and
constructors used during staging must not start background work or perform I/O.
Start expensive resources explicitly on use. Use `Lifetime` and retained cleanup
tasks so cancelled teardown still drains jobs, workers, observers, and stores.

Constructor changes do not populate old objects with new attributes. Adding
retained state requires an explicit migration or restart. Preserve existing
ownership when changing reset, resume, rewind, shutdown, or reload behavior.
Session journals record tool starts before dispatch; resume never replays a tool
whose execution outcome is uncertain.

## Understand Live Reload

Compatible Python edits stage as one candidate generation and apply between
complete model/tool batches. Existing class identities and owned services stay
alive. Compilation, imports, class layouts, required APIs, and the new registry
are checked before commit. A rejected candidate remains on disk while the last
accepted generation continues to run.

The authoritative restart boundary is `CORE_MODULES` in `runtime.py`. It includes
the stable runtime/configuration, wire/data, workspace, tool-base, MCP, lifecycle,
resource, work, and plugin-loader contracts, plus the semantic backend whose worker keeps native
model and database state. Changes to inheritance, slots, dataclass fields, or
removed classes also require restart. See the exact
[reload transaction contract](architecture.md#reload-transactions).

Editable prompt text is read at use time, including when Python live reload is
disabled. Prompt resources are also monitored by the reload frame. Saving source
or passing tests does not establish that a running session accepted a generation.
Check `/generations`, runtime diagnostics in the next request, or the context
inspector. Restart for rejected contract/layout changes once the files are valid.

## Verify Without Touching Working Data

Use this checkout's interpreter explicitly:

```bash
.venv/bin/python -m pytest -q -n 0 -p no:cacheprovider tests/test-semantic-tool.py
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m mypy
```

The first command is an example; choose a focused regression for the change.
The full suite uses up to four workers and leaves out tests marked `slow`;
add `--slow` to include them. Existing uv-created environments may
have no pip; use the environment installation steps in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#start-with-the-correct-environment).
Do not recreate an existing environment to work around a missing installer.

Tests use dummy credentials, stub models/transports, disposable files, and isolated
`SLIPAGENT_STATE_DIR` storage. Preserve `tests/offline` in child `PYTHONPATH` and
use `cli_environment()` for CLI subprocess environments. The guard blocks external
network access. Never remove it to make a test pass, use real provider credentials,
or give test edit tools working files. Real local-model evaluation is an explicit
benchmark, separate from pytest; see [semantic benchmarks](../benchmarks/semantic/README.md).

Check both the file changes and resulting behavior. For bugs, first reproduce a
meaningful failure, then verify the fix. For prompt changes, inspect what reaches
the actual request, rather than relying only on terminal output or a preview.
Keep examples aligned with the parser and report skipped checks plainly.

## Release And Update Baselines

Keep `pyproject.toml` and `src/slipagent/__init__.py` versions synchronized.
Before a push that creates new file hashes, increment the patch version before
hash generation. Before every commit, hash the exact staged contents of every
version-controlled application/supporting file with SHA-256 and update
`file-hashes.json`, keyed by checkout-relative path. Include added files, remove
deleted entries, and exclude the manifest itself, secrets, ignored local files,
Git internals, environments, and caches. Commit the manifest with the files it
describes; do not hash unrelated unstaged edits as a committed baseline.

The installer compares installed files with their installed hash baseline before
replacement. An unchanged install updates without edit-loss confirmation. Changed,
added, or deleted application files must be listed and replacement requires
confirmation. Missing or unreadable baselines are not evidence of an unmodified
installation. Updates preserve configuration, Git metadata, and the environment;
see the [installer documentation](installation.md#installation-and-data-locations).
