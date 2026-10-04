# Contributing to SlipAgent

This guide is for people and coding agents modifying the harness. Read the
workspace's applicable `AGENTS.md`, `CLAUDE.md`, and scoped instruction files
first. The [README](README.md) covers running the application;
[Internal Contracts](docs/architecture.md) explains how its parts fit together.

## Start With the Correct Environment

Use an editable installation in a project virtual environment. If one already
exists, inspect and use it; do not replace it or install into the global Python.
For a new checkout without an environment, create one explicitly:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -c "import sys; print(sys.executable); print(sys.prefix)"
```

On Windows the executable is `.venv\Scripts\python.exe`. Commands below use the
POSIX path. Starting SlipAgent through its own venv does not activate a different
project's environment. Use that project's explicit interpreter or `--python`.

For an existing environment, first check `.venv/bin/python -m pip --version`.
A venv created by uv may intentionally omit pip. If pip is missing and uv is
available, install into that exact environment with:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

uv does not require pip inside the target venv; see its
[environment targeting documentation](https://docs.astral.sh/uv/pip/environments/).
If uv is unavailable, check `.venv/bin/python -m ensurepip --version`. When that
succeeds, `.venv/bin/python -m ensurepip --upgrade` bootstraps pip into this venv
without downloading it, after which the pip install command above can run.
[ensurepip is optional](https://docs.python.org/3/library/ensurepip.html); if it is
missing or bootstrap fails, report that failure and determine the project's
environment-management setup. Do not recreate the venv or substitute global pip.

Tests use stub models, mock transports, loopback HTTP servers, temporary Git
repositories, and disposable source copies. They need no real API key. Set
`SLIPAGENT_NO_DOTENV=1` for manual offline experiments so real configuration is
not loaded accidentally. Never point test tools at working files they might
overwrite; give them temporary workspaces. Do not run the live REPL driver as
an offline smoke test.

## Find the Owner of the Change

The [module responsibility map in AGENTS.md](AGENTS.md#module-responsibilities)
is the shared index for humans and agents. It is kept in the project instruction
file so the model receives it before choosing which source files to inspect.

Read callers and tests before editing a contract. The model only knows the
instructions, definitions, history, and observations in the outgoing request.
A terminal notice, an archived post, a local variable, or an earlier API call
does not automatically become model knowledge.

## Make a Complete Change

1. Reproduce the behavior with a small, isolated test where practical. Assert
   observable results: whether a file survives failure, which tools execute,
   what the model receives, or what the user sees.
2. Change the component that owns the behavior. Keep local paths, call IDs,
   history post IDs, and provider parameters explicit rather than inferring them
   from a rendered string when structured state exists.
3. Spell out model-facing contracts. Explain every required field, whether an
   outcome is known or pending, what was omitted, and the exact recovery call.
   Include an example that passes the actual parser. Free models should not
   need to infer hidden state or remember a rejected response.
4. Preserve failure recovery. A failed operation may have partial effects;
   return them with an error status. Cancellation must release owned resources
   and keep assistant/tool history paired. Validate responses before displaying
   reply text or invoking tools.
5. Exercise the appropriate checks, then the full suite before presenting a
   complete change. Update affected commands, examples, help, and documentation
   with the implementation rather than leaving incompatible instructions behind.

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test-active-task.py
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m mypy
```

The focused file above is an example; select tests for the behavior changed.
Use `tests/test-runtime.py` for live-state changes, `test_cli_e2e.py` for command
flow, and the terminal tests for rendering and actual input-buffer behavior.
New response fields require current fixtures in the stub client, HTTP tests,
streaming tests, reload subprocesses, and offline demos. Do not loosen production
validation just to accommodate an obsolete fixture.

## Work on the Running Harness

Compatible edits are applied as a complete source generation between model/tool
batches. Classes keep their identity while their behavior is replaced. Built-in
tools are rebuilt; their session-owned environment and log services are borrowed
from the existing registry. MCP tools retain their connections. A rejected edit
stays on disk but does not become active behavior.

Use relative package imports in components. Imports must define classes/functions
without opening network connections, starting background tasks, or mutating the
workspace: the loader imports candidates before deciding whether to accept them.
Keep resources in explicit owners with cleanup methods. A reload candidate must
not close services borrowed from the live session on rejection or retirement.
Use `Lifetime` for owned tasks and cleanup callbacks; cancellation of a close
waiter must not abandon cleanup. Declare tools through the default registry so
its explicit built-in ownership survives reloads, regardless of module location.
Test an active background operation across both accepted and rejected reloads.

Changes to stable core modules, inheritance, slots, dataclass fields, or removed
classes require a restart. Editing a constructor does not initialize new attributes
on old persistent objects. For added state, choose an intentional migration or a
restart; do not scatter silent fallbacks that conceal an incompatible layout.
See the exact [reload contract](docs/architecture.md#reload-transactions).

For a self-edit, check both the resulting files and the next request's runtime
diagnostic. `/generations` and Ctrl+backslash expose the active generation/context.
Passing tests does not mean a running session has accepted the edit, and a successful
reload does not mean its new behavior is correct. Test rejected reloads and a later
valid recovery using disposable package copies when modifying this machinery.

## Preserve the Model's Information

- Store accepted originals independently from the context view. Compression,
  excerpts, and window selection must not alter the archive.
- Preserve durable message/tool-start ordering and resume without replay. Use
  disposable `SLIPAGENT_STATE_DIR` storage in every CLI/subprocess test. Check
  torn final records, rejected corruption, interrupted calls, summary callbacks,
  queued inputs, task boundaries, and retained command output.
- Keep repository maps bounded and optional. Post-edit checks inspect final batch
  state, use the selected project environment, and report skipped/failed checks
  to the model without implying that edits themselves failed.
- Store the original prompt, reply, tool calls, and results as independently
  retrievable parts. Recent turns use all originals; older turns use only one
  whole-turn summary. Shrink the window by whole turns when needed.
- Summarize each completed turn in a separate background request containing only
  those four parts. Never include the thread or project/task state. Test reset,
  cancellation, late results, and model/profile changes. Summary completion must
  not hide tool results before the working model receives them.
- Allow ordinary replies and null-content native calls without schema support.
  The optional strict schema contains only reply text and embedded calls where
  needed. Memory is not a required response envelope.
- Preserve the active goal and constraints independently of short follow-ups.
  Missing source instructions must be supplied before actions proceed; an
  excerpt or request to retrieve them does not prove the model has read them.
- Keep every observation's success/error status in model context. Report lost
  output honestly and distinguish a preview from retained data that can be paged.
- Supply reload failures, project environment selection, and scoped MCP guidance
  explicitly. Do not depend on the model seeing a terminal-only diagnostic.
- Project instruction previews do not establish delivery. Preserve the exact
  request snapshot used by the pre-edit guard; never mark newly read guidance
  delivered merely because a file tool discovered it during the same batch.
- Parallel tool safety is opt-in per argument set. Preserve exclusive barriers,
  ordered observations, journal-before-dispatch, and cancellation cleanup.
- Keep failed attempts in request diagnostics, outside accepted history. Working
  retries share the step budget and remain interruptible during backoff.
- Background jobs and optional language servers belong to session services.
  Never install a server implicitly or relaunch an archived job on resume.
- Normalize alternate response formats before validation. Keep native tool IDs,
  batch order, exact arguments, and matching observations intact. Duplicate
  representations must not execute twice; conflicting batches require a retry.
  Extend parser tests with complete wire responses, streamed fragments, and
  malformed cases before adding another format. Do not infer actions from prose.
- Fetch protocol capabilities at startup and explicit model selection, cache
  between requests, and preserve the working model/profile when selection fails.
  Exercise both native calls and the JSON-only fallback when changing history.

## Document for the Next Contributor

Comments should explain reasons, ownership, invariants, ordering, and non-obvious
failure cases close to the code that enforces them. Docstrings should make public
entry points usable without reverse-engineering their callers. Avoid line-by-line
narration, unexplained compatibility branches, or claims stronger than the code.

Keep the README oriented toward users: setup, choices, commands, model behavior,
storage lifetime, and recovery. Put deeper lifecycle and extension contracts in
the architecture document, and link them from the README. Update examples when
schemas change and verify them against the parser. Numeric limits need units,
defaults, scope, and what happens on exhaustion. Treat project instruction files
as user-controlled guidance; do not replace custom content to synchronize a guide.

Audit reports and scratch notes belong in `agents/` when that directory is used
by the project. They complement maintained documentation and should clearly label
historical findings versus current behavior. Keep real credentials, local project
settings, generated logs, and temporary workspaces out of commits.
