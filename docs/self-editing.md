# Editing the Running Harness

[Back to the README](../README.md)

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
[contributor guide](../CONTRIBUTING.md) and [internal contracts](architecture.md)
for the implementation workflow, state ownership, and failure behavior.
