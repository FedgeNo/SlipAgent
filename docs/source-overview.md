# Source Overview

[Back to the README](../README.md)

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
