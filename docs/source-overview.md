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
├── nvidia.py         NVIDIA hosted endpoint and reasoning controls
├── nvidiaprofiles.py reviewed NVIDIA model limits and parameters
├── google.py         Google's compatible API with catalog token limits
├── mistral.py        Mistral chat completions and model capabilities
├── groq.py           Groq chat completions and live catalog capabilities
├── kilo.py           Kilo Gateway and live catalog capabilities
├── capabilities.py   endpoint selection, native tools, JSON/reasoning, limits
├── terminal.py       fixed footer, scrolling transcript, menus, and panels
├── terminallayout.py layout assembly around retained panels and input
├── terminalbindings.py keyboard commands and focus filters
├── terminalmouse.py  mouse routing between feed, input, editor, menus, inspector
├── statusline.py     status priority, expiring notices, rotating tips
├── palette.py        shared terminal colors and ANSI foreground codes
├── transcript.py     file-backed transcript and incremental word wrapping
├── instructions.py   project guidance discovery before model activity
├── config.py         configuration resolution
├── workspace.py      symlink-aware path boundary
├── types.py          shared messages, tool calls, usage, wire types
├── mcp.py            persistent stdio MCP connections
├── plugins.py        user method overrides, fingerprints, and approvals
├── plugintools.py    /plugin copy, diff, and restamp
├── updater.py        --update: download the latest release and run its installer
├── semantic/         local embedding index, watcher, and retrieval
└── tools/
    ├── base.py       tool contract, argument validation, registry
    ├── answer.py     user-visible answers and answer-only completion
    ├── planning.py   bounded working-plan updates
    ├── blocking.py   owned worker threads for synchronous filesystem operations
    ├── files.py      read_file, write_file, edit_file
    ├── hashline.py   optional read_hashlines and edit_hashlines
    ├── paths.py      copy_path, move_path, delete_path, create_directory
    ├── editing.py    atomic replacement planning, diagnostics, bounded diffs
    ├── search.py     grep, glob
    ├── grep.py       isolated, killable regex worker using a source snapshot
    ├── navigate.py   list_dir
    ├── shell.py      run_command
    ├── output.py     session log quota, storage, and read_command_output
    ├── git.py        confined status, diff, log, staging, commits
    ├── python.py     create_python_environment
    ├── semantic.py   semantic_search over the session's index
    ├── http.py       http_fetch
    └── web.py        web_search, fetch_page
```

The tree above shows the central modules. Smaller components now own the
request pipeline and interactive inspection features:

| Responsibility | Modules in `src/slipagent/` |
| --- | --- |
| Request assembly and recovery | `contextbuilder.py`, `requests.py`, `toolbatch.py`, `contextlegacy.py`, `requestbody.py`, `requestfailures.py`, `responsecapture.py`, `upload.py`, `toolcompat.py`, `thoughts.py`, `calllabels.py` |
| Model routing, pricing, and scores | `modelrouting.py`, `compressionproviders.py`, `modelpricing.py`, `modelscores.py`, `startupcache.py` |
| History summaries and turn storage | `historymerge.py`, `turnstore.py` |
| User input awaiting dispatch | `submissions.py`, `inputqueue.py`, `diffcomments.py` |
| Command-output monitoring | `monitoring.py` |
| Headless JSON Lines output | `automation.py` |
| Session state and owned resources | `sessionstate.py`, `resources.py`, `work.py`, `events.py` |
| Slash-command behavior and help | `commands.py` |
| Saved inspection records | `artifacts.py`, `inspection.py`, `payloads.py`, `contextview.py`, `rowindex.py` |
| File-backed editing and coordination | `editor.py`, `editorbuffer.py`, `editorcontrol.py`, `editorui.py`, `useredits.py`, `editdelivery.py`, `fileio.py` |
| Read-only text and diff viewers | `viewers.py`, `viewerpanels.py` |
| Reusable lists, links, focus, and layout | `listpanels.py`, `links.py`, `focus.py`, `uiwindows.py`, `mainarea.py`, `menulifecycle.py`, `transcriptlayouts.py` |
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
