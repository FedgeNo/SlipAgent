# Browser Workspace

The browser workspace shares the CLI's running agent, tools, original history and
saved sessions. Start SlipAgent with a browser link:

```bash
slipagent --gui
```

Or enter `/gui` in an existing session, including while the agent is working.
Open the complete printed link. Repeating `/gui` reuses the host and prints its
link again. `--gui-port 8765` selects a fixed port; otherwise an available port
is chosen. `--gui` accompanies an interactive session, not a one-shot prompt.
The CLI session must stay open; quitting it shuts down the browser host too.

The left column contains the conversation and prompt. The other columns explore
collections and their children, or show the selected original content. Closing
the browser leaves the agent running. Refreshing reconnects to the same session
and reloads its retained conversation.

## Navigate The Hierarchy

Choose History, Sessions, Files, Plan, Tools, Models, Skills, Outputs, Restore or
Settings in the navigation rail.

- Click an item in the middle column to see its children in the right column.
- Double-click a collection, or use **Focus**, to make it the browsing root.
  This exposes another level of descendants; nesting has no fixed column depth.
- **Up** moves one parent level. Previously visited collections restore their
  selection, page and scroll position. Breadcrumbs jump to ancestors.
- **Back** follows navigation history, separately from the parent relationship.
- Arrow keys select list items; Alt+Enter or Right opens their collection.

Collections page their rows. Original text has Previous/Next controls for reading
all of it, including long tool results. Feed cards link to the corresponding
message and tool-call objects for further inspection.

Assistant cards show the recorded API provider and model beneath their heading.
Inspect exposes the same attribution under `response_source` for messages and
turns. Changing models does not relabel older responses. Older records display
only the attribution they actually contain; an unknown provider stays unknown.

History objects offer **Ask About This Step**, **Ask About This Call**,
**Ask About This Result**, or **Ask About This** for individual fields. The action
focuses the prompt box and adds a removable reference while preserving your draft.
Write your question and send it normally; during a run it joins the same queue.
The model receives the question and an indexed `recall_history` reference, rather
than a copy of a potentially large result. Chat shows your question with an
“About…” label; Inspect preserves the full reference. Changing sessions clears
the attachment and keeps your draft so it cannot target unrelated history.

## Work With The Agent

Send a prompt with Enter; Shift+Enter inserts a newline. During a run, **Queue**
adds a correction to the existing input queue. **View Full Queue** opens the
pending prompts as browsable objects. A queued prompt can be discarded before it
is submitted. Browser and terminal input use the same turn driver.

**Stop After Turn** finishes the current step and its tools. **Interrupt** cancels
active work immediately while retaining completed results. The feed follows new
activity only near its bottom; **Latest** returns to the newest messages.

The conversation shows ordinary prompts and replies, with compact tool names and
completion/error indicators. Tool arguments and raw results are available through
Inspect and the explorer, rather than printed as JSON in chat. Live reasoning
appears under **Thinking**, alongside streamed reply text. The live preview follows
new output while you are at its bottom; scroll up to pause following. Full saved
reasoning remains available through Inspect after the turn completes.

History exposes the original stored prompts, responses, reasoning, calls, results
and edit notifications. GUI browsing does not use the model's context window,
compressed summaries or recent-turn selection. Paging and a bounded browser cache
control memory usage; older originals remain retrievable in full.

Press **F3 to search**. The prompt area becomes a search field and results appear
in the conversation area. Matching text is highlighted near the center of a
single-line excerpt; long excerpts are clipped with an ellipsis. Click a result
to jump to its message. Search remains available while the agent runs. Escape
returns to the draft prompt. Search pages through originals, with one matching
excerpt per message.

## Sessions, Files And Controls

Sessions can be renamed, resumed, forked or deleted. Resuming a saved session
opens its original history through the existing journal loader without replaying
tools. The active session's collection contains its turns, messages, outputs and
checkpoints. Session changes wait for idle state; close an open editor first.

Select a file, then **View** or **Edit**. Editing uses the same file editor and
work-boundary hold as the CLI: it opens after the current turn, then the agent
waits until the editor closes. Large files have editor pages. Save File or Ctrl+S
saves staged pages, checks for changes on disk and records the diff for the
model. Close offers to discard unsaved edits. Disconnecting the owning tab closes
its editor and releases the agent; saved edits remain recorded.

Plan visualizes the existing planning tool's objective, stages, statuses and
evidence. Models groups choices by provider, with pricing and context metadata;
choose a model or enable Optimal Free Selection. Skills selects existing pinned
instruction packages. Outputs pages retained output/diff records. Restore browses
file checkpoints and confirms restoration through the chosen checkpoint. Settings
controls planning, overthinking, hashline editing and workspace confinement.
Provider-key setup, plugin management and advanced CLI diagnostics remain in the
terminal interface.

## SSH Forwarding

The host listens only on `127.0.0.1`.

The CLI prints an SSH tunnel command with both ports already filled in whenever
it prints the browser link. Run that command on your own computer, replacing
`YOUR_SSH_HOST` with the SSH destination you normally use. Forwarding the same
local and remote port makes the exact printed browser link work locally.

To choose a fixed port, start on the remote machine with:

```bash
slipagent --gui --gui-port 8765
```

From your own computer, forward that port through SSH:

```bash
ssh -N -L 8765:127.0.0.1:8765 your-server
```

Open the printed `http://127.0.0.1:8765/#…` link in your local browser. Replace
`your-server` with your SSH host. If your local port differs, replace the port in
the browser link accordingly. Keep the fragment after `#` intact.

## Access And Lifetime

The fragment is an access capability for the session. The browser removes it
from the address bar after reading it and keeps it in tab-scoped session storage
so refreshes can reconnect. It is not sent in request URLs. Use the current
`/gui` link if access is refused after restarting the host. No public listener,
external scripts, or separate Node server is needed.

Only authenticated clients can read session state, browse originals or submit
actions. The host checks browser origins and loopback hostnames. It accepts at
most eight WebSocket connections and disconnects slow consumers instead of
holding up agent work. Reconnecting uses a fresh state snapshot and reloads
original history from its existing owners.

For implementation contracts, see [the application interface](application.md)
and [browser transport ownership](architecture.md#browser-host-and-observer-transport).
