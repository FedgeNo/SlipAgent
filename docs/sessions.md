# Saved Sessions

[Back to the README](../README.md)

### Saved Sessions

Sessions, transcripts, command logs, artifacts, and preferences share the platform
data directory with global MCP settings. On Linux this is `$XDG_DATA_HOME/slipagent`,
defaulting to `~/.local/share/slipagent`; on macOS it is
`~/Library/Application Support/SlipAgent/data`; on Windows it is
`%LOCALAPPDATA%/SlipAgent`. Relative XDG paths are ignored. Each project gets a directory named `<md5>-<sha1>`, computed from
the same expanded absolute workspace path. Symlink spelling and case are
preserved; `project.json` retains the original path. `/sessions list` prints the
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
refreshed from the running installation. The feed shows its newest output as
soon as the saved records are read, with "Loading conversation…" on the status
line, which then gives way to the usual tips. Earlier blocks join above it,
newest first, each laid out in a worker thread and placed between redraws.
Nothing is laid out on the event loop, so keys, scrolling and streamed output
stay live throughout, and what you are reading never moves when rows arrive
above it. Type `continue` when ready; resume
does not start model requests or replay tools by itself. A started tool with
no saved result is marked as having an unknown outcome; an unstarted tool is
marked as not run. Inspect uncertain effects before retrying.

Use `/sessions` to manage saved sessions without typing their IDs. It lists them
by title and local date, marking the current one; an unnamed session shows the
start of its first prompt instead of the project path, as in `/resume`; choosing one offers Resume, Rename,
Delete, and Back. Deleting another session asks for confirmation, then removes its
journal, title, logs, diagnostics, and checkpoints; deleting the current one is
`/delete`. After renaming or deleting another session the list reopens, so several
can be cleared in turn. During work it queues until the current tool batch
finishes; the work then continues unless the menu resumed another session or
deleted the current one. Without an interactive terminal it prints the listing.

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
