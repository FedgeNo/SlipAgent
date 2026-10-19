# Terminal Guide

[Back to the README](../README.md)

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
| `/danger on` | Disable workspace path confinement; queues while working |
| `/danger off` | Restore workspace path confinement; queues while working |
| `/danger` or `/danger status` | Show the current access mode |
| `/overthinking [on\|off]` | Enable recent archived thoughts, up to 25 steps, with `/overthinking` or `/overthinking on`; disable with `/overthinking off`. Enabled by default; queues while working. |
| `/planning [on\|off]` | Enable planning with `/planning` or `/planning on`; disable with `/planning off`. Enabled by default; queues while working. Disabling preserves the saved plan. |
| `/hashline [on\|off]` | Show or change optional anchored editing for this session; enabled by default, queues while working. Start disabled with `--no-hashline`. |
| `/tools` | List available tools |
| `/model` | Choose from every `/model` option; each line starts with its subcommand, such as `[provider]` or `[next]` |
| `/model <model>` | Switch to an exact model (`provider::model` when ambiguous) and remember the choice for future launches |
| `/model <words>` | Choose from models matching every word; several words always search, even when the first is a subcommand |
| `/model <provider>` | Choose from that provider's models, free models first in the free color |
| `/model provider` | Choose a provider, then one of its models |
| `/model free` | Choose from zero-cost models |
| `/model optimal`, `/model next`, `/model reset` | Automatic selection, skip to the next ranked model, or retry from the top (same as `/optimal`) |
| `/models [filter]` | Shortcut: choose from the catalog, optionally filtered |
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
| `/new` (or `/reset`) | Start a fresh conversation/task/usage/log index; retain saved sessions and current guidance/model/settings |
| `/init` | Create a project-notes scaffold in `AGENTS.md` if missing and load project guidance into this session |
| `/status` (or `/config-show`) | Show the model and optimal state, workspace, harness interpreter, and current settings without exposing credentials |
| `/settings` | Choose a setting from a menu showing each one's command and current value; the change applies like the typed command and queues while working |
| `/reload` | Reload components now, or after the current model/tool batch |
| `/generations` | Show the current component reload generation |
| `/plugins` | List each plugin, its target, and the state of every override; then allow or reject overrides, and choose one plugin or a stacking order where several override the same method |
| `/plugin copy [module.Class[.method]] [file]` | Copy a class's methods, or one method, into a plugin file with fingerprints, save the copied source as its baseline, and approve it; without a target, choose one from menus |
| `/plugin diff [file]` | Show how each application method changed since a plugin copied it |
| `/plugin restamp <file> [method]` | After review, record current fingerprints and baselines, stamping unstamped overrides |
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
`/new` starts a new conversation with the default title. With `--no-session`,
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
| Skip a silent model (optimal mode) | — | Ctrl+N while the status line shows the countdown |
| Keep waiting on a silent model | — | Esc while the countdown shows; Esc again cancels the turn |
| Return to newest feed output | Scroll to Bottom link | Scroll to the end |
| Copy or reuse selected feed text | Select, right-click, choose an action | `/copy` for the latest reply |

On an interactive terminal, seven fixed rows sit at the bottom of the screen:
a pulsing activity indicator, five rows for wrapping input,
and the readout `<cwd>  │  Model: <provider>::<model>`. Background colors distinguish this area from the scrolling transcript. Output and redraws preserve
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

The startup screen lists the available commands and basic usage. Startup reports
OpenRouter's free-call count when it is available.

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

`/danger on` enables access outside the workspace for built-in file, search,
navigation, and Git tools. `/danger off` restores confinement; a bare `/danger`
or `/danger status` reports the mode. During work, changes queue until the current response and
complete tool batch finish, keeping that batch on one access policy.
While active, the bottom readout ends with
`| Danger Mode` in red. The current mode is supplied to the model on every request.
The working directory and relative-path base stay the same. OS permissions and
the tools' validation, atomic writes, and resource limits still apply.

For an unattended one-shot task, use `slipagent --danger -p "your task"`.
The flag enables the same mode at startup without a confirmation prompt.
The setting lasts for the current process, including `/new`, `/resume`, and
component reloads; saved sessions do not enable it in a future launch. A new
process starts confined unless `--danger` is supplied.
