# Tools

[Back to the README](../README.md)

## Tools

The model initially gets complete definitions for `read_file`, `write_file`,
`list_dir`, `grep`, `edit_file`, `run_command`, and `load_tools`. Other tools appear
in a compact names-and-purpose catalog. The model calls `load_tools` with selected
names to obtain their full descriptions and argument schemas; those definitions
join subsequent system prompts and API tool lists for the conversation.
Only tools whose full definitions are supplied in the current request can run.
If a response calls an unavailable tool, the entire batch returns errors without
executing any calls. Load deferred tools in a separate response before using them.

Every advertised tool, except `answer`, also accepts an optional `call_label`: a
short plain-language description of the call, such as "Reading the install guide",
for people who don't read code. The feed shows the label in the tool color with the
exact call beneath it in grey; a call without a label is shown alone in the tool
color. The label stays in the saved call but is removed before the tool runs, and
repetition checks ignore it. Tools that define their own `call_label` argument keep
theirs. The parameter's wording is in `prompts/call-label.md`.
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

Compatible reloads preserve jobs and logs. `/new`, `/resume`, `/fork`, `/delete`, and shutdown stop
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
