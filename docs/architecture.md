# Internal Contracts

This document describes current behavior for contributors and agents editing
SlipAgent. Read the [contributor workflow](../CONTRIBUTING.md) and applicable
project instructions alongside it. The [README](../README.md) documents commands,
configuration, response examples, and the user-visible behavior.

## Instruction Ownership

`AGENTS.md` contains SlipAgent's repository-specific development guidance:
architecture, module responsibilities, setup, verification, and reload contracts.
The [prompt ownership map](../AGENTS.md#prompt-ownership) identifies the runtime
source for each operating contract. Generic workflow, environment use, output
recovery, response formats, and history/task instructions are supplied by those
system prompts and tool definitions in every workspace. They do not depend on
the workspace having a copy of SlipAgent's own `AGENTS.md`.

Project instruction discovery supplies the selected workspace's documents under
a separate heading. `/init` creates a project-notes scaffold only when `AGENTS.md`
is absent, then reloads the project's guidance; it neither overwrites existing
instructions nor copies generic operating rules into the project file.

`ProjectInstructions` owns the root and visited ancestor scopes. It rereads
recognized files before each working request and records the exact snapshot
presented by `on_request`. A context preview never marks instructions delivered.
File-tool dispatch discovers new scopes; a write/edit is blocked if applicable
guidance differs from that delivered snapshot. The next request supplies the
new content, with scope labels and precedence guidance. Removed files disappear
from the snapshot. Discovery failures remain errors, and mandatory guidance is
included in the context budget. Shell/MCP paths are not inferred from command
strings. Legacy explicitly supplied system text remains supported, while CLI
project guidance comes from this refreshed service rather than a stale suffix.

## A Request Is the Model's Entire Memory

The transport sends one explicit list of messages plus tool definitions and
request parameters. No earlier API call, terminal line, file, or archive entry
is visible unless it is included in that request. Separate reasoning deltas are
displayed and joined into one string for that response's uncompressed record;
they are excluded from outgoing working context, compaction, and the active-task
record. Explicit history retrieval can still access the originals. The context inspector
captures the final outgoing body after capability settings are applied, without
the authentication headers.

Context construction proceeds in this order:

1. `ConversationHistory.sync` recognizes complete assistant/tool batches and
   assigns sequential post IDs. New user messages are associated with the next
   response post and also registered as active-task source instructions.
2. System messages are pinned and labelled. The last receives the response
   contract and named sections assembled by `PromptSections`: schema/tool/MCP
   guidance first, then current project instructions, environment, runtime
   state, and repair details. Current post IDs and task state follow those
   sections. Names are unique, owners and static/dynamic roles are explicit,
   and order is deterministic. Section registries belong to one request, so
   a rejected generation cannot leave global prompt registrations behind.
   Prompt prose retains real newlines. Blank lines separate headings, topic
   groups, task fields, tool guidance, examples, and assembled sections; source
   wrapping must not erase those boundaries with backslash continuations.
   Loaded instruction-file contents and original conversation text stay intact.
   The input uses a system prefix followed by JSON records. The inspector
   displays the final body rather than reconstructing it separately.
3. Up to 100 older records precede the recent full window. Each
   retained turn has one representation: either its complete prompt, response,
   calls and results, or its summary. A summary is selected only if its estimated
   token cost, including JSON fields, escaped values and message overhead, is lower.
   Both native-tool and embedded-tool profiles receive the same record format;
   ties keep originals, with complete tool batches intact. Every turn has its
   own JSON object and API user message, including consecutive summaries.
   The full window targets 50 calls by default, with a floor of 5 on the requested
   window size. Under model context pressure, selection drops the oldest older
   records first, then reduces the full window oldest-first, below 5 if necessary.
   Boundaries and representation caches belong to one view: later requests can
   restore omitted records as large calls age into smaller summaries. There is
   no persistent omission marker and no separate history token cap.
   `records.py` encodes full turns with `record_type`, `representation`, `user_prompt`,
   `agent_response`, `tool_calls`, and `tool_results`. Prompts remain an array so
   queued inputs retain their boundaries. Calls use `call_id`, `tool_name`, and
   argument objects; results use matching IDs, names, status, and content.
   The harness's exact observation envelope is unpacked, while arbitrary JSON
   inside tool output remains content. Unknown legacy status stays `unknown`.
   The selected records form a contiguous suffix of stored history. The current
   post number remains in the system prompt, so subtracting one for the latest
   record and counting backward gives each record's exact `recall_history` ID.
   Description keys never enter actual message strings. Reserved legacy response
   labels are removed from standalone prose
   lines in the context copy and in accepted model replies. Fenced/quoted examples
   remain intact; user and tool content, arguments, and archived originals are
   not rewritten. This avoids teaching the reply format through artificial
   assistant-message prefixes, including when resuming an older session.
4. A final `current_turn` JSON object contains current input verbatim. An empty
   prompt array with `continue_current_task=true` continues the supplied task.
   Continuation turns carry the active user
   prompt with their full records. Newly returned results cannot leave the full
   window before the working model receives them, even when their independent
   summary finishes first. Oversized observations use an `excerpt` record with
   original-part retrieval instructions as a last resort. Text fragments and
   omitted-character counts occupy separate fields; clipping never splices
   descriptive markers into message text.
5. `TaskMemory.prompt_supplement` ensures the current user prompt is supplied,
   even after its original post is compressed. It reuses a full copy already
   selected or fills the current-turn object's prompt array with the retained
   original. Its source post ID stays in the system
   task metadata, rather than the user-message label. Supplemental
   input counts against the endpoint budget. The active prompt is independent
   of the historical representation of the call that first received it.

The selected model's context/prompt limits constrain the whole request. The
budget reserves output, system instructions, tool definitions, response schema,
and a 15% estimation margin before fitting history. Text estimates use UTF-8
size, not a model tokenizer. History uses the remaining model allowance;
`--context-posts` controls its target full-call count, not a separate token budget.
Selection preflight uses a copy of history/task selection state so an incompatible
model cannot mutate the active conversation merely by being considered.

`ContextBudget` calibrates estimates against the final working request's measured
prompt tokens, including its tool/schema overhead. The multiplier can only
tighten limits and resets on model/profile replacement. It is applied to a fresh
estimate of each selected view; old absolute usage is never applied to a newly
compacted history. Compaction requests do not update this calibration.
The reduced fraction learned during overflow recovery persists until the
model/profile changes, avoiding the same oversized history on the next call.
Explicit provider input-context errors allow two smaller-budget retries, each
reserving a working request. They do not consume the malformed-response retry
counter, duplicate tools, or modify originals. Stop and request limits still
apply. If the next request would be identical, recovery reports that it cannot
shrink further instead of resending it.

`RepositoryMap` adds optional, budgeted orientation. Files are bounded by count,
traversal time, source bytes and output characters. Python ASTs supply definitions
without executing code; other supported source suffixes supply paths. Metadata
stamps invalidate cached outlines. Ranking combines task words and Python import
references. Git owns ignore-rule interpretation. Hidden/dependency directories
and symlinks are excluded only from this map. If orientation prevents fitting
mandatory input, selection retries without the map.

## Response Acceptance and Tool Execution

`capabilities.py` selects native tool calling from the live endpoints' `tools`
support, independently of their JSON-format support. Startup and explicit model
selection refresh the profile; ordinary requests reuse it. Native-capable routes
take priority. Schema-capable endpoints receive a small strict response schema.
Other endpoints receive no `response_format`; ordinary replies are accepted.
A JSON-object capability alone does not force JSON responses. Without native
support, a JSON-capable route uses embedded calls and text tool definitions,
while allowing plain final replies.
Selection preflight publishes neither the model nor its profile on failure.

`protocol.py` normalizes ordinary text, native calls, and explicit response/call
envelopes. The optional strict schema requires only `response` for native models,
or `response` plus `tool_calls` for embedded calls. There is no compressed-field
or task-field requirement. A null-content native call is valid. Older response
envelopes remain readable; optional legacy task updates still undergo validation.
The archive retains assistant calls followed by matching `role: tool` results.
Working requests package each complete turn as one JSON object, regardless of
native-tool support. The model's new response still uses the selected profile's
native or embedded call format. JSON history does not change the response schema
between calls. The budgeter measures the serialized records; recall retains the
originals, and the current prompt is retained independently of compression.

Normalization precedes validation and is independent of the requested mode.
Accepted call carriers are native OpenAI-style function calls, flat calls in the
JSON record, legacy `function_call`, text/`tool_use` content blocks, and explicit
JSON `<tool_call>`/`<function_call>` blocks adjacent to the record. Whole JSON
code fences are accepted. Arguments may be JSON strings or objects, with
`arguments`, `input`, or `parameters` as an unambiguous field name. Only absent
IDs are generated. Exact tool names and argument values are never inferred.

Multiple nonempty call sources must describe the same ordered batch, including
multiplicity. Native IDs win for identical repeated representations; differing
batches require a retry. Two intentionally identical calls with distinct IDs in
one batch remain two calls. JSON canonical comparison distinguishes booleans
from numbers, unlike Python object equality. Never union alternative batches or
scan response prose/quoted examples for executable calls.

Malformed or ambiguous executable envelopes, duplicate JSON keys, invalid
arguments, Unicode, non-finite numbers, and duplicate call IDs are rejected.
A batch terminated by the output-token limit is rejected before execution.
No `tool_choice` is forced. Ordinary prose and quoted examples remain text;
missing inline summaries do not trigger retries. Models that return a task
record through the legacy envelope must still supply a valid task record.

The transport concatenates readable reasoning only from the response currently
being received, preserving whitespace and avoiding duplicate copies when a
delta supplies both a plain reasoning field and reasoning details. JSON
completions use the same extraction. Each retry starts a fresh accumulator;
existing history is never reassembled. Accepted messages and `TurnPost.reasoning`
retain the resulting string. Full-history projections clear reasoning text and
provider details before budgeting, so thoughts cannot crowd out actual replies
and tool results. The client also removes both fields from outgoing message
dictionaries, covering direct calls and older stored records without mutating
them. No plaintext, signed, or encrypted reasoning blocks are automatically
replayed. Reasoning generation and visible streaming remain enabled where supported.

The normalization boundary follows the general approach documented by
[vLLM's tool parsers](https://docs.vllm.ai/en/stable/features/tool_calling/):
recognize specific call formats, then expose one internal call representation.
SlipAgent does not use a model-name allowlist or attempt arbitrary JSON repair.

Acceptance validates response structure and executable calls, including the
contents of optional task updates. Source acknowledgment and revision echoes
are not required; the harness assigns revision metadata when saving a record.
Until validation passes, only separately delivered reasoning can appear.
Reply text and all tool effects wait. A rejected response is not appended as
executed history. A bounded excerpt and precise diagnosis enter the next request
as invalid diagnostic data, with their size included in the context budget.
Three invalid attempts end the run; retries count against usage and the step cap.

Working calls set `single_attempt=True` on the client. The agent owns bounded
transport recovery, preventing nested HTTP retries from multiplying requests
beyond the step budget. The default allows three transient retries; each starts
with fresh stream accumulators and emits a retry event before interruptible
backoff. A registry-owned stop event wakes that wait. EOF without a finish marker,
transport failures, and retryable status codes are distinct from permanent API
errors. Reported partial usage is retained; unreported usage is not invented.
Direct metadata and isolated compaction calls retain their own client policy.

No tool runs before response acceptance. Diagnostic request/attempt records
can preserve failed partial responses, but those records do not enter working
history or summarization. Cancellation and stream failures end the visible
reasoning block before a retry begins.

`run_batch` executes up to four adjacent independent read tools concurrently.
`Tool.allows_concurrency(arguments)` defaults to false; audited file reads,
directory lists, grep and glob opt in. Writes, shell commands, state changes,
and unclassified MCP tools remain exclusive barriers. A read-only MCP hint
alone does not grant concurrency. Results commit in declared order, each with
its own invocation/log provenance and pre-execution journal marker. The tool
registry validates arguments and converts ordinary exceptions into error results.
`CancelledError` drains started tasks before the agent archives interrupted or
not-run observations, retaining completed results and one observation per call.
Errors carry explicit status and possible partial effects into model context.
User messages queued during the batch are appended after all matching results.
Before submitting a request, the agent includes every waiting prompt together,
including arrivals during context preparation. Drained messages retain a delivery
receipt keyed by their archive index until the request is sent; merely draining
the queue, preparing context, or reaching `/stop` cannot acknowledge them. These
indices are journaled so unsent prompts retain their queued labels on resume.
Submission emits one receipt per prompt. The terminal removes that prompt's
separate notice chunk and indexed display rows without reformatting surrounding
history, then adjusts scroll and pinned-prompt indices. Identical queued prompts
are acknowledged in arrival order. Plain stdout remains an append-only log.

Synchronous file reads, edits, writes, listings, glob scans, and instruction
preflight checks run in owned worker threads. The terminal event loop remains
available during filesystem waits. Cancellation drains each worker before
propagating; glob scans also receive a cooperative cancellation signal. Atomic
edits finish before their owner releases the batch boundary. Async HTTP, MCP,
and subprocess operations retain their event-loop resources.

State-changing slash commands queue in FIFO order while work is active. The
agent finishes its response and whole tool batch, then applies commands while
idle. Settings changes resume the same work with its original request budget
and repetition guard. Explicit stop/exit prevents that automatic continuation;
reset, resume, and starting a new task do not restart the replaced task. Command
text remains outside model history and session journals.

`/stop` finishes the active response and entire tool batch and prevents another
working model request. The completed turn can still be summarized in the
background. It does not kill that batch's processes. `/quit` finishes the active
run before closing resources and prevents automatic launch of another queued run.
Queued corrections from an earlier failed/stopped run are drained before the new
prompt on resumption. Timeout cleanup of an individual process is a separate
mechanism from these session controls.

## History and Task State

The working prompt asks for explicit findings, evidence, decisions, uncertainties,
and next steps in each response, including tool-call turns. These responses remain
in working history and their important conclusions are preserved in summaries.
Private reasoning is not supplied back to the working model or to the compactor.

`TurnPost` extends `HistoryPost` with independent `user_prompt`, `agent_response`,
`reasoning`, `tool_calls`, and `tool_results` references, plus one whole-turn `summary` and a
task-record snapshot. Canonical messages retain their original order and native
call/result structure. Continuation prompts repeat the active request under an
explicit heading; that heading cannot acknowledge a new task-source post.

After archiving a complete batch, `Agent._archive_turn` submits it to
`TurnCompactor`, a registry-owned service. It freezes only the original prompt,
response, tool calls and results, the selected model, and its capabilities. Every job starts asynchronously;
the working loop does not wait. A reply without tools also gets one job. The
compaction request contains exactly two messages: summarization instructions
with the post ID as private system metadata, and the JSON serialization of those
four parts without a post-number field. It contains no thread, prior
summaries, task state, project guidance, or reasoning. No tools or output schema
are requested; the reply is a plain summary capped at 6,000 characters. The
complete input is checked against the frozen endpoint budget without silently
truncating it. A context overflow or bad summary leaves the originals intact and
marks the summary failed with a visible warning and a recall reference.

Summary completion and observation delivery are independent. `TurnPost.observed`
protects the newest tool batch until an accepted working response has received
it; a fast compactor cannot replace unseen results. No working actions are
performed by the compactor. It does not emit context-inspector or reasoning
stream events. Its reported usage contributes to session totals, independently
of primary step accounting. It uses temperature 0.2 where supported, and the
frozen profile's reasoning/routing settings even after model reselection.

Reset increments the compactor generation and cancels all jobs. Late transport
results cannot publish summaries, errors, or usage into a new session. Shutdown
cancels and awaits jobs before closing the client; one-shot completion drains
its summaries first. Replacing the API key drains jobs before retiring their
client. The shared service survives behavior reloads. Its lazy initialization
is an explicit migration for sessions created before the service existed.

`recall_history` pages listings and original parts at character offsets. Every
list/search page includes `total_matches` for the entire filtered result, including
empty results and offsets beyond the listing. `next_offset` remains the character
cursor, and `total_characters` describes the complete listing text. A single `section`
selects `prompt`, `response`, `reasoning`, `tool_calls`, or `tool_results`; `sections` selects
several as one JSON object. Without a selector, it returns the full turn with its
task snapshot. `section="user"` returns only the original new user messages for
source recovery. `call_id` selects a tool observation by default, or one call
with `section="tool_calls"`. A batch ID is not session-unique: use
`(post_id, call_id)` as provenance. Invalid or conflicting selectors return a
tool error. The schema and direct `run` path share the allowed original-part
names; unknown singular or plural selectors are rejected before accessing the
record. CLI sessions journal these records by default; `--no-session` keeps them
only in memory until reset/exit. Embedded Agent callers opt in by attaching a
`SessionJournal` to registry services.

`TaskMemory` separately owns original user inputs, their post references, source
revision, and the optional working record. It records the substantive first
request and later user corrections; “continue” adds a reference without replacing
the original task identity. `/task new` changes the source boundary at the current
message index and makes subsequent input a separate task while retaining history.
Task `complete` is model-reported and requires no pending work or next steps.
Completion updates can share a batch with any other tools, including further
task updates. Inline legacy task records follow the same rule. Calls run in the
declared order. A follow-up can reopen that same task. `blocked` records needed user
input; it does not start a separate inference loop or bypass normal tool rules.

The latest user input remains available verbatim until new input replaces it.
Queued messages received together share a post and remain in their original
order. `current_prompt_post` identifies that source in every working request;
the model can use the number with `recall_history` for original user messages or
the whole uncompressed call. No model echo, acknowledgment, task update, or
mandatory retrieval is needed to keep working. Input is not automatically
classified as a new idea or an amendment. Earlier inputs remain in normal
history and the optional working record rather than being forcibly replayed.

`update_task` replaces the bounded working record when task state changes.
`task_update_instructions()` supplies one shared contract to the system prompt
and tool description before the first working request. Field descriptions come
from the schema; schema, validator, and prose share their size-limit constants.
The contract explains required fields, arrays and empty values, whole-record
replacement, completion conditions, batching, harness metadata, and examples.
The harness sets source revision and owns source IDs; the model supplies status,
goal, constraints, facts, pending work, and next steps. The prompt explains that
still-relevant entries must survive replacement. The record is pinned in every
working request, with an 8,000-character total cap, 2,000-character entries, and
24 entries per list. It is not included in isolated summarization requests.

Legacy parsing helpers remain for imported records and compatibility fixtures;
the active agent never requests inline compressed fields, XML memory envelopes,
or metadata inside tool arguments.

## Resource Ownership and Project Settings

`Session` owns the API client, registry, MCP manager, renderer, terminal lifecycle,
and optional runtime frame. Startup uses `AsyncExitStack` until ownership transfers
to the complete session, so cancellation/preflight failure closes acquired resources.
Shutdown cancels command tasks, closes MCP connections, closes tools and registry
services, closes the API client, and releases the runtime generation.

`ToolRegistry.services` holds the session's `ProjectEnvironment` and
`CommandArchive`, plus `TurnCompactor` after agent initialization. The loop lazily
adds `ContextBudget`, `LoopGuard` and `RepositoryMap`; the CLI adds its optional
`SessionJournal`. The original
registry owns them. Reload candidates borrow the
same dictionary with `owns_services=False`; they must not create a replacement
archive or close the live one when staging fails. HTTP clients owned by rebuilt
web tools have their own lifecycle and are closed when those tools retire.

Session services also own `ProjectInstructions`, `CommandJobs`,
`RequestDiagnostics`, and `LanguageServers`. `builtin_names` explicitly identifies
rebuilt tools; module-path guessing cannot distinguish a built-in defined in a
service module from a retained extension. The stable `Lifetime` primitive owns
tasks and LIFO cleanup callbacks. Its retained close task drains resources once,
even if multiple callers close it or a waiter is cancelled repeatedly. Cleanup
continues after an individual callback fails, then reports the failure.
Tools close before services; services close in reverse creation order, so jobs
finish draining before their command archive closes. Captured generation closers
release the resources they created. MCP disconnect uses the same retained-cleanup
principle, including cancellation during startup or shutdown.

Project settings come from `.slipagent/project.json` under the selected workspace.
The interpreter precedence is `--python`, then the explicit `python` setting,
then a single discovered `.venv` or `venv`. Discovery does not activate PATH or
create/install anything. A bounded isolated subprocess probe identifies the exact
interpreter's prefix, version, and pip/pytest/ensurepip availability. Preserve the lexical
venv executable path: resolving its symlink to the base binary can select global
Python accidentally. Selection/probe metadata is supplied explicitly to the model.

Malformed configuration is reported rather than ignored. Interpreter selection is
refreshed for model requests and probes are cached using executable/configuration
and import-location metadata. The probe contract itself participates in the cache
key so live reloads can introduce new metadata. Missing pip does not produce a
dead pip command: an available uv executable can target the selected venv, or a
separate ensurepip bootstrap command is offered for a venv. Unavailable install
and pytest commands are null. No discovery probe installs anything. The log quota
is a startup setting retained by the session service;
reloading code does not reset its consumed bytes or change its configured quota.

## Command Output Is Separate From the Context Preview

Shell and Git capture drain stdout/stderr concurrently. Their bounded observations
retain a beginning and end preview; the archive receives all decoded text while
quota permits. Process timeout cleanup kills the process group and drains its
pipes before reporting partial output. Cancellation also closes the log and reaps
the process. POSIX process groups cover ordinary descendants; shell execution is
not an operating-system security sandbox.

Each subprocess has a random log ID, with `(post_id, call_id)` copied from a
context-local invocation marker set by the agent. One tool can spawn multiple
subprocesses. The archive uses private generated filenames, never call IDs or
model-supplied paths. `read_command_output` resolves IDs through its own index.
It pages stdout/stderr with byte offsets and character limits, or lists bounded
metadata with record offsets. The response always states which units apply.

The quota defaults to 100 MiB across both streams of all commands in the session.
Retention is append-only until reset/close. Once a stream loses bytes, it retains
one contiguous prefix and counts subsequent bytes as lost. Earlier logs are not
evicted. Disk failure or quota exhaustion does not stop pipe draining or discard
the in-memory preview. Users/models get a retention error and lost-byte count;
they must not be told that discarded data is retrievable.

The archive stores UTF-8 text after replacement decoding of invalid process bytes.
Page boundaries preserve code points. Binary artifacts belong in workspace files.
Persistent sessions bind the archive to private durable files and journal log
metadata when commands start/finish. Resume copies retained streams to the new
session and preserves their IDs. Reset detaches old files without deleting them.
Without persistence, reset/shutdown removes temporary files; a forced kill can
leave the temporary directory behind. Resuming an older archive larger than a
newly configured quota preserves existing bytes and permits no additional bytes
until within budget; quota is never silently raised.

`activity.command_output` supplies a context-local display callback only for
`run_command`. `OutputProgress` throttles to ten updates per second, retaining at
most 32 KiB of pending display text and marking omissions. Capture/archive remain
independent of that display buffer. The renderer escapes terminal controls and
does not duplicate successful streamed output at completion. Reload occurs only
after the batch; callbacks finish within their invocation.

## Background Command Ownership

`run_command(background=True)` creates a `CommandLog` with the initiating
post/call provenance, then hands its execution coroutine to `CommandJobs`.
The manager owns the process through completion, timeout, cancellation, reset,
and shutdown; rebuilt shell tools borrow it. Process creation is shielded so
cancellation cannot lose a child between spawn and handle assignment. Capture
reuses the foreground implementation and drains pipes after killing descendants.

`command_jobs` provides list/status/wait/stop. Its bounded wait never cancels the
owned task; output is retrieved incrementally through `read_command_output`.
Four jobs can run at once and 128 handles are retained, evicting only completed
handles while preserving archived output. Execution keeps the normal timeout
limits. Completed jobs are announced once while idle or at an agent boundary, in both a user
notice and model-visible state, without scheduling another model call.
`/stop` preserves running jobs. CLI reset/resume drains jobs before clearing the
archive; direct `Agent.reset` rejects an active job. Shutdown also drains them.
Resume restores logs but never reconnects or relaunches a process; its recovery
note makes that uncertainty explicit.

## Optional LSP Ownership

`LanguageServers` owns configured stdio clients across compatible reloads.
`navigate_code` is registered only when project configuration enables servers;
manual reload refreshes tool availability. Selection uses configured extensions,
with an explicit server name for ambiguity. Configuration changes retire affected
clients before reuse; commands are argv arrays and no installation occurs.

The client implements the [LSP 3.17 lifecycle and message protocol](https://github.com/microsoft/language-server-protocol/blob/gh-pages/_specifications/lsp/3.17/specification.md),
with bounded Content-Length framing, concurrent response dispatch, stderr tails,
request cancellation and shutdown/exit followed by forced cleanup if needed.
Stopped or failed protocol readers are replaced by bounded discard readers
before process reaping, so a full stdout pipe cannot deadlock shutdown.
Startup/query deadlines are 30 seconds, writes 5 seconds, shutdown stages 1 second,
and incoming messages at most 8 MiB. Only UTF-16 wire positions are negotiated;
tool input uses 1-based Unicode character columns and converts to wire units.
Returned locations label their 1-based UTF-16 columns explicitly.

Queries serialize document synchronization, open current UTF-8 contents, and
close after the response. Workspace metadata scans before navigation notify
servers of changed, created, and deleted files, excluding generated/dependency
trees through the normal workspace traversal. The current file is always opened
explicitly even if excluded from recursive traversal. Configuration requests are
answered from explicit settings. Server-originated workspace edits are refused.
Locations and symlinks follow `Workspace.resolve`: confinement counts and omits
outside results, while danger mode permits them. Hover and location pages are bounded. Errors retire the failed client
so later explicit calls can start cleanly.

## Edit Recovery, Checks and Loop Detection

`tools/editing.py` plans every exact replacement against the same original text.
Zero/ambiguous matches, overlaps, empty edits and no-ops reject before the single
atomic write. LF/CRLF matching preserves the file's local newlines. Nearby source
and already-present replacement hints are bounded diagnostics, never fuzzy edits.
Successful edits return a bounded diff. Legacy single-replacement arguments and
the new `edits` array are mutually exclusive.

`checks.py` records successful built-in write/edit paths and checks their final
state only after all requested tools complete. Default Python AST checks use the
selected project interpreter without imports/execution. Explicit project `checks`
are argv arrays with whole-argument `{python}`/`{files}` expansion, extension
filters, and bounded timeouts/output. Nothing is installed. Missing environments,
invalid settings, timeouts, and failures are visible; results are attached under
a harness heading in the final observation before archival and compaction.
The write's success status remains separate from check diagnostics. These checks
do not cover shell/MCP edits or replace an explicit project test suite.

`LoopGuard` compares consecutive complete batches by tool name, canonical
arguments and actual result/status, excluding changing call/log IDs. The third
unchanged batch adds recovery instructions to the next request; the fourth
stops. A changed batch or user input clears the sequence. Individual calls are
never deduplicated. Explicit `run_command(poll=true)`, log reads and task updates
are exempt, allowing intentional external polling and normal bookkeeping.

## Durable Sessions and Recovery

`SessionJournal` stores versioned JSONL under an expanded absolute state root:
user-home `.SlipAgent` by default, or `SLIPAGENT_STATE_DIR`. Project directory
names combine MD5 and SHA1 of the same expanded absolute lexical workspace path,
without resolving symlink spelling or folding case. `project.json` and journal
headers retain that path for identity checks. New directories/files are private.
Credentials from configuration are not serialized, although user/tool text can
itself contain sensitive data.

The conversation title defaults to the full project path plus ` | SlipAgent`.
New journal headers carry their initial title. `/rename` atomically replaces a
private `<session-id>.title.json` file in the same project state directory;
the sidecar overrides the header without rewriting conversation records.
Listings read headers and title metadata only, rather than scanning each journal.
Malformed metadata reports an error; legacy journals without titles use their
project path. Resume embeds the restored title in the new journal's header,
so subsequent renames leave the parent untouched. Reset gets the default title.
The CLI uses that same title for terminal output; `Session.app_title` remains the
OpenRouter attribution setting. Without persistence, the name lives in
`Session.extensions`. Names cannot contain terminal control characters.

Messages append as deltas, with explicit replacement records for refreshed system
instructions or post-batch check annotations. State records hold task boundaries,
the working task record, usage, queued input, and selected settings. Post records
hold summary state and task snapshots, including asynchronous summary completion.
Reasoning is stored in the original message and remains excluded from context
and compaction. Writes flush/fsync; an unsavable tool-start marker stops dispatch.
Complete originals are never rewritten by compression.

Resume reads and validates the journal before changing the active conversation.
Only an unterminated final entry is ignored, with a recovery note; malformed
complete records fail. Missing tool observations are filled explicitly as
unknown outcomes for started calls and not-run outcomes for unstarted calls.
No saved tool call executes during restore. Original numbered posts, task
sources, summaries, queued input, usage and command logs are restored to a new
journal. The original remains available. Operating/project instructions and
model settings come from the current process. Interrupted summaries stay
labelled; restoration itself performs no model calls.

Interactive restore rebuilds the display from original saved messages and queued
input through the renderer, replaces the current transcript's temporary files,
and follows the final row. It retains the input draft/application. Readable
reasoning and tool observations are shown; system instructions, provider reasoning
signatures, and compaction summaries are excluded from ordinary scrollback.
Rendering never invokes archived tools. `--resume` defers this display work until
the REPL terminal exists; one-shot mode continues to print only its new answer.
This reconstructs the conversation, not transient progress/retry notices that
were never part of the saved messages.

The CLI exposes `/sessions`, `/resume [id|latest]`, `--resume [id]`, and
`--no-session`. `/reset` starts a new journal while retaining prior sessions.
On an interactive terminal, bare `/resume` passes saved titles and dates to
`TerminalUI.choose`, using full session IDs as selection values. The list keeps
the journal's newest-first order and marks the current session. The selected row
stays visible when moving down or up beyond the footer's edges. Escape returns
before compaction waits, job shutdown, or changes to the active conversation.
An explicit ID or `latest` uses the same restore path without opening the picker.
No automatic deletion policy is applied to saved journals. The command-output
quota is per session; it does not limit cumulative saved conversations.
Tests redirect storage to disposable directories, including inherited CLI
subprocesses. Session persistence is enabled at construction on the next launch;
compatible edits to its behavior can subsequently reload.

`RequestDiagnostics` uses a sidecar directory attached by `SessionJournal.begin`.
Each working attempt records its post/step, UTC time, request snapshot reference,
outcome, reported usage, and bounded response/error excerpts. Exact final request
JSON is gzip-compressed and deduplicated by SHA-256; no authentication headers are
captured. Pending records survive a crash and remain visibly unsettled. Atomic
owner-only writes and fsync preserve earlier records when storage fails. A
32 MiB quota stops further diagnostic writes with a warning rather than deleting
old requests or blocking ordinary work. These files are not history posts.
`/requests [attempt]` inspects current-session records; previous directories remain
beside their journals after reset/resume. `--no-session` uses temporary storage.

## Reload Transactions

`runtime.py` is the stable frame. Its `CORE_MODULES` set also pins package root,
configuration, wire types, workspace, tool base, MCP connections, and `lifecycle`.
These classes/state contracts must agree for the session's lifetime. Editing a
pinned module requires a restart; most agent, tool, UI, API, task, and context
behavior is reloadable when layouts remain compatible.

The frame snapshots source bytes, imports a complete candidate namespace through
the snapshot loader, validates required APIs and class layouts, constructs tools
with borrowed services, and checks name collisions before touching live objects.
Classes retain identity: new methods/descriptors are rebound to existing classes,
including `super()` closures. Module globals then point at the accepted generation.
The terminal rebuilds layout/bindings/styles around its existing application and
buffers. On failure, class/module/registry/presentation snapshots are restored.

Closers captured from the constructing generation release rejected candidates or
retired tools after the transaction. Stable services and MCP clients remain alive.
The regex worker is launched from the exact accepted source bytes, so an invalid
on-disk worker edit cannot leak into a running tool while its reload was rejected.

Method replacement does not rerun constructors on existing persistent objects.
Changing inheritance, slots, dataclass field layouts, or removing classes is
rejected. `Session.extensions` is available for deliberate extra session state;
resource-bearing services need explicit ownership and cleanup. Do not rely on
silently absent attributes as a migration plan. The runtime records reload results
in both user output and `registry.context_notes`, which the next request supplies
to the model along with the active generation number.

## Other Boundaries Worth Preserving

- **Provider routing:** aggregate endpoint rows with the same routing tag, use
  their shared capabilities/minimum limits, and exclude incompatible variants
  when a selected base tag can match them. Missing effort choices mean “enable
  when supported”; an explicitly disabled list must not become “enabled.” Stage
  profiles and preflight before publishing a model switch.
- **File/Git operations:** resolve paths against the workspace; writes publish
  atomically and avoid following hard links to overwrite external inodes. Git
  checks both worktree and metadata, rejects active staging filters, and disables
  hooks/signing/diff helpers/automatic fetching. Shell and MCP remain unrestricted
  subprocesses with the harness's operating-system permissions.
  The frozen workspace retains its root and owns a mutable `WorkspaceAccess`
  object shared by all tools and session services. `/danger` and `--danger` lift
  only the path boundary; `contains` still answers geometric containment and
  external display paths remain absolute. Changes queue during work and apply
  after the active response and tool batch finish. Reset, resume,
  and compatible component reloads preserve this process setting; journals do
  not restore it. Each request supplies a current Workspace Access section, and
  the footer reserves room for a red `| Danger Mode` suffix when active.
  External atomic writes retain descriptor-based traversal from the target's
  filesystem root. Git ancestor discovery stops at the filesystem root in danger
  mode. External instruction scopes include the target's ancestors and are
  omitted from snapshots when confinement is restored. Installing changes to
  `workspace.py` requires a process restart, as with other stable contracts.
- **Search:** directory listing includes dotfiles and environments. Recursive
  search prunes generated/dependency directories; explicit paths can target them.
  Regex matching lives in a killable child, while file traversal remains in the
  workspace-aware parent. A thread alone cannot interrupt pathological regex code
  that holds the interpreter lock.
- **MCP:** modern metadata belongs in `params._meta`; legacy servers use the
  initialize handshake. Send deadlines include pipe writes, not just response
  waits. Tools receive structural schema checks and registry validation; servers
  remain responsible for their full JSON Schema semantics. Structured content,
  server guidance, and tool errors must survive adaptation into model context.
- **Web:** validate every resolved destination and redirect, and pin the chosen
  public address while preserving Host/TLS identity. HTML extraction preserves
  code whitespace and link destinations. Fetching does not execute JavaScript.
- **Terminal:** `transcript.py` stores original chunks and completed styled rows
  in separate private session files, indexed by byte offsets. Block boundaries
  retain the renderer's spacing and ANSI resets. The ANSI decoder carries style
  and partial escapes across chunks. Wrapping retains the last two rows plus any
  unfinished word that can still alter the preceding row; completed rows are
  immutable. Adjacent characters with the same style are stored as one run.
  `TranscriptControl` supplies indexed rows through `UIContent.get_line`, with
  a 256-row cache; it does not build or hash a full transcript on each repaint.
  The application coalesces redraws at a maximum of 30 per second. Resize replays
  original chunks to rebuild the width-specific row file; ordinary appends
  process only new chunks. Explicit continuation indents are indexed by chunk
  and source line, and replayed on resize. The startup tool list uses this to
  align wrapped rows beneath its first item without changing ordinary output.
  During append/reflow, `WrappedTranscript` indexes source lines whose first
  character is `>` with decoded foreground color `#00ff00`. Binary search selects
  the nearest preceding matching line at the viewport top, including its wrapped
  rows. The unfinished tail participates without rescanning previous output.
  A one-row green overlay reuses the first wrapped row, adding an ellipsis at a
  word boundary when more text follows. Long words fall back to a display-width
  cutoff that preserves graphemes. The full prompt remains in the transcript;
  drawing the header never moves scroll offsets or scans the entire prompt.
  Scrolling in either direction selects the corresponding older/newer
  prompt. New submissions release the header until the new prompt reaches the
  top or explicit scrolling resumes. The header is hidden if there is no room
  for at least one output row below it. Context view keeps its own layout. Reloads
  rebuild the index from stored text and colors, including existing scrollback;
  no display copies enter the transcript or model history.
  Refresh explicitly migrates older in-memory block
  lists while preserving the application, draft, input queue, and scroll state.
  Closing the terminal releases both temporary files. These files serve display,
  not conversation persistence or model context. Password entry owns a separate
  history-free buffer/queue, not just a masking processor.
  `TerminalUI.choose(title, options)` temporarily replaces the six-row footer
  with a heading, four option rows, and a bottom key legend. A focused menu
  control owns navigation; the input buffer, history, queue, and transcript stay
  intact. It returns the selected value or `None` on Escape, Ctrl-C/Ctrl-D, or
  terminal closure. Cancellation restores input focus, and refresh preserves an
  open menu's selection. `/menu` dispatches complete command strings through the
  normal handlers, retaining busy-state checks and background execution for
  read-only network commands during a turn.

## Extending the Contracts

When adding a tool, define an accurate schema and recovery-oriented description,
return `ToolResult`, register it once, and test errors as well as success. Choose
whether its resources belong to a rebuilt tool or the session before constructing
them. Keep remote names scoped so disconnecting one server cannot remove another's
tools. Update the README tool table when the public tool set changes.

When changing the response schema, update validation, prompt explanations/example,
post storage, task/context budgets, stub responses, streaming/reload fixtures, and
offline demos together. When adding a command, update busy-state restrictions,
help, startup listing, README, and control-flow tests. Read-only network commands
must not block input processing while `/stop` is waiting.

For every stateful change, test the relevant combination of success, rejection,
cancellation, reset, and reload. Assert actual request bodies and external effects,
not just helper return values. Preserve user-owned instruction files; maintain
these explanatory guides separately from local operational conventions.
