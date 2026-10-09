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

The built-in model instructions and tool descriptions live in the checkout's top-level `prompts/` folder. Wheel builds include that folder as `slipagent/prompts`. `prompts.py` locates the application resources independently of the working project and reads UTF-8 text at use time. `agent.py` renders the main template's workspace and interpreter and refreshes it before each CLI model request, including with live reload disabled. Other dynamic templates use explicit `${name}` substitutions without interpreting inserted data as templates. The reload frame monitors these resources along with Python sources; compatible source reloads preserve their service owners. Missing resources and invalid templates report errors.
File reads and edits, command-output recovery, background jobs, Git previews,
and language-server navigation are described in their respective tool definitions.
Native-tool requests deliver those contracts through API tool definitions;
JSON-tool requests include the same definitions in the system's Available Tool
Definitions section.

The background-summary prompt is in `prompts/background-summary-prompt.md`. `compaction.py` reads it when building each isolated summary request. Protocol, history, environment, recovery, and tool guidance use the same loader. Tool specs read descriptions and argument-description mappings on each access; argument schemas and behavior remain in Python. See `prompts/README.md` for the editable file map.

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

Tool-call arguments, structured tool results, observation envelopes, input records,
and frozen compaction inputs remain dictionaries or lists internally. External JSON
text is decoded at entry. File contents and command output remain literal string
fields. Model-facing messages and terminal output render structured values as
labelled literal text, without JSON escaping. Journals and HTTP transport encode
JSON only at their destination boundaries. The request callback carries a
dictionary to diagnostics and token accounting. Complete history selections return
structured content; partial character pages return fragments of its literal-text presentation. Resume
accepts both legacy text observations and structured journal observations.

The transport sends one explicit list of messages plus tool definitions and
request parameters. No earlier API call, terminal line, file, or archive entry
is visible unless it is included in that request. Separate reasoning deltas are
displayed and joined into one string for that response's uncompressed record;
Overthinking Mode (enabled by default) includes filtered thoughts when ready,
otherwise the original string, as `reasoning` in selected records from the latest
25 completed steps by default. `--reasoning-history-steps` controls that depth.
Missing thoughts add no field. The exact JSON participates in context selection
and token budgeting; omitted steps and budget-limited excerpts omit reasoning.
Opaque provider metadata is excluded. Disabling the mode excludes all reasoning
from working context. Compaction processes reasoning separately in either mode.
Explicit history retrieval can still access the originals. The context inspector
captures the final outgoing body after capability settings are applied, without
the authentication headers.

Context construction proceeds in this order:

1. `ConversationHistory.sync` recognizes complete assistant/tool batches and
   assigns sequential step IDs. New user messages are associated with the next
   response step and also registered as active-task source instructions.
2. System messages are pinned without rewriting their contents. The last receives the response
   contract and named sections assembled by `PromptSections`: schema/tool/MCP
   guidance first, then current project instructions, environment, runtime
   state, and repair details. Current step IDs and task state follow those
   sections. Names are unique, owners and static/dynamic roles are explicit,
   and order is deterministic. Section registries belong to one request, so
   a rejected generation cannot leave global prompt registrations behind.
   Static headings belong to editable templates; `PromptSections` renders their
   content without adding wrappers. Markdown uses `#`, `##`, and `###` for the
   instruction hierarchy, fenced examples, and field tables. Only imported
   history retains opening and closing equals-sign warnings, loaded from
   `history-opening.md` and `history-closing.md`.
   Prompt paragraphs occupy one physical line, without fixed-width wrapping. Two newlines separate paragraphs, headings, topic groups, task fields, tool guidance, examples, and assembled sections. Single newlines remain for meaningful item boundaries, metadata fields, code, and other structured content.
   Loaded instruction-file contents and original conversation text stay intact.
   The input uses a system prefix followed by labelled records with literal text blocks. The inspector
   displays the final body rather than reconstructing it separately.
3. Up to 100 older records precede the recent full window. Each retained step
   occupies one object. Full records contain originals plus available compressed
   analysis. Older records use a summary if its estimated cost is lower than
   the original representation before adding analysis. The budget includes all
   fields actually sent, presentation indentation, and message overhead. Selection
   checks the fully assembled request, including excerpts, against the allowance.
   Both native-tool and embedded-tool profiles receive the same record format;
   ties keep originals, with complete tool batches intact. Every step has its
   own object in the system prompt's history list. Objects remain structured until
   `Message.to_api()` renders literal text through `data_text.py`; the HTTP client
   JSON-encodes the complete payload. Text fields are not embedded JSON strings.
   The full window targets 50 calls by default, with a floor of 5 on the requested
   window size. Under model context pressure, selection drops the oldest older
   records first, then reduces the full window oldest-first, below 5 if necessary.
   Boundaries and representation caches belong to one view: later requests can
   restore omitted records as large calls age into smaller summaries. There is
   no persistent omission marker. `--context-tokens` can cap total working context.
   `records.py` encodes full steps with `record_type`, `representation`, `user_prompt`,
   `agent_response`, `tool_calls`, and `tool_results`. Prompts remain an array so
   queued inputs retain their boundaries. Calls use `call_id`, `tool_name`, and
   argument objects; results use matching IDs, names, status, and content.
   The harness's exact observation envelope is unpacked, while arbitrary JSON
   inside tool output remains content. Unknown legacy status stays `unknown`.
   The selected records form a contiguous suffix of stored history. The current
   step number is supplied as metadata. Every full, compressed, excerpt,
   and current bundle also includes its exact `step_id` as metadata for
   `recall_history`, separate from the original message strings.
   Description keys never enter actual message strings. Reserved legacy response
   labels are removed from standalone prose
   lines in the context copy and in accepted model replies. Fenced/quoted examples
   remain intact; user and tool content, arguments, and archived originals are
   not rewritten. This avoids teaching the reply format through artificial
   assistant-message prefixes, including when resuming an older session.
4. A final `current_step` JSON object contains current input verbatim. An empty
   prompt array with `is_tool_result_response=true` marks a response to earlier tool results. If the supplied results satisfy the user's request, return the result with no tool calls to finish; request more tools only when necessary work remains.
   Continuation steps carry the active user
   prompt with their full records. Newly returned results cannot leave the full
   window before the working model receives them, even when their independent
   summary finishes first. Oversized observations use an `excerpt` record with
   original-part retrieval instructions as a last resort. Text fragments and
   omitted-character counts occupy separate fields; clipping never splices
   descriptive markers into message text.
5. `TaskMemory.prompt_supplement` ensures the current user prompt is supplied,
   even after its original step is compressed. It reuses a full copy already
   selected or fills the current-step object's prompt array with the retained
   original. Its source step ID stays in the system
   task metadata, rather than the user-message label. Supplemental
   input counts against the endpoint budget. The active prompt is independent
   of the historical representation of the call that first received it.

The selected model's context/prompt limits constrain the whole request. The
budget reserves output, system instructions, tool definitions, response schema,
and a 15% estimation margin before fitting history. Text estimates use UTF-8
size, not a model tokenizer. History uses the remaining model allowance;
`--context-steps` controls its target full-call count, not a separate token budget.
Selection preflight uses a copy of history/task selection state so an incompatible
model cannot mutate the active conversation merely by being considered.

`ContextBudget` calibrates estimates against the final working request's measured
prompt tokens, including its tool/schema overhead. The multiplier is the mean measured-to-estimated ratio of the latest five valid measurements with a 10% safety margin. It can rise or fall, resets on model/profile replacement, and does not change configured limits. Existing live instances initialize the measurement window at their next observation. It is applied to a fresh
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
traversal time, source bytes and output characters. Python ASTs and bundled
Tree-sitter grammars supply declaration outlines without executing code or fetching
parsers. `symbols.py` bounds traversal and labels incomplete syntax trees. Metadata
stamps invalidate cached outlines. Ranking combines task words with a personalized
reference graph; each file contributes at most 512 reference tags and roughly
256 target files. Symbol-name matches are hints rather than resolved bindings.
Git owns ignore-rule interpretation. Hidden/dependency directories
and symlinks are excluded only from this map. If orientation prevents fitting
mandatory input, selection retries without the map.

## Response Acceptance and Tool Execution

`LoopGuard` fingerprints tool names, arguments, status, and actual results while
ignoring invocation-specific command-log notices. A batch or cycle of up to six
batches receives recovery guidance after three repetitions and stops after four.
Explicit polling and progress-exempt tools do not establish repetition. Changed
results break the repeated cycle; new user input resets detection.

Each working response uses a fresh `StreamLoopGuard`. Six exact repetitions of
a prose segment of 80–2048 characters stop the response before any of its tools
execute. Detection operates on whitespace-normalized text, separately for content
and reasoning, with bounded buffers and chunk-independent comparisons. Content
code fences, tables, lists, and indented code are excluded. Rejected partial
responses stay in request diagnostics, outside accepted history, and are not
automatically retried. Stream closure and usage accounting follow the ordinary
request-error path. Background summaries do not use this working-response guard.

`capabilities.py` selects native tool calling from the live endpoints' `tools`
support, independently of their JSON-format support. Startup and explicit model
selection refresh the profile; ordinary requests reuse it. Native-capable routes
take priority. Schema-capable endpoints receive a small strict response schema.
Other endpoints receive no `response_format`; ordinary replies are accepted.
Native routes without schemas request plain reply text. JSON-only routes request
`response` and `tool_calls` on every step, including final answers.
Selection preflight publishes neither the model nor its profile on failure.

`protocol.py` detects response formats independently of request capabilities.
Plain replies, JSON envelopes, whole JSON fences, legacy summary records, tagged
JSON calls, and Qwen3-Coder calls share one normalization boundary. The API adapter
accepts native `tool_calls`, legacy `function_call`, and `tool_use` content blocks.
Separate calls and calls embedded in reply text can coexist. API carriers precede
text calls; identical alternate representations are deduplicated by multiplicity,
while distinct calls are combined. Conflicting shared IDs reject the batch.
Qwen argument conversion uses tool parameter schemas, preserving string whitespace
and entities. No model-specific response-format memory is used. Requests retain
their preferred output instructions and API capability profiles.
Complete streamed responses are validated before effects. Malformed or truncated
batches, conflicting fields, and duplicate IDs within a carrier reject the entire
response. Explanatory code examples and reasoning are never scanned for actions.
An empty reply is valid only when accompanied by tool calls.

The built-in `answer` tool returns its text as a user-visible answer. A successful
batch containing only answer calls ends the run without another working request;
mixed batches continue normally. Results are journaled and replayed through the
same renderer path. XML `<tool_call><answer>text</answer></tool_call>` maps to this
tool. Missing trailing XML closing tags are tolerated, but provider-reported
truncation and incomplete JSON arguments still reject the entire batch.

The archive retains assistant calls followed by matching `role: tool` results.
Working requests package each complete step as one JSON object, regardless of
native-tool support. JSON history does not change the selected response contract.
The budgeter measures serialized records; recall retains originals, and the
current prompt is retained independently of compression. Legacy archive readers
remain separate from active response acceptance.

Duplicate JSON keys, malformed arguments, invalid Unicode, non-finite numbers,
and duplicate call IDs are rejected. Output-token truncation rejects tool batches
before execution. No `tool_choice` is forced. Native prose and quoted examples
remain text and cannot execute tools.

The transport concatenates readable reasoning only from the response currently
being received, preserving whitespace and avoiding duplicate copies when a
delta supplies both a plain reasoning field and reasoning details. JSON
completions use the same extraction. Each retry starts a fresh accumulator;
existing history is never reassembled. Accepted messages and `CompletedStep.reasoning`
retain the resulting string. Working input records include readable thoughts
for up to 25 completed steps (configurable) when Overthinking Mode is enabled, with
their exact token cost included in selection. Provider details stay excluded.
The client removes native reasoning fields from outgoing message
dictionaries, covering direct calls and older stored records without mutating
them. Signed and encrypted provider blocks are never replayed. Reasoning generation
and visible streaming remain enabled where supported.

The normalization boundary follows the general approach documented by
[vLLM's tool parsers](https://docs.vllm.ai/en/stable/features/tool_calling/):
recognize specific call formats, then expose one internal call representation.
SlipAgent does not use a model-name allowlist or attempt arbitrary JSON repair.

Acceptance validates response structure and executable calls. Working plans
use the ordinary `update_plan` tool; no task envelope or revision echo is required.
Until validation passes, only separately delivered reasoning can appear.
Reply text and all tool effects wait. A rejected response is not appended as
executed history. A bounded excerpt and precise diagnosis enter the next request
as invalid diagnostic data, with their size included in the context budget.
Three invalid attempts end the run; retries count against usage and the step cap.

Working calls set `single_attempt=True` on the client. The agent owns bounded
transport recovery, preventing nested HTTP retries from multiplying requests
beyond the step budget. The default performs three silent transient retries, then reports the error with a cancellable countdown. Exponential retry intervals stop before exceeding 60 seconds; `Retry-After` values, including HTTP dates, cannot exceed that ceiling. Explicit finite `RetryPolicy` instances can disable extended recovery. Each attempt starts with fresh stream accumulators. Exhausted retries report the failure. A registry-owned stop event wakes the wait. EOF without a finish marker,
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
working model request. The completed step can still be summarized in the
background. It does not kill that batch's processes. `/quit` finishes the active
response/tool batch before closing resources and prevents another working request.
Queued corrections from an earlier failed/stopped run are drained before the new
prompt on resumption. Timeout cleanup of an individual process is a separate
mechanism from these session controls.

Outside a chooser, Escape invokes a direct terminal callback instead of enqueuing
a slash command. The session cancels its active agent/interactive command tasks,
read-only command tasks, background jobs, and compactor requests. Cleanup is
retained in `Session.extensions`; repeated Escape does not cancel cleanup itself.
The cancelled batch drains its tools and records completed, interrupted, and
unstarted calls in model order. Interrupted archival persists originals without
starting a summary request. Deferred commands are discarded; queued user prompts
remain pending and no automatic continuation occurs. Worker-thread operations
retain their safe drain contract. Chooser bindings capture Escape before the
interrupt binding, so closing a chooser never interrupts agent work.

## History and Task State

The working prompt asks for explicit findings, evidence, decisions, uncertainties,
and next steps in each response, including tool-call steps. These responses remain
in working history and their important conclusions are preserved in summaries.
Overthinking Mode supplies recent original or filtered reasoning to the working
model. The compactor processes thoughts separately from its factual summary.

Working history uses one JSON object per step. Full records include available
`compressed_summary` analysis alongside their original fields; compressed-only
records use the same key in place of those fields. Selection compares older base
representations before adding analysis, then budgets every field actually sent.
Current-turn input and original archives are unchanged.

`CompletedStep` extends `HistoryStep` with independent `user_prompt`, `agent_response`,
`reasoning`, `tool_calls`, and `tool_results` references, plus whole-step `summary`
and separate `reasoning_summary` fields. Canonical messages retain their original
order and native call/result structure. Continuation records retain the active
request without registering it as new user input.

After archiving a complete batch, `Agent._archive_step` submits it to
`StepCompactor`, a registry-owned service. It freezes the original prompt,
response, tool calls, results and thoughts, up to 25 preceding completed steps,
the selected model, and its capabilities. Every job starts asynchronously;
the working loop does not wait. A reply without tools also gets one job. The
compaction request contains exactly two messages: summarization instructions
and a structured record separating the completed step's fields from contextual
`history`, rendered as labelled literal text at the API boundary.
History uses available summaries or full records, without project guidance.
Only the current step is compressed into two JSON string fields: `summary`
and `reasoning_summary`, each capped at 6,000 characters. Both are validated
before publication. An empty reasoning summary means nothing useful survived;
`None` means no filtered result is available. Overthinking prefers the filtered
field and otherwise uses raw thoughts. Originals remain immutable and recallable.
Both fields persist in the journal; old journals fall back to original thoughts.
The frozen endpoint budget is checked before sending. Oldest contextual records
are omitted first with an explicit count; the current step is never truncated.
A context overflow or bad summary leaves the originals intact and
marks the summary failed with a visible warning and a recall reference. Each
summary attempt has a 20-minute timeout. Retry waits and later attempts can
extend the job beyond that duration; final failure preserves the originals.

Summary completion and observation delivery are independent. `CompletedStep.observed`
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
several as one JSON object. Without a selector, it returns the full step with its
original user prompt. `section="user"` returns only the original new user messages for
source recovery. `call_id` selects a tool observation by default, or one call
with `section="tool_calls"`. A batch ID is not session-unique: use
`(step_id, call_id)` as provenance. Invalid or conflicting selectors return a
tool error. The schema and direct `run` path share the allowed original-part
names; unknown singular or plural selectors are rejected before accessing the
record. CLI sessions journal these records by default; `--no-session` keeps them
only in memory until reset/exit. Embedded Agent callers opt in by attaching a
`SessionJournal` to registry services.

`TaskMemory` owns original user inputs and their source step references. The latest user input remains available verbatim until new input replaces it. Queued inputs received together share a source step. `/task` displays original prompts, and `/task new` changes the source boundary without deleting history. Session resume reconstructs prompt retention from original messages and ignores legacy model-maintained task snapshots.

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
`CommandArchive`, plus `StepCompactor` after agent initialization. The loop lazily
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

Each subprocess has a random log ID, with `(step_id, call_id)` copied from a
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
metadata when commands start/finish. Fork copies retained streams and preserves
their IDs; resume reuses saved log storage. Reset detaches files without deleting them.
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
step/call provenance, then hands its execution coroutine to `CommandJobs`.
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
stops. It also detects repeating cycles of up to six batches. New user input
resets detection. This guard does not deduplicate calls. Explicit
`run_command(poll=true)`, command-log reads and job-control calls are exempt.

## Durable Sessions and Recovery

`SessionJournal` stores versioned JSONL under an expanded absolute state root:
user-home `.SlipAgent` by default, or `SLIPAGENT_STATE_DIR`. Project directory
names combine MD5 and SHA1 of the same expanded absolute lexical workspace path,
without resolving symlink spelling or folding case. `project.json` and journal
headers retain that path for identity checks. New directories/files are private.
Credentials from configuration are not serialized, although user/tool text can
itself contain sensitive data.

Startup attaches the journal service without beginning a conversation. `record`
creates the journal on the first user message or queued prompt, before model/tool
dispatch, and otherwise ignores startup-only state changes. A pre-prompt rename
keeps its title in memory until the first journal header is written. Resume reopens
the selected journal; only `/fork` creates a populated child journal.

The conversation title defaults to the full project path plus ` | SlipAgent`.
New journal headers carry their initial title. `/rename` atomically replaces a
private `<session-id>.title.json` file in the same project state directory;
the sidecar overrides the header without rewriting conversation records.
Listings read headers and title metadata only, rather than scanning each journal.
Malformed metadata reports an error; legacy journals without titles use their
project path. Resume retains the saved title and its sidecar. Fork embeds that
title in the child header, so renaming a fork leaves the parent untouched. Reset gets the default title.
The CLI uses that same title for terminal output; `Session.app_title` remains the
OpenRouter attribution setting. Without persistence, the name lives in
`Session.extensions`. Names cannot contain terminal control characters.

Messages append as deltas, with explicit replacement records for refreshed system
instructions or step-batch check annotations. State records hold task boundaries,
usage, queued input, and selected settings. Working plans persist as tool results. Step records
hold summary state, including asynchronous summary completion.
Reasoning is stored in the original message, included in recent context when
Overthinking Mode is enabled, and supplied separately to compaction. Writes flush/fsync;
an unsavable tool-start marker stops dispatch.
Complete originals are never rewritten by compression.

Resume reads and validates the journal before changing the active conversation.
Only an unterminated final entry is ignored, with a recovery note; malformed
complete records fail. Missing tool observations are filled explicitly as
unknown outcomes for started calls and not-run outcomes for unstarted calls.
No saved tool call executes during restore. Original numbered steps, task
sources, summaries, queued input, usage and command logs are restored from the
selected journal, which resume reopens. Fork creates a child. Operating/project instructions and
model settings come from the current process. Interrupted summaries stay
labelled; restoration itself performs no model calls.

`/fork` drains compaction and stops jobs, checkpoints the current journal, and
uses the validated load/restore path to create a populated child. Its parent
stays immutable after the fork; copied command output remains independently
available. The transcript stays in place and subsequent work targets the child.

`/delete` requires an interactive Enter/Escape chooser before any deletion.
After confirmation it drains summaries and jobs, clears the conversation with
`reset(new_session=False)` to detach log/diagnostic owners, and removes only the
current journal, title sidecar, logs, request diagnostics, and file checkpoints.
The journal service returns to its pre-prompt state and the transcript is replaced
by the startup banner. Other sessions and project files remain untouched.
Filesystem deletion errors report potentially partial removal and disable further
writes to that journal. Cancellation preserves the current session.

Interactive restore rebuilds the display from original saved messages and queued
input through the renderer, replaces the current transcript's temporary files,
and follows the final row. It retains the input draft/application. Readable
reasoning and tool observations are shown; system instructions, provider reasoning
signatures, and compaction summaries are excluded from ordinary scrollback.
Rendering never invokes archived tools. `--resume` defers this display work until
the REPL terminal exists; one-shot mode continues to print only its new answer.
This reconstructs the conversation, not transient progress/retry notices that
were never part of the saved messages.

`FileCheckpoints` is a session service, created lazily and retained across component
reloads. The agent binds it through a context variable for each complete tool
batch. Built-in atomic writes durably save original bytes and intended result
hashes before publication, then record successful completion. Failed and interrupted
writes remain distinguishable; a pending write can be restored only when its actual
contents match its recorded original or intended result. Backups are SHA-256 blobs
with private permissions; checkpoint indexes use atomic, flushed writes.

`/rewind` restores the selected batch and all subsequent active checkpoints. It
validates every affected file and backup before the first mutation, and checks
again before individual writes. Conflicts preserve the workspace. I/O failure
during restoration reports any files already restored. Original bytes and modes
are restored; files originally absent are removed. The conversation is retained,
and a factual context notice identifies restored files. Resume restores checkpoint
storage and notices; fork copies backups and indexes into its own sidecar directory.
Project files remain shared between forks. Shell/MCP effects and external edits
are not checkpointed. Reset detaches storage; deletion removes the selected sidecar.
Without journals, temporary storage lasts until reset or shutdown. Queued rewind
never resumes the interrupted run automatically.

The CLI exposes `/sessions`, `/resume [id|latest]`, `/fork`, `/delete`, `/rewind`, `--resume [id]`, and
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
Each working attempt records its run-step and history-step IDs, UTC time, request snapshot reference,
outcome, reported usage, and bounded response/error excerpts. Exact final request
JSON is gzip-compressed and deduplicated by SHA-256; no authentication headers are
captured. Pending records survive a crash and remain visibly unsettled. Atomic
owner-only writes and fsync preserve earlier records when storage fails. A
32 MiB quota stops further diagnostic writes with a warning rather than blocking
ordinary work. These files are not history steps. `/requests [attempt]` inspects
saved attempts. Reset starts a new directory; resume reuses the saved session's
directory, continues after its highest attempt number, and counts existing file
sizes toward the quota. Even incomplete attempt files reserve their numbers.
`--no-session` uses temporary storage.

## Reload Transactions

`runtime.py` is the stable frame. Its `CORE_MODULES` set also pins package root,
configuration, wire types, their `data_text` renderer, workspace, tool base, MCP
connections, and `lifecycle`.
These classes/state contracts must agree for the session's lifetime. Editing a
pinned module requires a restart; most agent, tool, UI, API, task, and context
behavior is reloadable when layouts remain compatible.

The frame snapshots and hashes source bytes in a worker thread, and compiles and
imports a complete candidate namespace in that thread through
the snapshot loader, validates required APIs and class layouts, constructs tools
with borrowed services, and checks name collisions before touching live objects.
Classes retain identity: new methods/descriptors are rebound to existing classes,
including `super()` closures. Module globals then point at the accepted generation.
The terminal rebuilds layout/bindings/styles around its existing application and
buffers. On failure, class/module/registry/presentation snapshots are restored.

Candidate preparation leaves the terminal input loop responsive. Live-state
validation and the atomic commit stay on the event-loop thread. If new model/tool
work or a command starts during preparation, an idle reload discards its candidate
and retries at a later safe boundary. Cancellation drains the staging thread before
removing its import namespace. This change to the stable frame requires a restart.

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
  when a selected base tag can match them. Endpoint status values do not filter
  the chosen model's providers; actual request results establish availability.
  Missing effort choices mean “enable
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
- **Terminal:** `markdown.py` parses assistant replies independently of literal harness
  output. CommonMark tokens become semantic terminal fragments; Pygments lexer
  tokens provide code syntax roles without using its output formatters. Themes
  choose role-specific colors and backgrounds. Source and rendered views replay
  the same originals. Incremental Markdown keeps a bounded mutable preview and
  reconciles the complete response at finalization, including reference links.
  A separate copy sidecar maps visible character boundaries to readable text,
  excluding code wrap markers and background padding. Clipboard commands can
  copy readable replies, Markdown source, or exact fenced code bodies.
  Structured model responses retain their validation boundary before display.
  `transcript.py` also stores original literal chunks and completed styled rows
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
  character is `>` with a recognized historical green foreground or the current
  white-on-green prompt style. Binary search selects
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
  intact. It returns the selected value or `None` on Escape, Ctrl-D, or
  terminal closure. Cancellation restores input focus, and refresh preserves an
  open menu's selection. `/menu` dispatches complete command strings through the
  normal handlers, retaining busy-state checks and background execution for
  read-only network commands during a step.

## Extending the Contracts

When adding a tool, define an accurate schema and recovery-oriented description,
return `ToolResult`, register it once, and test errors as well as success. Choose
whether its resources belong to a rebuilt tool or the session before constructing
them. Keep remote names scoped so disconnecting one server cannot remove another's
tools. Update the README tool table when the public tool set changes.

The registry validates parameter-schema structure before registering tools or
replacing its collection. Invalid schemas leave the existing collection intact;
remote servers remain responsible for full JSON Schema semantics.

When changing the response schema, update validation, prompt explanations/example,
step storage, task/context budgets, stub responses, streaming/reload fixtures, and
offline demos together. When adding a command, update busy-state restrictions,
help, startup listing, README, and control-flow tests. Read-only network commands
must not block input processing while `/stop` is waiting.

For every stateful change, test the relevant combination of success, rejection,
cancellation, reset, and reload. Assert actual request bodies and external effects,
not just helper return values. Preserve user-owned instruction files; maintain
these explanatory guides separately from local operational conventions.

## API Provider Modules

`api.py` owns `APIClient`, common errors, HTTP/SSE transport, retries, and completion normalization. `providers.py` maps provider names to implementation classes in separate files and exposes `create_client`. Providers subclass `APIClient`; they supply identity/defaults, capability discovery, optional account metadata, headers, request preparation, and options retained during key replacement. Importing modules does not create connections. `openrouter.py` keeps the previous error aliases for callers using those imports.

The agent and compactor use the common `RequestProfile` contract, not a provider class or routing format. `ModelCapabilities` selects OpenRouter routes from live metadata. NVIDIA's explicit profiles describe verified hosted models; unsupported catalog entries stay visible without invented capabilities. A supplied frozen compaction profile remains independent of later selections. Provider request preparation runs before diagnostics so the inspector sees the final body.

`ModelInfo.selector` is `provider::model-id`. Search combines catalogs only for providers with a nonblank environment or session key, preserves duplicate IDs, and reports partial failures. Missing-key providers are not queried or listed. Changes to the active provider set refresh the catalog cache. `Session.select_model` preflights capabilities and context. Same-provider changes preserve pending summaries; provider changes cancel and drain them before closing their client. Originals survive cancellation. Preflight failures close candidate resources and preserve active state. Credentials, URLs, and remembered models are provider-specific. `/key` uses authoritative metadata where available and explicitly reports when verification must wait for inference.

Provider behavior files participate in the existing reload transaction. Shared state contracts in `types.py`, configuration, package exports, and changes to class inheritance require restart. Adding a provider requires a module, registry entry, documented configuration, and offline wire/selection tests.

Common transport and agent recovery share one retry schedule. Working requests remain `single_attempt=True`, so each attempt consumes exactly one working request. Three retries are silent; subsequent waits update the terminal activity row once per second, clearing it after success, stop, or cancellation. Noninteractive output retains the error and periodic countdown notices. No interval exceeds one minute. Background compaction uses the same transient-error schedule; malformed summaries retain a separate finite repair limit. HTTP 200 overload envelopes and SSE errors are retryable; partial tool calls never execute before a complete accepted response.
