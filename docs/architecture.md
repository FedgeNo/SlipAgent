# Internal Contracts

This document describes current behavior for contributors and agents editing
SlipAgent. Read the [contributor workflow](../CONTRIBUTING.md) and applicable
project instructions alongside it. The [README](../README.md) and the user guides it links document commands,
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
Working requests include complete contracts for core and loaded tools in the
system's Available Tool Definitions section; other eligible tools appear in a
short discovery catalog. Native-tool requests also carry the selected definitions in
the API tool field required for native calls.

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
fields. Working-model user input serializes one complete, indented JSON object; system data
sections, compression inputs, and terminal output use labelled literal text.
Journals and HTTP transport encode JSON only at their destination boundaries.
Completed turns have three disk representations: `NNNNNNNN.raw.json`, an
individual `NNNNNNNN.compressed.json`, a `FIRST-LAST.grouped.json` narrative
for each run-aligned block of up to 20 turns made from individual summary files, and a
`FIRST-LAST.era.json` for each five consecutive blocks made from their narratives.
Grouped records retain covered turn numbers, call IDs, and artifact references. The
append-only session journal remains the recovery authority; resume rebuilds the
file-backed indexes, including for older sessions. Completed Calls inspector
records also use file-backed strings; opening a call streams its fields into
the viewer, and clearing the feed releases that inspector storage.

`turnstore.py` retains small metadata and file ranges in memory. Large strings
are read on demand by explicit history retrieval or streamed during selection
and serialization. Private hard links pin immutable versions while summaries
and late artifact metadata update their canonical files. Session teardown drains
background jobs before removing these temporary pins. Session deletion also
removes its turn files.

`sources.py` provides bounded text and JSON iterators. `requestbody.py` serializes
selected history and rendered prompt sections into one immutable temporary
request file before transport. Its small index references strings in that file;
HTTP sends chunks of at most 64 KiB with an exact Content-Length. Retries replay
the same snapshot. This disk spool prevents prompt edits or completed summaries
from changing an in-flight request. Dynamic templates and active input still
use memory while being rendered; the complete serialized request does not.
Request and viewer spools live under the application's data directory,
avoiding RAM-backed system temporary space.

The internal request callback passes this snapshot to diagnostics, token
accounting, and the context viewer. An explicit legacy `on_request` callback can
still request a materialized dictionary. Token estimates count serialized UTF-8
bytes incrementally, preserving the existing selection policy and calibration.
Complete history selections return structured content without character paging. Resume
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

`tools/discovery.py` owns the session's compact tool catalog and retained loaded
names. `Agent` applies the explicit allowlist before selecting core and loaded
definitions. Both the system definitions and native API tool list use this
selection. System definitions group core and additional loaded tools by name,
with indented JSON schemas; the remaining eligible tools have short table entries without
argument schemas. `load_tools` returns structured definitions and retains the
selected names for subsequent requests. It does not execute tools or expand the
allowlist. The selected definitions also form the tool batch's execution
allowlist, frozen before the response. `ToolBatch` rejects every call in a batch
before dispatch if any name is outside that set, returning an error for each call
without invoking tools or starting checkpoints. Loading a definition in the same
batch does not authorize its use. Recorded calls retain their definitions on
subsequent requests.
Reset clears discovery state, and saved load results/calls restore it on resume.
Reload candidates borrow the same catalog through registry services. MCP server
guidance enters requests when a tool from that server is eligible in the catalog,
including before its full definition is loaded.

The optional `tools/hashline.py` reader/editor pair is deferred and session-gated.
`/hashline off` unregisters both tools and clears their discovery selections;
registry rebuilds preserve the service-owned switch. Their `load_tools` results
acknowledge loading without repeating definitions in history. Full descriptions
and schemas appear in the following request only while enabled. The reader returns
line-number/content anchors and a SHA-256 revision of the full file bytes. The
editor validates that revision, all original anchors, and nonoverlapping edits
before publishing through the ordinary atomic-write/checkpoint path. Successful
edits participate in batch syntax checks. New lines use the file's line ending;
unchanged text and the final newline state are preserved.

`Agent` coordinates the pipeline. `ContextBuilder` receives immutable
`ContextSettings`, the conversation, token budget, and registry resources;
previews copy history/task and budget state. `RequestDriver` owns attempt-local
retry, overflow, repair, diagnostics, and acceptance state. Its typed hooks expose
queued input, the shared request budget, usage accounting, and stop controls.
`RequestAccepted` and `RequestStopped` distinguish outcomes without inserting
rejected calls as executable history. `ToolBatch` owns invocation provenance, ordered
observations, checkpoint scope, and post-batch checks. Its interruption callback
archives paired results before checkpoint cleanup and exception propagation.
Archival and compaction scheduling remain with the conversation coordinator.

Shared event contracts live in `events.py`. Earlier inline-summary readers live
in `contextlegacy.py`, with compatibility exports in `context.py`; decoded
legacy response envelopes pass directly to the protocol adapter. These class
relocations require restarting processes created before the transition.

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
   instruction hierarchy, fenced examples, and field tables. History occupies
   separate user-input metadata and result blocks rather than a system-prompt section.
   Prompt paragraphs occupy one physical line, without fixed-width wrapping. Two newlines separate paragraphs, headings, topic groups, task fields, tool guidance, examples, and assembled sections. Single newlines remain for meaningful item boundaries, metadata fields, code, and other structured content.
   Loaded instruction-file contents and original conversation text stay intact.
   The input uses system instructions followed by JSON user-input metadata and literal tool-result blocks in the same user message. The inspector
   displays the final body rather than reconstructing it separately.
3. Economical older records precede the recent full window. Each retained step
   occupies one object. Full records contain originals plus available compressed
   analysis. Older records use a summary if its estimated cost is lower than
   the original representation before adding analysis. The budget includes all
   fields actually sent, presentation indentation, and message overhead. Selection
   checks the fully assembled request, including excerpts, against the allowance.
   Both native-tool and embedded-tool profiles receive the same record format;
   ties keep originals, with complete tool batches intact. Every step has its
   own object in the user input's `history` list. `JSONInput` keeps the input
   structured until `Message.to_api()` renders metadata and tool-result blocks through
   `data_text.py`; the HTTP client then encodes the provider transport envelope.
   Tool-result text is copied verbatim into length-labelled blocks, with references in the JSON metadata. Explicitly structured results use pretty-printed JSON blocks. JSON-looking strings are never implicitly decoded. Literal rendering remains available
   for system data sections, compression input, and archive retrieval.
   Every step is sent in full while the history fits the budget. Over budget,
   `ConversationHistory.view` applies reductions from the oldest end until the
   estimate fits and the assembled request measures within the limit: step
   summaries up to two thirds of the steps, then each wholly summarized
   narrative, then each era, then summaries in the newest third (never
   the unseen batch), then omission from the very beginning. A summary replaces
   its original only when smaller. A narrative or era replaces exactly the
   records covering its listed source steps. Representation costs are cached per
   archived step and record, so selection is one pass. Each request starts again
   from full originals; there is no persistent omission marker.
   `--context-tokens` can cap total working context.
   `records.py` represents saved full steps with `record_type`, `representation`, `user_prompts`,
   `agent_response`, `tool_calls`, and `tool_results`. Prompts remain an array so
   queued inputs retain their boundaries. Calls use `call_id`, `tool_name`, and
   argument objects; results use matching IDs, names, status, and content.
   The harness's exact observation envelope is unpacked, while arbitrary JSON
   inside tool output remains content. `linked_tool_results` projects observations
   onto their matching calls for the working input: `result` for success, `error`
   for failure, and `unclassified_result` for unknown legacy status. It preserves
   raw values and leaves saved records unchanged. Legacy observations without a
   call retain their ID and name without invented arguments.
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
4. Current-step metadata in the system prompt supplies `step_id` and
   `is_tool_result_response`. One user-input JSON metadata object contains `user_prompts`
   (new messages in chronological order, or `[]` without new input),
   `retained_user_prompts`, `history`, and `errors`. Later user messages supersede
   conflicting earlier instructions. Tool errors belong to their
   calls through references to literal result blocks; `errors` holds harness corrections not attached to a call. Keys label
   content without inserting labels into its values. If the supplied results
   satisfy the user's request, return the result with no tool calls to finish;
   request more tools only when necessary work remains.
   Newly returned results cannot leave the full
   window before the working model receives them, even when their independent
   summary finishes first. Oversized observations use an `excerpt` record with
   original-part retrieval instructions as a last resort. Text fragments and
   omitted-character counts occupy separate fields; clipping never splices
   descriptive markers into message text. Excerpted calls use `result_excerpt`,
   `error_excerpt`, or `unclassified_result_excerpt`; other bounded messages
   occupy `response_excerpts`. Request assembly does not append its JSON envelope,
   tool definitions, or system guidance to archived turns.
5. `TaskMemory` retains original requests independently of compression. Request
   assembly supplies the active source messages in `retained_user_prompts`,
   with their source step ID in system task metadata. Retained input counts
   against the endpoint budget. The active prompt is independent
   of the historical representation of the call that first received it.

The selected model's context/prompt limits constrain the whole request. The
budget reserves output, system instructions, tool definitions, response schema,
and a 15% estimation margin before fitting history. Text estimates use UTF-8
size, not a model tokenizer. History uses the remaining model allowance;
`--context-steps` is retained as a deprecated compatibility option; the newest
third of retained individual posts determines the full-call target.
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

Working requests also pass a 120-second first-output deadline to the streaming
transport. It covers connection, upload, and waiting for output; upload pacing
waits suspend it. Keepalives,
role-only deltas, and tool-call IDs without a name or arguments leave it armed.
Content, reasoning, or tool-call content disarms it so active generation can
continue. Expiry closes the stream, saves captured diagnostics, and records a
failed history turn. It bypasses transport retries and reaches the existing
optimal fallback; manual mode retains its model and reports the error. Background
compression omits this deadline and keeps its own timeout.

`capabilities.py` selects native tool calling from the live endpoints' `tools`
support, independently of their JSON-format support. Startup and explicit model
selection refresh the profile; ordinary requests reuse it. Native-capable routes
take priority. Schema-capable endpoints receive a small strict response schema
unless disabled by the model profile.
Other endpoints receive no `response_format`; ordinary replies are accepted.
Native routes without schemas request plain reply text. JSON-only routes request
`response` and `tool_calls` on every step, including final answers.
Selection preflight publishes neither the model nor its profile on failure.

`modelscores.py` stores calibrated text-coding scores; `ModelInfo.score` exposes
the same model/version estimate across providers. Higher scores sort first,
after free/paid grouping, with provider/model IDs breaking ties. Zero is a valid
score; `None` indicates missing evidence. Catalog discovery announces newly
encountered unscored free models once, asking the user to update SlipAgent.
Reviewed evidence gaps and generic routers remain manual choices without a
new-release notice. Cached catalogs refresh known scores from the curated table
on each access, so compatible table edits apply without a catalog request.
The offline maintenance script uses saved benchmark calibration; runtime code
loads only final scores and exact aliases. See [model scoring](model-scores.md).
Catalog inventory expires after five minutes; working and compression selection
share the routing lock and refreshed inventory. Successful catalogs replace each
provider's available IDs, including empty catalogs. Transient catalog failures
(connections, rate limits, and retryable HTTP errors) preserve that provider's
cached entries; definitive authentication failures do not. The curated score table
retains removed IDs so a returning model can be selected without another ranking
update.

`Session.model_routing` owns optimal mode, purpose-specific route failures,
provider authentication failures, credential identities, and catalog locking.
CLI startup enables optimal mode unless `--model` or `--provider` explicitly
selects manual mode. Fetch candidate catalogs alongside the other startup
services, then preflight selection after MCP, environment, and history restore
finish. Startup failure closes any provider client selected during this phase.
Automatic eligibility requires a configured key, verified free pricing/tier,
and a known score. Billed Google accounts are excluded. `/model optimal` and
`/model reset` clear all exclusions and preflight candidates by descending score;
manual selection disables working fallback until `/model optimal` is used again. This mode and
failure state are process-local; automatic selection does not save the model, but whether optimal is on is a remembered preference.
Credential/endpoint changes clear that provider's exclusions; conversation reset
clears all exclusions. The migration from rank to score in the stable `ModelInfo`
dataclass requires a restart. Subsequent score-table edits are reloadable.

`Agent._step` checks current eligibility before requesting a response. Normal
request recovery happens first. When a working request exhausts recovery,
optimal mode excludes its route and selects the next model by score, rebuilding context
with retained task/history and durable failed attempts. A provider HTTP 401
excludes all its models for both purposes. If no route remains, further prompts
stay stopped until a reset, credential change, or manual selection. The fallback
boundary encloses model requests only: accepted tool batches and test failures
do not trigger it or replay tools. The replacement stays selected; score alone
does not move it back to a previously failed model. Stop and request limits still
take precedence.

Transport exceptions use `APIConnectionError` and retain the selected model
after ordinary retry exhaustion. Metadata connection failures also leave routing
exclusions unchanged. Explicit provider errors and the two-minute silent-output
deadline remain eligible for fallback. Queued `/model`, `/models`, and `/provider` commands apply between
batches in FIFO order without requesting a task stop.

`prompts/model-profiles.json` applies exact-model prompt sections, optional
tool-channel overrides, and a boolean `structured_output` switch at request
construction. `protocol.model_response_format` supplies the same effective
constraint to prompt construction, context budgeting, and the outgoing API body.
Setting the switch to `false` omits `response_format`; native-tool models receive
plain reply instructions and retain native tools. Embedded-tool models retain
their JSON call instructions and local validation. Omission or `true` follows
endpoint capabilities. Both Dots3 Note Preview IDs disable the constraint.
GLM-5.3 and GLM-5.3 Flash use the `glm_xml` channel: their system prompt lists
tools and the `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`
call format exactly as GLM's chat template does, earlier calls are shown in that
form, and no API tool definitions or response format are sent. Native API calls
are still accepted if returned. Other models retain
their capability-selected channel. Profile edits and model switches apply when
the next request is built; no model-specific instructions are hard-coded in Python.

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
Complete streamed responses are normalized before effects. Calls need a tool
name and object arguments; arguments may arrive as JSON text or an object, and
omitted arguments default to an empty object. Extra metadata and optional type
labels do not reject calls. Nested function fields take precedence over flat
fields; argument aliases are read in order: `arguments`, `input`, `parameters`.
Usable IDs are preserved, while missing, invalid, or repeated IDs receive unique
local IDs. Streams use supplied indexes, otherwise call IDs or delta positions.
Malformed or truncated argument JSON still rejects the response. Explanatory
code examples and reasoning are never scanned for actions.
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
`streaming.py` incrementally previews plain prose and top-level JSON `response`
strings, including JSON fences. Its lexer retains split escapes and surrogate
pairs, skips other fields and nested values, and never repairs malformed JSON.
Line-prefix filtering keeps bookkeeping and top-level call carriers out of
previews while preserving fenced examples. Other carriers wait for acceptance.
Providers returning a complete JSON HTTP response instead of SSE also wait for
acceptance; the transport does not publish a buffered reply as a content delta.
Reply previews and separately delivered reasoning can appear before validation;
all tool effects wait. A rejected response is not appended as
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
history or summarization wholesale. A separate failed-turn record does enter
history, with explicit unexecuted status and a bounded excerpt. Cancellation and stream failures end the visible
reasoning/reply block before a retry begins. Failed reply previews receive an
explicit warning. Each attempt starts with a fresh decoder and renderer preview
record; successful final reply events remain available to automation and saved
history, while the renderer suppresses a duplicate of already streamed text.

`requestfailures.py` validates failed-turn metadata and derives retry counts from
the archived steps. `RequestDriver` invokes the coordinator's recorder through
registry services before retrying or returning a failed attempt. The recorder
archives a labelled harness notice with no executable calls, attaches
`request_failure` to the completed step, persists it, and emits `step_end` with
that attempt's usage. Each attempt receives its own global turn number and
consumes one working request. A deterministic failure summary replaces the
ordinary background summary call; partial model prose remains diagnostic data.

Rebuilt requests carry the failed turn, consecutive and session failure counts,
and `request-retry.md` guidance to honor user stop/no-retry instructions. Earlier
tool results remain unacknowledged until an accepted response follows them;
intervening failures cannot move them into a summary before the model receives
them. At small budgets, explicit excerpts preserve both that batch and the latest
failure. Provider overflow recovery compares the smaller view before adding new
failure bookkeeping so an unchanged history does not trigger another request.

Session journals store failed-turn metadata in `request_failure` events alongside
the canonical notice and step summary. Resume and fork validate the outcome,
counters, diagnostic reference, and absence of executed tools. They restore the
metadata and any preceding unacknowledged batch; archived attempts never replay.

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
indices are journaled so unsent prompts remain pending on resume.
`InputQueue` gives unsent messages transient identities for `/queue` inspection
and selective discard, including duplicates and messages already drained into
the conversation. Submitted messages cannot be discarded. Queue signatures
invalidate a prepared request when input changes before submission. A single
`discard_input` journal event removes the unsent entry and adjusts queue/task
indices atomically; accepted steps and their compaction metadata retain their
identities. Resume reconstructs original task sources without discarded input.
Submission emits one receipt per prompt. The terminal holds waiting prompts in
a separate area immediately above the bottom controls, with one clipped line
per prompt and no queue notice. Its width follows the main pane when the
inspection sidebar is visible. Receipts remove matching previews in arrival
order, including identical prompts, and append the full user text to the feed
before the submitted request's output. Queuing alone does not interrupt streamed
prose or reformat transcript history. Reload migrates older unsent feed entries
into the waiting area. Plain stdout remains an append-only log.

Synchronous file reads, edits, writes, listings, glob scans, and instruction
preflight checks run in owned worker threads. The terminal event loop remains
available during filesystem waits. Cancellation drains each worker before
propagating; glob scans also receive a cooperative cancellation signal. Atomic
edits finish before their owner releases the batch boundary. Async HTTP, MCP,
and subprocess operations retain their event-loop resources.

State-changing slash commands queue in FIFO order while work is active. The
agent finishes its response and whole tool batch, then applies commands while
idle. Settings changes resume the same work with its original request budget
and repetition guard. Explicit stop holds earlier input until a new submission;
exit prevents automatic continuation.
Reset, resume, and starting a new task do not restart the replaced task. Command
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
starting a summary request. Deferred commands are discarded. After an active
agent run finishes cancellation, queued user prompts start a new run unless the
session is exiting; an empty queue leaves it idle. Worker-thread operations
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

`fileio.py` opens canonical regular-file paths through nonblocking descriptors
and validates the opened file type. POSIX directory traversal refuses substituted
symlinks. File tools, checkpoint reads, editor loading and indexed viewers share
that reader. Built-in text reads and exact-edit inputs keep the 32 MB bound.

After archiving a complete batch, `Agent._archive_step` submits it to
`StepCompactor`, a registry-owned service. One dispatcher runs step summaries
(oldest first), then due narratives and eras, with at most one job per
background provider at a time; a single slot runs jobs in order on the
dispatcher task. The ready queue holds at most 32 original-step references. Further steps retain deferred request
metadata on their originals and refill the queue as it drains. Only the active
request copies its immutable original prompt, response, calls, results, thoughts,
and up to 25 earlier steps. CLI sessions attach `CompressionProviders` and the
session-owned `ModelRouting` service. Both compactors use one selection/retry
loop, rotating through each configured provider's best ranked free model.
The selected coding provider is always excluded. `CompressionProviders.slots`
counts the remaining providers; selection reserves an idle provider and waits for
a release notification when every eligible alternative is busy. Completion and
cancellation release the reservation and wake waiting requests. With no eligible
background provider, a notice asks for another provider's key and summaries are
marked unavailable. Each attempt uses its destination's own
profile; every failure advances rotation, including failures during capability
lookup, context checking, and response validation. Concurrent jobs also avoid
their own last failed provider when alternatives remain. The compactor owns
and closes these separate clients. Without an eligible free route, originals
remain available and a notification reports compression unavailable. Direct embedded
callers without routing retain their explicitly supplied model/profile.
The working loop does not wait. A reply without tools is scheduled the same way. The
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
of primary step accounting. Both compactors omit temperature regardless of the
working model's explicit setting, and use the selected destination's
reasoning/routing settings for each attempt.

`historymerge.py` provides the narrative and era jobs. `run_numbers` derives each
turn's run from saved records: a turn with user input starts a run, as does a turn
after a reply without tool calls. `blocks` cuts history into narrative blocks that
close at a run boundary once they hold ten turns, or at 20 turns within a run; a
block closes using only its own turns and the next one, so its boundaries never
move. Sessions with fixed 20-turn narratives (records without `runs`) keep fixed
blocks through the last of them. A block is ready once every turn has a
first-pass summary or a failed first pass (a gap); five consecutive blocks are
ready for an era once all their narratives exist. The complete
source must fit the provisional route budget; sources are never truncated. All
jobs call `StepCompactor.request`:
transient transport failures retry persistently with shared API backoff. Other
failures advance providers and exclude that route for compression; a context
fit failure excludes it only from this job. Failures are scoped to provider/model,
except HTTP 401, which excludes the provider for both uses. Failed routes are
reused only after healthy candidates run out. Stop after more than the greater
of five or the remaining ranked free candidate count nontransient failures.
Explicitly unrouted embedded callers retain five invalid-response retries.
Validation stays inside the retry loop and publication follows validation. A final failure retains the
attempt marker; that block is attempted again once 20 further turns have passed.
One response contains only a `summary`, capped at 12,000 characters and required
to cost less than the records it replaces. A narrative is a cross-turn account,
not a list of per-step recaps. Its prompt preserves attribution, uncertainty,
source turn citations and exact retrieval identifiers.

The block's last `CompletedStep` owns its `historical_summary` record, and the
last step of an era's fifth block owns its `era_summary`. Narratives retain
`step_ids`, `runs`, and `sources` (turn IDs, runs, call IDs, names and artifact references)
copied by the harness, independent of generated prose; eras retain `step_ids` and `runs`.
Anchors store `history_merge_attempt` and `era_attempt` before dispatch; journal
`history_merge` events persist all four fields. Resume validates IDs and metadata
shape; a narrative without `runs` must match a fixed 20-turn block, and one that
does not is not restored, so its turns are summarized again. Fork retains the records and
attempt markers. These optional
attributes initialize lazily on existing mutable completed-step objects, so no
stable dataclass layout changes. The existing compactor owns all merge tasks,
generation guards, usage callbacks and reset/shutdown cancellation. No working
tools run during merging. Originals and individual summaries are never rewritten.

Reset increments the compactor generation and cancels all jobs. Late transport
results cannot publish summaries, errors, or usage into a new session. Shutdown
cancels and awaits jobs before closing the client. `Agent.wait_for_compaction`
gives the entire queue a 120-second grace period for fork, resume, deletion, and
one-shot completion. `StepCompactor.settle` shields the drain until the deadline,
then invalidates the generation, cancels active/deferred jobs, and finishes their
cleanup. Caller cancellation also drains cleanup before propagating. Originals
survive even when no summary finishes. Replacing the API key cancels and drains jobs before retiring their
client. The shared service survives behavior reloads. Its lazy initialization
is an explicit migration for sessions created before the service existed.

`recall_history` returns complete selections without character limits or paging.
List/search results include `total_matches` for the entire filtered result,
including empty results. A single `section`
selects `prompt`, `response`, `reasoning`, `tool_calls`, or `tool_results`; `sections` selects
several as one JSON object. Without a selector, it returns the full step with its
original user prompt. `section="user"` returns only the original new user messages for
source recovery. `call_id` selects a tool observation by default, or one call
with `section="tool_calls"`. A batch ID is not session-unique: use
`(step_id, call_id)` as provenance. Invalid or conflicting selectors return a
tool error. The schema and direct `run` path share the allowed original-part
names; unknown singular or plural selectors are rejected before accessing the
record. CLI sessions journal these records by default; `--no-session` keeps them
in temporary files until reset/exit. Embedded Agent callers opt in by attaching a
`SessionJournal` to registry services.

`TaskMemory` owns original user inputs and their source step references. The latest user input remains available verbatim until new input replaces it. Queued inputs received together share a source step. `/task` displays original prompts, and `/task new` changes the source boundary without deleting history. Session resume reconstructs prompt retention from original messages and ignores legacy model-maintained task snapshots.

Legacy parsing helpers remain for imported records and compatibility fixtures;
the active agent never requests inline compressed fields, XML memory envelopes,
or metadata inside tool arguments.

## Resource Ownership and Project Settings

`sessionstate.Session` owns the API client, registry, MCP manager, renderer, terminal lifecycle,
and optional runtime frame. Startup uses `AsyncExitStack` until ownership transfers
to the session, so cancellation/preflight failure closes acquired resources.
Interactive terminals open in Working before service initialization. Welcome lines
appear 60 ms apart, followed by progress and individual MCP connection results.
Catalog/capability discovery, quota, repository-map warming, Python discovery, MCP,
and session recovery overlap; context validation follows their completion. Provider
capability discovery owns catalog loading, avoiding duplicate catalog requests.
The CLI uses four default executor workers and at most four concurrent MCP starts.
Prompts and settings entered during startup queue until initialization completes;
Esc, `/exit`, `/quit`, or EOF cancel startup and drain owned work before shutdown.
The repository worker finishes before its resources close. Reload watching begins
after startup settles, preserving the editor's boundary callback when present.
MCP negotiation prefers modern per-request metadata and retains the legacy
handshake fallback. If initial discovery failed but the handshake rejection
identifies a supported modern version, discovery is retried once on that same
connection. Disjoint versions and a failed recovery remain visible errors.
Shutdown cancels command tasks, closes MCP connections, closes tools and registry
services, closes the API client, and releases the runtime generation.

`SessionResources` supplies typed resource references over the authoritative
`ToolRegistry.services` compatibility map. `SessionWork` similarly exposes active
tasks and deferred commands over `Session.extensions`; writes through either
interface refer to the same objects. New code uses typed references for known
owners, while extensions and runtime metadata can retain dictionary entries.
`WorkCoordinator` owns the agent-boundary gate and the exact reload frame held
by an editor. `UserEdits` specializes it with editor opening and diff delivery.
Changing focus never releases that gate. Session layout, resource adapters, and
work-gate contracts require a restart from the earlier architecture.

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
Tools close before services. `SessionResources.shutdown_keys` explicitly closes
editing, compaction, jobs and language servers before their durable stores;
jobs drain before artifacts and the command archive close. Unknown extension
owners retain reverse insertion order, and shared identities close only once.
Captured generation closers
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

`CreatePythonEnvironmentTool` borrows the session's `ProjectEnvironment` owner
and is exclusive. It returns an existing selection unchanged, refuses ambiguous
discovery, and creates a missing target directory exclusively before running
standard `venv` in a bounded subprocess. Existing target directories are only
validated, never recreated or repaired. A successful virtual-environment probe
precedes session selection, whose source is labelled `create_python_environment`.
The tool does not install project dependencies or modify project settings;
custom selection paths therefore need explicit configuration to survive restart.
Failure and cancellation preserve partial files, and retries cannot overwrite them.

## Command Output Is Separate From the Context Preview

Shell and Git capture drain stdout/stderr concurrently. Their bounded observations
include up to 32,000,000 characters per stream, keeping the beginning and end
when that limit is exceeded; the archive receives all decoded text while
quota permits. Timeout and cancellation kill the process group, with a shared
two-second deadline for process exit and pipe draining. If detached descendants
keep pipes open, cleanup closes the subprocess transport and settles the readers,
preserving captured previews and decoded logs. Timeout results and retained log
metadata report incomplete output and potentially surviving detached processes;
the number of uncaptured bytes is unknown. Spawn-cancellation cleanup and
background-job shutdown use the same bounded path. POSIX process groups cover
ordinary descendants; shell execution is not an operating-system security sandbox.

Each subprocess has a random log ID, with `(step_id, call_id)` copied from a
context-local invocation marker set by the agent. One tool can spawn multiple
subprocesses. The archive uses private generated filenames, never call IDs or
model-supplied paths. `read_command_output` resolves IDs through its own index.
It reads stdout/stderr with byte offsets and a default and maximum limit of
32,000,000 characters, or lists bounded
metadata with record offsets. The response always states which units apply.

The quota defaults to 1 GiB across both streams of all commands in the session.
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

## Shared Command Payload Format

Live logs retain private append-only stream files. After closure, `payloads.py`
copies each retained UTF-8 stream through a bounded decoder and SHA-256 digest
into the artifact store's `payloads/<digest>` directory. Existing identical
payloads are verified and reused. No original stream is removed during staging.

Version-two artifact headers use `<record_id>.v2.json`; their flat JSON contains
`format_version: 2`, output/error digest strings and a literal retention-error
suffix. The index records each payload's digest, byte count and character count.
`record_fields` validates the header against the index, streams verified payloads
to the existing bounded page/snapshot writers, and appends the error suffix.
Version-one inline JSON records retain their reader. Unknown versions, invalid
digests, symlinks, mismatches and corrupt payloads are errors, not fallbacks.

`CommandLog.consolidate` publishes the payloads, header and index before changing
its stream paths. The existing journal callback then persists stream `payload`
identifiers before `remove_legacy_output` removes redundant raw files. Version
two uses file fsync and, on POSIX, directory fsync before that barrier. Failures
after index publication preserve its referenced payloads and the old files.
Failures before publication roll back newly unreferenced payloads. Recovery
validates leftover headers and copies before retrying migration or cleanup.

Resume/fork resolves journal payload IDs through the session artifact directory;
old journal events can also resolve the current artifact index after an interrupted
migration. Legacy output is compared incrementally before conversion, retaining
record IDs and provenance. Fork copies the artifact directory once, so shared
streams do not get a second log-directory copy and do not depend on parent files.
Command quotas count retained bytes; inspection quotas charge header plus logical
payload bytes per record. Shared physical storage is not the sum of those counters.
Non-command artifacts and conversation journals retain their existing formats.

## Background Command Ownership

`run_command(background=True)` creates a `CommandLog` with the initiating
step/call provenance, then hands its execution coroutine to `CommandJobs`.
The manager owns the process through completion, timeout, cancellation, reset,
and shutdown; rebuilt shell tools borrow it. Process creation is shielded so
cancellation cannot lose a child between spawn and handle assignment. Capture
reuses foreground capture and its bounded cleanup of processes and pipes.

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

## Command Event Subscriptions

`CommandJobs.monitors` explicitly migrates a retained job owner to a lazy
`CommandMonitors` instance. `monitor_command` borrows that owner across reloads.
It records `current_invocation` as the initialization turn/call, replays a bounded
retained tail without yielding, then attaches a live callback to `CommandLog.write`.
The callback runs after retention accounting, including when quotas discard bytes;
matching live text and reporting storage loss are separate concerns. Literal
matching retains at most a pattern-sized suffix per stream. It never executes
regular expressions or joins stdout and stderr.

Each subscription owns deadline and coalescing timer handles. A callback on the
existing job task observes completion even for a job launched before the feature
was loaded. Cancel/timeout completes only the subscription. Ready results contain
bounded output tails, explicit omissions, loss counters, and initialization
provenance, and are saved through the session artifact store. Acknowledgement
removes only the live reminder. Eight retained monitors bound memory and prompt
growth; unacknowledged results are never evicted to admit another watch.

The existing idle/batch-boundary notification path displays monitor notices
without launching requests. Context assembly independently snapshots live state:
setup instructions when empty, active reminders while watching, and inline ready
results until acknowledged. Previews do not acknowledge or consume results. All
three sections obey tool availability. Output is labelled as untrusted data.
Repeated notices coalesce for five seconds; terminal outcomes flush immediately.
Raw command output follows the existing retention quota regardless of suppression.

Jobs settle before monitor timers are cleared and archives close. Compatible
reloads retain subscriptions; reset/resume/fork/deletion/exit clear live state.
Saved result artifacts retain initialization metadata after restart, but prior
process subscriptions are not reattached or replayed automatically.

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
and incoming messages at most 256 MB. Only UTF-16 wire positions are negotiated;
tool input uses 1-based Unicode character columns and converts to wire units.
Returned locations label their 1-based UTF-16 columns explicitly.

Queries serialize document synchronization, open current UTF-8 contents, and
close after the response. Workspace metadata scans before navigation notify
servers of changed, created, and deleted files, excluding generated/dependency
trees through the normal workspace traversal. The current file is always opened
explicitly even if excluded from recursive traversal. Configuration requests are
answered from explicit settings. Server-originated workspace edits are refused.
Locations and symlinks follow `Workspace.resolve`: confinement permits the project
and configured system temp directory, while danger mode permits other paths.
Hover text and locations fit within the incoming message limit; locations from
the requested offset onward are returned without a separate page-size cap. Errors retire the failed client
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
shared platform data directory by default, or `SLIPAGENT_STATE_DIR`. Project directory
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
While a journal keeps that default, requests carry the dynamic
`response-session-title.md` section, and the agent saves the reply's optional
`session_title` field through the same rename path. A title already set, by
`/rename` or an earlier reply, is never replaced, and the section is then omitted.
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

`/fork` settles compaction with the two-minute grace period and stops jobs, checkpoints the current journal, and
uses the validated load/restore path to create a populated child. Its parent
stays immutable after the fork; copied command output remains independently
available. The transcript stays in place and subsequent work targets the child.

`/delete` requires an interactive Enter/Escape chooser before any deletion.
After confirmation it settles summaries with the same grace period and stops jobs, clears the conversation with
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
Backup copying, integrity checks, and restore writes stream 64 KiB blocks. Rewind
plans retain hashes and permission modes instead of file contents. Diff previews
read small files within a 200,000-byte combined limit and describe larger restores
without loading their contents into memory.

`/rewind` restores the selected batch and all subsequent active checkpoints. It
validates every affected file and backup before the first mutation, and checks
each destination again immediately before publication. Each restored backup is
also hashed while copying into the temporary file, before atomic replacement.
Conflicts preserve the workspace. I/O failure
during restoration reports any files already restored. Original bytes and modes
are restored; files originally absent are removed. The conversation is retained,
and a factual context notice identifies restored files. Resume restores checkpoint
storage and notices; fork copies backups and indexes into its own sidecar directory.
Project files remain shared between forks. Shell/MCP effects and external edits
are not checkpointed. Reset detaches storage; deletion removes the selected sidecar.
Without journals, temporary storage lasts until reset or shutdown. Queued rewind
never resumes the interrupted run automatically.

The CLI exposes `/sessions`, `/resume [id|latest]`, `/fork`, `/delete`, `/rewind`, `--resume [id]`, and
`--no-session`. `/new` starts a new journal while retaining prior sessions.
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
outcome, reported usage, and complete response/error details. A separate compressed
response captures all provider fields before parsing. Streaming choices, calls,
and reasoning blocks are assembled by index; text fragments join into continuous
fields. Interrupted and rejected responses are retained too. `responsecapture.py`
wraps the HTTPX response byte stream before content decoding, recording each chunk
in an owner-only `NNNNNNNN-stream.bin.gz` file. Each chunk flushes for inspection
while a request is pending; closing finalizes and fsyncs the gzip stream even on
parser failure or cancellation. The attempt records HTTP status, response headers,
and whether an injected/prefetched response was already decoded. Real transport
captures retain provider content encoding, comments, empty events, malformed bytes,
and line endings; TLS and HTTP transfer framing are outside this capture. A logging
write failure reports an error while allowing delivery and transport cleanup.
Exact final request JSON is gzip-compressed and deduplicated by SHA-256; request
authentication headers are excluded. Pending records survive a crash and remain visibly unsettled. Atomic
owner-only writes and fsync preserve earlier records when storage fails. The
archive targets 32 MiB by default, evicting oldest attempts after each durable
write. Eviction removes attempt metadata, assembled and raw responses, and request snapshots
whose last reference was removed. Shared request snapshots remain readable.
The newest attempt stays complete even when it alone exceeds the target;
later attempts can evict it. Unknown files are preserved. Storage errors warn
without blocking ordinary work. Compatible reload clears the former hard-limit
error and retains custom byte targets and unrelated I/O errors. A subsequent
successful attempt clears the stale model-facing diagnostic warning. These files
are not history steps. `/requests [attempt]` inspects
saved attempts. Reset starts a new directory; resume reuses the saved session's
directory, continues after its highest attempt number, and counts existing file
sizes toward the rolling target. Even incomplete attempt files reserve their numbers.
`--no-session` uses temporary storage.

## Terminal Components and Invocation Identity

`terminalbindings.py` builds keyboard commands and their focus filters;
`terminalmouse.py` routes pointer events among visible regions;
`terminallayout.py` assembles the layout and styles around retained controls.
`TerminalUI` delegates through its existing presentation entry points, so reload
rebuilds callbacks without replacing the application, buffers, or panel owners.
These modules hold no independent session state.

`StatusLine` retains notification expiry and flashing, retry and command status,
navigation hints, rotating-tip timing, and marquee positions independently of
layout rebuilds. `TerminalUI` supplies contextual tip choices and clickable
actions. Older terminals migrate their existing status fields on first use.

`MainArea` selects the editor, viewer, context, or feed and supplies its focus
and scroll targets. It owns inspector visibility and split-width calculations,
shared by the layout, queued prompts, editor header, pointer routing, and
transcript cache warming. Pane buffers and viewer resources retain their existing
owners. `FocusCoordinator.panel` builds their common title/body structure.

`MenuLifecycle` builds footer menus and restores focus, clears notifications,
and invalidates the display on acceptance, cancellation, or failure. List choices
and settings fields retain their own pending futures across layout refreshes.
`TranscriptWindow.seek` resets viewport position and pending scrolling for feed,
viewer, and context navigation.

`terminalmouse.limit_mouse_reports` replaces the VT output's mouse-enable sequence so the
terminal reports clicks, drags, and the wheel but not every pointer movement, which
keeps a slow link from flooding the input parser.

`ContextView` stores the latest request's display blocks and wrapped rows in
private temporary files. Opening it builds rows once per width; repaint reads
through a 128-row cache. Replacing the request closes the old snapshot, and
terminal shutdown closes the current one. Legacy in-memory blocks migrate on
refresh while retaining the context window. Full request contents remain
inspectable, including message metadata and provider options.

Request accounting retains a scalar token estimate during inference rather
than a second payload tree. Internal request callbacks receive the file-backed
body; legacy dictionary callbacks materialize an isolated copy. Exact diagnostics
and context snapshots are captured before transport.

Slash-command definitions live in `commands.py`. Each `Command` descendant
owns syntax, help, menu insertion text, aliases, execution, and argument-dependent
scheduling policy. Shared presentation and saved-session bases supply common
preconditions; descendants implement the specific operation. CLI entry points
remain adapters for reload and callers. Help, startup command listings, and
menus are generated from the definitions; descriptions are never parsed to
decide behavior. Credential commands remain outside model history and journals.

`ListPanel` owns stable-key selection, clamping, selection visibility, rendering,
and mouse activation. `TouchedPanel`, `CallsPanel`, and `ChoicePanel` supply their
items, labels, and opening policy. `ListActions` supplies explicit host callbacks;
the components do not depend on `TerminalUI`. Wrapped manual scrolling belongs
to `uiwindows.TranscriptWindow`. Sidebar wheels scroll the viewport; bottom-menu
wheels move selection. `ListWindow` defers selection visibility until rendering
supplies the viewport dimensions, then scrolls the minimum number of wrapped
screen rows needed to reveal the whole selected item. Items taller than the
viewport reveal their beginning. Resizing recalculates visibility; manual sidebar
scrolling preserves its position until selection moves. Those policies remain distinct.

`FocusCoordinator` owns a forward route through the prompt, active feed/editor/viewer/context,
links and lists, and footer actions. Tab includes indexed feed links and reveals
each selected item. Shift+Tab skips individual feed links and selects the next
UI component; it also works from the editor, whose plain Tab inserts indentation.
Lists are single stops with their own arrow navigation. `WrappedTranscript`
indexes linked rows as they are written, so discovery does not scan scrollback.
Tabbing temporarily shows the component-selection tip without flashing or
discarding the latest event notification.
`TranscriptLayouts` retains at most two file-backed wrapped layouts and prepares
the alternate Inspect width in an application-owned task, yielding between
records. Both layouts consume new output incrementally. Theme/source changes,
transcript replacement, and reload invalidate them; closing cancels prewarming
before its files close. Each rendered feed snapshot checks that its layout is
still active and retained. Retired snapshots return empty rows, and mouse routing
rejects their hit maps until redraw. Keyboard link highlighting preserves that
validity check, so a stale click cannot activate a replacement action.
`ActionControl` retains rendered handlers for keyboard activation and focus styling;
keyboard Enter invokes the same enabled action as a left click.
Mouse presses use the same focus owner before native mouse dispatch, so input
controls also position their cursor on the first click. Panel titles target
their panel bodies; list-title focus reveals the selected item without opening it.
The coordinator also owns restoration of invoking controls and reconciliation
when resize hides the focused region. Menus and options retain their invoking
control and restore it only while its region remains available. Save completion
does not move focus. Navigation bindings check focus rather than visibility.
Editor and options operations use that
owner rather than choosing focus destinations independently. Focus never
changes the editor's hold on agent activity. `ActionLink` supplies common hover
and left-click behavior; `CallbackLink` invokes synchronous actions and
`TaskLink` submits asynchronous actions to an explicit application task owner.
Links cover their rendered labels; list hit testing remains a list concern.

`ViewerDeck` owns the read-only viewport and asynchronous-open tokens.
`TextViewerPanel` owns its indexed snapshot and renderer; `DiffViewerPanel`
selects the diff renderer, and `FileViewerPanel` owns source-change checks.
Replacing or closing a panel releases its snapshot exactly once. A completed
check for a replaced panel cannot alter the current panel. Calls and ordinary
output use the shared text panel. This ownership transition requires restarting
processes with the earlier viewer state; reload rejects that state explicitly.

Saved diff panels retain a compact record/step/call reference for their whole-diff
comment action. The action attaches the reference to the existing prompt draft
and focuses it without submitting or holding the agent. `TerminalUI` migrates the
draft attachment and prompt-label processor during layout rebuilds. Existing
panels without a reference gain the action when a saved diff is opened again.
The attachment is consumed only by an ordinary nonempty prompt submission;
commands and secret input do not consume or inherit it. Detaching retains the
draft, and clearing the feed clears the attachment.

`diffcomments.py` serializes references in a labelled footer of the ordinary user
message. Queue acceptance/discard, failure retries, original-request retention,
and journal resume therefore use their existing paths without separate delivery
state or a new journal format. `ContextBuilder` supplies `diff-comment.md` only
for valid references in the latest contiguous group of actual user messages.
It persists across that group's tool calls and disappears for a later ordinary
turn; tool-result text cannot activate it. The system section identifies whole
saved changes, whose current files may have changed since the original edit.

`AgentEvent.step` is the conversation step ID, matching archival, checkpoints,
tool provenance, and artifact references. A new user request does not restart
that identity. Run-local request budgets and diagnostic attempt numbers remain
separate. A tool invocation is identified within its session by `(step_id,
call_id)`; a call ID alone cannot identify saved output across turns.

The list extraction moves the scrolling window class and persistent selection
state. Processes started before that transition need a restart; compatible
behavior reloads after it preserve the component owners and their selections.

## Reload Transactions

`runtime.py` is the stable frame. Its `CORE_MODULES` set also pins package root,
configuration, wire types, their `data_text` renderer, workspace, tool base, MCP
connections, `lifecycle`, typed `resources`, `work` ownership contracts, and the
`plugins` loader. These classes/state contracts must agree for the session's lifetime.

User plugins in `data_directory()/plugins/` are watched with the package source
under separate `:plugins/` keys and are never imported as components. Each
reload stages the candidate, then `plugins.overlay` applies the approved members
of every plugin onto the candidate classes before the component API, layout,
registry, and terminal checks; the normal merge then carries them to the live classes, and
a rejected generation leaves them as they were. A plugin that fails to load or
apply is rolled back on its own and reported once; the generation continues.
The `original()` registry distinguishes wrapper code by object identity, so
identical wrappers from separate plugin files or reload generations retain their
own predecessors. Weak-reference cleanup removes only that wrapper's registration;
failed overlays remove only the registrations they created.
Each property override registers its getter, setter, and deleter code separately
against the matching accessor of the replaced property. `original()` invokes that
accessor directly; an absent predecessor accessor raises `AttributeError`.
Startup applies the same overlay to the live classes once per process before the
session is built. Before compiling or executing a plugin, the loader checks a
file-level approval against the SHA-256 digest of the exact source snapshot.
Unapproved or changed files are listed without executing Python. Once file code
is approved, member decisions separately govern which overrides apply.
Approvals and conflict choices live in `plugins/.approvals.json`,
which is watched with the plugin files: an override without a decision stays off,
and a method that several approved overrides target applies none of them until a
run order is chosen. Malformed approval fields invalidate all saved decisions and
produce a diagnostic without aborting startup. The REPL asks about unapproved
file versions, undecided overrides, and conflicts once the
terminal is up, and `/plugins` reopens every decision; without a reload frame the
menu reapplies plugins to the live classes after restoring their original members.
`--all-plugins` explicitly permits execution and applies everything in file-name
order without consulting the saved choices. `plugintools.py`, a reloadable module,
reads plugin files statically
for `/plugin copy`, `/plugin diff`, and `/plugin restamp`; baselines of copied
application source live in `plugins/.baseline/<file>.json`. `--no-plugins` or `SLIPAGENT_NO_PLUGINS=1` skips them. Editing a
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
Retained callbacks can outlive their generation's import namespace. The snapshot
loader forwards their lazy imports into the current accepted generation after
retirement; rejected or unsaved-on-disk code cannot supply those imports. This
loader behavior is part of the stable frame and requires a process restart to adopt.
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
  only the path boundary. The configured `tempfile.gettempdir()` is accessible
  in either mode and its resolved path is supplied in every request.
  `contains` still answers geometric containment and
  external display paths remain absolute. Changes queue during work and apply
  after the active response and tool batch finish. Reset, resume,
  and compatible component reloads preserve this process setting; journals do
  not restore it. Each request supplies a current Workspace Access section, and
  the footer reserves room for a `  │  Danger Mode` suffix (readout-colored divider, red label) when active.
  External atomic writes retain descriptor-based traversal from the target's
  filesystem root. Git ancestor discovery stops at the filesystem root in danger
  mode. External instruction scopes include the target's ancestors and are
  omitted from snapshots when inaccessible under the current policy. Installing changes to
  `workspace.py` requires a process restart, as with other stable contracts.
- **Search:** directory listing includes dotfiles and environments. Recursive
  search prunes generated/dependency directories; explicit paths can target them.
  Regex matching lives in a killable child, while file traversal remains in the
  workspace-aware parent. A thread alone cannot interrupt pathological regex code
  that holds the interpreter lock. Grep filesystem batches use owned workers and
  a per-search cancellation event, checked before and after advancing the file
  iterator. All candidate files count toward the batch limit, including files
  skipped by the include filter or rejected during reading. Cancellation and
  timeout drain an active filesystem operation before returning; a blocked
  filesystem operation must return before its thread can stop.
- **MCP:** modern metadata belongs in `params._meta`; legacy servers use the
  initialize handshake. Send deadlines include pipe writes, not just response
  waits. The stdout reader routes replies to pending request IDs, discards
  notifications and unrelated or duplicate replies immediately, and declines
  unsupported server requests even while idle. Declines have a bounded write
  deadline. A disconnected reader releases pending requests and rejects further
  sends; one undelivered read error is retained for the next request. Server
  stderr retains only its last 20 lines, capped at 2,000 characters per line plus
  a truncation marker. Tools receive structural schema checks and registry
  validation; servers remain responsible for their full JSON Schema semantics. Structured content,
  server guidance, and tool errors must survive adaptation into model context.
  Server guidance is included for both loaded tools and deferred tools in the
  current catalog, which already applies the explicit tool allowlist. Guidance
  therefore precedes tool loading and disappears when a server disconnects.
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
  excluding code wrap markers, continuation indentation, and background padding.
  Soft-wrap separators remain in copied text without appearing at the next row's
  start; literal feed selections preserve source line breaks rather than display
  wrapping. Clipboard commands can
  copy readable replies, Markdown source, or exact fenced code bodies.
  Optional response JSON `suggestions` contains up to three verbatim prompts.
  Final replies expose these as feed links; a click sends a distinct prompt input
  through the REPL without editing the draft or dispatching slash commands.
  Busy runs, open viewers/editors, and modal input prevent suggestion submission.
  Suggestions have no tool/sidebar records and do not execute automatically.
  Structured model responses retain their validation boundary before tool effects
  and history storage; provisional reply text can stream before acceptance.
  `transcript.py` also stores original literal chunks and completed styled rows
  in separate private session files, indexed by byte offsets. Block boundaries
  retain the renderer's spacing and ANSI resets. The ANSI decoder carries style
  and partial escapes across chunks. Wrapping retains the last two rows plus any
  unfinished word that can still alter the preceding row; completed rows are
  immutable. Adjacent characters with the same style are stored as one run.
  `TranscriptControl` supplies indexed rows through `UIContent.get_line`, with
  a 256-row cache; it does not build or hash a full transcript on each repaint.
  Copy-row mappings use two packed integer arrays in `RowIndex`, with binary
  search for sparse rows. Retraction removes mappings and shifts later row keys;
  existing dictionary indexes migrate when first used after reload.
  The application coalesces redraws at a maximum of 30 per second. Resize replays
  original chunks to rebuild the width-specific row file; ordinary appends
  process only new chunks. Explicit continuation indents are indexed by chunk
  and source line, and replayed on resize. The startup tool and command lists use
  this to align wrapped rows beneath their first item without changing ordinary output.
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
  UI choices use the shared footer selector by default.
  `TerminalUI.choose(title, options)` temporarily replaces the seven-row footer
  with a heading, five option rows, and a bottom key legend. The feed and
  sidebar meet the footer without spacer rows. The footer background is dark blue, menus
  use dark purple, and the prompt uses dark green. Activity and notices share
  the footer heading; `notify` starts three flashes only when the event notice
  changes. Only the notice text flashes; activity indicators, links, headings,
  redraws, resizing, repeated notices, and rotating tips do not restart flashing.
  The bottom readout has no separate notice area.
  Overflowing activity text and notices scroll in their available footer width,
  pausing for one second before moving five graphemes per second and looping
  through a three-space gap. Text or width changes restart the scroll; spinner
  and style changes do not. Grapheme boundaries, styles, and mouse handlers
  survive clipping. Fitting text stays still and reuses the existing repaint timer.
  Very short terminals reduce the input/list rows to retain feed
  output. The displayed **back** action cancels the selector like Escape.
  A focused menu
  control owns navigation; the input buffer, history, queue, and transcript stay
  intact. It returns the selected value or `None` on Escape, Ctrl-D, or
  terminal closure. Cancellation restores the invoking region if it is available,
  and refresh preserves an open menu's selection. `/menu` inserts the selected
  command into the prompt draft for submission. Submitted commands and prompts
  share the ordered queue; stop and exit retain their immediate cancellation paths.
  Feed dragging retains a selection. Copying and highlighting share the same
  bounds, including the last selected cell in reading order; a click without a
  drag leaves no text selected. Right-click uses the same selector for
  copying text, appending it to the draft, and copying recognized reference
  paths. Arbitrary selected prose does not become a path. File links retain
  source-line targets through wrapping and reflow. Diff gutters index old/new
  coordinates on disk, independently of viewport traversal order.
  File-reference, saved-output, and diff links underline and accept mouse events
  only over their labels, excluding indentation and trailing feed space. Saved
  edit diffs stay in the viewer; their links follow file links on the same line,
  including in restored transcripts. `edit_file` emits its file link after
  successful completion, using the first changed new-file line from its retained
  unified diff. Multiple replacements omit the displayed line number while
  retaining that first-change target. A result without a recoverable location
  links to the file without a line label. Tool calls and result
  links stay together; the next non-tool entry supplies or receives a blank line.
  `editor.py` owns file snapshots, bounded pages, changed-page files, and atomic
  publication with disk-signature checks. Each `IndexedText` snapshot serializes
  its seek/read operations across text and line indexes, preventing redraws from
  changing the save worker's read position. Locks cover individual bounded reads,
  and existing snapshots acquire the same protection after a compatible reload.
  `editorcontrol.py` renders neighboring
  sections around the editable buffer and translates mouse coordinates back to
  their section. Its window scrolls by displayed rows, preserving the cursor's
  screen edge across wrapped lines and section boundaries. `editorui.py` owns the editable buffer,
  navigation, save/reload choices, and unsaved-change handling. `editorbuffer.py`
  keeps document undo/redo checkpoints on private disk, referring to immutable
  changed-page versions. Navigation and saves preserve them; close and reload
  start a new history. Boundary Backspace/Delete join logical lines across
  storage pages. Cursor, selection and deletion use complete graphemes;
  indexed segments also end at grapheme boundaries without normalizing bytes.
  `useredits.py`
  owns the session gate: the existing agent boundary finishes the complete tool
  batch and waits while editing is active. No new working-model requests or tool
  calls run during the hold, including read-only calls. Existing managed jobs
  and isolated history compression continue independently; opening an editor
  does not stop those jobs. Closing it releases the gate for subsequent batches.
  The reload frame is busy until the editor closes. Cancellation drains an
  in-progress save before releasing its resources. Read-only viewers bypass
  this gate. Each changed save archives its full cumulative diff and creates a
  distinct revision, including reversions. `editdelivery.py` projects pending
  revisions into a jointly budgeted `user_edits` input field with bounded diff
  prefixes and artifact references; deferred observations remain pending.
  Previews and failed requests do not acknowledge revisions. `requests.py`
  captures the exact projection on acceptance; the agent journals the accepted
  message and delivery receipt before acknowledging matching pending revisions
  or executing tools. Receipt replay reconciles a torn pending-state update.
  Delivered observations belong to that completed step's retrievable parts and
  compaction input; compressed history and merged narratives retain artifact
  references. They remain user observations, not fabricated user messages or
  tool calls. Separate editing sessions retain separate observations. A failed
  archive retains the full diff privately and leaves truncated delivery pending.
  `read_artifact` streams validation of the stored JSON while retaining only the
  requested character page. Record listings and command metadata are bounded;
  `next_offset` identifies the next character or record position.
  `options.py` owns the selector and isolated folder-command field;
  `uipreferences.py` preserves unrelated entries in the existing preferences
  file. Folder commands receive literal argument lists without a shell.

## Extending the Contracts

When adding a tool, define an accurate schema and recovery-oriented description,
return `ToolResult`, register it once, and test errors as well as success. Choose
whether its resources belong to a rebuilt tool or the session before constructing
them. Keep remote names scoped so disconnecting one server cannot remove another's
tools. Update the tool table in `docs/tools.md` when the public tool set changes.

The registry validates parameter-schema structure before registering tools or
replacing its collection. Invalid schemas leave the existing collection intact;
remote servers remain responsible for full JSON Schema semantics.

When changing the response schema, update validation, prompt explanations/example,
step storage, task/context budgets, stub responses, streaming/reload fixtures, and
offline demos together. When adding a command, update busy-state restrictions,
help, startup listing, the command table in `docs/terminal.md`, and control-flow tests. Read-only network commands
must not block input processing while `/stop` is waiting.

For every stateful change, test the relevant combination of success, rejection,
cancellation, reset, and reload. Assert actual request bodies and external effects,
not just helper return values. Preserve user-owned instruction files; maintain
these explanatory guides separately from local operational conventions.

## API Provider Modules

`api.py` owns `APIClient`, common errors, HTTP/SSE transport, retries, and completion normalization. `providers.py` maps provider names to implementation classes in separate files and exposes `create_client`. Providers subclass `APIClient`; they supply identity/defaults, capability discovery, optional account metadata, headers, request preparation, and options retained during key replacement. Importing modules does not create connections. `openrouter.py` keeps the previous error aliases for callers using those imports.

The agent and compactor use the common `RequestProfile` contract, not a provider class or routing format. `ModelCapabilities` selects OpenRouter routes from live metadata. NVIDIA's explicit profiles describe verified hosted models; unsupported catalog entries stay visible without invented capabilities. Compression profiles come from the background destination rather than the working model. Provider request preparation runs before diagnostics so the inspector sees the final body.

`ModelInfo.selector` is `provider::model-id`. Search combines catalogs only for providers with a nonblank environment or session key, preserves duplicate IDs, and reports partial failures. Missing-key providers are not queried or listed. Changes to the active provider set refresh the catalog cache. `Session.select_model` preflights capabilities and context. Independent routed summaries continue across working-provider changes; embedded summaries borrowing the working client cancel and drain before it closes. Originals survive cancellation. Preflight failures close candidate resources and preserve active state. Credentials, URLs, and remembered models are provider-specific.

`providers.key_provider` identifies prefixes and Kilo JWT metadata locally; it does not authenticate credentials. `/key` asks for an explicit provider for unknown formats and rejects conflicts with a detected provider. `Session.use_api_key(..., verify=True)` verifies through account metadata or an authenticated catalog before changing credentials. Providers without either defer authentication to inference. Accepted keys cancel and drain pending compression, clear the catalog and that provider's routing failures, and update session `provider_options`. The active provider also replaces its connection and clears quota state; adding another provider preserves the working model and client. Catalogs, model selection, and compression consume the remembered provider credentials and endpoint. Saving changes only the chosen provider's dotenv entry after confirmation.

Provider behavior files participate in the existing reload transaction. Shared state contracts in `types.py`, configuration, package exports, and changes to class inheritance require restart. Adding a provider requires a module, registry entry, documented configuration, and offline wire/selection tests.

Common transport and agent recovery share one retry schedule. Working requests remain `single_attempt=True`, so each attempt consumes exactly one working request. Three retries are silent; subsequent waits update the terminal activity row once per second, clearing it after success, stop, or cancellation. Noninteractive output retains the error and periodic countdown notices. Working recovery stops before an interval exceeds one minute. Both background compactors use the same transient-error schedule with persistent recovery: after the working retry limit, waits continue at the configured ceiling (at most 60 seconds), honoring longer provider `Retry-After` cooldowns. Reset and shutdown cancel these waits. Malformed individual and grouped summaries share the bounded repair budget described above, expanded when needed to cover ranked candidates. HTTP 200 overload envelopes and SSE errors are retryable; partial tool calls never execute before a complete accepted response.
