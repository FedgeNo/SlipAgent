# Model Context, Protocol, and Reasoning

[Back to the README](../README.md)

### Context View and Model Input

Press **Ctrl+backslash (`Ctrl+\`)** to toggle the context view. It shows the
latest outgoing working request, with every system, user, assistant, and tool
message in its sent order. System prompts appear in yellow (`#FFFF00`); tool
definitions, the response schema, and other request fields are also shown.
The view updates for each request, including retries, using the selected full
or compressed history actually sent. Scroll with the mouse wheel, arrow keys,
Page Up/Down, or Home/End. The footer, draft, and transcript scroll position
remain available. Isolated background summaries do not replace this view.
Before the first working request, the view indicates that none has
been sent.

System instructions, tool definitions, and current-step metadata appear in the
system prompt. Native tools also use the API tool field. A single user message
begins with JSON metadata containing these fields, followed by literal tool-result blocks:

| Field | Content |
| --- | --- |
| `user_prompts` | New messages in chronological order, or `[]` without new input; later messages supersede conflicting earlier instructions |
| `retained_user_prompts` | Full ordered array of user messages received before tool use began; supplied only after tools when no new user messages are present |
| `history` | Selected completed steps, oldest to newest |
| `errors` | Harness errors and response corrections not attached to a call |
| `user_edits` (when present) | Saved user file edits, with revision IDs, bounded diff prefixes and artifact references |
| `user_edits_omitted` (when present) | Pending saved-edit notifications deferred by the shared budget |

Saved-edit guidance is included only when the request contains pending edit
notifications or selected history with saved edits. Requests without them omit
both the fields and that guidance.

Each full history record contains `user_prompts`, `agent_response`, and `tool_calls`. `user_prompts` lists only the messages the user sent in that step.
Each call retains its `call_id`, `tool_name`, and argument object, with its raw
outcome reference under `result` for success or `error` for failure. Legacy observations
without a recorded status use `unclassified_result`. Labels belong to keys;
no classification prefixes are inserted into result contents. Each reference identifies a numbered block, its character count, and its `text` or `json` format. Textual results are copied verbatim, even when they resemble JSON. Explicitly structured results use pretty-printed JSON blocks.

Records remain raw objects and lists internally. At the message boundary the
user-input metadata is serialized and exact result blocks are appended without extra escaping or indentation. The HTTP client then encodes the
provider's required transport envelope; native tool-call arguments also use the
JSON string required by that protocol. Quotes, backslashes, and newlines round-trip
without changing the values or allowing text to create sibling input fields.
Metadata and length-labelled blocks provide structural separation, not a guarantee against prompt injection.
`prompts/system-history.md` explains that historical text and tool observations
are evidence, not system instructions. Request assembly does not archive the
input envelope or system guidance as conversation turns.

`representation` distinguishes `full`, `compressed`, and `excerpt` history records.
Full records also include available `compressed_summary` analysis; compressed
records contain summaries and metadata instead of originals. Every record has a
`step_id` for retrieving originals with `recall_history`. Excerpts use
`result_excerpt`, `error_excerpt`, or `unclassified_result_excerpt` on calls,
with separate response excerpts, omission counts, and retrieval guidance.
Task-source IDs and the current step's `is_tool_result_response` appear in system
metadata. Original retrieval and prompt retention remain independent of history
selection. All rendered fields, text, and message overhead count toward the context budget. This input format
is separate from the selected model's response contract, which stays consistent
between calls. Native tool calls remain available for responses from models that
support them; the harness does not ask models to echo its history-record format.
Echoes of former harness response labels are removed from ordinary reply text
and from assistant replies reused as context, without rewriting saved originals.
Quoted or fenced examples, user input, tool arguments and file content remain intact.

The prompt stays live while the agent works. Consecutive waiting prompts join the next
model request together, after the current tool batch finishes. Their previews
stay just above the bottom controls until their request is sent. Each
preview occupies one line within the main pane, stopping at the visible
inspection sidebar and adding an ellipsis when it is too long. Each queued prompt
or command has a Cancel link at the right, separated from the preview by a space:
click it, or focus it and press Enter, to
remove that specific item. API keys are hidden in command previews. On submission,
the full prompts enter the feed in arrival order before that request's output.
If context preparation fails or `/stop` prevents that request, the previews stay.
Use `/queue` to inspect pending messages and select one to view in full or discard.
In plain output it lists the complete messages; `/queue <number>` views one and
`/queue discard <number>` removes one. Numbers refer to the current queue order.
The interactive chooser retains each message's identity while open: if that
message has already been submitted, discarding it changes nothing. Duplicate
messages can be discarded individually. Discards are saved with the session and
preserve accepted history, command logs, and the current task. `/task new` remains
blocked while any unsent user input is pending.
Blank lines separate submitted prompts from the surrounding transcript.
Each model reply starts with a blank line, including replies preceded only by
tool output. Its following tool calls and results stay together until the next
model reply; each command response is another unit.
The model can explain its work alongside batched tool calls.
The interactive feed shows command activity and saved-output links instead of
printing stdout and stderr inline. Open the output viewer to inspect them;
the model still receives command results. Saved edit diffs also stay in the viewer,
with their existing View Diff links beside the file links. One-shot progress uses stderr, and
command archives retain decoded output within their quota.

### Repeated Actions Without Progress

Three repetitions of an unchanged tool batch or cycle of up to six batches
produce model-only recovery guidance with the exact calls. A fourth repetition
stops the run with history intact. New user input resets detection. Intentional
external polling can use `run_command` with `poll=true`; command-log reads and
job-control calls are exempt.

### Reasoning and Overthinking

Readable reasoning supplied through a separate provider field streams under a
gray `Thinking:` label. Each response's reasoning is concatenated into one
string and saved with that step's uncompressed originals. Existing history is
not rewritten. **Overthinking Mode** is enabled by default: the latest 25 completed
steps include supplied thoughts as a `reasoning` string in each step's input
record, subject to `--reasoning-history-steps`. Steps without thoughts gain no extra field. Use `/overthinking off` to
disable this mode, or `/overthinking` or `/overthinking on` to enable it. Thoughts count
toward the context budget; omitted steps and budget-limited excerpts omit them.
Opaque provider reasoning metadata is never included. Background compression returns
separate `summary` and `reasoning_summary` fields. It receives the completed step's
thoughts and up to 25 preceding steps as context, but compresses only that completed
step. Overthinking uses filtered thoughts when available, otherwise the originals.
An empty filtered result stays empty. Original thoughts remain available through
`recall_history`; streaming thoughts still appear in the terminal.
For OpenRouter, the harness selects the highest reasoning effort
explicitly listed by the API. When reasoning is supported without listed effort
choices, it enables reasoning; when unsupported, it sends no reasoning setting.
Reply text streams as it arrives, including text inside an unfinished JSON
`response` string. JSON escapes are decoded across chunk boundaries; tool
arguments and bookkeeping stay out of previews. These previews are provisional:
failed or rejected attempts are marked in the transcript. Tool calls still wait
for complete response validation, and only accepted responses enter history.
The terminal displays the reply text defined by the selected response contract, followed by tool activity. Background summaries
stay out of the transcript. One-shot mode keeps the final answer on stdout,
with streamed reply previews, thoughts, and tool activity on stderr.

### Response Protocol

All built-in model prompts and tool guidance are in the top-level [`prompts/`](../prompts/README.md) folder for direct human editing. The `.md` templates contain Markdown headings, lists, tables, and fenced examples; static headings and response-schema descriptions live in these resources rather than Python. Only imported history retains the large opening and closing warning banners. SlipAgent rereads templates when building requests and tool definitions, including with live reload disabled. Start with `prompts/system-prompt.md` for general behavior or `prompts/background-summary-prompt.md` for compression. Preserve placeholder names and each prompt's output contract; Markdown instructions can still require JSON replies or plain-text summaries. The installer copies this folder, and wheels bundle it inside the package. See its README for the resource map and formatting conventions.

For model-specific instructions, map the exact model ID to Markdown files in `prompts/model-profiles.json`. Profiles can also select embedded tool calls, XML call compatibility, and structured reply formatting. Set `structured_output` to `false` to omit the API reply-format constraint for one model; set it to `true` or omit it to follow endpoint capabilities. Dots3 Note Preview has this setting disabled for both its standard and `:free` IDs, keeping native tools enabled. GLM-5.3 and GLM-5.3 Flash receive tools in GLM's own XML template format. See [Model Profiles](../prompts/README.md#model-profiles) for the settings.

Runtime guidance distinguishes exploratory questions from authorized implementation and keeps work within the user's requested scope. Background summaries prioritize user requests, corrections, and boundaries over agent plans, preserve source attribution, and distinguish completed or rejected actions from unfinished requested work. Historical reasoning and optional agent suggestions do not authorize additional work.

At startup and every model selection, including reselection, SlipAgent refreshes the selected provider's capabilities. OpenRouter uses live endpoint metadata; NVIDIA uses explicit model profiles because its catalog omits those fields. Subsequent calls reuse the profile until another selection. A failed or incompatible selection reports an error and keeps the current provider, model, and conversation.
Startup model-check failures display a warning and leave the interface available, so users can select another model with `/models` or `/model <slug>`. A model that fails its startup check is not saved as a new preference.

Temperature is omitted unless explicitly set with `/temperature <value>` or
`--temperature`, leaving each provider to choose its default. `/temperature`
shows the explicit value or `provider default`. An explicit session setting
carries across model switches and applies wherever the selected endpoints
support it. Unsupported changes report an error and preserve that setting;
requests to unsupported endpoints omit it. Individual-turn and grouped-history
compression always omit temperature, including when the working model has an override.

Supported top-p/top-k/min-p defaults remain model-specific for original Qwen3
hybrid checkpoints and Nemotron 3.5 Lightning; explicit request values take precedence.

SlipAgent uses [OpenRouter's native tool-call format](https://openrouter.ai/docs/guides/features/tool-calling)
when the selected endpoints advertise `tools`. Tool definitions go in the API's
`tools` parameter; calls arrive in `message.tool_calls`, separately from the
response text. The harness preserves each call ID and stores its result with
the matching `call_id` in that step's `tool_results` array. The next request
supplies the completed step in the user-input metadata, with each literal result block referenced
by its call under `result` or `error`. Native calls remain the model's
response format; the input records describe completed history. SlipAgent does
not force a `tool_choice` setting.

Endpoint metadata, including reasoning support and provider limits, remains
cached between selections. Model-profile formatting overrides are reread when
building working requests.
Selection prefers native-capable endpoints;
models without them can use the JSON call format below if they support JSON
output. Their tool definitions enter the system prompt, and their history uses
the same JSON input records, without unsupported native tool parameters.

**Native-tool models without schema support use plain reply text.** They receive
no `response_format` parameter and need no JSON content record. Unless disabled
in the model profile, JSON-only models receive
`response_format: {"type":"json_object"}` when schemas are unavailable.
Reply text is encouraged on tool turns but may be empty when calls are present.
Unless disabled in the model profile, endpoints advertising `structured_outputs`
receive a small [strict schema](https://openrouter.ai/docs/guides/features/structured-outputs):
`response` contains the plain terminal reply. Models using embedded calls also
include `tool_calls`, an ordered array, empty for a final answer. Neither mode
requires compressed fields or a task record in every response.

For example, a schema-capable native-tool response uses this content:

```json
{"response": "I will read the file."}
```

A model without schema support simply writes `I will read the file.`. In both
cases, the accompanying native `message.tool_calls` value can be:

```json
[
  {
    "id": "read-1",
    "type": "function",
    "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
  }
]
```

When native calls are unavailable, requests ask for a JSON object containing
`response` and `tool_calls`. Response parsing independently detects plain replies,
JSON envelopes (including complete JSON code fences), legacy summary envelopes,
tagged JSON calls, and Qwen3-Coder's `<tool_call><function=...><parameter=...>`
format, plus `<tool_call><answer>text</answer></tool_call>`. Complete tagged JSON
may omit its outer closing tag. Answer carriers also tolerate omitted trailing
tags; executable Qwen calls require their parameter and function closing tags.
These forms require a provider-completed response;
incomplete JSON arguments and provider-reported truncation still reject the batch.
API `tool_calls`, legacy `function_call`, and `tool_use` content blocks
are also recognized. No per-model response-format record is needed.

Calls may arrive separately, inside answer text, or through both channels in the
same response. Distinct calls are combined in API-carrier order followed by text
calls; mirrored calls execute once, while intentional repetitions within a batch
are preserved. Conflicting calls sharing an ID reject the complete response.
Qwen parameters use the advertised tool schema for types and preserve string
content without XML entity decoding. Streamed calls wait for the complete response.
Malformed arguments, duplicate keys, invalid Unicode, non-finite numbers, duplicate
IDs within a carrier, and truncated batches reject the complete response before
execution. Quoted and fenced examples within explanatory prose remain text.
Request-side API capabilities still determine which parameters can be sent.
Three consecutive invalid responses stop the run; retries count against the
working-request limit. Streamed thoughts and provisional reply text remain visible;
failed reply previews are marked as unaccepted.

Interrupted streams, overloads inside streamed/JSON responses, and transient HTTP failures receive three silent retries. If failures continue, SlipAgent displays the error and a countdown while retrying with increasing delays. It gives up before scheduling an interval longer than **60 seconds**; a server `Retry-After` beyond that limit also stops recovery. The minute is an interval limit, not a total request deadline. Permanent authentication, credit, and request errors are not retried. Every working attempt counts against `--max-steps`, with no nested retry loop. `/stop` and Esc interrupt waits. Each attempt starts fresh:
partial tool arguments and rejected replies never execute or enter history as accepted actions.
Authentication, credit, and permanent request failures remain errors.

Every failed working request is preserved as a numbered history turn, including
provider/transport errors, explicit context overflows, rejected responses,
repetition stops and cancellations. Its `request_failure` record preserves the
error, a bounded rejected/partial-response excerpt, the diagnostic attempt number,
and consecutive/session failure counts. It explicitly marks that no tools from
the attempt executed. Failures receive a deterministic summary rather than an
extra model call. Full excerpts remain retrievable through `recall_history`.
Retry requests rebuild their context with those failed turns and `request_retry`
metadata identifying the failure count and turn IDs. Guidance tells the model
to honor instructions to stop or not retry rather than treat each attempt as the
first. The request budget still counts every attempt; recovery waits remain
cancellable. Failed-turn records and counts survive resume and fork, and reset
clears them. Raw request/response diagnostic files remain separate.

`/requests` lists the latest 20 attempts; `/requests 3` displays attempt 3's
final request body, outcome, and complete response/error details. Complete provider
responses are captured before parsing, with streaming text and reasoning joined
into continuous fields rather than individual fragments. Provider metadata and
tool-call fields are retained, including partial responses on failure. Request JSON
is deduplicated and gzip-compressed in private files beside the session journal,
without request authentication headers. Each attempt also captures its raw HTTP
response body in `NNNNNNNN-stream.bin.gz`, with status and response headers in
the attempt metadata. This preserves keepalives, empty events, malformed bytes,
and original line endings before HTTP content decoding or stream parsing. Capture
writes incrementally; cancellation or disconnect retains the received prefix.
The readable `/requests` view continues to show the assembled response and names
its raw file. These diagnostics are never supplied as history
or sent for compaction. Diagnostics use a **32 MiB rolling target**: oldest attempts
are removed as new ones are saved. The newest attempt keeps its complete request
and responses, including raw bytes, even if it alone exceeds the target. Shared request snapshots remain
until their last retained attempt is removed. Conversation history is unaffected.
Storage failures display a warning and preserve existing records. Reset starts a
new directory. Resuming continues the saved attempt numbering and rolling retention.
`--no-session` uses temporary files removed on reset/exit. Treat request logs as
private project data, since they contain the same text sent to the model.

OpenRouter requests set `provider.require_parameters` and restrict routing to compatible
endpoint providers. Context budgets use their actual context and prompt limits.
Unsupported reasoning and temperature settings are omitted; requested output
limits are checked. A model without available native-tool or JSON-capable
endpoints is rejected. Metadata cannot guarantee that every advertised parameter
combination works at every provider.

`/stop` lets the current response and every tool call in that response finish,
keeps their results in the conversation, and prevents another working model request. Background summarization of that
completed step still runs.
Managed background commands also continue until completion, their execution
timeout, or an explicit stop. Their completion cannot restart the agent.
Input queued before `/stop` stays pending. A new prompt or queued command entered
afterward resumes work, including while the current step is finishing. A new
prompt also starts work from idle; no special resume phrase is required.
SlipAgent defines no Ctrl-C or SIGINT key binding.
Ctrl-D, `/exit`, or its alias `/quit` exits the session. Pipes and `TERM=dumb`
use plain output.

### Active-Task Memory

The model retains nothing between API requests unless SlipAgent sends it again.
New input arrives in one chronological `user_prompts` array. Later messages
supersede earlier conflicting instructions; queued messages are not relegated
to a secondary field. With no new input, `user_prompts` is empty and
`retained_user_prompts` carries the full original array for the active request,
after tool use and on any other continuation. A new set of messages leaves the
retained array empty.

The harness stores original source messages independently of compression.
`current_prompt_step` identifies the step that received them, and the model can
retrieve its originals with `recall_history`. Retained messages count toward the
request budget. Tool-continuation guidance explains why control returned and
asks the model to assess the results against those requests.

Original user prompts and their source step IDs remain available independently of history compression. `/task` displays these sources and the current original prompt. `/task new` starts a new source boundary with the next prompt while preserving history; `/new` clears prompt retention and history. Older session task-record snapshots are ignored during resume.

The step limit caps working model requests per user run. Background summary
requests are separate and their reported usage contributes to session totals. Reaching it preserves the
conversation for continuation and exits one-shot mode with status 1.

### Rolling Conversation Context

Each accepted model response forms one numbered step: its active user prompt,
response, requested tool calls, and complete tool results. Those four originals
are stored separately, alongside the response's reasoning as one string when
available. Recent steps present the full prompt, response, calls, and results,
with supplied reasoning included for the latest 25 steps in Overthinking Mode.
Older steps use their whole-step summary only when its
estimated token cost is smaller than the original; otherwise they retain their
full prompt, response, tool calls, and results. The comparison includes JSON fields,
escaped content, and message overhead. Ties retain originals.
Each full step also includes its available `compressed_summary` in the same JSON
object, alongside the original fields. Compressed-only records use that same key.
Both originals and their accompanying analysis count toward the context budget;
the representation comparison above uses the original before adding its summary.

Each background compression attempt has a 20-minute timeout instead of the normal HTTP timeout. Single-turn and multi-turn compression keep retrying rate limits, transient provider errors, and transport failures until recovery or cancellation. They share the working request retry schedule, then continue with waits of up to 60 seconds, honoring longer provider `Retry-After` cooldowns. Every retry selects the next available provider. Invalid replies and permanent failures use a bounded retry budget, expanded when needed to try the ranked candidates. Retry waits and additional attempts can extend the job beyond 20 minutes. Exhaustion preserves original steps for `recall_history`. Working requests retain their configured timeout and retry limit; reset and shutdown cancel pending summaries.

Forking, resuming, deleting a session, and finishing a one-shot run give all pending
compression up to two minutes to finish, including retries. At that deadline,
SlipAgent cancels and drains unfinished jobs before continuing. Original turns
remain available unless the session itself is deleted.

Each working request identifies the source step of the active requests in system guidance; their exact text belongs in the structured user-input object. Steps after tool batches explicitly explain that control returned automatically without a new user instruction: assess results against that goal and return an answer without tools when it is fulfilled. This guidance is assembled for the outgoing request only; it is not saved in conversation history or background summaries. Historical requests provide background rather than independently requesting more work.

**History is sent in full until it would exceed the context budget.** The budget
is 85% of the model's context allowance, subject to `--context-tokens`, after
reserving instructions, tools, output and the current input. Sizes are estimated
from the bytes of each record and scaled by measured calibration.

When the full history does not fit, SlipAgent compresses **from the oldest end,
only as far as needed**, and stops as soon as the request fits:

1. Individual steps are replaced by their summaries, oldest first, until up to
   **two thirds** of the steps are summarized. A step whose summary is missing or
   no smaller keeps its original.
2. Each fixed block of **20 steps** (1–20, 21–40, …) that lies wholly in the
   summarized part is replaced by its **narrative**, oldest first.
3. Each fixed block of **five narratives** (steps 1–100, 101–200, …) is replaced
   by its **era**, oldest first.
4. Steps in the newest third are summarized, oldest first. The latest tool
   results the model has not seen stay full.
5. Only then is history omitted, starting at the very beginning of the
   conversation.

Selection starts fresh on every request, so steps return to full text whenever
room allows, for example after a large result has moved into summarized history.
`--context-steps` remains accepted for compatibility but does not control
selection. Newly returned tool results reach the working model intact, or as
explicit excerpts if even the latest batch cannot fit. Originals remain
retrievable even when omitted. Project instructions and the retained user prompt
stay pinned outside the history.

Narratives and eras are written in the background as soon as their inputs
exist: a narrative once all 20 steps of its block have first-pass summaries
(a step whose summary failed is left out as a gap), and an era once all five
narratives of its block exist. Every level is kept, so any request can use
whichever level fits. Each narrative is built from its steps' individual
summaries and each era from its five narratives. The prompts require source turn
citations; narratives also keep the source turn list, call IDs, tool names and
artifact references copied by the harness. `recall_history` can retrieve exact
turns or search the full archive when a turn number is unknown. A failed block
is attempted again after 20 more turns. Results and attempt markers persist
across resume and fork; reset cancels pending work. Invalid or non-shrinking
narratives and eras leave the summaries they would replace intact.

**One separate background request starts after each complete tool batch returns**,
or immediately after a reply without tools. It receives that step's prompt,
response, calls, results and thoughts, plus up to 25 preceding completed steps
as contextual history. Its instructions restrict compression to the current step.
Oldest contextual records are omitted if needed to fit; the current step remains complete.
Tool arguments and structured results remain dictionaries or lists internally;
the complete summary input is serialized once for the model. Complete history
retrievals return complete objects or arrays without character paging.
**Both individual-turn and grouped-history compression round-robin through
the best available ranked free model from each eligible provider.** Every job
and retry uses one shared rotation, in both normal and optimal modes. Providers
participate when their own API key and a ranked free model are available:
OpenRouter, NVIDIA, Kilo, Mistral, and Google. Each attempt uses its destination's
own capabilities and context/output limits.

The selected coding provider serves only working requests, whatever its model
costs. Background summaries use the other providers. If no other provider is
available, nothing is summarized: requests send originals, omit the oldest
history when needed, and SlipAgent asks for another provider's API key.

Any compression failure sends the retry to another provider when possible.
Transient failures retain the model's eligibility. Permanent/model-format
failures exclude that compression model, so its provider contributes the next
ranked model on a later rotation. A request too large for one model tries another
without disabling that model for smaller jobs. Failed candidates are reused only
when no unfailed option remains. If no eligible free route exists, originals
remain available and SlipAgent reports why compression is unavailable.
Background clients are separate from the working client and close with the session.

The summarizer returns a JSON object with separate factual `summary` and filtered
`reasoning_summary` strings, each limited to 6,000 characters. Short steps
should get short summaries. The working model continues without waiting.
One dispatcher runs every background summary: step summaries oldest first, then
narratives, then eras. It runs **at most one job per background provider at a
time**, so with one background provider every summary runs in turn. The ready
queue holds at most 32 step references. Further steps wait on their retained
originals and enter the queue as it drains. Only an active job copies its payload,
and each job selects its provider when it starts.

Unseen tool results reach the next working request even if their summary has
already finished. An oversized new batch uses JSON excerpt records with separate
omission metadata and retrieval instructions as a last resort. The latest request
stays verbatim.
Neither excerpts nor summaries overwrite any captured originals.

Pending or failed summaries are labelled in context with a retrieval reference.
Compaction failures show a warning and preserve the originals; they do not
regenerate or repeat the agent's work. Normal compaction is silent. `/new`
cancels outstanding jobs and prevents late results from entering the new
conversation. Session shutdown cancels remaining jobs; one-shot mode waits for
its summaries before shutting down. These additional requests use their selected
compression provider's key and contribute to its quota use, session tokens, and
reported cost.

The model's `recall_history` tool searches captured originals or reads a step
by ID, returning the complete selected object without character limits or paging.
List/search results include `total_matches`, the number of matching steps (zero
when none match). Select `section="prompt"`,
`"response"`, `"reasoning"`, `"tool_calls"`, or `"tool_results"` to retrieve one original part;
`sections=["prompt", "tool_results"]` selects several together. Omit the selector
for the full step. `call_id` selects one observation, or a call when paired with
`section="tool_calls"`. `section="user"` returns the original new user messages
associated with that step, for task-source recovery. Shell and Git observations show the
full output up to 32,000,000 characters per stream, then the beginning and end
for larger output. The full
decoded streams are retained separately within the session log quota and are
retrieved with `read_command_output`. Timeout results preserve partial output and
state that effects may be partial. Both history versions are journaled for
resume by default; `--no-session` uses temporary files removed on reset/exit.
The terminal transcript continues to show the full conversation.

Token counts use a UTF-8 size estimate rather than a model-specific tokenizer.
Reported prompt usage calibrates that estimate using the average measured-to-estimated ratio from the latest five working requests, plus a 10% safety margin. The multiplier can rise or fall as measurements change; the configured context and output limits remain unchanged. Calibration resets when the model or endpoint
profile changes; background-summary usage does not calibrate working context.
Explicit provider context overflow triggers up to two retries with smaller
history budgets, counted against the step limit. An identical, unshrinkable
request is not resent. Originals remain available throughout.
The reduced budget persists for that model/profile so the next call does not
immediately restore the rejected context size.
Startup and model selection fetch live catalog and endpoint capabilities,
require a valid context length, and check that the current instructions, tool
definitions, and reserved output fit before accepting the model. The harness
uses the selected endpoints' conservative limits to reduce the budget for
smaller models. The request reserves output tokens and a 15% estimation margin;
the remaining space accommodates instructions, tools, and selected history.
Older records are omitted before the recent full window is reduced. Both stored
versions remain intact, and a summary is used only when it saves tokens.
It reports a limit if mandatory input and instructions cannot fit even after
older history is omitted and oversized observations use excerpts.

A bounded repository map supplies file paths and declaration outlines for Python,
JavaScript, TypeScript/TSX, PHP, Go, Rust, C/C++, Java, C#, Ruby, and other supported
grammars. Python uses its built-in AST; other languages use bundled Tree-sitter
grammars with no parser downloads during a run. Ranking combines words in recent
user input with a graph of symbol references and Python imports. Symbol matches
are orientation hints, not compiler-verified bindings. Cached outlines refresh
when file metadata changes, and syntax errors label the outline as partial.
The map excludes dependencies, hidden directories, symlinks, and Git-ignored
files without changing file-tool visibility. Limits on scanning, source size,
and output keep it partial by design: use `glob`, `grep`, and `read_file` for
details. It counts against the request budget and is dropped if needed to fit
the current input. The map does not authorize edits or replace source reads.

When the map does not show where behavior lives, the model can load
`semantic_search` through `load_tools`. It returns five complete ranked excerpts
by default, with relative paths and line numbers. Short queries of five or six
meaningful words work best. Ranking combines the local quantized BGE encoder
with SQLite keyword matches and filename context; scores are relative rankings,
not confidence estimates. Use ordinary search when results are weak, and read
current source before editing an indexed excerpt.

Semantic search starts only on its first call. It reuses locally cached model
assets or downloads missing assets, builds a persistent index under the private
user-data directory's `semantic/` folder, and watches changes for the rest of the
session. Source and queries stay local. Updates coalesce save events, hash affected
files, and reconcile metadata every 30 seconds; queued changes flush before a
lookup. Reopening a session reuses unchanged vectors. Dependencies, hidden paths,
symlinks, Git-ignored files, and the harness's data directory are excluded.
The first build can take time on large projects. CPU inference uses one chunk
per call and the physical core count, with a two-thread fallback. No model or
indexing worker is loaded just to display the tool catalog.

Source files are hashed into one temporary snapshot beside the index, then read
in bounded blocks and chunked incrementally. Long lines are split into complete
consecutive excerpts; no source is discarded to meet a file-size cutoff. The
snapshot is removed when processing finishes or fails. Source/chunk buffers stay
bounded independently of file size; the stored vector index still grows with the
number of chunks.

The [semantic retrieval benchmarks](../benchmarks/semantic/README.md) retain the
complete evaluation data and reproduction commands.

Useful flags:

| Flag | Effect |
| --- | --- |
| `-h, --help` | Show all command-line options and examples. |
| `-p, --prompt` | Supply a one-shot task instead of positional task text. |
| `-m, --model` | Select and remember a model slug; start in manual mode. |
| `-w, --workspace` | Project root and default path boundary. |
| `--danger` | Disable workspace path confinement at startup, including unattended one-shot tasks; no confirmation prompt. |
| `--max-steps` | Cap working model requests per run, including response retries (default 200); background summaries are separate. |
| `--context-steps` | Deprecated compatibility option; it does not control history selection. |
| `--context-tokens` | Optional total working-context cap, including output reserve; at least 4096, bounded by endpoint limits. Archived originals are preserved. |
| `--reasoning-history-steps` | Archived thoughts from 0–25 recent steps (default 25); 0 supplies outcomes without archived thoughts. Does not disable current-turn reasoning. |
| `--raw-reasoning-history` | Use original thoughts instead of filtered thoughts for controlled comparisons. Compression still runs. |
| `--planning` / `--no-planning` | Planning is enabled by default. The latest successful plan is injected on subsequent turns and recovered from journaled results on resume. Use `--no-planning` to start with it disabled. |
| `--tools` | Explicit comma-separated tool allowlist, useful for focused tasks or large MCP installations. History retrieval and enabled planning remain available. Unlisted calls are rejected. |
| `--python` | Explicit project interpreter, overriding project-file settings and discovery. |
| `--temperature`, `--max-tokens` | Explicit temperature and output limits; temperature applies only to working-model requests. |
| `-v, --verbose` | Show full tool output and per-step token usage. |
| `--provider` | Select `openrouter`, `nvidia`, `kilo`, `mistral`, or `google`; start in manual mode. |
| `--update` | Download the latest SlipAgent from GitHub and run its installer, which checks for local edits first. |
| `--list-models` | Print catalogs for providers with configured API keys and qualified model selectors. |
| `--api-key` | Override the API key for this process; an environment variable avoids putting it in shell history. |
| `--base-url` | Override the API base URL, for example for a local test gateway. |
| `--no-color` | Disable ANSI color output. |
| `--no-mcp` | Skip globally configured MCP servers. |
| `--no-reload` | Disable automatic source reloads for this process. |
| `--no-plugins` | Skip every file in the plugins folder for this process; `SLIPAGENT_NO_PLUGINS=1` does the same. |
| `--resume [id]` | Restore a saved project session; omitted ID means latest. |
| `--no-session` | Disable durable conversation and command-log storage. |
| `--mcp` | Connect MCP servers even in one-shot mode. |

In one-shot mode the final answer goes to **stdout** and progress goes to
**stderr**, so the answer can be piped to another program. Model output is
rendered Markdown in the terminal and readable plain text when redirected.
Use `--markdown` to retain Markdown source in stdout or start the REPL in source
view. LaTeX is not rendered; equations use plain text.

Assistant replies support CommonMark, tables, strikethrough, and task checkboxes.
Headings share one bright, bold style without added spacing; body and bold text
share a foreground color. Language-tagged code uses Pygments lexer tokens for
syntax colors, with a theme-specific background and no language label. Wrapped
code has a continuation marker that is excluded from copied text. Narrow tables
become header/value records. Links display `label (destination)`.

Choose `--theme dark|light|monochrome|ironbow`, or `/theme NAME` in the REPL.
The default feed is white on black. Teal marks user input and success, violet
marks tools and panel titles, cyan marks navigation, amber marks activity and
warnings, and coral marks errors. Menus, the footer, and the editor use distinct
dark backgrounds. Reasoning stays grey, with a subtle background on inline code.
`/markdown [source|rendered]` toggles the view. `/copy` copies the latest reply,
automatically choosing exact code when it contains one code block; source view
copies Markdown. `/copy text`, `/copy markdown`, and `/copy code [number]`
choose explicitly. F2 toggles the view and F3 copies the latest reply. Dragging
selects feed text; right-click the selection to copy it without wrap decorations.
Clipboard access uses an installed local utility or the terminal's OSC 52 support.

The renderer updates incoming Markdown chunks as they arrive, including provisional
text decoded from JSON response strings. Tool calls remain buffered until the
complete response is validated; previews do not authorize tool execution.
