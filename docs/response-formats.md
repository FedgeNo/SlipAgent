# Response Format Compatibility

SlipAgent normalizes complete calls into the same tool validation and execution
path. Native API calls retain their IDs. Mirrored calls across native and text
formats execute once; intentional repetitions within a batch remain separate.

## Accepted Calls

| Format | Example | Availability |
| --- | --- | --- |
| Native API | `tool_calls`, legacy `function_call`, or `tool_use` content blocks | All models |
| JSON envelope | `{"response":"Reading.","tool_calls":[{"name":"read_file","arguments":{"path":"app.py"}}]}` | All models |
| Bare JSON | `{"name":"read_file","arguments":{"path":"app.py"}}` or an array of calls | All models |
| Tagged JSON | `<tool_call>{"name":"read_file","arguments":{"path":"app.py"}}</tool_call>` | All models |
| Qwen tags | `<tool_call><function=read_file><parameter=path>app.py</parameter></function></tool_call>` | All models |
| Invoke XML | `<function_calls><invoke name="read_file"><parameter name="path">app.py</parameter></invoke></function_calls>` | XML profile |
| Wrapped XML | `<tool_calls><call><invoke name="read_file"><parameter name="path">app.py</parameter></invoke></call></tool_calls>` | XML profile |
| Singular invoke XML | `<tool_call><invoke name="read_file"><parameter name="path">app.py</parameter></invoke></tool_call>` | Tagged calls; XML profile also accepts whole XML fences |
| Mistral array | `[TOOL_CALLS][{"name":"read_file","arguments":{"path":"app.py"}}]` | All models |
| Mistral arguments | `[TOOL_CALLS]read_file[ARGS]{"path":"app.py"}` | All models |
| DeepSeek V3.1 | `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file<｜tool▁sep｜>{"path":"app.py"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>` | All models |
| Kimi K2 | `<\|tool_calls_section_begin\|><\|tool_call_begin\|>functions.read_file:0<\|tool_call_argument_begin\|>{"path":"app.py"}<\|tool_call_end\|><\|tool_calls_section_end\|>` | All models |
| Bare function tags | `<function=read_file><parameter=path>app.py</parameter></function>` | All models |

JSON carriers may follow prose on a new unquoted, unindented line. The response
must then contain a complete JSON carrier, with any supported tagged suffix.
Ordinary JSON data and nested objects inside that data remain literal.

XML opening tags accept attributes and whitespace before `>`. Closing tags
accept whitespace before `>`. Singular invoke wrappers contain exactly one
invoke. Explicit call wrappers contain exactly one invoke. Parameter strings
preserve their whitespace and decoded XML entities; other argument types use
the active tool's schema. Missing names, duplicate parameters, and broken XML
reject the entire batch, including native calls in the same response.

The model-marker adapters follow the published
[Mistral tokenization examples](https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/tool_calling.md),
[Mistral chat template](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/chat_template.jinja),
the [DeepSeek V3.1 tool format](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base/blob/main/README.md#toolcall),
and [Moonshot's Kimi K2 guidance](https://github.com/MoonshotAI/Kimi-K2/blob/main/docs/tool_call_guidance.md#manually-parsing-tool-calls).
JSON boundaries are decoded before marker boundaries, so marker text inside
arguments does not split a call. Mistral's terminal `</s>` and DeepSeek's terminal
`<｜end▁of▁sentence｜>` are accepted after their complete batches.
Kimi accepts an optional argument-end marker before each call-end marker.

## Argument and Name Compatibility

Complete explicit JSON carriers tolerate literal control characters inside
strings, invalid backslash escapes, and trailing commas. Existing valid JSON
escapes retain their meaning; invalid escapes retain their backslash as literal
data. Missing quotes, values, and structural delimiters remain errors. Duplicate
fields, nonfinite numbers, and invalid Unicode are rejected. An argument object
encoded as JSON twice is decoded with a bounded extra pass.

Tool names are matched against definitions supplied in the current request.
Exact registered names take priority. Whitespace, `functions.`/`tools.` prefixes,
case differences, dotted namespaces, and MCP suffixes are normalized only when
the match is unique. No fuzzy spelling repair or cross-tool semantic alias is
used. Normalization cannot make an undisclosed or restricted tool available.

The selected tool's schema permits unambiguous conversions of numeric and
boolean strings and JSON-encoded objects or arrays, including nested fields,
local references, and selected nullable/union schemas. A string accepted by a
union remains a string. Code, paths, and other declared string fields retain
their content. Unknown fields remain present so validation can report them;
fields and missing values are never silently discarded or invented. Conversion
recursion is bounded to twelve levels.

Bare or wrapped function tags also recognize the malformed parameter opening
`<parameter=name=value</parameter>` when its name and text boundary are explicit.
Parameter and function closing tags are still required for executable tools.
The existing text-only `answer` carrier retains its tolerance for missing
trailing tags. A leading `</think>` left by a provider's reasoning separation
is accepted before the answer.

Before any tool in an accepted batch executes, the complete batch is checked
for available tools, argument schemas, and callable signatures. One failing
call produces an error result for every call and executes none of them. The
existing tool checks still run again at invocation time.

## Provider Streaming Modes

The default preserves ordinary streamed identity fragments as deltas. NVIDIA
direct MiniMax calls use snapshot identity fields, reflecting the provider
behavior documented in Hermes' accumulator. Empty or null identity fields do
not erase established identities. Conflicting nonempty snapshot identities
reject the response rather than retargeting the call. Arguments always remain
ordered deltas.
A finish event freezes that choice; subsequent usage records are retained,
while late reply or tool deltas cannot reopen it.

Exact model entries in `prompts/model-profiles.json` can override this behavior:

```json
{
    "provider/model-id": {
        "stream_tool_identity": "snapshot",
        "stream_tool_indices": "id"
    }
}
```

`stream_tool_identity` accepts `delta` or `snapshot`; an explicit `delta` also
overrides the NVIDIA MiniMax default. `stream_tool_indices` accepts `index`
(default) or `id`. The `id` mode accommodates gateways that reuse one raw index
for multiple calls: IDs select separate slots, and ID-less continuations use
the last slot associated with that raw index. It requires complete IDs and
must not be enabled for fragmented-ID streams. Settings are validated and read
for each streamed request, including after model selection or prompt edits.
Integer IDs are represented as strings for subsequent API history.

## Context Limits for Individual Providers

An exact model profile can reduce the local context allowance for an
OpenRouter provider without changing its advertised capabilities:

```json
{
    "dots-studio/dots-3-note-preview:free": {
        "provider_context_limits": {
            "atlas-cloud": 256000
        }
    }
}
```

Limits are positive integers in tokens. Base provider tags cover their suffix
variants, such as `atlas-cloud/fp8`; an exact variant entry takes precedence.
If several providers are eligible to receive a request, the smallest applicable
cap governs that request. Routes containing only other providers retain their
normal allowance. Existing smaller model, endpoint, or user limits still apply.
Profiles are read on every context build, including previews, recovery, and
background step summaries, so
changes apply without refreshing cached provider metadata. History selection
uses the reduced budget while retaining archived originals for retrieval.

Dots3-Note Preview's free AtlasCloud route defaults to 256,000 tokens after
repeated generic HTTP 400 responses when a user message exceeded approximately
1 MiB. This is a conservative workaround; those errors did not confirm a
specific provider byte limit.

## Reasoning, Examples, and Completion

Complete leading `<think>…</think>` blocks are separated before detecting the
answer's format. Their contents never request tools. Unfinished blocks and
thought-only replies remain terminal text. Separate API reasoning fields retain
their existing handling.

Quoted, indented, and prose-surrounded fenced examples remain literal. An explicit
`Example:`, `Sample:`, or `Illustration:` label also protects unfenced examples.
A response consisting entirely of a JSON fence is a call carrier when its body
is a call. Whole XML fences require the model's `xml_tool_calls` profile setting.
Tool markers at the start of a JSON reply string are also recognized; XML
profiles additionally inspect their reply strings for supported carriers.

Preview streaming never executes tools. Potential calls and leading thoughts
are held until the complete response is validated. Ordinary reply text can
continue streaming; ambiguous text is displayed after acceptance.

A nonempty accepted reply without calls ends the turn, including when completion
metadata says `tool_calls`. This compatibility layer does not infer new work
from completion metadata. Invalid explicit call batches use the existing bounded
response-correction path before any calls execute. Streaming identity modes are
explicit; the normalizer does not guess from repeated text. No additional
inference is used to reinterpret prose as an intended operation.

## Integration of Researched Approaches

These implementations were inspected on 2026-10-12. SlipAgent implements the
applicable mechanisms locally, without depending on the other harnesses.

| Harness | Integrated mechanism | Behavior deliberately excluded |
| --- | --- | --- |
| [OpenCode](https://github.com/anomalyco/opencode/blob/15b24ae12ddb8b7ae3e923b286b4032761bee510/packages/core/src/plugin/tool-input-repair.ts) | Schema-directed conversion | Dropping undeclared fields or treating optional nulls as absent |
| [Goose](https://github.com/aaif-goose/goose/blob/69de89c50cc01d4e0187d6645bf3618729873f74/crates/goose/src/providers/toolshim.rs) | Kimi markers and unique registered-name normalization | Interpreter-model inference of actions from prose |
| [Pi](https://github.com/badlogic/pi-mono/blob/7d75a0e3eb273dff753d49c066ea29fbe7b3b6f6/packages/ai/src/utils/json-parse.ts) | JSON string-encoding tolerance; schema conversion | Executing a permissive partial JSON parse |
| [Hermes](https://github.com/NousResearch/hermes-agent/blob/427d605d4985c2d49a795493ef6c5d66413057d1/agent/turn_tool_validation.py) | Trailing commas, explicit streaming modes, numeric IDs, errors to the model | Executing valid calls from an invalid batch; completing truncated structures |
| [OpenHands SDK](https://github.com/OpenHands/software-agent-sdk/blob/acca181af1f353975965bcf6486e1d0a58f444c0/openhands-sdk/openhands/sdk/llm/mixins/fn_call_converter.py) | Bare function calls, reasoning boundary, selected parameter-tag repair | Inventing missing closing tags for executable tools or mapping different tools by legacy semantics |
| [Roo-Code](https://github.com/RooCodeInc/Roo-Code/blob/b867ec9145750d0ae1ff7f02d35406e9bf2a0b16/src/core/assistant-message/presentAssistantMessage.ts) | Finalized validation and matching error results | Restricting every model to native calls |
| [Kimi Code](https://github.com/MoonshotAI/kimi-cli/blob/9ab1286b8fe4e6bcd116949a27ce5e0ac3389c82/packages/kosong/src/kosong/tooling/simple.py) | Literal control-character tolerance and structured errors | General arbitrary JSON reconstruction |
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness/blob/d743267388641bc76f17c45ce8b4c231aed1d32c/packages/llm/llm/src/assembler.ts) | Retaining identities under empty continuations; rejecting interrupted/truncated calls | Dispatch before the whole response is accepted |
| [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness/blob/4399efacd4755ff6feccc727d3180a97428e76cf/pydantic_ai_harness/repair_tool_arguments/_capability.py) | Bounded syntax normalization before schema validation | Broad heuristic repair that guesses missing data |
