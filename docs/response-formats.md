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
and [DeepSeek V3.1 tool format](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base/blob/main/README.md#toolcall).
JSON boundaries are decoded before marker boundaries, so marker text inside
arguments does not split a call. Mistral's terminal `</s>` and DeepSeek's terminal
`<｜end▁of▁sentence｜>` are accepted after their complete batches.

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
response-correction path before any calls execute. Streaming API fields remain
deltas; the normalizer does not guess whether repeated fields are snapshots.
