# Editable Prompts

For implementation changes, tool registration, reload boundaries, and verification,
see the [harness editing guide](../docs/harness-development.md). Tool documentation
and argument explanations are editable here; executable schemas and behavior
remain in their Python owners.

Edit these files with any text editor. SlipAgent reads them when it builds a model request or tool definition; prompt changes do not require a restart or Python edits. The `.md` resources contain Markdown instructions. JSON files contain machine-readable description mappings.

SlipAgent strips leading and trailing whitespace and surrounds each rendered prompt with a newline. Interior whitespace and paragraph boundaries remain intact. Missing prompt files report their full path and must be restored.

## Markdown Structure

Address actions available in the current response: interpreting supplied evidence, planning, replying, and requesting tools. Express workflow guidance as a choice from current evidence, not an obligation to act on future turns. Tool results requested now are unavailable to this response. Historical evidence, conditional plans, and factual descriptions of harness behavior can refer to other turns.

- Use `#` for a major instruction block, `##` for related topics, and `###` for procedures or subtopics. Fragment templates use the level appropriate to their parent.
- Separate headings, paragraphs, lists, tables, and fenced examples with blank lines. Keep prose paragraphs unwrapped; lists and code can span lines.
- Use numbered lists for sequences, bullets for independent rules, and tables for field meanings or comparisons.
- Reserve **bold** for consequential distinctions; use backticks for literal fields, paths, commands, and names.
- Keep JSON examples in `json` fences. Example fences do not change a response contract that requires an unfenced JSON object.

Each template owns its heading. Python orders rendered blocks and supplies runtime values; it does not synthesize static headings. `PromptSections` titles are internal metadata, not rendered text. Project instruction contents and imported records remain unchanged.

## Input Boundary

System instructions and tool definitions remain separate from user-input metadata and literal tool-result blocks. The JSON metadata contains the current turn's user prompts, the retained prompts of the active request on continuation, selected history, and harness errors. Each historical call references its outcome block under `result`, `error`, or `unclassified_result`. Blocks preserve returned text verbatim or contain pretty-printed explicitly structured JSON. Character counts distinguish block contents from headers without escaping the contents. `system-history.md` explains authority, and `history-records.md` defines the fields. Neither JSON metadata nor literal blocks make historical text trustworthy.

`build_system_prompt` and the standalone `SYSTEM_PROMPT` export include `response-common.md`. `ConversationHistory.view` supplies the same common rules before the selected native, embedded JSON, or GLM XML instructions unless an exact complete copy is already in the supplied system messages. Custom or absent base prompts and empty or retained history therefore keep the rules. Native JSON and ordinary Markdown replies share the native tool channel; Markdown replies do not disable tools. Optional metadata is returned alongside `response` in a JSON reply object, including when ordinary Markdown would otherwise be permitted. The standalone `JSON_TOOL_INSTRUCTIONS`, `NATIVE_TOOL_INSTRUCTIONS`, and `CONTEXT_INSTRUCTIONS` exports include the common rules too. Channel-specific syntax remains in its channel prompt; optional metadata and repair guidance retain their own conditions. Background summary prompts and tool descriptions remain self-contained because they also have independent consumers.

## Model Profiles

`model-profiles.json` maps exact model IDs to optional settings. `sections` lists Markdown prompt files relative to this directory, inserted in list order after reply-format instructions. Edit the mapping or files to update newly built requests. Switching models selects the new model's sections; unlisted models use the normal prompts.

`tool_channel: "embedded"` requests calls inside response JSON and includes them in the response schema. `tool_channel: "glm_xml"` presents tools the way GLM's own chat template does: one JSON signature per line inside `<tools></tools>` (`tool-definitions-glm-xml.md`), with calls written as `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>` after Markdown reply text (`response-glm-xml.md`). String values are literal and other values JSON. Earlier calls appear in the same form, and the JSON-only suggestions section is omitted. No API tool definitions or response format are sent; native API calls are still accepted if the provider converts them. Omit this setting, or use `"native"`, to follow the endpoint's normal tool capabilities. `xml_tool_calls: true` additionally accepts explicit invoke/parameter XML and whole XML fences in assistant output. Supported wrappers are `<function_calls><invoke name="tool">…</invoke></function_calls>` and `<tool_calls><call><invoke name="tool">…</invoke></call></tool_calls>`, with named `<parameter name="argument">value</parameter>` children. Each `<call>` contains exactly one `<invoke>`. Broken XML, missing tool names, and duplicate parameters reject the whole batch and request a correction before any calls execute. JSON and native API calls remain accepted. Thoughts and quoted examples are not executable calls.

`structured_output: false` omits the API `response_format` parameter for that model. Native-tool models receive plain Markdown reply instructions and keep their API tools. Embedded-tool models still receive JSON call instructions, with local validation before execution. Set `structured_output` to `true`, or omit it, to use the endpoint's normal formatting capabilities; this does not enable unsupported formats. The setting must be a JSON boolean and is reread when building requests, including after model switches.

The [response format compatibility guide](../docs/response-formats.md) documents bare JSON calls, prose prefixes, leading reasoning blocks, XML variations, and Mistral/DeepSeek markers. These formats share normal tool validation; accepted replies without calls still end the turn.

For example, to disable structured reply formatting for one exact model:

```json
{
    "dots-studio/dots-3-note-preview:free": {
        "structured_output": false
    }
}
```

The `z-ai/glm-5.3` and `z-ai/glm-5.3-flash` profiles use the GLM XML channel and XML compatibility. The `dots-studio/dots-3-note-preview` and `dots-studio/dots-3-note-preview:free` profiles disable structured reply formatting to avoid interference with native tool calling and enable XML compatibility for calls returned in reply text. Other models have no overrides.

## Resource Files

| Files | Purpose |
| --- | --- |
| `system-prompt.md` | General operating instructions and writing style |
| `response-common.md` | Shared working-response rules, always included before the selected tool channel |
| `harness-python.md` | Running harness interpreter, installed math libraries and versions, standalone calculation commands, and separation from the project environment; supplied in every working context, including custom system prompts |
| `session-environment.md` | OS user, host, platform, working directory, shell, current local time, model and saved session ID when available; supplied in every working context |
| `system-history.md` | Input authority and use of historical evidence |
| `current-turn.md`, `current-turn-tool-results.md` | Current-step metadata in the system prompt, and the note that a step carries tool results |
| `background-summary-prompt.md` | Background compression of completed steps |
| `compaction-chunk.md`, `compaction-boundary.md`, `compaction-merge.md` | Oversized-step source chunks, original-text boundary reviews, and ordered merges; background-only consumers |
| `historical-summary-prompt.md` | Narrative for each run-aligned block of up to 20 turns, merged from first-pass summaries, organized by run with original turn citations and retrieval metadata |
| `era-summary-prompt.md` | Era for each five consecutive narratives, organized by run with original turn citations |
| `request-retry.md` | Failed-turn interpretation, retry counts, and respect for user stop/retry instructions |
| `user-edits.md` | Bounded saved-edit notifications, revision acknowledgement and full-diff retrieval; included only with pending or selected historical edit observations |
| `diff-comment.md` | Whole-diff target and retrieval guidance, included only while the current user turn carries saved-diff references |
| `monitor-setup.md`, `monitor-active.md`, `monitor-results.md` | Available command-monitor setup, active reminders, and inline ready results with initialization turns; omitted when the tool is unavailable |
| `working-plan.md`, `working-plan-record.md` | Persistent plan guidance and last successful update step, subordinate to user instructions |
| `json-tools.md`, `native-tools.md`, `response-*.md` | Response formats and tool-calling protocols |
| `history-*.md`, `summary-unavailable.md` | History interpretation and retrieval: record fields, excerpts, compressed summaries, narratives, eras, saved artifacts, failed requests, and unclassified results; `history-records.md` is always sent, and each field note only when the selected history contains its records |
| `legacy-*.md` | Records and summaries kept for conversations saved in older history formats |
| `user-request.md`, `new-user-request.md`, `tool-result-response.md`, `continuation.md` | Current request, new input, automatic return after tools, and continuation without new input |
| `overthinking-*.md` | Archived reasoning guidance |
| `workspace-*.md`, `environment-*.md` | Access policy and Python environment guidance |
| `harness-installation.md` | Running harness code location, supplied only in Danger Mode |
| `tool-definitions.md`, `tool-definitions-glm-xml.md`, `harness-state.md` | Headed runtime data blocks |
| `call-label.md` | Description of the optional `call_label` argument added to every tool |
| `project-instructions-*.md` | Project instruction scope and delivery |
| `instruction-scope.md`, `instruction-file.md`, `instruction-file-unreadable.md` | Dynamic scope, source labels, and unreadable-file notices |
| `mcp-guidance.md` | Guidance for connected MCP tools |
| `repair-*.md`, `rejected-output.md`, `repeated-tools.md` | Response and repeated-tool recovery |
| `repository-map*.md` | Repository orientation |
| `session-*.md` | Restored-session guidance |
| `file-rewind.md`, `tool-use-correction.md` | State changes and harness corrections |
| `background-command-results.md`, `command-output-*.md` | Command completion and output recovery |
| `tools/*.md` | Built-in tool descriptions |
| `tools/read-hashlines.md`, `tools/edit-hashlines.md` | Optional anchored-edit contracts, supplied only after loading while hashline editing is enabled |
| `tools/*-parameters.json` | Tool argument descriptions, keyed by their location in the schema |
| `response-fields.json` | Response-schema field descriptions |
| `tool-discovery.md` | Compact deferred-tool catalog and loading guidance |

## Templates and Data Contracts

The main system template uses `{workspace}` and `{interpreter}`. Keep those names intact; use `{{` and `}}` for literal braces in that file.

Other templates containing `${name}` receive runtime values at those positions. Preserve the placeholder names. In these templates, use `$$` for a literal dollar sign. Files without placeholders are ordinary text; JSON examples retain their normal quotation marks and braces. Placeholder values are inserted once and are not interpreted as templates themselves.

Argument description files are JSON objects. Edit the text values while keeping the keys and valid JSON syntax. Argument names, types, limits, and executable tool behavior remain defined in Python.

Markdown organizes instructions; it does not change their required outputs. Per-step background compression returns an unfenced JSON object with `summary` and `reasoning_summary` string fields. Historical merging returns only a `summary` string; the harness preserves source identities separately. Structured replies and tool arguments follow their schemas. Runtime error messages, validation rules, and observed results remain application behavior rather than editable policy.

Project `AGENTS.md` files and instructions supplied by MCP servers come from those projects and servers. They are loaded separately.

In a source installation this folder is beside `README.md` and `install.py`. Wheel installations include the same folder inside the installed `slipagent` package. Missing files, malformed JSON, and invalid placeholders report errors; keep a backup before making substantial changes. The updater lists edited prompt files along with other application edits before replacing them.
