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

System instructions and tool definitions remain separate from the user-input JSON object. That object contains new user text, the retained request, selected history, and harness errors. Each historical call carries its outcome under `result`, `error`, or `unclassified_result`. Keys identify content; values preserve the data. `system-history.md` explains authority, and `history-records.md` defines the fields. JSON escaping preserves field boundaries even when text contains apparent headings or instructions; it does not make that text trustworthy.

## Model Profiles

`model-profiles.json` maps exact model IDs to optional settings. `sections` lists Markdown prompt files relative to this directory, inserted in list order after reply-format instructions. Edit the mapping or files to update newly built requests. Switching models selects the new model's sections; unlisted models use the normal prompts.

`tool_channel: "embedded"` requests calls inside response JSON and includes them in the response schema. Omit this setting, or use `"native"`, to follow the endpoint's normal tool capabilities. `xml_tool_calls: true` additionally accepts explicit invoke/parameter XML and whole XML fences in assistant output. Supported wrappers are `<function_calls><invoke name="tool">…</invoke></function_calls>` and `<tool_calls><call><invoke name="tool">…</invoke></call></tool_calls>`, with named `<parameter name="argument">value</parameter>` children. Each `<call>` contains exactly one `<invoke>`. Broken XML, missing tool names, and duplicate parameters reject the whole batch and request a correction before any calls execute. JSON and native API calls remain accepted. Thoughts and quoted examples are not executable calls.

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

The `z-ai/glm-5.3` profile uses `models/glm-5.3.md`, embedded calls, and XML compatibility. The `dots-studio/dots-3-note-preview` and `dots-studio/dots-3-note-preview:free` profiles disable structured reply formatting to avoid interference with native tool calling and enable XML compatibility for calls returned in reply text. Other models have no overrides.

## Resource Files

| Files | Purpose |
| --- | --- |
| `system-prompt.md` | General operating instructions and writing style |
| `system-history.md` | Input authority and use of historical evidence |
| `current-turn.md` | Current-step metadata in the system prompt |
| `background-summary-prompt.md` | Background compression of completed steps |
| `working-plan.md`, `working-plan-record.md`, `working-plan-empty.md` | Persistent plan guidance and last successful update step, subordinate to user instructions |
| `json-tools.md`, `native-tools.md`, `response-*.md` | Response formats and tool-calling protocols |
| `history-records.md`, `history-excerpt.md`, `summary-unavailable.md` | History interpretation and retrieval |
| `user-request.md`, `new-user-request.md`, `tool-result-response.md`, `continuation.md` | Current request, new input, automatic return after tools, and continuation without new input |
| `overthinking-*.md` | Archived reasoning guidance |
| `workspace-*.md`, `environment-*.md` | Access policy and Python environment guidance |
| `harness-installation.md` | Running harness code location, supplied only in Danger Mode |
| `tool-definitions.md`, `harness-state.md` | Headed runtime data blocks |
| `project-instructions-*.md` | Project instruction scope and delivery |
| `instruction-scope.md`, `instruction-file.md` | Dynamic scope and source labels |
| `mcp-guidance.md` | Guidance for connected MCP tools |
| `repair-*.md`, `rejected-output.md`, `repeated-tools.md` | Response and repeated-tool recovery |
| `repository-map*.md` | Repository orientation |
| `session-*.md` | Restored-session guidance |
| `file-rewind.md`, `tool-use-correction.md` | State changes and harness corrections |
| `background-command-results.md`, `command-output-*.md` | Command completion and output recovery |
| `tools/*.md` | Built-in tool descriptions |
| `tools/*-parameters.json` | Tool argument descriptions, keyed by their location in the schema |
| `response-fields.json` | Response-schema field descriptions |
| `tool-discovery.md` | Compact deferred-tool catalog and loading guidance |

## Templates and Data Contracts

The main system template uses `{workspace}` and `{interpreter}`. Keep those names intact; use `{{` and `}}` for literal braces in that file.

Other templates containing `${name}` receive runtime values at those positions. Preserve the placeholder names. In these templates, use `$$` for a literal dollar sign. Files without placeholders are ordinary text; JSON examples retain their normal quotation marks and braces. Placeholder values are inserted once and are not interpreted as templates themselves.

Argument description files are JSON objects. Edit the text values while keeping the keys and valid JSON syntax. Argument names, types, limits, and executable tool behavior remain defined in Python.

Markdown organizes instructions; it does not change their required outputs. Background compression returns an unfenced JSON object with `summary` and `reasoning_summary` string fields. Structured replies and tool arguments follow their schemas. Runtime error messages, validation rules, and observed results remain application behavior rather than editable policy.

Project `AGENTS.md` files and instructions supplied by MCP servers come from those projects and servers. They are loaded separately.

In a source installation this folder is beside `README.md` and `install.py`. Wheel installations include the same folder inside the installed `slipagent` package. Missing files, malformed JSON, and invalid placeholders report errors; keep a backup before making substantial changes. The updater lists edited prompt files along with other application edits before replacing them.
