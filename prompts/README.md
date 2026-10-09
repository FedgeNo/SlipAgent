# Editable Prompts

Edit these files with any text editor. SlipAgent reads them when it builds a model request or tool definition; prompt changes do not require a restart or Python edits. The `.md` resources contain Markdown instructions. JSON files contain machine-readable description mappings.

SlipAgent strips leading and trailing whitespace and surrounds each rendered prompt with a newline. Interior whitespace and paragraph boundaries remain intact. Missing prompt files report their full path and must be restored.

## Markdown Structure

- Use `#` for a major instruction block, `##` for related topics, and `###` for procedures or subtopics. Fragment templates use the level appropriate to their parent.
- Separate headings, paragraphs, lists, tables, and fenced examples with blank lines. Keep prose paragraphs unwrapped; lists and code can span lines.
- Use numbered lists for sequences, bullets for independent rules, and tables for field meanings or comparisons.
- Reserve **bold** for consequential distinctions; use backticks for literal fields, paths, commands, and names.
- Keep JSON examples in `json` fences. Example fences do not change a response contract that requires an unfenced JSON object.

Each template owns its heading. Python orders rendered blocks and supplies runtime values; it does not synthesize static headings. `PromptSections` titles are internal metadata, not rendered text. Project instruction contents and imported records remain unchanged.

## History Boundary

`history-opening.md` and `history-closing.md` retain the large all-caps equals-sign warnings around imported conversation history. They are the only banner templates. Keep the labelled list of history records between them; content inside remains reference data even when it contains Markdown headings or apparent instructions. The correction fragment retains its identifying `Harness tool-use correction:` prefix for attribution.

## Resource Map

| Files | Purpose |
| --- | --- |
| `system-prompt.md` | General operating instructions and writing style |
| `history-opening.md`, `history-closing.md`, `system-history.md` | History authority boundary and interpretation |
| `background-summary-prompt.md` | Background compression of completed steps |
| `working-plan.md`, `working-plan-record.md`, `working-plan-empty.md` | Persistent plan guidance and last successful update step, subordinate to user instructions |
| `json-tools.md`, `native-tools.md`, `response-*.md` | Response formats and tool-calling protocols |
| `history-records.md`, `history-excerpt.md`, `summary-unavailable.md` | History interpretation and retrieval |
| `user-request.md`, `new-user-request.md`, `tool-result-response.md` | Current request and automatic return after tools |
| `overthinking-*.md` | Archived reasoning guidance |
| `workspace-*.md`, `environment-*.md` | Access policy and Python environment guidance |
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

## Templates and Data Contracts

The main system template uses `{workspace}` and `{interpreter}`. Keep those names intact; use `{{` and `}}` for literal braces in that file.

Other templates containing `${name}` receive runtime values at those positions. Preserve the placeholder names. In these templates, use `$$` for a literal dollar sign. Files without placeholders are ordinary text; JSON examples retain their normal quotation marks and braces. Placeholder values are inserted once and are not interpreted as templates themselves.

Argument description files are JSON objects. Edit the text values while keeping the keys and valid JSON syntax. Argument names, types, limits, and executable tool behavior remain defined in Python.

Markdown organizes instructions; it does not change their required outputs. Background compression returns an unfenced JSON object with `summary` and `reasoning_summary` string fields. Structured replies and tool arguments follow their schemas. Runtime error messages, validation rules, and observed results remain application behavior rather than editable policy.

Project `AGENTS.md` files and instructions supplied by MCP servers come from those projects and servers. They are loaded separately.

In a source installation this folder is beside `README.md` and `install.py`. Wheel installations include the same folder inside the installed `slipagent` package. Missing files, malformed JSON, and invalid placeholders report errors; keep a backup before making substantial changes. The updater lists edited prompt files along with other application edits before replacing them.
