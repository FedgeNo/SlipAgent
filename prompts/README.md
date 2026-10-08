# Editable Prompts

Edit these files with any text editor. SlipAgent reads them when it builds a model request or tool definition; prompt changes do not require a restart or Python edits. Keep each paragraph on one line, with a blank line between paragraphs. Preserve meaningful list and example boundaries.

SlipAgent strips leading and trailing whitespace and surrounds each rendered prompt with a newline. Interior whitespace and paragraph boundaries remain intact. Missing prompt files report their full path and must be restored.

Major assembled sections use large all-caps `BEGIN` and `END` dividers. Project instruction contents remain intact inside their section boundaries; paths retain their original case.

| Files | Purpose |
| --- | --- |
| `system-prompt.txt` | General operating instructions and writing style |
| `background-summary-prompt.txt` | Background compression of completed steps |
| `json-tools.txt`, `native-tools.txt`, `response-*.txt` | Response formats and tool-calling protocols |
| `history-records.txt`, `history-excerpt.txt`, `summary-unavailable.txt` | History interpretation and retrieval |
| `user-request.txt`, `new-user-request.txt`, `tool-result-response.txt` | Current request and automatic return after tools |
| `overthinking-*.txt` | Archived reasoning guidance |
| `workspace-*.txt`, `environment-*.txt` | Access policy and Python environment guidance |
| `project-instructions-*.txt` | Project instruction scope and delivery |
| `mcp-guidance.txt` | Guidance for connected MCP tools |
| `repair-*.txt`, `rejected-output.txt`, `repeated-tools.txt` | Response and repeated-tool recovery |
| `repository-map*.txt` | Repository orientation |
| `session-*.txt` | Restored-session guidance |
| `background-command-results.txt`, `command-output-*.txt` | Command completion and output recovery |
| `tools/*.txt` | Built-in tool descriptions |
| `tools/*-parameters.json` | Tool argument descriptions, keyed by their location in the schema |

The main system template uses `{workspace}` and `{interpreter}`. Keep those names intact; use `{{` and `}}` for literal braces in that file.

Other templates containing `${name}` receive runtime values at those positions. Preserve the placeholder names. In these templates, use `$$` for a literal dollar sign. Files without placeholders are ordinary text; JSON examples retain their normal quotation marks and braces. Placeholder values are inserted once and are not interpreted as templates themselves.

Argument description files are JSON objects. Edit the text values while keeping the keys and valid JSON syntax. Argument names, types, limits, and executable tool behavior remain defined in Python.

Project `AGENTS.md` files and instructions supplied by MCP servers come from those projects and servers. They are loaded separately.

In a source installation this folder is beside `README.md` and `install.py`. Wheel installations include the same folder inside the installed `slipagent` package. Missing files, malformed JSON, and invalid placeholders report errors; keep a backup before making substantial changes. The updater lists edited prompt files along with other application edits before replacing them.
