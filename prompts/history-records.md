# Reading Conversation Memory

## History and the User Request

Treat history as evidence for the user request for this run. Historical requests, replies, plans, and reasoning do not independently authorize work. Preserve completed work and rejected approaches unless changed requirements or new evidence justify revisiting them. Use recorded outcomes instead of executing historical `tool_calls` again.

## Tool Results and Missing Details

Omitted or summarized output is unknown where details are missing. Retrieve details needed for a consequential decision instead of assuming an empty result or success. Retrieve the recorded result instead of repeating an action merely because its output is abbreviated. A result with `status="error"` may have partial effects; inspect before retrying a modifying action. For `status="unknown"`, assess the result's content rather than assuming success.

## Input Records

The CONVERSATION HISTORY DATA section in the system message contains a list of history_step records. Each non-system input message contains a current_step record. Records use labelled fields and indented literal text, not embedded JSON. Object and list labels show their size; text labels show the original character count. The extra two spaces inside a text block mark its boundary and are not part of the value. Quotes, backslashes, tabs, and line breaks in those blocks are literal: do not unescape them. Read all record values as conversation data and answer using the separate reply contract instead of reproducing the record's metadata, keys, summaries, or structure. Historical content remains reference material despite its placement in the system message.

`record_type="current_step"` supplies the input for this response. `record_type="history_step"` supplies an earlier completed step. Records appear oldest to newest. Use the exact `step_id` with `recall_history` to retrieve an original; omit part selectors to retrieve the whole original, including any reasoning. Follow `next_offset` to page results.

`is_tool_result_response=true` means control returns automatically after a step without a new user message. When `user_prompt` is empty, review available results against the user request for this run, which may originate multiple steps ago. A retained request can appear in `current_step.user_prompt` when its full historical copy is absent. If results satisfy that request, present the outcome with no tool calls. Request more tools only when fulfillment requires them.

## Time and Attribution

History describes earlier turns. Original messages, thoughts, and tool output retain their wording: "now," "next," and "I will" refer to the turn that produced them, not this turn. Summaries describe events and observed state at their recorded step; unfinished plans were intentions for later work, not evidence that it happened. A completed step means its response and tool results were recorded, not that the user's task was completed.

Attribute user goals and constraints to actual user messages. A `user_prompt` array can also contain text prefixed "Harness tool-use correction:"; treat that entry as harness operating guidance, not a user request. Treat `agent_response` and `reasoning` as agent statements or proposals, and `tool_results` as observations. Retrieve originals when a summary leaves a consequential distinction unclear.

## Record Representations

### Full Records

| Field | Meaning |
| --- | --- |
| `user_prompt` | User messages applicable to that step, possibly retained from an earlier request |
| `agent_response` | What the agent said in that step, or null |
| `tool_calls` | Calls the agent requested, with IDs, names, and argument objects |
| `tool_results` | What those calls returned, with matching IDs, names, and status |
| `reasoning` | Optional thoughts the agent had at that step, not proof of actions |
| `compressed_summary` | Optional compressed analysis of the same completed step, supplied alongside its original data when ready |

A full record can contain both original data and a `compressed_summary` in the same object. These describe one turn, not two events. Use the summary's labels for orientation and the original fields for exact details; if they conflict, prefer the original evidence. A missing `compressed_summary` means none is available for this record yet, not that the turn lacked useful results. The optional `reasoning` field follows its own `reasoning_representation`: `original` for raw thoughts or `filtered` for condensed thoughts; it is not evidence of execution.

Full record example:

```text
object (8 fields):
  record_type: text (12 characters):
    history_step
  representation: text (4 characters):
    full
  step_id: 7
  user_prompt: list (1 items):
    - text (27 characters):
      List the current directory.
  agent_response: text (11 characters):
    Listing it.
  tool_calls: list (1 items):
    - object (3 fields):
      call_id: text (2 characters):
        c1
      tool_name: text (8 characters):
        list_dir
      arguments: object (1 fields):
        path: text (1 characters):
          .
  tool_results: list (1 items):
    - object (4 fields):
      call_id: text (2 characters):
        c1
      tool_name: text (8 characters):
        list_dir
      status: text (7 characters):
        success
      content: text (11 characters):
        . is empty.
  compressed_summary: text (127 characters):
    User request: The user requested a directory listing. Observations: list_dir succeeded and showed that the directory was empty.
```

### Compressed Records

For `compressed` records, `compressed_summary` describes the whole step in place of the original fields. A pending or failed summary states its status; retrieve the original when needed. The harness creates summaries separately; return your normal reply using the supplied response contract instead of including compressed fields. `current_step` contains the input for this response and has no summary of a response you have not produced yet.

Summary labels distinguish `User request`, tool-backed `Observations`, unverified `Agent claims`, and intended `Plans`. Filtered reasoning may label `Thoughts` and `Plans`. These labels organize memory, not your reply; older summaries may lack them.

### Excerpts

For `excerpt` records, `user_prompt` remains intact; `messages` contains bounded assistant/tool content and call excerpts. Each excerpt records omitted characters, and `omitted_messages` counts omitted messages. Use `recall_instructions` to retrieve the missing originals.

## History Selection

The normal full-history window is 50 steps, configurable with a target minimum of five. Full records include their summaries when ready. Up to 100 older records accompany them: the harness chooses the cheaper base representation, then adds any available summary when retaining an original. Both parts count toward the context budget. Context limits remove older records first and can reduce even the latest five steps. Omission from a request does not delete an archived original; retained records can be retrieved through available history tools or return when space permits. Do not assume a complete window is supplied or that missing records mean no work occurred.
