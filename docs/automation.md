# Headless Automation and Evaluations

[Back to the README](../README.md)

## Headless Automation and Evaluations

One-shot runs can stream machine-readable JSON Lines instead of rendered answers:

```bash
slipagent --json-events --prompt-file task.txt --workspace /path/to/trial --no-reload > events.jsonl
```

`--prompt-file PATH` reads a UTF-8 prompt, preserving its line endings and trailing
whitespace. Use `--prompt-file -` to read UTF-8 from stdin. It cannot be combined
with `-p` or positional prompt text. `--json-events` also works with the existing
`-p` and positional forms; it requires a one-shot task and cannot list models or
start the interactive terminal. Without this flag, normal output is unchanged.

Each stdout line is one complete JSON object, flushed immediately. Human progress
and diagnostics remain on stderr. Records share `schema_version` (currently 1),
`record_type`, a sequential `sequence`, and `elapsed_seconds`:

| Record Type | Contents |
| --- | --- |
| `run_start` | Selected model/provider, workspace, and session ID |
| `event` | Agent event `kind` and step, text, structured tool call/result, or usage as applicable |
| `run_end` | Completion status, exit code, exact answer, error if any, working request count, and session usage |

Context events carry the outgoing request as a `request` **object**, rather than
a JSON string inside JSON. Tool arguments and result objects retain their data
types. Match tool observations by their step and call ID. Per-token reasoning and
reply fragments and incremental command-output chunks are omitted; completed
reply/tool events, request snapshots, and the existing saved sessions retain the
corresponding observations. This avoids enormous fragment-by-fragment logs.

`run_end.status` is `completed`, `step_limit`, `stopped`, or `error`. Completed
runs exit 0; execution errors, step limits, and early stops exit 1. Setup failures
exit 2 and emit an error `run_end` with `phase="setup"`; no `run_start` is required
in that case. Argument or prompt-file errors use argparse's stderr diagnostics
and exit 2 without starting a run. A crash or externally terminated process may
leave no final record: treat a missing `run_end` as incomplete.

**Completed means the harness finished, not that its answer or code was correct.**
Tool errors stay labelled in their event even if the model recovers. Grade the
resulting files and behavior independently. Session usage includes background
compression; `requests` counts working requests, including retries.

Automation mode does not remember its selected model in your preferences. Session
persistence, providers, tool execution, context assembly, and compaction use the
same path as normal one-shot runs. MCP remains opt-in with `--mcp`. Use
`--no-reload` when comparing a fixed code version and a separate
`SLIPAGENT_STATE_DIR` for trial records. These records contain private project
text and complete model requests; keep them out of public commits.

### Repeated Trials

Use a fresh workspace and state directory per trial. The following Python example
copies a fixture, invokes the installed harness three times, and retains results
without parsing terminal text. It uses your normal configured provider and may
consume API quota or credits:

```python
import json
import os
from pathlib import Path
import shutil
import subprocess

fixture = Path("fixtures/coding-task").resolve()
prompt = Path("task.txt").read_text(encoding="utf-8")
records = Path("agents/trials").resolve()
records.mkdir(parents=True, exist_ok=False)
for number in range(1, 4):
    trial = records / str(number)
    workspace = trial / "workspace"
    shutil.copytree(fixture, workspace)
    # Set provider credentials in the environment for isolated state directories.
    environment = {**os.environ, "SLIPAGENT_STATE_DIR": str(trial / "state")}
    with (trial / "events.jsonl").open("w", encoding="utf-8") as events, \
         (trial / "stderr.txt").open("w", encoding="utf-8") as errors:
        process = subprocess.run(
            ["slipagent", "--json-events", "--no-reload", "--no-mcp",
             "--workspace", str(workspace), "--max-steps", "20", "-p", prompt],
            stdout=events, stderr=errors, env=environment,
        )
    outcome = {"process_exit_code": process.returncode}
    (trial / "process.json").write_text(json.dumps(outcome, indent=2) + "\n")
    # Run independent behavior checks against workspace here.
```

Use small fixture directories with ordinary files, not a working installation,
symlinks to shared writable files, or a copied virtual environment. Preserve the
source version, prompt, model ID, sampling settings, grading command, and limits
with your evaluation records. Compare several trials per condition; keep
infrastructure failures separate from incorrect model behavior.
