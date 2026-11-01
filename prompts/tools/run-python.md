# `run_python`

Execute a Python script in a fresh subprocess using SlipAgent's own Python and its advertised math libraries.

Use this for involved arithmetic, symbolic or numerical mathematics, simulations, parsing, filtering, aggregation, and related multi-step data transformations. Combine dependent calculations in one script to save model round trips. Print the required answers, summaries, units, and validation checks; intermediate variables stay inside the script. Use dedicated tools for project edits, navigation, Git, or remote calls when they suffice.

## Execution

Supply `code` as Python source, at most 262,144 characters. Each call starts fresh: variables, imports, and functions do not persist between calls. Use ordinary imports for available libraries. Expressions are not displayed automatically; use `print`. Scripts run with the workspace root as their working directory, with stdin closed. Python's safe-path option prevents implicit script/current-directory imports; explicitly configured `PYTHONPATH` still applies.

This is the harness interpreter, not the selected project interpreter. Use the project's explicit interpreter through `run_command` for project code, tests, and dependencies. Missing libraries and Python errors are returned as failures; do not install project dependencies into the harness environment.

## Output And Lifetime

Execution defaults to 120 seconds and is capped at 600 seconds. Output is captured and archived using the command log service. Each stream returns at most 32,000,000 characters, preserving beginning and end if larger; use `read_command_output` with the log ID for retained middle output. The context budget may further limit what appears in later requests.

Timeout or cancellation terminates the process group with the command tool's bounded cleanup. The temporary script is removed after the process settles. Code runs with the harness process's permissions, as shell commands do; this is not an OS sandbox. Keep generated files within the workspace or its allowed temporary directory, and perform only operations authorized by the user's task. Failed or interrupted code can have partial effects; inspect them before retrying. Selecting this tool does not grant access to other SlipAgent tools inside the script.
