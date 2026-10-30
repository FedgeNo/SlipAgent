# Session Environment

These are the current local session facts. The user is the operating-system account running the harness; the machine name identifies the host executing tools. Time includes the local UTC offset. Use these facts when choosing commands and interpreting paths or dates.

`process_working_directory` is the harness process's directory, which may differ from the selected `workspace`. Tools use the workspace for relative paths; commands use `command_working_directory` by default. An unavailable process directory is reported as null with its error and does not change the workspace. Harness dependency setup runs in the harness Python environment directory, independently of the process directory.

```text
${environment}
```
