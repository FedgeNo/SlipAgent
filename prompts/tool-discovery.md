# Additional Tool Catalog

The tools below are available to load, but their complete definitions are not
included in this request. Use `load_tools` with the exact names you need, then
wait for the next request before calling those tools with the supplied argument
schemas. A call to any tool without a supplied full definition rejects the entire
batch: no tools in that response execute, including `load_tools`. Prefer the
supplied native tools when they suffice; load a suitable tool rather than
substituting shell commands merely because its definition is deferred. Load only
what helps the current task.

Loaded definitions remain available across turns and model changes in this
conversation. Starting a new conversation clears them; resuming a saved one
restores tools that were loaded or called in its history. Tool availability
still depends on current configuration, connections, and session restrictions.

| Tool | Purpose |
| --- | --- |
${catalog}
