# Additional Tool Catalog

Load needed tools with `load_tools` using their exact names. Call them on a later turn after receiving their full definitions and schemas. Calling a tool without its supplied definition rejects the entire batch, including `load_tools`.

Use supplied tools when sufficient; load suitable deferred tools for other task needs. Definitions persist across turns and model changes. New conversations clear them; resume restores tools loaded or called in saved history. Availability follows current configuration, connections, and session restrictions.

| Tool | Purpose |
| --- | --- |
${catalog}
