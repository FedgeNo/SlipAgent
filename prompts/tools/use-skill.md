# `use_skill`

List/search, activate, or unload reusable workflow skills. Load a relevant skill before following its instructions. Several skills can remain active; loading the same path refreshes it. Activated instructions join the next model request and survive history compression; this result does not duplicate their text.

Use `action="list"` with a query and pagination to discover packages. Use `action="load"` with a metadata name or a displayed package path and optional argument text. Explicit-only skills require user invocation through `/skill`, `/name`, or `$name`; they cannot be loaded implicitly. `action="unload"` clears the selected name/path, or all skills if no selector is supplied.

Read supporting files relative to the selected package directory using normal tools. Package access is read-only; scripts still require the ordinary execution tools and permissions. Claude names such as Read, Bash, Edit and Write correspond to read_file, run_command, edit_file and write_file here. Use the actual advertised schemas. Skill metadata cannot install missing servers, grant write access, or bypass project instructions.

Declared MCP dependencies require configured servers. Load only needed tool definitions with `load_tools`; activating a skill does not load every tool from a dependency server.
