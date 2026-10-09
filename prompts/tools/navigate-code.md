# `navigate_code`

## Operations

Use an explicitly configured language server for definition, references, implementation, or hover.

## Coordinates

Input `line`/`column` are 1-based Unicode character positions.

Results label UTF-16 columns explicitly.

## Paths and Paging

Locations follow the current Workspace Access mode. Returns all locations from `offset` onward and complete hover text. Source files may be up to 32 MB.

## Prerequisites

Requires `language_servers` in `.slipagent/project.json` and an already installed stdio server. Use `grep` and `read_file` for ordinary discovery; follow the explicit position units.
