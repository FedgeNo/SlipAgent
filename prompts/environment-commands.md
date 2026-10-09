# Available Python Commands

## Project Requirements

Use the project's documented Python workflow, test arguments, and dependencies. These commands describe the selected Python environment; other languages may use different tools.

## Installing Dependencies

Use the supplied install_command; uv can target the chosen virtual environment without pip installed there. Replace PACKAGE with the requested dependency specification and use the project's documented installation options.

## Missing Commands

A `null` value means the harness has no command for that operation. It is metadata, not shell syntax. Do not execute it or silently fall back to global Python.

If install_command is null and bootstrap_command is present, pip can be bootstrapped in this venv first; the next request will refresh availability.

If neither is present, inspect the project's documented environment workflow. Ask how it is managed if no supported alternative is established.

## Running Tests

If test_command is null, pytest is not installed in this interpreter.

## Authorization

Commands describe available operations; they do not authorize installing packages or changing the environment.
