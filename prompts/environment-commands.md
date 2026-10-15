# Available Python Commands

Use the project's documented workflow, test arguments, dependencies, and installation options. These commands target the selected Python environment.

Replace PACKAGE in `install_command` with the authorized dependency specification. uv can target a venv lacking pip. A `null` command means the operation is unavailable.

If `install_command` is null and `bootstrap_command` exists, bootstrap pip in this venv first; command availability refreshes on the next request. If both are absent, inspect documented environment management and ask when no supported alternative is established. A null `test_command` means this interpreter lacks pytest.

Environment changes and package installation require user authorization.
