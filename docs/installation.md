# Installation and Data Locations

[Back to the README](../README.md)

## Quick Start

Requires **Python 3.11+**, an API key for your selected provider, and Git for the Git tools.

Terminology: a message is one user, assistant, system, or tool message. A step is one model response plus any requested tool batch and its results. A run processes a user request until a reply without tools, a successful answer-only batch, a stop, or a limit. A session is the conversation containing runs, saved by default. Model context carries one JSON object per history step.
Clone or download this repository, then run the installer from its root:

```bash
python3 install.py
```

On Windows, use `py -3 install.py`. Installations are per-user and require no administrator privileges. The installer copies the application source, creates a private virtual environment, installs SlipAgent in editable mode, and places its launcher on your user PATH. It uses `uv` if available, otherwise Python's `venv` and pip. Add `--dev` to include the project's testing and development dependencies.

To update a managed installation, run `slipagent --update`. It downloads the latest version from GitHub and runs that version's installer, with the local-edit check described below; it reports when you already have the latest version. A development checkout is updated with Git instead.

The same script also updates an existing installation. Download the desired version and run `python3 install.py` from that download (Windows: `py -3 install.py`). It replaces installed application files and refreshes dependencies and the launcher; it does not download updates or run Git commands. Running inside the installed checkout only refreshes dependencies and the launcher. Unchanged installations update without warnings or confirmation. If application files have been modified, added, or deleted, the installer lists every affected path, warns that updating may lose local edits, and asks you to back up changes before confirming with `y` or `yes`. Enter, EOF, or Ctrl+C cancels. A missing or invalid baseline also requires confirmation and lists files to inspect. Changes from separate checkouts are not merged. First installations and `--dry-run` need no confirmation.

The installer compares SHA-256 hashes against `installed-file-hashes.json` in the installation directory, falling back to the checkout's `file-hashes.json` when available. Successful installations or updates from another checkout record the installed file hashes. Running inside the installed checkout retains the existing baseline rather than treating local edits as an unmodified release. Git internals, secrets, virtual environments, and caches are excluded from comparisons; the hash manifest itself is excluded to avoid self-reference.

| OS | Installation directory | PATH launcher |
| --- | --- | --- |
| Linux | `~/.local/lib/slipagent` | `~/.local/bin/slipagent` |
| macOS | `~/Library/Application Support/SlipAgent` | `~/.local/bin/slipagent` |
| Windows | `%LOCALAPPDATA%\Programs\SlipAgent` | Installation directory + `\bin\slipagent.exe` |

Linux and macOS use a symlink. Windows also uses a symlink when permitted, with an executable launcher fallback that does not require elevation. Persistent PATH entries are added to bash, zsh, sh, ksh, or fish configuration on Linux/macOS, or to the Windows user PATH. Open a new terminal when PATH changes; other shells receive instructions to add the directory manually. Use `--dry-run` to inspect the destinations, or `--install-dir`, `--bin-dir`, and `--python` to override them.

The installer replaces recognized SlipAgent launchers and saves a backup; it refuses unrelated commands or nonempty unmanaged installation directories. The permanent working checkout lives at `source` inside the installation directory, with its own `.venv`. Initial copies include application files, uncommitted edits, and Git metadata, but exclude existing virtual environments, caches, and secrets. Updates from another checkout replace the application directories and supplied root files while preserving installed Git metadata, the virtual environment, and configuration. Removed application directory contents are removed during updates too. Existing Git history is retained, so new downloaded files appear as working-tree changes. Failed updates restore the previous application and dependencies as described below. Older release directories are retained during migration. On Linux/macOS, `current` points to `source`; `installation.json` records the source, interpreter, and launcher on all platforms. Edit the permanent checkout for live changes and normal Git development.

An existing checkout `.env` is copied to the private user data directory on the first install, with owner-only file permissions on Linux/macOS. Subsequent installs preserve that configuration. SlipAgent reads only its own user-data `.env`; old installation `.env` files remain a compatibility fallback. Project `.env` files belong to their projects and are never read. The installer does not move or delete the original checkout or project sessions.

### Installation and Data Locations

Application files and private user data have different owners and lifetimes:

| Location | Contents |
| --- | --- |
| Installation directory + `source/` | Editable source, prompts, documentation, Git metadata when supplied, and the harness `.venv` |
| Installation directory | Managed-install marker, installation metadata, launcher backups, installed hash baseline, and the POSIX `current` link |
| Platform data directory | Saved sessions, command output, inspection artifacts, preferences, global `mcp.json`, and saved `.env` credentials |
| Project workspace | Your application's files and optional `.slipagent/project.json` |

| OS | Default Private Data Directory |
| --- | --- |
| Linux | `$XDG_DATA_HOME/slipagent`, or `~/.local/share/slipagent` |
| macOS | `~/Library/Application Support/SlipAgent/data` |
| Windows | `%LOCALAPPDATA%\SlipAgent` |

`SLIPAGENT_STATE_DIR` overrides the data location. Changing it does not move
existing data or relocate the program. On Linux, the default editable checkout
is `~/.local/lib/slipagent/source`; the executable on PATH points into that
checkout's virtual environment. Installing from a downloaded archive does not
create Git history if the download has no `.git` directory.

Managed updates back up source, the virtual environment, activation metadata and
the launcher before changing them. A failed copy, dependency installation,
verification or activation restores those components. Backup failures stop the
update before replacement. If rollback itself fails, the installer retains the
backup directory and reports its location. Private user configuration is preserved.

### Running From a Development Checkout

For development directly in this checkout instead of a managed installation, use the existing project environment or create one if none exists:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Create an ignored `.env` file containing `OPENROUTER_API_KEY=your-key`. Run
`slipagent` to select the highest-ranked eligible free model automatically, or
select NVIDIA Nemotron 3 Ultra (free) explicitly:

```bash
slipagent --model nvidia/nemotron-3-ultra-550b-a55b:free
```

For another project or a single task:

```bash
slipagent --workspace /absolute/path/to/myproject --model nvidia/nemotron-3-ultra-550b-a55b:free
slipagent --model nvidia/nemotron-3-ultra-550b-a55b:free -p "add type hints to src/parser.py and run the tests"
```

The editable install (`-e`) is essential for working on the running harness:
Python loads the source from its editable installation, so edits in that source tree reach the source
watcher. The `[dev]` extra installs the test and type-checking tools; use
`python -m pip install -e .` if you only need the application.

Reuse an existing project environment instead of recreating it. If that venv
has no pip, `uv pip install --python .venv/bin/python -e ".[dev]"` targets it
explicitly without needing pip inside it. See the
[environment setup and bootstrap alternatives](../CONTRIBUTING.md#start-with-the-correct-environment).
