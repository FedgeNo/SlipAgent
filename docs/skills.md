# Skill Packages

Skills are reusable workflow instructions selected explicitly by the user.
They do not modify Python classes or automatically run scripts.

## Discover And Select

`/skills` lists names, descriptions, source paths, and disabled packages from:

1. The current project's `.slipagent/skills/` directory.
2. The `skills/` directory under SlipAgent's user data directory, including any
   `SLIPAGENT_STATE_DIR` override. See [installation locations](installation.md#installation-and-data-locations).
3. SlipAgent's bundled `prompts/skills/` directory, included in source and
   installed packages.

Each package is an immediate child directory containing `SKILL.md`. Discovery
does not recursively search resources or follow linked package directories.
Invalid or unreadable metadata is reported without hiding other packages.

Use `/skill release` to select a package by its metadata name. Name precedence
is **project → user → bundled**, including disabled packages: a disabled project
or user override prevents selection of a lower-priority namesake. Duplicate names within the selected scope require
an explicit path. `/skills` displays all paths so duplicates remain identifiable.

`/skill .slipagent/skills/release` or `/skill PATH/TO/SKILL.md` selects by path.
Paths are relative to the workspace unless absolute; quote a path when needed.
Catalogued user and bundled packages are permitted instruction sources. Other explicit
paths follow ordinary workspace access rules. Missing, invalid, or disabled
packages leave the current selection unchanged.

One skill is active at a time; selecting another replaces it. `/skill` shows
the selected name, source path, and source SHA-256. `/skill off` clears it.
Selection and clearing queue at the normal agent boundary while work is active;
listing and status remain available during a turn. There is no model-selected
activation or automatic catalog injection into model requests.

## Bundled Workflows

These optional starter workflows provide concise procedures with completion
conditions and recovery guidance. They have been reviewed and checked for
loading and selection, without comparative model-performance testing or claims
of proven improvements for lower-end models. They activate only when selected.

| Selection | Purpose |
| --- | --- |
| `/skill fix-bug` | Reproduce a bug, trace its cause, correct it, and verify the result. |
| `/skill implement-feature` | Define observable behavior, follow project patterns, and complete integration. |
| `/skill review-changes` | Find evidence-backed defects and report review coverage. |
| `/skill prepare-commit` | Review and verify a coherent commit within the user's authorized scope. |
| `/skill investigate-failure` | Diagnose a command, job, service, or workflow through bounded probes. |

The review skill asks for status, summary, coverage, and severity-ordered findings
with path/line, evidence, and uncertainty. It guides ordinary conversation and
does not restrict tools. `/review [scope]` runs a dedicated review using the same
agent with file-reading and local Git inspection tools only. Shell/Python
execution, edits, MCP tools, and running tests are unavailable in that mode.
Its structured `answer` tool validates findings; a prose-only or interrupted
review is incomplete. `/review results` opens the report or a finding's source
line, warning when a file has changed since inspection. The original tools and
selected skill are restored when review ends. Saved review records remain
available through `/view`; the latest-results picker is session-local.

Create a package with the same name in a higher-priority root to customize or
disable a bundled workflow. Selecting a package by its displayed path bypasses
name precedence. Bundled packages contain instructions only, with no scripts.

## Package Format

```text
.slipagent/skills/release/
├── SKILL.md
├── resources/
│   └── checklist.md
└── scripts/
    └── inspect-release.py
```

Example `SKILL.md`:

```markdown
---
name: release
description: Review a release using the project's checklist.
enabled: true
---
# Release Review

Read resources/checklist.md relative to this package directory.
Review the pending changes and report any unmet release requirements.
Run scripts/inspect-release.py only when authorized by the user's task.
```

YAML front matter must begin on the first line and end with `---`. `name` and
`description` are required nonempty strings, limited to 128 and 2,048 characters
respectively. Descriptions can use YAML multiline scalars. `enabled` is an
optional boolean, defaulting to `true`; `enabled: false` disables selection.
Metadata is limited to 16 KiB and the full selected file to 256 KiB. Instructions
after the front matter must be nonempty UTF-8 text. Metadata uses a safe YAML
parser, without Python object construction.

Discovery reads metadata only. Selection reads and pins the instructions;
referenced resources and scripts are loaded only when the agent needs them
through its existing tools. Their paths are resolved relative to the package
directory supplied in context. Skill selection grants no additional file/tool
access. In particular, user-package resources outside the workspace still need
the access permitted by the current tools and access mode.

## Lifetime And Context

The active snapshot includes its instructions, package directory, source path,
and source hash. It is included in every working request, including custom
system prompts and repair requests, independently of compressed history. The
instructions consume the normal input budget and are not silently truncated;
an oversized current input is reported through the existing context error.
Background summarizers do not activate skills.

Edits, renames, deletion, or disabling the original package do not silently
rewrite an active snapshot. Select it again to read the new version, or clear
it explicitly. Compatible code reloads preserve the active selection. Saved
sessions and forks restore its recorded snapshot without reopening or executing
the package. `/new` and session reset clear it; restoring a session without a
saved skill also clears any previous selection.

Selected instructions remain subordinate to higher-priority instructions,
explicit user directions, and existing access rules. Merely listing a package
does not activate it, and selecting it does not promote its referenced files
into instructions.
