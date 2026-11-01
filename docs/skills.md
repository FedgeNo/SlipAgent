# Skill Packages

Skills are reusable workflow instructions selected explicitly by the user.
They do not modify Python classes or automatically run scripts.

## Discover And Select

`/skills` lists names, descriptions, source paths, and disabled packages from:

1. The current project's `.slipagent/skills/` directory.
2. The `skills/` directory under SlipAgent's user data directory, including any
   `SLIPAGENT_STATE_DIR` override. See [installation locations](installation.md#installation-and-data-locations).

Each package is an immediate child directory containing `SKILL.md`. Discovery
does not recursively search resources or follow linked package directories.
Invalid or unreadable metadata is reported without hiding other packages.

Use `/skill release` to select a package by its metadata name. Project packages
take precedence over user packages with the same name, including when the
project package is disabled. Duplicate names within the selected scope require
an explicit path. `/skills` displays all paths so duplicates remain identifiable.

`/skill .slipagent/skills/release` or `/skill PATH/TO/SKILL.md` selects by path.
Paths are relative to the workspace unless absolute; quote a path when needed.
Catalogued user packages are permitted instruction sources. Other explicit
paths follow ordinary workspace access rules. Missing, invalid, or disabled
packages leave the current selection unchanged.

One skill is active at a time; selecting another replaces it. `/skill` shows
the selected name, source path, and source SHA-256. `/skill off` clears it.
Selection and clearing queue at the normal agent boundary while work is active;
listing and status remain available during a turn. There is no model-selected
activation or automatic catalog injection into model requests.

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
