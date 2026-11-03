# Skill Packages

Skills are reusable workflow instructions using the Agent Skills `SKILL.md`
format, with common Claude and Codex metadata supported. They do not modify
Python classes or automatically execute scripts.

## Discover And Select

`/skills` lists names, descriptions, paths, and disabled packages from:

1. `.slipagent/skills`, `.agents/skills`, `.claude/skills`, `.github/skills`,
   `.copilot/skills`, and `.codex/skills` in the workspace and its ancestors up
   to the nearest Git root. Outside a Git repository only the workspace is used.
2. SlipAgent's user-data `skills/` directory, then the same hidden directories
   under the user's home. `SLIPAGENT_STATE_DIR` still controls SlipAgent data.
3. System-wide `/etc/codex/skills/`, then bundled `prompts/skills/`.

Each package is an immediate child directory containing `SKILL.md`. Linked
packages and linked `SKILL.md` files resolve to their physical package directory;
identical physical files are listed once. Invalid metadata is reported without
hiding other packages. Name precedence is **project → user → system → bundled**.
Duplicate names within the winning scope require an explicit path. Disabled overrides
continue to shadow lower-priority namesakes.

Use `/skill release arguments`, `/release arguments`, or `$release arguments`
to activate a named workflow. `/skill "PATH/TO/SKILL.md" arguments` accepts an
explicit package path, including outside the workspace. Built-in slash commands
win name collisions; `/skill` and `$name` remain available. `/name` is a CLI
shortcut; `$name` works in CLI and browser chat. Arguments retain their original
text, with shell-style quoting used only for positional/named substitution.

Up to 16 skills can be active. Selecting an existing path refreshes it; selecting
another adds it. `/skill` shows active snapshots; `/skill off` clears all.
Commands apply at the normal agent boundary while work is active. Chat
invocations activate when that input is prepared for its next request.

The model receives a bounded metadata catalog and the `use_skill` tool. It can
list/search packages, load an applicable workflow, or unload active skills.
Instructions join the next request; tool results contain activation status
without duplicating the instruction body. Listing alone does not activate a
package. Missing, invalid, disabled, or incompatible packages leave active
snapshots unchanged.

## Bundled Workflows

These optional starter workflows provide concise procedures with completion
conditions and recovery guidance. They have been reviewed and checked for
loading and selection, without comparative model-performance testing or claims
of proven improvements for lower-end models. They activate through user selection or model loading.

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
---
# Release Review

Read resources/checklist.md relative to this package directory.
Review the pending changes and report any unmet release requirements.
Run scripts/inspect-release.py only when authorized by the user's task.
```

For portable Agent Skills packages, YAML front matter starts on the first
line and ends with `---`. Names use lowercase letters, digits and single hyphens,
up to 64 characters; descriptions contain up to 1,024 characters. Matching the
package directory name gives strict portability. Claude-style aliases are
accepted with a warning. Headerless Claude workflows and omitted names or
descriptions use the directory name and first nonempty body line as defaults.
Metadata is limited to 16 KiB and a selected file or expanded body to 256 KiB.
Instructions must be nonempty UTF-8 text. YAML uses a safe parser.

Metadata is retained in saved snapshots and model context, including `license`,
`compatibility`, `metadata`, `allowed-tools`, invocation controls, and
`agents/openai.yaml` interface, policy, and MCP dependencies. The legacy
SlipAgent `enabled: false` disables a package; omit this nonstandard key from
packages intended for strict external validators.

`disable-model-invocation: true` or OpenAI
`policy.allow_implicit_invocation: false` requires explicit user invocation.
`user-invocable: false` allows model loading only. `allowed-tools` describes host
preapproval and does not grant permissions or form a whitelist in SlipAgent.
`disallowed-tools` removes matching tools, with common Claude names mapped to
SlipAgent names and optional command/path patterns checked at invocation.
These restrictions supplement existing access rules; they are not an OS sandbox.
OpenAI MCP dependencies require already configured servers; missing servers
produce a configuration error rather than an automatic install. The model uses
`load_tools` to select needed definitions through normal deferred discovery.

Bodies support `$ARGUMENTS`, `$ARGUMENTS[N]`, `$N`, and named `$argument` values
from the optional `arguments` list or space-separated names. Indices start at
zero; missing positional values expand to empty text. If arguments are supplied
and no argument placeholder is used, the original text is appended as
`ARGUMENTS: ...`. Substitution is a single pass and preserves unrelated shell
variables. Escape a placeholder with a backslash to keep it literal.
`${CLAUDE_SKILL_DIR}`, `${CLAUDE_PROJECT_DIR}`, and `${CLAUDE_SESSION_ID}` expand
to the package directory, workspace root, and current session ID.

Discovery reads metadata only. Selected packages grant read-only access for
`read_file`, `list_dir`, `grep`, and `glob` to their supporting resources, including
packages outside the workspace. References resolve against the package directory.
Clearing a skill revokes its extra resource access. Writes and script execution
retain their existing restrictions; loading never executes a script.

Host-specific `context: fork`, agent selection, model overrides other than
`inherit`, hooks, dynamic `!`-backtick command preprocessing, and Claude plugin
path substitutions require runtimes SlipAgent does not implement. These produce
explicit compatibility errors instead of silently ignoring requested behavior.
Use a portable inline workflow for these packages. OpenAI interface fields are
preserved as metadata; they do not install another host's UI or plugins.

## Lifetime And Context

Each active snapshot includes expanded instructions, arguments, metadata,
package directory, source path, and a hash of its source plus OpenAI sidecar.
It is included in every working request, including custom system prompts and
repair requests, independently of compressed history. The
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
