# Harness Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[docs/harness-development.md](docs/harness-development.md) for editing, reload,
verification, and commit requirements. Follow applicable scoped instructions.
Explicit user instructions take precedence.

Use [docs/architecture.md](docs/architecture.md) to find component owners and
their contracts, and [docs/source-overview.md](docs/source-overview.md) for the
source map. Keep audit reports and local work notes in `agents/`.

Provider pricing conventions are documented under [API Provider Modules](docs/architecture.md#api-provider-modules).
Groq marks free models by omitting pricing; do not classify that as unknown.
Compaction uses only free models and must never fall back to a paid primary model.

## Prompt Ownership

[prompts/README.md](prompts/README.md) defines prompt structure, input boundaries,
and which consumers receive each instruction. Preserve required guidance for
every consumer when consolidating prompts, including standalone exports,
custom system prompts, repair requests, and background summaries.

Editable wording belongs in `prompts/`; executable schemas and runtime behavior
belong in their Python owners. Update the owning documentation when changing a
contract rather than duplicating its rules here.

## Retained Development Agents

The user retains sessions named **SlipAgent Codex** and **SlipAgent Claude** for
this project. They coordinate with `global/global-coordinator`, independently of
IDWT Principal. Read the private local [coordination guide](agents/coordination.md)
for mailbox identities, configuration and commands. Other sessions must not
impersonate these retained identities.

Both retained agents must run their central mail watcher throughout active
sessions, including while doing other work. Attach background execution to the
session's tool notification mechanism; rearm after each delivery or timeout.
A detached orphan process cannot wake an ended model turn. Read exact mail before
acknowledging it; acknowledgment does not complete a task or expand authorization.

Mail the coordinator significant outcomes, blockers and user decisions. Consolidate
related completions into one report rather than sending duplicates for every
instruction. Put concrete user decisions in the dashboard's database-backed
Needs Your Input list. Preserve each other's edits and current assignments.
