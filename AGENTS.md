# Harness Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[docs/harness-development.md](docs/harness-development.md) for editing, reload,
verification, and commit requirements. Follow applicable scoped instructions.
Explicit user instructions take precedence.

Use [docs/architecture.md](docs/architecture.md) to find component owners and
their contracts, and [docs/source-overview.md](docs/source-overview.md) for the
source map. Keep audit reports and local work notes in `agents/`.

## Prompt Ownership

[prompts/README.md](prompts/README.md) defines prompt structure, input boundaries,
and which consumers receive each instruction. Preserve required guidance for
every consumer when consolidating prompts, including standalone exports,
custom system prompts, repair requests, and background summaries.

Editable wording belongs in `prompts/`; executable schemas and runtime behavior
belong in their Python owners. Update the owning documentation when changing a
contract rather than duplicating its rules here.
