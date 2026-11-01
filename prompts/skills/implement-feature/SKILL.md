---
name: implement-feature
description: Turn a requested behavior into a complete change using existing project patterns and focused verification.
---

# Implement A Feature

## Define The Change

1. Translate the request into observable behavior: what triggers it, what the user sees, and what counts as success.
2. Inspect the owning component, its callers, a comparable feature, and the project's required workflow. Resolve routine details from those patterns; ask about missing requirements only when they materially affect the result.
3. Identify the affected interfaces, state lifetime, failure handling, and documentation. For multi-step work, keep this scope and the current step in the available planning tool.

## Implement And Verify

1. Make a coherent change through the existing owners. Carry the behavior through its callers and user-facing entry points rather than stopping at a helper function.
2. Preserve existing edits and unrelated behavior. Account for initialization, repeated use, and cleanup when the feature retains state.
3. Exercise the main user path and relevant boundary or failure cases. Prefer small fixtures and focused checks that demonstrate the promised behavior.
4. Compare the result with the original success conditions. Complete missing integration, help, or documentation before declaring the feature done.

## Finish

Explain how to use the feature, what was verified, and any material limitation. Report deferred checks accurately and follow the requested commit and delivery workflow.
