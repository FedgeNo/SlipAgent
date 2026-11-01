---
name: fix-bug
description: Reproduce a reported bug, trace its cause, make a focused correction, and verify the result.
---

# Fix A Bug

## Establish The Failure

1. Identify the expected behavior, observed behavior, and trigger from the user's report and available evidence.
2. Inspect the relevant code, callers, and existing tests. Look for comparable working behavior before choosing a correction.
3. Obtain a small reproduction when practical. Record the exact failure so a later successful run can be compared with it. If reproduction is unavailable, state what evidence supports the suspected cause.

## Correct And Verify

1. Trace the failure to the code that owns the behavior. Choose a correction that explains the evidence; avoid changing unrelated code to test several guesses at once.
2. Add a focused regression check when useful, ideally observing it fail before the correction. Assert the expected behavior rather than the internal implementation.
3. Apply the smallest coherent fix, preserving existing work. Update affected contracts or documentation when the behavior changes.
4. Run the focused check and relevant neighboring checks required by the project. If they fail, inspect the failure before changing either the implementation or the test.

## Finish

Report the cause, correction, actual verification results, and any remaining uncertainty. Distinguish a confirmed fix from an unverified hypothesis. Follow the user's and project's instructions for broader checks and commits.
