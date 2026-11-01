---
name: investigate-failure
description: Diagnose a failing command, job, service, or workflow through bounded probes and verified hypotheses.
---

# Investigate A Failure

## Capture The Evidence

1. Identify the failing operation, its actual error or exit status, the expected result, and the relevant environment. Use current logs and state rather than assuming an earlier observation still holds.
2. Check whether the operation failed, timed out, was cancelled, or completed with an unexpected result. Determine what effects may already have occurred before repeating an operation that changes state.
3. Inspect the responsible component and its immediate dependencies. Distinguish an application defect from configuration, permissions, unavailable dependencies, or resource contention using evidence.

## Test A Hypothesis

1. State the leading explanation and the observation that would support or contradict it.
2. Use the smallest useful probe, preferably a read-only inspection or disposable reproduction. Change one relevant variable at a time and retain the result.
3. When evidence contradicts the explanation, revise it. Do not repeat the same failing probe without a new reason to expect a different result.
4. Apply a correction within the authorized task, or report the evidence and specific information needed to proceed. Verify recovery through the original operation when appropriate; a successful restart alone may not establish recovery.

## Finish

Report the established cause or remaining hypotheses, the actions taken, and the observed outcome. Keep confirmed recovery, partial recovery, and unresolved failure distinct.
