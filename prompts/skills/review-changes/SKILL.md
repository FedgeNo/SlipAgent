---
name: review-changes
description: Review a change and its callers for evidence-backed defects, with clear coverage and actionable findings.
---

# Review Changes

## Establish Scope

1. Identify the requested revision, diff, or working changes. Include uncommitted work when it belongs to that scope.
2. Read the changed code and enough surrounding code, callers, contracts, and tests to understand its intended behavior. A suspicious line alone is not evidence of a defect.
3. Keep the review read-only unless the user also authorizes fixes.

## Investigate Findings

1. Trace concrete inputs and state transitions through the change. Check relevant error paths, boundaries, and compatibility with existing callers.
2. For each suspected issue, identify its trigger, expected behavior, actual behavior, and consequence. Verify it with a focused test or direct code evidence when practical.
3. Separate demonstrated defects from design preferences or optional improvements. Do not invent a problem solely because another implementation is possible.
4. Track the reviewed scope and any parts that could not be assessed. Passing tests support a conclusion but do not establish that unexamined behavior is correct.

## Finish

Present actionable findings in severity order, each with a useful code location and supporting evidence. State when no defects were found in the reviewed scope, and identify material coverage limits. Do not silently fix or discard the work under review.
