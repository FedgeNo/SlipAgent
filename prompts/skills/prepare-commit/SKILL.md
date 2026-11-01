---
name: prepare-commit
description: Review pending changes, verify a coherent commit, and commit only within the user's authorized scope.
---

# Prepare A Commit

## Inspect And Verify

1. Read the project's commit, versioning, generated-file, and verification requirements. Inspect repository status and both staged and unstaged changes.
2. Identify the logical change the user wants committed. Include existing work that belongs to it; preserve other work. Do not infer ownership from whether a file is already staged.
3. Check the diff for unintended changes and incomplete integration. Run the required checks appropriate to the change and the user's current testing instructions. Report failures and unrun checks accurately.

## Prepare The Commit

1. Stage the intended paths or hunks explicitly. Inspect the staged diff, including generated files and release metadata required by the project.
2. Write a concise message describing the resulting software change. Keep independently useful changes in separate runnable commits when requested.
3. Create the commit when authorized by the user's task. If the request is only to prepare or review, leave the result ready for that next step.
4. Verify the commit result and remaining working changes. Push only when the user has authorized that delivery step, with its required checks complete.

## Finish

Report what was committed or prepared, its verification status, and any remaining changes. Include the commit identifier when one was created. Do not amend, discard, or rewrite unrelated history as part of preparation.
