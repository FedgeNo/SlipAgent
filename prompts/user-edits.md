# Saved User File Changes

`user_edits` contains saved changes the user made in the editor, separate from their messages and your tool actions. Each notification identifies its file, editing session, revision, and optional saved diff artifact. Its `diff` is a bounded beginning of the diff; `diff_truncated`, `diff_characters`, and `next_offset` describe omitted text. Retrieve more with `read_artifact(record_id, field="output", offset=next_offset, limit=...)`, whose offsets count decoded characters. Retrieve omitted changes when needed. A reversion to the editor's baseline is a real user change even when its cumulative diff is empty.

`user_edits_omitted` counts other pending changes that did not fit this request. They remain pending. A missing archive is reported explicitly; a truncated unarchived change remains pending with its full data retained. Notifications leave the pending input only after an accepted response to the request that included that exact revision. Older delivered notifications and artifact references belong to ordinary retrievable, compressible history. These observations describe user changes.

Step labels identify the original save. Use these observations to update the current action or plan. Retrieve a history step's `user_edits`
section with `recall_history` for its original notification. A new save or
reversion has a new revision.
