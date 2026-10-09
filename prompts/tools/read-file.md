# `read_file`

## Contents and Display

Read a text file allowed by the current Workspace Access mode.

Returns numbered lines so they can be cited; omit the displayed line numbers when calling `edit_file`.

Line endings are displayed as LF.

## Selecting Ranges

Read the code you intend to change before editing it. Include the needed sections of all known files in the same read batch.

Omit `limit` for ordinary files; for larger files, use `offset`/`limit` and include all known needed ranges in this batch.

## Limitations

Binary files are rejected.
