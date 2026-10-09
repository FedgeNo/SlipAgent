# `read_file`

## Contents and Display

Read a text file allowed by the current Workspace Access mode.

Returns numbered lines so they can be cited; omit the displayed line numbers when calling `edit_file`.

Line endings are displayed as LF.

## Selecting Ranges

Read the code you intend to change before editing it. Include the needed sections of all known files in the same read batch.

Omit `offset` and `limit` to read the whole file. With `offset` alone, read through the end. Use `offset`/`limit` when only a particular range is needed; there is no fixed line cap. Include all known needed ranges in this batch.

## Limitations

Files must be UTF-8 text and no larger than 32 MB (32,000,000 bytes). The model's context budget still applies to the combined results; oversized history may be excerpted, with originals retained for retrieval.
