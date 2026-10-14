# `move_path`

Move or rename a regular file or complete directory tree to an exact new destination on the same filesystem. Destination parents must exist. Existing destinations are never merged or replaced. Cross-filesystem moves fail; use copy_path and verify the copy before an authorized deletion. Symlink paths and special files are unsupported. Paths are literal.

Results report both old (`moved_from`) and new (`moved_to`) paths under `changes`, with each path's `kind`. These operations are not covered by file-edit rewind.
