# `copy_path`

Copy a regular file (including binary files) or a complete directory tree to an exact new destination. Destination parents must exist. Existing destinations are never merged or replaced. Files retain permission bits; directory permissions use the process defaults. Symlink paths and special files are unsupported; a tree containing a symbolic link is refused, so copy the link target explicitly or remove the link first. Paths are literal.

Results contain `changes` with `path`, `action`, and `kind`. Errors include changes already made; a failed copy may leave a partial destination. Inspect these results before retrying. These operations are not covered by file-edit rewind.
