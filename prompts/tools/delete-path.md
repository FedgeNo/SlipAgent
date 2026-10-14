# `delete_path`

Delete an exact file or empty directory. Set `recursive=true` explicitly to delete a nonempty directory tree. Delete only within the user's authorized scope. Symlink paths, special files, and workspace/filesystem roots are unsupported. Paths are literal.

Results contain `changes` with deleted paths and their `kind`. Errors report deletions already completed; they are not rolled back. **Deletion is not covered by file-edit rewind.**
