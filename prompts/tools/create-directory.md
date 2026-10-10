# `create_directory`

Create an exact directory. Set `parents=true` to create missing parent directories too. Existing targets produce an error. Permissions use process defaults. Symlink paths are unsupported. No glob expansion occurs.

Results contain `changes` with created directory paths. Errors report directories already created. Directory creation is not covered by file-edit rewind.
