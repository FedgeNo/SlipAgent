Create and select a project Python virtual environment when Python work needs one and none is selected. Check project instructions and package-manager configuration first; follow existing Poetry, Conda, or other managed setups instead when applicable.

Defaults to `.venv` in the workspace, using SlipAgent's Python. Supply `path` for a nested project's environment and `python` for a different installed Python executable. Check the returned version against the project's requirements.

**Idempotent:** an already selected interpreter is returned unchanged, regardless of arguments. A complete environment at the target is validated and reused. Existing incomplete or unrelated directories are never overwritten or repaired. Ambiguous or invalid configured environments require explicit selection or correction.

Creates the environment with Python's standard `venv` module and bundled pip where available; installs no project dependencies and makes no network requests. Returns whether it was created and the validated interpreter, version, installer, and test-command availability. Selection lasts for this session; default root `.venv` discovery also works after restart. Custom paths require project configuration for selection after restart.

Creation has a 120-second timeout. Failure or interruption leaves partial files for inspection; repeating the call does not recreate them.
