# Plugins

[Back to the README](../README.md)

A plugin changes SlipAgent's behavior by overriding methods of its classes. Plugins
are Python files in a folder that updates never touch, so your changes survive every
`--update`. Only the methods a plugin defines are replaced; everything else stays
SlipAgent's and keeps receiving upstream fixes.

## Where Plugins Live

Plugins are the `*.py` files in the `plugins` folder of the data directory, for
example `~/.local/share/slipagent/plugins/` on Linux. The folder follows
`SLIPAGENT_STATE_DIR`. Plugins are user-level only: they apply to every project.

- Files load in file-name order (`10-base.py` before `20-extra.py`).
- A file whose name starts with `_` or `.` is ignored, so renaming a file to
  `_name.py` disables it.
- `/plugins` shows the folder's path.

## Workflow: Copy, Trim, Edit, Save

1. **Copy.** `/plugin copy cli.Renderer.handle` writes that method into a plugin
   file, here `cli-renderer.py`. `/plugin copy cli.Renderer` copies every method of
   the class, and `/plugin copy` with no target lets you choose a class and then a
   method from menus. A second name chooses the file: `/plugin copy
   cli.Renderer.emit quiet` adds `emit` to `quiet.py`, next to anything already
   copied there.
2. **Trim.** Delete the methods you don't want to change. Fewer copied methods means
   more of SlipAgent's own code, and its fixes, stays in effect.
3. **Edit** the methods that remain.
4. **Save and approve.** Saving a plugin marks its new contents for approval.
   `/plugins` approves that version and applies it through a live reload. With
   `--no-reload`, approval reapplies plugins directly.

`/plugin copy` approves the methods it writes. A wholly generated new file also
receives code approval; appending to an existing file requires approval of the
resulting file through `/plugins`. This prevents copying one method from granting
trust to unrelated Python already in the file.

## How a Plugin Looks

```python
from slipagent.plugins import overrides, original, replaces


@replaces("cli.Renderer")
class Renderer:
    @overrides("3f9c1d2a7b44")
    def handle(self, event):
        if event.kind == "status":
            return                      # handle this case entirely
        return original(self, event)    # everything else: SlipAgent's own handle

    def _quiet(self):                   # a new helper; needs no fingerprint
        return True
```

- `@replaces("<module>.<Class>")` names the class to change. The class name in your
  file does not matter, but `/plugin copy` keeps the original name.
- Every method and attribute your class defines replaces, or is added to, the target
  class. The rest of the target class is untouched.
- `original(self, ...)` calls the implementation your method replaced. Calling it
  makes your override a filter; not calling it makes it a full replacement. It works
  from closures and lambdas inside the method too.
- `@overrides("<fingerprint>")` records which version of the method you wrote against.
  See [Update Warnings](#update-warnings).
- Copied code runs unmodified. Names resolve from your file first, then the target
  module, then Python's builtins, so a method copied out of `cli.py` still sees that
  module's imports and helpers without new imports.
- Zero-argument `super()` works and reaches the target class's parent.

## Rules

- **Targets:** classes of reloadable modules. The stable frame (`runtime`, `config`,
  `types`, `plugins`, and the rest of `runtime.CORE_MODULES`) can't be overridden.
  Module-level functions can't be overridden; override the class methods that
  call them. Entry points such as `protocol.parse_agent_response` delegate to a
  class (`protocol.ResponseParser`, `api.ResponseReader`, `tools.shell.ProcessCapture`,
  `context.HistoryText`, and others), so each step is one method to override.
- **One target module per file.** A file may replace several classes of `cli`, but
  not classes of both `cli` and `agent`, because names resolve from a single target
  module. Use two files for that.
- **Layout:** the same limits as a live reload. Leave out base classes or repeat the
  target's exactly; don't add `__slots__` or dataclass fields. Keep new instance
  state in `self.__dict__.setdefault(...)`. An overridden `__init__` affects objects
  created afterwards.

## Approval

Approval has two parts. First, **Approve This Version** permits execution of the
entire Python file, including imports, decorators, class bodies, and top-level
statements. Until then, SlipAgent lists the file without importing it. Review the
displayed file path before approving. **Keep This Version Disabled** records a
rejection without executing the file; Esc leaves the decision pending.

Code approval records the SHA-256 digest of the exact file contents. Every edit
requires approval again, including changes outside methods. Older approvals that
only name methods keep those choices but require an initial code approval.
`--all-plugins` explicitly bypasses both approval stages for that run.

After code approval, SlipAgent loads the file to discover its overrides and asks
about methods without a decision:

- **Allow all** enables every new override in the plugin;
- **Choose** asks about each one, labelled `<file> → <Class>.<method>`;
- **None** rejects the listed overrides. If no overrides remain approved, it also
  disables execution of this file version.

Esc leaves the remaining overrides undecided; they stay off and SlipAgent asks again
next time. Method choices survive edits, while code approval must match the new
contents. If a plugin appears or changes while SlipAgent is running, a notice points
to `/plugins`. One-shot runs (`-p`) never ask: only previously approved file contents
can execute, and undecided overrides stay off with a notice. Disabling a version
prevents later imports; effects of code already executed cannot be undone.

## Conflicts

When more than one approved override targets the same method, SlipAgent asks which
should apply:

- **Only `<file>`** keeps that plugin's override;
- **Stacked: `a.py`, then `b.py`** runs `a.py`'s override first, whose `original` runs
  `b.py`'s, whose `original` runs SlipAgent's method.

A stacking order is offered only when every override but the last calls `original`;
otherwise the ones after it could never run. Until you choose, none of the competing
overrides apply. SlipAgent asks again only if the set of competing plugins changes.

## Saved Decisions and `/plugins`

Approvals and conflict choices are saved in `plugins/.approvals.json`. Changing it
applies immediately, like saving a plugin.

Invalid approval data disables saved decisions and reports the error while keeping
SlipAgent startable. New decisions made through `/plugins` replace the invalid data.

`/plugins` lists every plugin file and its code approval state. Approved files also
show their target module and each override's state: approved, awaiting approval,
rejected, conflict unresolved, or another plugin chosen, plus "changed upstream" or
"unstamped" where that applies. It then reopens the
menus so you can change any decision. Esc in the plugin list moves on to the conflict
menus.

## Update Warnings

Compatibility is tracked per method. A plugin declares no SlipAgent version; each
`@overrides` records a fingerprint of the method it replaces, so an update warns only
when a method you override actually changed.

A fingerprint is the first 12 hex digits of a SHA-256 hash of the method's syntax
tree. Comments, formatting, and docstrings don't change it; code changes do.

| Situation | Result |
| --- | --- |
| The fingerprint matches | Nothing is shown |
| The method changed upstream | A warning; the override still applies |
| The method no longer exists | The plugin is disabled, with the reason |
| The class or module no longer exists | The plugin is disabled, with the reason |
| An override of an existing method has no `@overrides` | A warning that includes the current fingerprint to paste |
| The method is your own addition | Nothing is shown |

After a warning:

1. `/plugin diff <file>` shows how each changed method differs from the source
   `/plugin copy` saved when you copied it (in `plugins/.baseline/`). Without a file,
   it checks every plugin.
2. Bring your override up to date if it needs it.
3. `/plugin restamp <file> [method]` records the current fingerprint and source.
   Restamping also adds `@overrides` to unstamped overrides. Approve the resulting
   file version through `/plugins`.

The absence of a warning is not a guarantee. A method can break without its own code
changing, for example when a method it calls takes different arguments or an
attribute it reads is renamed.

## When a Plugin Breaks

- A plugin that fails to load or apply (a syntax error, a bad target, a layout change,
  a missing method) is disabled as a whole, with a notice, because its methods may
  depend on each other. SlipAgent and the other plugins continue.
- A plugin that breaks a component contract during a reload is rejected with that
  reload, like a faulty source edit, and the previous code stays active.
- An override that never runs, because one that runs before it doesn't call
  `original`, is reported. This can happen with `--all-plugins`.
- Exceptions inside plugin methods are ordinary exceptions; tracebacks show your
  plugin file.

## Command-Line Switches

| Option | Effect |
| --- | --- |
| `--no-plugins` | Skip every plugin for this run. `SLIPAGENT_NO_PLUGINS=1` does the same. Use it if a plugin prevents startup. |
| `--all-plugins` | Execute every plugin file and apply every override without approval, stacking overrides of one method in file-name order. Saved decisions stay unchanged. |

## Commands

| Command | Use |
| --- | --- |
| `/plugins` | List plugins and override states, then approve overrides or resolve conflicts |
| `/plugin copy [module.Class[.method]] [file]` | Copy a class's methods, or one method, into a plugin file |
| `/plugin diff [file]` | Show how application methods changed since they were copied |
| `/plugin restamp <file> [method]` | Record current fingerprints after reviewing a change |
