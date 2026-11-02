# Shared Application Interface

SlipAgent has one engine per session. The terminal and browser interfaces
attach to that engine through `application.py`; see the [browser guide](gui.md).
The [architecture contract](architecture.md#shared-application-controls) defines
ownership and frontend boundaries.

## Attach To An Existing Session

```python
from slipagent.application import application

controller = application(session)
unsubscribe = controller.subscribe(handle_agent_event)
unsubscribe_state = controller.states.subscribe(handle_state)

try:
    current = controller.state()
    task = controller.start("Inspect the requested changes")
    # Closing a UI waiter leaves the owned task running.
    answer = await controller.wait()
finally:
    unsubscribe()
    unsubscribe_state()
```

`start` returns an `asyncio.Task`; retain it when its result is needed after it
finishes. `wait` waits for the currently owned task. `run` is the convenience
operation that starts and awaits a task directly; cancellation of `run` cancels
that run. Explicitly call `interrupt` to cancel active work and
`wait_for_interrupt` to await cleanup. `close` shuts down the application, rather
than detaching one interface.

Input controls are `submit(text)`, `queued_input()`, and `discard_input(id)`.
Queue IDs come from the existing input owner and remain shared across interfaces.
`submit` signals `input_ready`, waking an idle frontend driver. The CLI waits on
that event alongside terminal input and uses its existing `run_request` loop for
both sources. A browser does not create a competing driver or agent task.
`stop()` requests completion of the current step and tools. `next_run` decides
whether queued work continues with the existing request limit or a fresh run.

Session actions take structured arguments: `reset()`, `new_task()`, `rename(title)`,
`saved_sessions()`, `resume(id)`, `fork()`, `delete_current()`, `delete_saved(id)`,
`rename_saved(id, title)`, and `rewind(checkpoint_id)`. Planning, overthinking,
path confinement, hashline editing, and skill selection have corresponding
typed methods. Frontends present confirmations before destructive operations.
`select_model(selector)` and `enable_optimal()` reuse provider verification and
preference persistence. Session changes also reject an active user-edit hold.
An operation requiring idle state raises `RuntimeError` if agent work, another
mutation, or an interactive command owned by another task is active. The CLI
already queues those actions using its existing ordered command mechanism.

## Construct Without The CLI

```python
from slipagent.config import Config
from slipagent.sessionfactory import SessionOptions, create_session
from slipagent.application import application

config = Config.from_env(workspace="/path/to/project")
session = await create_session(config, SessionOptions(), on_event=handle_agent_event)
controller = application(session)
try:
    answer = await controller.run("Inspect this project")
finally:
    await controller.close()
```

The factory constructs providers, workspace tools, environment owners, optional
MCP ownership, and the agent. It does not run the CLI's startup sequence: metadata
verification, quota refresh, MCP connection, journal setup/resume, and reload
startup remain explicit launch operations. A browser opened from the CLI uses
the already initialized session instead of repeating that startup.

Subscriptions deliver existing event objects synchronously without creating an
output archive or replay buffer. Large tool results stay under their existing
storage owners. Transport adapters must queue notifications without blocking the
engine, bound their buffers, and request archived content on demand. Browser
protocols must preserve these boundaries. Browser notifications provide
authenticated connections and fresh snapshots on reconnect, without retaining a
second event history. `guibrowser` retrieves full original records through bounded
pages, independently of the context view prepared for the model. `guieditor`
temporarily borrows the shared edit coordinator's port, restoring the terminal
adapter when the browser editor closes.
