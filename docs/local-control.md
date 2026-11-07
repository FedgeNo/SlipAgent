# Local Control

[Back to the README](../README.md)

Every running CLI session publishes a local control endpoint after startup.
Linux and macOS use a Unix-domain socket; Windows uses a named pipe. No browser,
provider key, password, or copied token is needed by the controller. One-shot
tasks do not open endpoints. Use `--no-control` to disable the service.

## Access

Normally only the same OS user can connect and control the instance. Enabling
Danger Mode allows **any local user** to control that instance as well as lifting
workspace confinement. Turning Danger Mode off restricts new connections and
disconnects other-user clients, including subscribers. Previously accepted
operations may already have taken effect. Control endpoints do not listen on TCP;
Windows pipes explicitly reject remote clients.

An instance prints its endpoint at startup. Unix endpoints live under
`/tmp/slipagent-UID/INSTANCE.sock`, with mode `0600` normally and `0666` in Danger
Mode. Peer UIDs are also checked. Windows pipe DACLs grant access to the owner's
SID normally and permit other users to exchange data in Danger Mode. Other users
cannot create additional server instances through that permission. Native Windows
clients can use access mask `0x12019B` (data, attributes, read permissions and
synchronization), or a smaller read/write-data mask. Generic write access also
requests pipe-instance creation and is therefore not granted to other users.
The bundled client handles this access mask automatically.

`SLIPAGENT_CONTROL_DIR` overrides the Unix discovery root. Use an absolute path short enough
for the OS socket-path limit and traversable by other users if Danger Mode should
allow them access. This variable does not change the Windows pipe namespace.

## Command-Line Client

```sh
slipagent control list
slipagent control INSTANCE status
slipagent control INSTANCE submit '{"text":"Run the focused tests"}'
slipagent control INSTANCE interrupt
slipagent control INSTANCE watch
```

`list` returns JSON records containing instance IDs, endpoints, session IDs,
titles, workspaces, models/providers, activity, and queued-input counts. Private
instances belonging to other users and stale endpoints are omitted. Each running
instance has a separate endpoint and queue. `INSTANCE` accepts a unique instance
ID prefix from this list or the full endpoint. One controller can address as many
instances as needed; closing the controller does not stop them.

On SSH, run the client on the host: `ssh HOST slipagent control list`. No GUI
tunnel or token exchange is involved.

The third positional argument is a JSON object. `--session-id ID` refuses an
action if the instance changed conversations. `--request-id UUID` lets a client
retry the same logical request without duplicating its effects while its response
is still cached. A controller should generate a fresh UUID for every new action.

## Protocol Version 1

The transport carries UTF-8 newline-delimited JSON-RPC 2.0 objects. Each request
requires a nonempty string `id`; use UUIDs to avoid collisions between controllers.
Notifications as incoming commands and JSON-RPC batches are not supported.

```json
{"jsonrpc":"2.0","id":"a-new-uuid","method":"submit","params":{"text":"Inspect the failing test"},"sessionId":"optional-current-session-id"}
```

The response preserves `id` and contains either `result` or `error`.
`submit` returns `{"accepted":true}` when queued; this does **not** mean the model
has finished or that requested actions succeeded. Execution uses the same driver,
interruption rules, tool permissions, and input queue as CLI and GUI submissions.
Submitting during a run supplies input through that queue without creating a
second agent or duplicate tool execution.

| Method | Parameters | Result |
| --- | --- | --- |
| `status` | None | Instance identity and current application state; `protocolVersion` is 1 |
| `submit` | `text` | Queue acceptance |
| `stop` | None | Request a graceful stop |
| `interrupt` | None | Request immediate cancellation and owned cleanup |
| `queue` | None | Up to 100 queued identifiers with 160-character text previews |
| `discard` | `identifier` | Whether that queued item was removed |
| `rename` | `text` | New session title |
| `setting` | `setting`, boolean `enabled` | Apply `danger`, `planning`, `overthinking`, or `hashline` |
| `model` | `identifier` | Select a catalog model by provider-qualified selector |
| `sessions` | Optional `offset`, `limit` | Saved-session metadata pages; unavailable without session persistence |
| `resume` | `identifier` | Resume a saved conversation while idle |
| `fork` | None | Fork the conversation and return its new session ID |
| `reset` | None | Start a fresh conversation while idle |
| `new_task` | None | Establish a task boundary while idle |
| `history` | `stepId`, optional `offset`, `limit` | A character page of the original step's JSON |
| `plan` | Optional `offset`, `limit` | A character page of the latest saved plan's JSON |
| `context` | Optional `offset`, `limit` | A character page of the latest outgoing working request body as JSON, with `stepId`, `attempt`, `sentAt`, `model`, and calibrated `estimatedTokens` |
| `subscribe` | None | Current status, followed by live notifications |
| `unsubscribe` | None | Stop live notifications on this connection |

History and plan offsets start at zero. Their default and maximum page size is
16,384 characters. Responses provide `text` and `nextOffset`; concatenate pages
before decoding the original JSON. History step IDs run from 1 through `status.steps`.
Original history remains available regardless of model-context compression.
`context` pages the same request the context view shows (`Ctrl+\`): the full
provider body of the most recent working attempt, including retries, with the
selected history representations actually sent. It is replaced by each new
attempt, is unavailable before the first request of a conversation, and is as
private as the request logs. The host serves the request it retained when the
body was sent, or otherwise the newest attempt in the request diagnostics
(the same files `/requests` reads), which also covers a resumed conversation
whose last request predates this process. `attempt` matches `/requests`
numbering. `estimatedTokens` is the calibrated estimate, not
provider-reported usage.
Supply `sessionId` while paging to reject reads that cross a conversation change.
Saved-session pages use item offsets with a default and maximum limit of 100,
returning `items` and `nextOffset`.
Session/model/setting changes use the shared application's idle and editor guards.
This protocol does not expose the browser editor or a direct tool-execution bypass.

Subscriptions receive JSON-RPC notifications with method `event` or `state`.
Their `params` include `instanceId` and a monotonic `sequence`. Event text is
limited to 2,048 characters and reports `truncated`; use original history for full
completed outputs. Outgoing model-request context bodies and raw tool payloads
are excluded from live events; read the latest body with `context`. Reconnect with `status` and retrieve history to
resynchronize; notifications are not replayed.

## Bounds And Lifetime

Messages are limited to 128 KiB, with at most 16 connected clients and 64 queued
outgoing messages per client. Slow subscribers are disconnected instead of
blocking the agent. Socket/pipe I/O and provider work remain asynchronous.
History paging reads archives through bounded worker operations.
Request IDs accept at most 128 characters, method names 64, and `text` fields
65,536. Invalid fields are rejected before invoking the application action.

Each instance retains the latest 128 request IDs and responses in memory.
Concurrent retries share the same in-flight action. Reusing an ID for different
parameters is an error. Disconnecting a client does not cancel an accepted action.
Deduplication is **not durable**: after cache eviction or instance restart, inspect
the current state before deciding whether to resubmit. The instance ID changes
on restart and remains stable when its saved conversation is resumed or forked.

Application shutdown closes the endpoint and its clients before closing engine
resources. Unix socket files are removed on orderly shutdown. A crashed process
may leave a stale socket file; discovery skips it without deleting other instances'
files. No saved sessions or model histories are removed by endpoint cleanup.

## Python Controllers

Code running in an environment with SlipAgent installed can use the bundled
client without handling platform-specific transports:

```python
import asyncio
from slipagent.controlclient import instances, request

async def main():
    for instance in await instances():
        print(instance['title'], instance['workspace'], instance['instanceId'])
        # Choose targets before submitting work.
        response = await request(instance['endpoint'], 'status')
        print(response['result']['activity'])

asyncio.run(main())
```

Native Windows security uses Microsoft's [named-pipe access rules](https://learn.microsoft.com/en-us/windows/win32/ipc/named-pipe-security-and-access-rights)
and [CreateNamedPipeW](https://learn.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-createnamedpipew).
