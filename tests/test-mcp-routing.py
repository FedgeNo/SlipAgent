"""MCP traffic routing and idle memory bounds using in-memory pipes."""

import asyncio
import json
import tracemalloc
from types import SimpleNamespace

import pytest

from slipagent.mcp import MCPClient, MCPError, ServerSpec


def send(reader, message):
    reader.feed_data((json.dumps(message) + "\n").encode())


@pytest.fixture
async def connection():
    stdout = asyncio.StreamReader()
    outgoing = asyncio.Queue()

    async def drain():
        pass

    stdin = SimpleNamespace(write=lambda raw: outgoing.put_nowait(json.loads(raw)), drain=drain)
    client = MCPClient(ServerSpec("local", "unused"), timeout=1)
    client._process = SimpleNamespace(stdin=stdin, stdout=stdout, returncode=None)
    client._pump = asyncio.create_task(client._read_stdout())
    try:
        yield client, stdout, outgoing
    finally:
        client._pump.cancel()
        await asyncio.gather(client._pump, return_exceptions=True)


@pytest.mark.parametrize("traffic", ["notifications/message", "notifications/progress", "response", "invalid"])
async def test_idle_traffic_does_not_accumulate(connection, traffic):
    client, stdout, outgoing = connection
    message = {"jsonrpc": "2.0", "method": traffic, "params": {"data": "x" * 4096}}
    if traffic == "response":
        message = {"jsonrpc": "2.0", "id": 9999, "result": "x" * 4096}
    raw = (json.dumps(message) + "\n").encode()
    if traffic == "invalid":
        raw = b"invalid " + b"x" * 4096 + b"\n"
    await asyncio.sleep(0)
    tracemalloc.start()
    try:
        for _ in range(1000):
            stdout.feed_data(raw)
            await asyncio.sleep(0)
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert retained < 256_000, f"Idle MCP traffic retained {retained:,} bytes"
    assert outgoing.empty()
    assert not client._pump.done()


@pytest.mark.parametrize("server_id", [1, "server-request"])
async def test_idle_server_request_is_declined_immediately(connection, server_id):
    client, stdout, outgoing = connection
    send(stdout, {"jsonrpc": "2.0", "id": server_id, "method": "sampling/createMessage"})
    reply = await asyncio.wait_for(outgoing.get(), .5)
    assert reply["id"] == server_id
    assert reply["error"]["code"] == -32601
    assert "sampling/createMessage" in reply["error"]["message"]


async def test_reply_survives_notifications_server_requests_duplicates_and_eof(connection):
    client, stdout, outgoing = connection
    task = asyncio.create_task(client._request("tools/list", {}, None))
    try:
        request = await outgoing.get()
        for key in (None, True, 9999, [], {}, str(request["id"])):
            send(stdout, {"jsonrpc": "2.0", "id": key, "result": "unrelated"})
        send(stdout, {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}})
        send(stdout, {"jsonrpc": "2.0", "id": request["id"], "method": "sampling/createMessage"})
        send(stdout, {"jsonrpc": "2.0", "id": request["id"], "result": {"tools": []}})
        send(stdout, {"jsonrpc": "2.0", "id": request["id"], "result": "duplicate"})
        stdout.feed_eof()
        response = await asyncio.wait_for(task, .5)
        assert response["result"] == {"tools": []}
        assert (await outgoing.get())["error"]["code"] == -32601
        with pytest.raises(MCPError, match="exited before answering tools/list"):
            await client._request("tools/list", {}, None)
        assert outgoing.empty()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_late_reply_after_timeout_or_cancel_cannot_replace_next_response(connection, cancel):
    client, stdout, outgoing = connection
    task = asyncio.create_task(client._request("tools/call", {}, None, timeout=.02))
    try:
        first = await outgoing.get()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else MCPError):
            await task
        notice = await outgoing.get()
        assert notice["method"] == "notifications/cancelled"
        assert notice["params"]["requestId"] == first["id"]
        send(stdout, {"jsonrpc": "2.0", "id": first["id"], "result": "late"})
        task = asyncio.create_task(client._request("tools/call", {}, None))
        second = await outgoing.get()
        send(stdout, {"jsonrpc": "2.0", "id": first["id"], "result": "still late"})
        send(stdout, {"jsonrpc": "2.0", "id": second["id"], "error": {"code": -32602, "message": "current error"}})
        response = await task
        assert response["id"] == second["id"]
        assert response["error"]["message"] == "current error"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("idle", [False, True])
@pytest.mark.parametrize("failure", ["json", "limit", "read", "eof"])
async def test_reader_failures_reach_pending_or_next_request(connection, idle, failure):
    client, stdout, outgoing = connection
    task = None
    try:
        if not idle:
            task = asyncio.create_task(client._request("tools/list", {}, None))
            await outgoing.get()
        if failure == "json":
            stdout.feed_data(b"bad json\n")
            expected = "server wrote non-JSON"
        elif failure == "limit":
            stdout.feed_data(b"x" * 70_000 + b"\n")
            expected = "line longer than"
        elif failure == "read":
            stdout.set_exception(OSError("broken pipe reader"))
            expected = "broken pipe reader"
        else:
            stdout.feed_eof()
            expected = "exited before answering tools/list"
        await asyncio.sleep(0)
        if idle:
            task = asyncio.create_task(client._request("tools/list", {}, None))
        with pytest.raises(MCPError, match=expected):
            await asyncio.wait_for(task, .5)
        if idle:
            assert outgoing.empty()
        if failure == "json":
            task = asyncio.create_task(client._request("tools/list", {}, None))
            request = await outgoing.get()
            send(stdout, {"id": request["id"], "result": {"tools": []}})
            assert (await task)["result"] == {"tools": []}
        else:
            with pytest.raises(MCPError, match="exited before answering tools/list"):
                await client._request("tools/list", {}, None)
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("idle", [False, True])
async def test_server_request_reply_has_a_write_deadline(connection, monkeypatch, idle):
    client, stdout, outgoing = connection
    client.timeout = .05
    task = None

    async def blocked():
        await asyncio.Event().wait()

    try:
        if not idle:
            task = asyncio.create_task(client._request("tools/list", {}, None, timeout=1))
            await outgoing.get()
        monkeypatch.setattr(client._process.stdin, "drain", blocked)
        send(stdout, {"jsonrpc": "2.0", "id": "server-request", "method": "sampling/createMessage"})
        done, _ = await asyncio.wait({client._pump}, timeout=.5)
        assert done
        client._pump.result()
        assert (await outgoing.get())["id"] == "server-request"
        if idle:
            task = asyncio.create_task(client._request("tools/list", {}, None))
        with pytest.raises(MCPError, match="did not read.*reply"):
            await task
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_stopping_reader_releases_pending_request(connection):
    client, stdout, outgoing = connection
    task = asyncio.create_task(client._request("tools/list", {}, None))
    try:
        await outgoing.get()
        client._pump.cancel()
        await asyncio.gather(client._pump, return_exceptions=True)
        with pytest.raises(MCPError, match="exited before answering tools/list"):
            await asyncio.wait_for(task, .5)
        assert outgoing.empty()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
