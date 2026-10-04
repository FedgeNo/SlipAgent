"""A throwaway MCP server for tests, speaking both protocol eras."""

from __future__ import annotations

import json
import sys
import threading


def _reply(message_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\n")
    sys.stdout.flush()


def _error(message_id, code, message, data=None):
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "error": error}) + "\n")
    sys.stdout.flush()


TOOLS = [
    {
        "name": "echo",
        "description": "Echo the message back.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "boom",
        "description": "Always fails.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

MODERN_VERSION = "2026-07-28"


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "legacy"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        message_id = message.get("id")
        metadata = message.get("params", {}).get("_meta", {})
        if mode.startswith("modern") and message_id is not None:
            if "_meta" in message or "io.modelcontextprotocol/protocolVersion" not in metadata:
                _error(message_id, -32602, "Required metadata belongs in params._meta")
                continue

        if method == "server/discover":
            if mode == "legacy":
                # A legacy server answers an unknown pre-init request with a
                # non-modern error, which is the signal to fall back.
                _error(message_id, -32601, "Method not found")
            elif mode == "modern-old" and metadata.get("io.modelcontextprotocol/protocolVersion") != "2025-11-25":
                _error(
                    message_id,
                    -32022,
                    "Unsupported protocol version",
                    {"supported": ["2025-11-25"], "requested": MODERN_VERSION},
                )
            else:
                _reply(
                    message_id,
                    {
                        "resultType": "complete",
                        "capabilities": {"tools": {"listChanged": False}},
                        "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "stub", "version": "1.0"}},
                        "supportedVersions": [MODERN_VERSION, "2025-11-25"],
                        "instructions": "Use echo before boom.",
                    },
                )
            continue

        if method == "initialize":
            if mode == "modern":
                # A server that only speaks modern rejects the legacy handshake.
                _error(message_id, -32601, "initialize is not supported")
                continue
            _reply(
                message_id,
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "stub", "version": "1.0"},
                    "instructions": "Use echo before boom.",
                },
            )
            continue

        if method == "notifications/initialized":
            continue

        if method == "tools/list":
            _reply(message_id, {"tools": TOOLS})
            continue

        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "echo":
                _reply(
                    message_id,
                    {
                        "content": [{"type": "text", "text": f"echo: {arguments.get('message')}"}]
                    },
                )
            elif name == "boom":
                _reply(message_id, {"content": [{"type": "text", "text": "it exploded"}], "isError": True})
            else:
                _error(message_id, -32602, f"Unknown tool: {name}")
            continue


if __name__ == "__main__":
    main()
