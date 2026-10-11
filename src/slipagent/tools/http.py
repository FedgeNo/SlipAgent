"""Structured HTTP requests using the web tools' public-address transport."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from ..prompts import load_prompt
from .base import ToolResult
from .web import PublicHttpTool, MAX_FETCH_BYTES, _public_address


class HttpFetchTool(PublicHttpTool):
    """Reuse pinned HTTP transport and cleanup without page-text extraction."""

    name = "http_fetch"
    description_prompt = "tools/http-fetch.md"
    description = load_prompt(description_prompt)
    parameter_prompts = "tools/http-fetch-parameters.json"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string"},
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "params": {"type": "object", "additionalProperties": {"type": "string"}},
            "body": {"type": "string"},
            "json_body": {"type": ["object", "array"]},
            "timeout": {"type": "number", "minimum": 1, "maximum": 600},
        },
        "required": ["url"],
    }

    async def run(
        self, url: str, method: str = "GET", headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None, body: str | None = None,
        json_body: Any = None, timeout: float = 120,
    ) -> ToolResult:
        if body is not None and json_body is not None:
            return ToolResult.error("Provide body or json_body, not both.")
        try:
            target = httpx.URL(url)
            address = await _public_address(target)
            async with asyncio.timeout(timeout):
                async with self._client.stream(
                    method, target, headers=headers, params=params,
                    content=body.encode("utf-8") if body is not None else None,
                    json=json_body, timeout=timeout, follow_redirects=False,
                    extensions={"validated_address": address},
                ) as response:
                    raw = bytearray()
                    truncated = False
                    async for chunk in response.aiter_bytes():
                        remaining = MAX_FETCH_BYTES - len(raw)
                        raw.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            truncated = True
                            break
                    result: dict[str, Any] = {
                        "url": str(response.url), "method": response.request.method,
                        "status_code": response.status_code,
                        "headers": [list(pair) for pair in response.headers.multi_items()],
                        "truncated": truncated,
                    }
                    try:
                        result["body"] = raw.decode(response.encoding or "utf-8")
                    except (UnicodeError, LookupError):
                        result["body_base64"] = base64.b64encode(raw).decode("ascii")
                    return ToolResult(result, is_error=response.is_error)
        except (httpx.HTTPError, ValueError, OSError, asyncio.TimeoutError) as exc:
            return ToolResult.error({"error": f"HTTP request failed: {exc}"})
