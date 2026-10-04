"""Web tools: `web_search` (Exa) and `fetch_page` (HTML to text).

`web_search` needs an `EXA_API_KEY`. Without one the tool still registers but
returns a clear setup error, so the model can tell the user what is missing
instead of silently failing. `fetch_page` needs nothing.
"""

from __future__ import annotations

import os
import re
import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from itertools import groupby
from urllib.parse import urljoin, urlsplit

import httpx

from .base import Tool, ToolResult

EXA_SEARCH_URL = "https://api.exa.ai/search"
DEFAULT_TIMEOUT = 30.0
MAX_FETCH_BYTES = 5_000_000
MAX_PAGE_CHARS = 40_000
MAX_SEARCH_RESULTS = 10

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "template"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3",
    "h4", "h5", "h6", "blockquote", "pre", "hr", "header", "footer", "main",
}


class _TextExtractor(HTMLParser):
    """Pull readable text out of HTML, dropping scripts and markup noise."""

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[tuple[str, bool]] = []
        self._skip_depth = 0
        self._title: str | None = None
        self._in_title = False
        self._preserve_depth = 0
        self._base_url = base_url
        self._links: list[tuple[str, int]] = []

    def _append(self, text: str) -> None:
        self._chunks.append((text, self._preserve_depth > 0))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif not self._skip_depth:
            if tag in _BLOCK_TAGS:
                self._append("\n")
            if tag in {"pre", "code"}:
                self._preserve_depth += 1
            if tag == "a":
                href = dict(attrs).get("href") or ""
                url = urljoin(self._base_url, href) if href else ""
                if urlsplit(url).scheme not in {"", "http", "https"}:
                    url = ""
                self._links.append((url, len(self._chunks)))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif not self._skip_depth:
            if tag in {"pre", "code"}:
                self._preserve_depth = max(0, self._preserve_depth - 1)
            if tag == "a" and self._links:
                url, start = self._links.pop()
                label = "".join(text for text, _ in self._chunks[start:]).strip()
                if url and url != label:
                    self._append(f" ({url})")
            if tag in _BLOCK_TAGS:
                self._append("\n")

    def handle_data(self, data: str) -> None:
        # <title> lives inside <head>, which is otherwise skipped, so it has
        # to be handled before the skip check.
        if self._in_title:
            self._title = (self._title or "") + data.strip()
            return
        if self._skip_depth:
            return
        self._append(data)

    def text(self) -> str:
        parts: list[tuple[str, bool]] = []
        for protected, group in groupby(self._chunks, key=lambda chunk: chunk[1]):
            text = "".join(chunk[0] for chunk in group)
            if not protected:
                text = re.sub(r"[ \t\r\f\v]+", " ", text)
                text = re.sub(r" *\n *", "\n", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                if not parts:
                    text = text.lstrip()
            parts.append((text, protected))
        if parts and not parts[-1][1]:
            parts[-1] = (parts[-1][0].rstrip(), False)
        return "".join(text for text, _ in parts).strip("\n")


def html_to_text(markup: str, *, base_url: str = "") -> tuple[str, str | None]:
    """Return `(visible_text, title)` for a chunk of HTML."""
    parser = _TextExtractor(base_url)
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML must not kill the tool
        pass
    return parser.text(), parser._title


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web and return ranked results with titles, URLs, and text "
        "snippets. Use it for library documentation, error messages, API "
        "references, and anything about code you have not seen before. Prefer "
        "reading the most relevant result with fetch_page before concluding. "
        "Requires an EXA_API_KEY."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "num_results": {
                "type": "integer",
                "description": f"Number of results (1-{MAX_SEARCH_RESULTS}). "
                               f"Defaults to 5.",
                "minimum": 1,
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def run(self, query: str, num_results: int = 5) -> ToolResult:
        if not self.api_key:
            return ToolResult.error(
                "web_search is unavailable: no EXA_API_KEY configured. "
                "Set EXA_API_KEY in the environment or .env to enable it. "
                "fetch_page still works without it."
            )
        if not query.strip():
            return ToolResult.error("query must not be empty.")

        limit = max(1, min(int(num_results), MAX_SEARCH_RESULTS))
        try:
            response = await self._client.post(
                EXA_SEARCH_URL,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "query": query,
                    "numResults": limit,
                    "type": "auto",
                    "contents": {"text": {"maxCharacters": 1200}},
                },
            )
        except httpx.HTTPError as exc:
            return ToolResult.error(f"Web search request failed: {exc}")

        if response.status_code >= 400:
            detail = _error_detail(response)
            hint = ""
            if response.status_code in (401, 403):
                hint = " Check that EXA_API_KEY is valid."
            elif response.status_code == 429:
                hint = " Exa rate limit or credits exhausted."
            return ToolResult.error(
                f"Web search failed ({response.status_code}): {detail}{hint}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return ToolResult.error(f"Web search returned invalid JSON: {exc}")

        results = payload.get("results") or []
        if not results:
            return ToolResult.ok(f"No results for {query!r}.")

        blocks = [f"{len(results)} result(s) for {query!r}:"]
        for index, item in enumerate(results, start=1):
            title = item.get("title") or "(untitled)"
            url = item.get("url", "")
            snippet = (item.get("text") or "").strip()
            blocks.append(f"[{index}] {title}\n    {url}")
            if snippet:
                blocks.append(f"    {snippet[:600]}")

        blocks.append(
            "Use fetch_page on the most relevant URL to read it in full."
        )
        return ToolResult.ok("\n".join(blocks))


async def _public_address(url: httpx.URL) -> str:
    if url.scheme not in {"http", "https"} or not url.host or url.username or url.password:
        raise ValueError("url must be an absolute HTTP(S) URL without credentials")
    try:
        literal = ipaddress.ip_address(url.host)
    except ValueError:
        records = await asyncio.wait_for(asyncio.to_thread(
            socket.getaddrinfo, url.host, url.port or (443 if url.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        ), timeout=DEFAULT_TIMEOUT)
        addresses = [ipaddress.ip_address(record[4][0]) for record in records]
    else:
        addresses = [literal]
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("url resolves to a non-public address")
    return str(addresses[0])


class _PublicTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        address = request.extensions.get("validated_address")
        if not isinstance(address, str):
            raise ValueError("request has no validated public address")
        # Pin DNS while retaining the original Host header and TLS identity.
        pinned = httpx.Request(
            request.method, request.url.copy_with(host=address), headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": request.url.host},
        )
        return await super().handle_async_request(pinned)


class FetchPageTool(Tool):
    name = "fetch_page"
    description = (
        "Fetch a public URL over HTTP(S) and return its readable text, with scripts and "
        "markup removed. Works on documentation, articles, and raw files; it does not run "
        "JavaScript, so client-rendered pages may come back thin."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http:// or https:// URL."},
            "max_chars": {
                "type": "integer",
                "description": f"Truncate output to this many characters "
                               f"(default {MAX_PAGE_CHARS}).",
                "minimum": 500,
            },
        },
        "required": ["url"],
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT, trust_env=False,
            transport=_PublicTransport(limits=httpx.Limits(max_keepalive_connections=0)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def run(self, url: str, max_chars: int = MAX_PAGE_CHARS) -> ToolResult:
        candidate = url.strip()
        if not candidate.startswith(("http://", "https://")):
            return ToolResult.error(
                f"url must start with http:// or https://, got {url!r}"
            )

        raw_body = bytearray()
        bytes_cut = False
        try:
            current = httpx.URL(candidate)
            for hop in range(11):
                address = await _public_address(current)
                async with self._client.stream(
                    "GET", current, follow_redirects=False,
                    headers={"User-Agent": "SlipAgent/0.1", "Accept-Encoding": "identity"},
                    extensions={"validated_address": address},
                ) as response:
                    if response.has_redirect_location:
                        if hop == 10:
                            return ToolResult.error("Fetch exceeded the 10-redirect limit")
                        current = current.join(response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        return ToolResult.error(f"Fetch failed ({response.status_code}) for {candidate}")
                    content_type = response.headers.get("Content-Type", "").lower()
                    async for chunk in response.aiter_bytes():
                        remaining = MAX_FETCH_BYTES - len(raw_body)
                        raw_body.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            bytes_cut = True
                            break
                    break
        except (httpx.HTTPError, ValueError, OSError, asyncio.TimeoutError) as exc:
            return ToolResult.error(f"Could not fetch {candidate}: {exc}")
        raw = bytes(raw_body)

        if "json" in content_type:
            body = raw.decode("utf-8", errors="replace")
            heading = f"{candidate} (application/json)"
        elif "html" in content_type or raw[:200].lstrip()[:1] == b"<":
            markup = raw.decode("utf-8", errors="replace")
            body, title = html_to_text(markup, base_url=str(current))
            heading = f"{candidate}"
            if title:
                heading = f"{title} — {candidate}"
            if not body:
                return ToolResult.error(
                    f"{candidate} returned no readable text (it may be "
                    f"JavaScript-rendered)."
                )
        else:
            body = raw.decode("utf-8", errors="replace")
            heading = candidate

        limit = max(500, min(int(max_chars), MAX_PAGE_CHARS))
        truncated = bytes_cut or len(body) > limit
        if truncated:
            body = body[:limit] + "\n… [truncated]"

        return ToolResult.ok(f"{heading}\n---\n{body}")


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300] or "no response body"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return str(payload)[:300]


def web_tools(api_key: str | None = None) -> list[Tool]:
    return [WebSearchTool(api_key or os.environ.get("EXA_API_KEY")),
            FetchPageTool()]
