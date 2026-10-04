"""Web tools: HTML extraction, fetch_page, and web_search."""

from __future__ import annotations

import httpx
import pytest

from slipagent.tools.base import ToolRegistry
from slipagent.tools.web import FetchPageTool, WebSearchTool, html_to_text, web_tools


def test_html_retains_code_indentation_and_resolves_links():
    text, _ = html_to_text('<p>Example</p><pre>if ready:\n    act()\n\n\n    finish()</pre><p><a href="../reference#api">API</a></p>',
                           base_url="https://example.com/guide/start")
    assert "if ready:\n    act()\n\n\n    finish()" in text
    assert "https://example.com/reference#api" in text

SAMPLE_HTML = """<!doctype html>
<html>
  <head>
    <title>Example Page</title>
    <style>body { color: red; }</style>
    <script>console.log("tracking");</script>
  </head>
  <body>
    <h1>Heading</h1>
    <p>First paragraph.</p>
    <p>Second paragraph.</p>
    <noscript>enable js</noscript>
    <a href="/next">Next &amp; last</a>
  </body>
</html>"""


def transport_for(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def rebind(tool, handler) -> None:
    """Point a tool's client at a mock transport."""
    tool._client = httpx.AsyncClient(transport=transport_for(handler))


@pytest.fixture(autouse=True)
def public_test_dns(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 443))
    ])


# --------------------------------------------------------------------------- #
# html_to_text
# --------------------------------------------------------------------------- #


def test_extracts_title_and_text() -> None:
    text, title = html_to_text(SAMPLE_HTML)

    assert title == "Example Page"
    assert "Heading" in text
    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_drops_script_style_and_noscript() -> None:
    text, _ = html_to_text(SAMPLE_HTML)

    assert "tracking" not in text
    assert "color: red" not in text
    assert "enable js" not in text


def test_unescapes_entities() -> None:
    text, _ = html_to_text(SAMPLE_HTML)

    assert "Next & last" in text


def test_collapses_whitespace() -> None:
    text, _ = html_to_text("<html><body><p>a   \n\n\n\n   b</p></body></html>")

    assert text == "a\n\nb"


def test_malformed_html_does_not_raise() -> None:
    text, _ = html_to_text("<div><p>unclosed <b>bold")

    assert "unclosed" in text


# --------------------------------------------------------------------------- #
# fetch_page
# --------------------------------------------------------------------------- #


async def test_fetch_page_returns_readable_text() -> None:
    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(
        200, html=SAMPLE_HTML, headers={"Content-Type": "text/html"}
    ))

    result = await tool.invoke({"url": "https://example.com/page"})
    await tool.aclose()

    assert not result.is_error
    assert "Example Page — https://example.com/page" in result.content
    assert "First paragraph." in result.content
    assert "tracking" not in result.content


async def test_fetch_page_passes_through_json() -> None:
    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(
        200, json={"version": "1.2.3"},
        headers={"Content-Type": "application/json"},
    ))

    result = await tool.invoke({"url": "https://api.example.com/v1"})
    await tool.aclose()

    assert "application/json" in result.content
    assert "1.2.3" in result.content


async def test_fetch_page_passes_through_plain_text() -> None:
    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(
        200, text="key = value\n", headers={"Content-Type": "text/plain"}
    ))

    result = await tool.invoke({"url": "https://example.com/f.txt"})
    await tool.aclose()

    assert "key = value" in result.content


async def test_fetch_page_truncates() -> None:
    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(
        200, html="<html><body>" + "x" * 5000 + "</body></html>",
        headers={"Content-Type": "text/html"},
    ))

    result = await tool.invoke({"url": "https://example.com/big", "max_chars": 600})
    await tool.aclose()

    assert "[truncated]" in result.content
    assert len(result.content) < 1200


async def test_fetch_page_rejects_bad_scheme() -> None:
    result = await FetchPageTool().invoke({"url": "ftp://example.com"})

    assert result.is_error
    assert "must start with http" in result.content


async def test_fetch_page_reports_http_error() -> None:
    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(404, text="nope"))

    result = await tool.invoke({"url": "https://example.com/gone"})
    await tool.aclose()

    assert result.is_error
    assert "404" in result.content


async def test_fetch_page_reports_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    tool = FetchPageTool()
    rebind(tool, handler)

    result = await tool.invoke({"url": "https://example.com/"})
    await tool.aclose()

    assert result.is_error
    assert "Could not fetch" in result.content


async def test_fetch_page_reports_javascript_only_page() -> None:
    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(
        200, html="<html><body><div id='app'></div></body></html>",
        headers={"Content-Type": "text/html"},
    ))

    result = await tool.invoke({"url": "https://spa.example.com"})
    await tool.aclose()

    assert result.is_error
    assert "JavaScript-rendered" in result.content


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #


def search_response() -> dict[str, object]:
    return {
        "results": [
            {
                "title": "asyncio — Python docs",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "text": "Asynchronous I/O support.",
            },
            {
                "title": "Real Python: asyncio",
                "url": "https://realpython.com/asyncio-python/",
                "text": "A guide to asyncio.",
            },
        ]
    }


async def test_web_search_formats_results() -> None:
    tool = WebSearchTool(api_key="test-key")
    rebind(tool, lambda r: httpx.Response(200, json=search_response()))

    result = await tool.invoke({"query": "python asyncio"})
    await tool.aclose()

    assert not result.is_error
    assert "[1] asyncio — Python docs" in result.content
    assert "https://docs.python.org/3/library/asyncio.html" in result.content
    assert "[2] Real Python: asyncio" in result.content
    assert "Use fetch_page" in result.content


async def test_web_search_requires_api_key() -> None:
    result = await WebSearchTool(api_key=None).invoke({"query": "anything"})

    assert result.is_error
    assert "EXA_API_KEY" in result.content
    assert "fetch_page still works" in result.content


async def test_web_search_reports_bad_key() -> None:
    tool = WebSearchTool(api_key="bad-key")
    rebind(tool, lambda r: httpx.Response(
        401, json={"error": {"message": "Invalid API key"}}
    ))

    result = await tool.invoke({"query": "anything"})
    await tool.aclose()

    assert result.is_error
    assert "Check that EXA_API_KEY is valid" in result.content


async def test_web_search_reports_rate_limit() -> None:
    tool = WebSearchTool(api_key="k")
    rebind(tool, lambda r: httpx.Response(429, json={"error": {"message": "slow"}}))

    result = await tool.invoke({"query": "anything"})
    await tool.aclose()

    assert result.is_error
    assert "credits exhausted" in result.content


async def test_web_search_handles_no_results() -> None:
    tool = WebSearchTool(api_key="k")
    rebind(tool, lambda r: httpx.Response(200, json={"results": []}))

    result = await tool.invoke({"query": "obscure nonsense"})
    await tool.aclose()

    assert not result.is_error
    assert "No results" in result.content


async def test_web_search_rejects_empty_query() -> None:
    result = await WebSearchTool(api_key="k").invoke({"query": "   "})

    assert result.is_error


async def test_web_search_caps_result_count() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as json_module

        captured["body"] = json_module.loads(request.content)
        return httpx.Response(200, json=search_response())

    tool = WebSearchTool(api_key="k")
    rebind(tool, handler)

    await tool.invoke({"query": "x", "num_results": 999})
    await tool.aclose()

    assert captured["body"]["numResults"] <= 10


# --------------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------------- #


async def test_web_tools_are_registered_by_default() -> None:
    registry = ToolRegistry(web_tools(None))

    assert registry.names == ["fetch_page", "web_search"]
    await registry.aclose()


async def test_registry_aclose_closes_tool_clients() -> None:
    registry = ToolRegistry(web_tools("k"))
    search = registry.get("web_search")

    await registry.aclose()

    assert isinstance(search, WebSearchTool)
    assert search._client.is_closed


async def test_fetch_json_obeys_character_limit() -> None:
    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(200, json={"text": "x" * 4000}))
    try:
        result = await tool.invoke({"url": "https://example.com", "max_chars": 500})
        assert "[truncated]" in result.content
        assert len(result.content) < 800
    finally:
        await tool.aclose()


async def test_fetch_stops_reading_at_byte_limit(monkeypatch) -> None:
    from slipagent.tools import web
    monkeypatch.setattr(web, "MAX_FETCH_BYTES", 1000)

    class LargeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 1001
            raise AssertionError("read beyond byte budget")

    tool = FetchPageTool()
    rebind(tool, lambda r: httpx.Response(200, stream=LargeStream()))
    try:
        result = await tool.invoke({"url": "https://example.com"})
        assert not result.is_error
        assert "[truncated]" in result.content
    finally:
        await tool.aclose()


@pytest.mark.parametrize("host", ["127.0.0.1", "169.254.169.254", "10.0.0.1", "[::1]"])
async def test_fetch_rejects_private_addresses(host) -> None:
    tool = FetchPageTool()
    requests = []
    rebind(tool, lambda r: requests.append(r) or httpx.Response(200, text="private"))
    try:
        result = await tool.invoke({"url": f"http://{host}/"})
        assert result.is_error
        assert not requests
    finally:
        await tool.aclose()


async def test_fetch_rejects_private_dns_and_redirect(monkeypatch) -> None:
    import socket
    tool = FetchPageTool()
    requests = []
    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/secret"})
    rebind(tool, handler)
    try:
        result = await tool.invoke({"url": "https://example.com"})
        assert result.is_error
        assert requests == ["https://example.com"]
        requests.clear()
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))])
        result = await tool.invoke({"url": "https://private.example.com"})
        assert result.is_error
        assert not requests
    finally:
        await tool.aclose()


def test_void_tags_inside_skipped_html_do_not_hide_body() -> None:
    text, title = html_to_text('<head><title>Title</title><meta><link></head><body>Visible</body>')
    assert text == "Visible"
    assert title == "Title"


def test_self_closing_skipped_tag_does_not_hide_following_text() -> None:
    text, _ = html_to_text("<svg/>Visible")
    assert text == "Visible"


async def test_network_transport_pins_address_and_preserves_host(monkeypatch) -> None:
    from slipagent.tools.web import _PublicTransport
    captured = []
    async def handle(transport, request):
        captured.append(request)
        return httpx.Response(200, text="ok")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    transport = _PublicTransport()
    try:
        request = httpx.Request("GET", "https://docs.example.com:8443/path?q=1",
                               extensions={"validated_address": "93.184.215.14"})
        await transport.handle_async_request(request)
        assert str(captured[0].url) == "https://93.184.215.14:8443/path?q=1"
        assert captured[0].headers["host"] == "docs.example.com:8443"
        assert captured[0].extensions["sni_hostname"] == "docs.example.com"
    finally:
        await transport.aclose()
