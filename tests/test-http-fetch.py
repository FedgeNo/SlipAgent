"""HTTP request contracts without real network access."""

import json

import httpx
import pytest

from slipagent.tools.http import HttpFetchTool


@pytest.mark.asyncio
async def test_request_and_error_body_are_preserved():
    def handle(request):
        assert request.method == "POST"
        assert request.url.params["q"] == "a & b"
        assert request.headers["authorization"] == "Bearer dummy"
        assert json.loads(request.content) == {"text": "a\\b's\nvalue"}
        return httpx.Response(422, text="literal <html>error</html>",
                              headers=[("x-example", "one"), ("x-example", "two")])

    tool = HttpFetchTool()
    await tool.aclose()
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        result = await tool.run("https://93.184.215.14/api", method="POST",
                                params={"q": "a & b"}, headers={"authorization": "Bearer dummy"},
                                json_body={"text": "a\\b's\nvalue"})
        assert result.is_error
        assert result.content["status_code"] == 422
        assert result.content["body"] == "literal <html>error</html>"
        assert ["x-example", "one"] in result.content["headers"]
        assert ["x-example", "two"] in result.content["headers"]
    finally:
        await tool.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body", [(302, b"redirect"), (200, b"\xff\xfe")])
async def test_redirects_and_binary(status, body):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, content=body, headers={"location": "http://127.0.0.1/"})

    tool = HttpFetchTool()
    await tool.aclose()
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        result = await tool.run("https://93.184.215.14/")
        assert not result.is_error
        assert len(requests) == 1
        if status == 200:
            assert result.content["body_base64"] == "//4="
        else:
            assert result.content["body"] == "redirect"
    finally:
        await tool.aclose()


@pytest.mark.asyncio
async def test_validation_and_download_limit(monkeypatch):
    monkeypatch.setattr("slipagent.tools.http.MAX_FETCH_BYTES", 4)
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, text="123456")

    tool = HttpFetchTool()
    await tool.aclose()
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        assert (await tool.run("http://127.0.0.1/")).is_error
        assert (await tool.run("file:///tmp/test")).is_error
        assert (await tool.run("https://93.184.215.14/", body="x", json_body={})).is_error
        assert not requests
        result = await tool.run("https://93.184.215.14/")
        assert result.content["body"] == "1234"
        assert result.content["truncated"] is True
    finally:
        await tool.aclose()
