"""Discover and control running instances without provider keys or GUI access."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from typing import Any

from . import localipc
from .localcontrol import encode
from .tools.blocking import run_blocking


async def request(endpoint: str, method: str, params: dict[str, Any] | None = None, *,
                  request_id: str | None = None, session_id: str | None = None,
                  timeout: float = 30) -> dict[str, Any]:
    reader, writer = await asyncio.wait_for(localipc.connect(endpoint), 5)
    try:
        payload: dict[str, Any] = {'jsonrpc': '2.0', 'id': request_id or uuid.uuid4().hex,
                                   'method': method, 'params': params or {}}
        if session_id is not None:
            payload['sessionId'] = session_id
        writer.write(encode(payload))
        await asyncio.wait_for(writer.drain(), 5)
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout)
            if not line:
                raise ConnectionError('Local control disconnected.')
            response = json.loads(line)
            if not isinstance(response, dict) or response.get('jsonrpc') != '2.0':
                raise ValueError('Invalid local control response.')
            if response.get('id') == payload['id']:
                return dict(response)
    finally:
        writer.close()
        writer.transport.abort()
        await writer.wait_closed()


async def instances() -> list[dict[str, Any]]:
    gate = asyncio.Semaphore(8)

    async def inspect(endpoint: str) -> dict[str, Any] | None:
        async with gate:
            try:
                response = await request(endpoint, 'status', timeout=1)
                result = response.get('result')
                if (not isinstance(result, dict) or result.get('protocolVersion') != 1
                        or not isinstance(result.get('instanceId'), str)
                        or not (endpoint.endswith(result['instanceId'] + '.sock')
                                or endpoint.endswith('-' + result['instanceId']))):
                    return None
                return {**result, 'endpoint': endpoint}
            except (OSError, ValueError, asyncio.TimeoutError):
                return None  # Other users' private endpoints and dead instances are not accessible.

    endpoints = await run_blocking(localipc.discover)
    results = await asyncio.gather(*(inspect(endpoint) for endpoint in endpoints))
    return sorted((item for item in results if item is not None), key=lambda item: item['instanceId'])


async def resolve(target: str) -> str:
    if target.startswith('\\\\.\\pipe\\') or target.endswith('.sock'):
        return target
    matches = [item['endpoint'] for item in await instances() if item['instanceId'].startswith(target)]
    if len(matches) != 1:
        raise ValueError('Choose one instance ID from `slipagent control list`, or give its full endpoint.')
    return str(matches[0])


async def watch(endpoint: str) -> None:
    reader, writer = await localipc.connect(endpoint)
    try:
        writer.write(encode({'jsonrpc': '2.0', 'id': uuid.uuid4().hex, 'method': 'subscribe'}))
        await writer.drain()
        while line := await reader.readline():
            print(json.dumps(json.loads(line), ensure_ascii=False), flush=True)
    finally:
        writer.close()
        writer.transport.abort()
        await writer.wait_closed()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog='slipagent control', description='Control local running SlipAgent instances.')
    parser.add_argument('target', help='Instance ID, full endpoint, or list.')
    parser.add_argument('method', nargs='?', default='status', help='Control method, or watch for live events.')
    parser.add_argument('params', nargs='?', default='{}', help='JSON object of method parameters.')
    parser.add_argument('--request-id', help='Reuse this ID to retry an operation without resubmitting it.')
    parser.add_argument('--session-id', help='Refuse the action if the target changed conversations.')
    args = parser.parse_args(argv)

    async def run() -> int:
        if args.target == 'list':
            print(json.dumps(await instances(), ensure_ascii=False, indent=2))
            return 0
        endpoint = await resolve(args.target)
        if args.method == 'watch':
            await watch(endpoint)
            return 0
        params = json.loads(args.params)
        if not isinstance(params, dict):
            raise ValueError('Parameters must be a JSON object.')
        result = await request(endpoint, args.method, params,
                               request_id=args.request_id, session_id=args.session_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int('error' in result)

    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, asyncio.TimeoutError) as exc:
        print(f'slipagent control: {exc}', file=sys.stderr)
        return 1
