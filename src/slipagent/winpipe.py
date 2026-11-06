"""Local-only Windows pipes with explicit user ACLs and overlapped asyncio I/O."""

from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes as w
from collections.abc import Callable
from typing import Any
import weakref

from .lifecycle import Lifetime
from .localipc import LIMIT

CLIENT_ACCESS = 0x12019B  # Read/write data and attributes; excludes creating pipe instances.


class SecurityAttributes(ctypes.Structure):
    _fields_ = [('length', w.DWORD), ('descriptor', w.LPVOID), ('inherit', w.BOOL)]


class WindowsAPI:
    def __init__(self) -> None:
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]
        self.security = ctypes.WinDLL('advapi32', use_last_error=True)  # type: ignore[attr-defined]
        signatures = [
            (self.kernel, 'GetCurrentProcess', [], w.HANDLE),
            (self.kernel, 'GetNamedPipeClientProcessId', [w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
            (self.kernel, 'OpenProcess', [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            (self.kernel, 'CloseHandle', [w.HANDLE], w.BOOL),
            (self.kernel, 'LocalFree', [w.LPVOID], w.LPVOID),
            (self.kernel, 'CreateNamedPipeW', [w.LPCWSTR, w.DWORD, w.DWORD, w.DWORD,
                w.DWORD, w.DWORD, w.DWORD, ctypes.POINTER(SecurityAttributes)], w.HANDLE),
            (self.kernel, 'CreateFileW', [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID,
                w.DWORD, w.DWORD, w.HANDLE], w.HANDLE),
            (self.security, 'OpenProcessToken', [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)], w.BOOL),
            (self.security, 'GetTokenInformation', [w.HANDLE, ctypes.c_int, w.LPVOID,
                w.DWORD, ctypes.POINTER(w.DWORD)], w.BOOL),
            (self.security, 'ConvertSidToStringSidW', [w.LPVOID, ctypes.POINTER(w.LPWSTR)], w.BOOL),
            (self.security, 'ConvertStringSecurityDescriptorToSecurityDescriptorW',
                [w.LPCWSTR, w.DWORD, ctypes.POINTER(w.LPVOID), w.LPVOID], w.BOOL),
            (self.security, 'SetKernelObjectSecurity', [w.HANDLE, w.DWORD, w.LPVOID], w.BOOL),
        ]
        for library, name, arguments, result in signatures:
            function = getattr(library, name)
            function.argtypes = arguments
            function.restype = result

    def check(self, result: Any) -> Any:
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        return result

    def token_user(self, token: Any) -> str:
        size = w.DWORD()
        self.security.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        self.check(self.security.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)))
        sid = ctypes.cast(buffer, ctypes.POINTER(w.LPVOID))[0]
        text = w.LPWSTR()
        self.check(self.security.ConvertSidToStringSidW(sid, ctypes.byref(text)))
        try:
            return str(text.value)
        finally:
            self.kernel.LocalFree(ctypes.cast(text, w.LPVOID))

    def descriptor(self, owner: str, danger: bool) -> Any:
        # Other users may exchange data, but cannot create impersonating server instances.
        acl = f'O:{owner}D:P(A;;GA;;;{owner})'
        if danger:
            acl += f'(A;;0x{CLIENT_ACCESS:x};;;WD)'
        descriptor = w.LPVOID()
        self.check(self.security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            acl, 1, ctypes.byref(descriptor), None))
        return descriptor


def current_user() -> str:
    api, token = WindowsAPI(), w.HANDLE()
    api.check(api.security.OpenProcessToken(api.kernel.GetCurrentProcess(), 8, ctypes.byref(token)))
    try:
        return api.token_user(token)
    finally:
        api.kernel.CloseHandle(token)


def pipe_user(handle: int) -> str:
    api, token, pid = WindowsAPI(), w.HANDLE(), w.DWORD()
    api.check(api.kernel.GetNamedPipeClientProcessId(handle, ctypes.byref(pid)))
    process = api.check(api.kernel.OpenProcess(0x1000, False, pid.value))
    try:
        api.check(api.security.OpenProcessToken(process, 8, ctypes.byref(token)))
        try:
            return api.token_user(token)
        finally:
            api.kernel.CloseHandle(token)
    finally:
        api.kernel.CloseHandle(process)


class PipeListener:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.api = WindowsAPI()
        self.owner = current_user()
        self.danger = False
        self.handles: weakref.WeakSet[Any] = weakref.WeakSet()
        self.lifetime = Lifetime('local control pipe')

    def create(self, first: bool = False) -> Any:
        from asyncio.windows_utils import PipeHandle  # type: ignore[attr-defined]
        descriptor = self.api.descriptor(self.owner, self.danger)
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        try:
            handle = self.api.kernel.CreateNamedPipeW(self.endpoint,
                3 | 0x40000000 | 0x40000 | (0x80000 if first else 0), 8, 255, 65536, 65536, 0,
                ctypes.byref(attributes))  # Duplex, overlapped, byte mode, reject remote clients.
            if handle == w.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
            pipe = PipeHandle(handle)
            self.handles.add(pipe)
            return pipe
        finally:
            self.api.kernel.LocalFree(descriptor)

    async def start(self, connected: Callable[[asyncio.StreamReader, asyncio.StreamWriter], None], danger: bool) -> None:
        loop = asyncio.get_running_loop()
        if not hasattr(loop, '_proactor'):
            raise RuntimeError('Local control pipes require the Windows Proactor event loop.')
        self.danger = danger
        pipe = self.create(first=True)
        self.lifetime.spawn(self.accept(pipe, connected))

    async def accept(self, pipe: Any, connected: Callable[..., None]) -> None:
        loop: Any = asyncio.get_running_loop()
        try:
            while True:
                try:
                    await loop._proactor.accept_pipe(pipe)
                except BrokenPipeError:
                    pipe.close()
                    pipe = self.create()
                    continue
                accepted, pipe = pipe, self.create()
                reader = asyncio.StreamReader(limit=LIMIT)
                protocol = asyncio.StreamReaderProtocol(reader)
                transport = loop._make_duplex_pipe_transport(accepted, protocol)
                connected(reader, asyncio.StreamWriter(transport, protocol, reader, loop))
        finally:
            pipe.close()

    def set_access(self, danger: bool) -> None:
        descriptor = self.api.descriptor(self.owner, danger)
        try:
            for pipe in tuple(self.handles):
                try:
                    handle = pipe.fileno()
                except ValueError:
                    continue  # A detached StreamWriter may still reference a closed PipeHandle.
                if handle != -1:
                    self.api.check(self.api.security.SetKernelObjectSecurity(handle, 4, descriptor))
            self.danger = danger
        finally:
            self.api.kernel.LocalFree(descriptor)

    async def aclose(self) -> None:
        await self.lifetime.aclose()


async def connect_pipe(endpoint: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    from asyncio.windows_utils import PipeHandle  # type: ignore[attr-defined]
    api = WindowsAPI()
    loop: Any = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while True:
        handle = api.kernel.CreateFileW(endpoint, CLIENT_ACCESS, 0, None, 3,
            0x40000000 | 0x100000 | 0x10000, None)  # Overlapped; identification-only impersonation.
        if handle != w.HANDLE(-1).value:
            break
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        if error != 231 or loop.time() >= deadline:
            raise ctypes.WinError(error)  # type: ignore[attr-defined]
        await asyncio.sleep(.02)
    pipe = PipeHandle(handle)
    reader = asyncio.StreamReader(limit=LIMIT)
    protocol = asyncio.StreamReaderProtocol(reader)
    transport = loop._make_duplex_pipe_transport(pipe, protocol)
    return reader, asyncio.StreamWriter(transport, protocol, reader, loop)
