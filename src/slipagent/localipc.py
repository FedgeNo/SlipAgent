"""OS-owned local endpoints shared by control servers and clients."""

from __future__ import annotations

import asyncio
import os
import socket
import stat
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

LIMIT = 131072


def socket_root() -> Path:
    return Path(os.environ.get('SLIPAGENT_CONTROL_DIR', '/tmp'))


def endpoint_name(instance: str) -> str:
    if os.name == 'nt':
        return rf'\\.\pipe\slipagent-{instance}'
    directory = socket_root() / f'slipagent-{os.geteuid()}'
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    entry = directory.lstat()
    if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_mode & 0o022:
        raise PermissionError(f'Unsafe local control directory: {directory}')
    directory.chmod(0o755)
    return str(directory / f'{instance}.sock')


def discover() -> list[str]:
    if os.name == 'nt':
        return [rf'\\.\pipe\{name}' for name in os.listdir('\\\\.\\pipe\\')
                if name.startswith('slipagent-')]
    return [str(path) for path in socket_root().glob('slipagent-*/*.sock')
            if path.is_socket()]


def owner_identity() -> str:
    if os.name == 'nt':
        from .winpipe import current_user
        return current_user()
    return str(os.geteuid())


def peer_identity(writer: asyncio.StreamWriter) -> str:
    if os.name == 'nt':
        from .winpipe import pipe_user
        return pipe_user(writer.get_extra_info('pipe').fileno())
    connection = writer.get_extra_info('socket')
    if hasattr(socket, 'SO_PEERCRED'):
        _, uid, _ = struct.unpack('iII', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        return str(uid)
    if hasattr(connection, 'getpeereid'):
        return str(connection.getpeereid()[0])
    # macOS exposes getpeereid in libc rather than Python's socket wrapper.
    import ctypes
    uid, gid = ctypes.c_uint(), ctypes.c_uint()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.getpeereid(connection.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot identify local control peer')
    return str(uid.value)


class UnixListener:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.server: asyncio.AbstractServer | None = None
        self.identity: tuple[int, int] | None = None

    async def start(self, connected: Callable[[asyncio.StreamReader, asyncio.StreamWriter], None], danger: bool) -> None:
        # Bind in an owner-only staging directory, then publish atomically.
        # This avoids exposing a permissive socket during an await or umask race.
        parent = Path(self.endpoint).parent
        staging = parent / (Path(self.endpoint).stem + '.pending')
        staging.mkdir(mode=0o700)
        temporary = staging / 'socket'
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(temporary))
            temporary.chmod(0o666 if danger else 0o600)
            os.rename(temporary, self.endpoint)
            info = Path(self.endpoint).lstat()
            self.identity = (info.st_dev, info.st_ino)
            listener.listen(16)
            listener.setblocking(False)
            self.server = await asyncio.start_unix_server(connected, sock=listener, limit=LIMIT)
        except BaseException:
            listener.close()
            self.unlink()
            raise
        finally:
            temporary.unlink(missing_ok=True)
            staging.rmdir()

    def set_access(self, danger: bool) -> None:
        info = Path(self.endpoint).lstat()
        if not stat.S_ISSOCK(info.st_mode) or (info.st_dev, info.st_ino) != self.identity:
            raise OSError('The local control socket was replaced.')
        os.chmod(self.endpoint, 0o666 if danger else 0o600)

    def unlink(self) -> None:
        try:
            info = Path(self.endpoint).lstat()
            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self.identity:
                Path(self.endpoint).unlink()
        except FileNotFoundError:
            pass

    async def aclose(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        self.unlink()


def listener(endpoint: str) -> Any:
    if os.name == 'nt':
        from .winpipe import PipeListener
        return PipeListener(endpoint)
    return UnixListener(endpoint)


async def connect(endpoint: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    if os.name == 'nt':
        from .winpipe import connect_pipe
        return await connect_pipe(endpoint)
    return await asyncio.open_unix_connection(endpoint, limit=LIMIT)
