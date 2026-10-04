"""Test-only network guard, also loaded at Python subprocess startup.

This directory belongs on test children's PYTHONPATH, never the application's
installation path. Audit hooks run before DNS/socket operations and survive
monkeypatches used to test the application's own network validation.
"""

import ipaddress
import os
from pathlib import Path
import socket
import sys

GUARD_DIRECTORY = str(Path(__file__).resolve().parent)


class ExternalNetworkBlocked(BaseException):
    """Fail the test, bypassing application retry/error handlers."""


def _check_host(host):
    if host is None:
        return
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and address.is_loopback:
        return
    raise ExternalNetworkBlocked(f"Tests cannot access external network host {host!r}; use a mock or loopback server")


def _audit(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        _check_host(args[0])
    elif event == "socket.getnameinfo":
        _check_host(args[0][0])
    elif event in {"socket.connect", "socket.sendto", "socket.sendmsg"}:
        connection, address = args
        if connection.family in {socket.AF_INET, socket.AF_INET6} and address is not None:
            _check_host(address[0])
    elif event == "subprocess.Popen":
        env = args[3]
        if env is not None and GUARD_DIRECTORY not in env.get("PYTHONPATH", "").split(os.pathsep):
            raise ExternalNetworkBlocked("Test subprocess environment must preserve the offline PYTHONPATH guard")


sys.addaudithook(_audit)
