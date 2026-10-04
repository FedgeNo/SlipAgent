"""External access must fail before DNS or network I/O, including child CLIs."""

import os
import socket
import subprocess
import sys

import pytest

from offline.sitecustomize import ExternalNetworkBlocked
from test_cli_e2e import cli_environment


def test_external_dns_is_blocked():
    with pytest.raises(ExternalNetworkBlocked, match="external network"):
        socket.getaddrinfo("openrouter.ai", 443)


@pytest.mark.parametrize("operation", ["connect", "connect_ex", "sendto"])
def test_external_addresses_are_blocked_without_dns(operation):
    kind = socket.SOCK_DGRAM if operation == "sendto" else socket.SOCK_STREAM
    with socket.socket(socket.AF_INET, kind) as connection:
        address = ("203.0.113.1", 443)
        with pytest.raises(ExternalNetworkBlocked, match="external network"):
            if operation == "sendto":
                connection.sendto(b"test", address)
            else:
                getattr(connection, operation)(address)


def test_child_python_is_guarded_and_has_only_dummy_configuration(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import os,socket; "
         "assert os.environ['OPENROUTER_API_KEY'] == 'test-key'; "
         "assert os.environ['SLIPAGENT_NO_DOTENV'] == '1'; "
         "socket.getaddrinfo('openrouter.ai', 443)"],
        cwd=tmp_path, env=cli_environment(), capture_output=True, text=True, timeout=5,
    )
    assert result.returncode != 0
    assert "ExternalNetworkBlocked" in result.stderr
    assert "AssertionError" not in result.stderr


def test_explicit_subprocess_environment_cannot_drop_guard():
    env = dict(os.environ)
    env.pop("PYTHONPATH")
    with pytest.raises(ExternalNetworkBlocked, match="preserve the offline"):
        subprocess.run([sys.executable, "-c", "pass"], env=env, check=True)
