"""Download the latest SlipAgent from GitHub and run its change-checked installer.

The downloaded `install.py` performs the update, so local-edit detection and its
confirmation prompt apply exactly as for a manual download.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from dataclasses import dataclass

from . import __version__
from .config import data_directory, save_private_text
from .tools.blocking import run_blocking

REPOSITORY = "https://github.com/FedgeNo/SlipAgent"
API_URL = 'https://api.github.com/repos/FedgeNo/SlipAgent'
CHECK_TIMEOUT = 10.0
MAX_CHECK_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class Release:
    commit: str
    version: str


def _commit(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('GitHub did not return a full commit SHA.')
    return value


def installed_identity() -> dict[str, Any] | None:
    found = installation()
    if found is None:
        return None
    root, metadata = found
    changed = False
    if not metadata.get('version'):
        metadata['version'] = __version__
        changed = True
    if 'commit' not in metadata:
        metadata['commit'] = None
        source = Path(metadata.get('source', root / 'source'))
        if (source / '.git').exists():
            try:
                result = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True, timeout=2, check=False)
                if result.returncode == 0:
                    metadata['commit'] = _commit(result.stdout.strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        changed = True
    if changed:
        save_private_text(root / 'installation.json', json.dumps(metadata, indent=2) + '\n')
    return metadata


async def _read_check(client: Any, url: str, *, accept: str = 'application/vnd.github+json') -> bytes:
    async with client.stream('GET', url, headers={'Accept': accept}) as response:
        response.raise_for_status()
        data = bytearray()
        async for chunk in response.aiter_bytes():
            if len(data) + len(chunk) > MAX_CHECK_BYTES:
                raise ValueError('Update metadata exceeds its size limit.')
            data.extend(chunk)
        return bytes(data)


async def check_for_updates(*, transport: Any = None) -> Release | None:
    """A whole-check deadline; this task never belongs on startup's critical path."""
    import httpx

    async with asyncio.timeout(CHECK_TIMEOUT):
        identity = await run_blocking(installed_identity)
        if identity is None:
            return None
        async with httpx.AsyncClient(transport=transport, timeout=CHECK_TIMEOUT, follow_redirects=False) as client:
            raw = await _read_check(client, API_URL + '/commits/master', accept='application/vnd.github.sha')
            commit = _commit(raw.decode('ascii').strip())
            installed = identity.get('commit')
            if installed == commit:
                return None
            raw_version = await _read_check(client, f'https://raw.githubusercontent.com/FedgeNo/SlipAgent/{commit}/pyproject.toml')
            version = _parse_version(raw_version.decode('utf-8'))
            if installed is None:
                return Release(commit, version) if version_key(version) > version_key(identity['version']) else None
            installed = _commit(installed)
            try:
                comparison = await _read_check(client, API_URL + f'/compare/{installed}...{commit}?per_page=1')
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                # An unpublished local commit cannot be ordered by the public graph.
                return Release(commit, version) if version_key(version) > version_key(identity['version']) else None
            result = await run_blocking(json.loads, comparison)
            if not isinstance(result, dict) or result.get('status') not in ('ahead', 'behind', 'identical', 'diverged'):
                raise ValueError('GitHub returned an invalid commit comparison.')
            return Release(commit, version) if result['status'] == 'ahead' else None


def latest_commit() -> str:
    """Resolve again for every explicit update; never reuse a startup observation."""
    import httpx

    with httpx.stream('GET', API_URL + '/commits/master', headers={'Accept': 'application/vnd.github.sha'},
                      timeout=CHECK_TIMEOUT, follow_redirects=False) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_bytes():
            if len(data) + len(chunk) > 1024:
                raise ValueError('GitHub commit metadata is unexpectedly large.')
            data.extend(chunk)
    return _commit(data.decode('ascii').strip())


def installation() -> tuple[Path, dict[str, Any]] | None:
    """The managed installation running this code, from its `installation.json`."""
    source = Path(__file__).resolve().parents[2]
    root = source.parent
    try:
        metadata = json.loads((root / "installation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(metadata, dict) or Path(str(metadata.get("source", ""))).resolve() != source:
        return None
    return root, metadata


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def download(url: str, destination: Path) -> None:
    import httpx

    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response, destination.open("wb") as target:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise OSError("The downloaded archive is unexpectedly large.")
            target.write(chunk)


def extract(archive: Path, destination: Path) -> Path:
    """Extract regular files and directories under one top-level folder."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (member.name.startswith("/") or ".." in parts or not (member.isfile() or member.isdir())):
                raise OSError(f"Refusing unsafe archive entry: {member.name}")
        if len(roots) != 1:
            raise OSError("The archive does not contain a single release folder.")
        if hasattr(tarfile, "data_filter"):
            source.extractall(destination, filter="data")
        else:
            source.extractall(destination)
    return destination / roots.pop()


def released_version(checkout: Path) -> str:
    return _parse_version((checkout / 'pyproject.toml').read_text(encoding='utf-8'))


def _parse_version(text: str) -> str:
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise OSError("The downloaded release does not declare a version.")
    return match.group(1)


def update(*, fetch: Callable[[str, Path], None] = download,
           resolve: Callable[[], str] = latest_commit,
           run: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run) -> int:
    found = installation()
    if found is None:
        print("This copy of SlipAgent is not a managed installation. Update a development checkout "
              "with Git, or run install.py from a downloaded release.", file=sys.stderr)
        return 2
    root, metadata = found
    parent = data_directory()
    parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="update-", dir=parent))
    try:
        archive = scratch / "slipagent.tar.gz"
        try:
            commit = _commit(resolve())
            if metadata.get('commit') == commit:
                print(f"SlipAgent {__version__} is up to date.", file=sys.stderr)
                return 0
            archive_url = REPOSITORY + f'/archive/{commit}.tar.gz'
            print(f"Downloading {archive_url}", file=sys.stderr)
            fetch(archive_url, archive)
            checkout = extract(archive, scratch / "release")
            latest = released_version(checkout)
        except Exception as exc:
            print(f"Could not download the latest release: {exc}", file=sys.stderr)
            return 1
        print(f"Updating SlipAgent {__version__} to {latest}.", file=sys.stderr)
        command = [getattr(sys, "_base_executable", sys.executable), str(checkout / "install.py"),
                   "--install-dir", str(root), '--source-commit', commit]
        launcher = metadata.get("launcher")
        if isinstance(launcher, str) and launcher:
            command += ["--bin-dir", str(Path(launcher).parent)]
        return run(command, cwd=checkout).returncode
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
