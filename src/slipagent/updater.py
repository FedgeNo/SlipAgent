"""Download the latest SlipAgent from GitHub and run its change-checked installer.

The downloaded `install.py` performs the update, so local-edit detection and its
confirmation prompt apply exactly as for a manual download.
"""

from __future__ import annotations

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

from . import __version__
from .config import data_directory

REPOSITORY = "https://github.com/FedgeNo/SlipAgent"
ARCHIVE_URL = REPOSITORY + "/archive/refs/heads/master.tar.gz"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


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
    match = re.search(r'^version = "([^"]+)"', (checkout / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise OSError("The downloaded release does not declare a version.")
    return match.group(1)


def update(*, fetch: Callable[[str, Path], None] = download,
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
        print(f"Downloading {ARCHIVE_URL}", file=sys.stderr)
        try:
            fetch(ARCHIVE_URL, archive)
            checkout = extract(archive, scratch / "release")
            latest = released_version(checkout)
        except Exception as exc:
            print(f"Could not download the latest release: {exc}", file=sys.stderr)
            return 1
        if version_key(latest) <= version_key(__version__):
            print(f"SlipAgent {__version__} is up to date.", file=sys.stderr)
            return 0
        print(f"Updating SlipAgent {__version__} to {latest}.", file=sys.stderr)
        command = [getattr(sys, "_base_executable", sys.executable), str(checkout / "install.py"),
                   "--install-dir", str(root)]
        launcher = metadata.get("launcher")
        if isinstance(launcher, str) and launcher:
            command += ["--bin-dir", str(Path(launcher).parent)]
        return run(command, cwd=checkout).returncode
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
