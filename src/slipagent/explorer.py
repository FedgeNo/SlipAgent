"""Discover and launch the desktop's native folder opener."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def folder_opener() -> tuple[str, ...] | None:
    if sys.platform == "win32":
        return () if hasattr(os, "startfile") else None
    if sys.platform == "darwin":
        command = shutil.which("open")
        return (command,) if command else None
    if sys.platform.startswith("linux") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        command = shutil.which("xdg-open")
        if command:
            return (command,)
        command = shutil.which("gio")
        if command:
            return command, "open"
    return None


def open_folder(directory: Path, opener: tuple[str, ...]) -> bool:
    """Return whether the opener started without an immediate failure."""
    try:
        if not opener:
            getattr(os, "startfile")(str(directory))
            return True
        process = subprocess.Popen([*opener, str(directory)], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=os.name != "nt")
        try:
            return process.wait(timeout=1) == 0
        except subprocess.TimeoutExpired:
            # A file manager can remain open until its window closes.
            return True
    except OSError:
        return False
