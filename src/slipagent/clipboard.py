"""Copy user-selected text through an installed clipboard utility or OSC 52."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess


def copy_to_clipboard(text: str) -> bool:
    """Return whether a local clipboard utility accepted the text."""
    choices = []
    if os.environ.get("WAYLAND_DISPLAY"):
        choices.append(["wl-copy"])
    if os.environ.get("DISPLAY"):
        choices.extend([["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]])
    if os.name != "nt":
        choices.append(["pbcopy"])
    for command in choices:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(command, input=text.encode("utf-8"), timeout=2, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            continue
        return True
    return False


def clipboard_sequence(text: str) -> str:
    """Construct a trusted clipboard control, never accepting raw escape data."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"\x1b]52;c;{encoded}\x07"
