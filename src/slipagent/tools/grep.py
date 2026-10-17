"""Standalone regex worker; only the parent opens workspace files.

The parent launches a snapshot of this source, so rejected on-disk edits cannot
change an already active tool. Keep this worker independent of package imports.
Each batch is a JSON header line with file byte sizes, followed by the files'
raw bytes; each output line is its JSON result. Files decode as UTF-8 with
replacement, and lines break at LF, CRLF, or CR, as in editors.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def worker_source() -> bytes:
    return _SOURCE_BYTES


def main() -> None:
    source = sys.stdin.buffer
    config = json.loads(source.readline())
    try:
        pattern = re.compile(config["pattern"], re.IGNORECASE if config["ignore_case"] else 0)
    except re.error as exc:
        print(json.dumps({"error": f"Invalid regular expression: {exc}"}), flush=True)
        return
    print(json.dumps({"ready": True}), flush=True)
    breaks = re.compile(r"\r\n|\r|\n")
    # Without anchors, boundaries, or lookarounds, any match inside a line is
    # also a match in the whole text, so a file without one can be skipped.
    context = ("^", "$", "\\A", "\\Z", "\\z", "\\b", "\\B", "(?=", "(?!", "(?<=", "(?<!")
    whole = None if any(token in config["pattern"] for token in context) else pattern
    while header := source.readline():
        request = json.loads(header)
        budget = request["budget"]
        data = source.read(sum(request["sizes"]))
        offset = 0
        matches: list[list[object]] = []
        for index, size in enumerate(request["sizes"]):
            text = data[offset:offset + size].decode("utf-8", errors="replace")
            offset += size
            if whole is not None and whole.search(text) is None:
                continue
            lines = breaks.split(text)
            if lines[-1] == "":
                lines.pop()
            for number, content in enumerate(lines, 1):
                if pattern.search(content):
                    matches.append([index, number, content])
                    if len(matches) >= budget:
                        break
            if len(matches) >= budget:
                break
        print(json.dumps({"matches": matches}), flush=True)


if __name__ == "__main__":
    main()
else:
    # The runtime supplies the exact bytes it validated, including during a
    # transactional reload. Ordinary imports take their initial file snapshot.
    _SOURCE_BYTES: bytes = globals().get("__source__", b"")
    if not _SOURCE_BYTES:
        _SOURCE_BYTES = Path(__file__).read_bytes()
