"""Standalone regex worker; only the parent opens workspace files.

The parent launches a snapshot of this source, so rejected on-disk edits cannot
change an already active tool. Keep this worker independent of package imports.
Each input line is a JSON object; each output line is its JSON result.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def worker_source() -> bytes:
    return _SOURCE_BYTES


def main() -> None:
    config = json.loads(sys.stdin.readline())
    try:
        pattern = re.compile(config["pattern"], re.IGNORECASE if config["ignore_case"] else 0)
    except re.error as exc:
        print(json.dumps({"error": f"Invalid regular expression: {exc}"}), flush=True)
        return
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        matches = []
        for number, text in enumerate(request["text"].splitlines(), 1):
            if pattern.search(text):
                matches.append([number, text])
                if len(matches) >= request["budget"]:
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
