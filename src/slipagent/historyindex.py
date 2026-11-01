"""Additive wire sizes for immutable history representations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .data_text import RenderedJSON
from .records import linked_tool_results
from .sources import FileText, json_chunks

RESULT_FIELDS = ('result', 'error', 'unclassified_result', 'result_excerpt', 'error_excerpt', 'unclassified_result_excerpt')


def number_bytes(count: int) -> int:
    """Digits used to number consecutive result blocks from one through count."""
    total, first, width = 0, 1, 1
    while first <= count:
        total += (min(count + 1, first * 10) - first) * width
        first *= 10
        width += 1
    return total


@dataclass(frozen=True)
class HistorySize:
    # Remove local numbering and the nonempty-list wrapper so records add.
    content_bytes: int
    blocks: int
    result_bytes: tuple[int, ...]


class HistorySizeIndex:
    """Cache each representation once for one immutable source revision.

    The owner discards entries when originals, summaries, or metadata change.
    Disk sidecars are derived diagnostics, rebuilt on resume, never replay data.
    """

    def __init__(self, measure: Callable[[list[Any]], int]) -> None:
        self.measure = measure
        self.empty = measure([])
        probe = {'record_type': 'history_step'}
        self.join_bytes = 2 * (measure([probe]) - self.empty) - (measure([probe, probe]) - self.empty)
        self.entries: dict[tuple[Any, ...], HistorySize] = {}
        self.dirty = False

    def measure_record(self, record: dict[str, Any]) -> HistorySize:
        results = [call[field] for call in linked_tool_results(record).get('tool_calls', [])
                   for field in RESULT_FIELDS if field in call]
        sizes = tuple(sum(len(chunk.encode('utf-8')) for chunk in json_chunks(
            result if isinstance(result, (str, FileText)) else RenderedJSON(result), indent=None)) - 2
            for result in results)
        delta = self.measure([record]) - self.empty
        return HistorySize(delta - self.join_bytes - 2 * number_bytes(len(results)), len(results), sizes)

    def get(self, key: tuple[Any, ...], record: dict[str, Any]) -> HistorySize:
        if key not in self.entries:
            self.remember(key, self.measure_record(record))
        return self.entries[key]

    def remember(self, key: tuple[Any, ...], size: HistorySize) -> None:
        if self.entries.get(key) != size:
            self.entries[key] = size
            self.dirty = True

    def total_bytes(self, content: int, blocks: int, records: int) -> int:
        return content + (self.join_bytes if records else 0) + 2 * number_bytes(blocks)

    def record_bytes(self, size: HistorySize) -> int:
        return self.total_bytes(size.content_bytes, size.blocks, 1)

    def snapshot(self) -> dict[str, Any]:
        return {'version': 1, 'join_bytes': self.join_bytes,
                'representations': [{'form': list(key), **asdict(size)} for key, size in self.entries.items()]}
