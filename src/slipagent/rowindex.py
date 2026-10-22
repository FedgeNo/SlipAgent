"""Compact sparse row-to-record indexes for disk-backed text."""

from array import array
from bisect import bisect_left
from collections.abc import Iterable


class RowIndex:
    def __init__(self, entries: Iterable[tuple[int, int]] = ()) -> None:
        self.keys = array("Q")
        self.values = array("Q")
        for key, value in entries:
            self[key] = value

    def __setitem__(self, key: int, value: int) -> None:
        if not self.keys or key > self.keys[-1]:
            self.keys.append(key)
            self.values.append(value)
            return
        index = bisect_left(self.keys, key)
        if self.keys[index] == key:
            self.values[index] = value
        else:
            self.keys.insert(index, key)
            self.values.insert(index, value)

    def get(self, key: int) -> int | None:
        index = bisect_left(self.keys, key)
        return self.values[index] if index < len(self.keys) and self.keys[index] == key else None

    def items(self) -> Iterable[tuple[int, int]]:
        return zip(self.keys, self.values)

    def prepend(self, other: "RowIndex", rows: int, records: int) -> None:
        """Place another index's entries first; this index's rows and records shift past them."""
        self.keys = other.keys + array("Q", (key + rows for key in self.keys))
        self.values = other.values + array("Q", (value + records for value in self.values))

    def remove_rows(self, start: int, stop: int) -> None:
        left, right = bisect_left(self.keys, start), bisect_left(self.keys, stop)
        del self.keys[left:right]
        del self.values[left:right]
        for index in range(left, len(self.keys)):
            self.keys[index] -= stop - start
