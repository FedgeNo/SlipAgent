"""Compact transcript mappings preserve copy records after row edits."""

from slipagent.rowindex import RowIndex
from slipagent.transcript import WrappedTranscript


def test_sparse_index_updates_removals_and_appends():
    index = RowIndex([(8, 80), (2, 20), (5, 50)])
    index[5] = 51
    assert [index.get(n) for n in (0, 2, 5, 8, 9)] == [None, 20, 51, 80, None]
    index.remove_rows(3, 7)
    assert index.get(2) == 20 and index.get(4) == 80
    assert index.get(5) is None and index.get(8) is None
    index[10] = 100
    index.remove_rows(0, 3)
    assert index.get(1) == 80 and index.get(7) == 100


def test_existing_transcript_copy_mapping_migrates_without_losing_text():
    flow = WrappedTranscript(20)
    try:
        flow.append_markdown("first\n\nsecond", first=True, final=True)
        before = [flow.copy_row(n).copy_text for n in range(len(flow.rows))]
        records = flow._copy_records()
        flow._copy_index = dict(zip(records.keys, records.values))
        assert [flow.copy_row(n).copy_text for n in range(len(flow.rows))] == before
        flow.remove_rows(0, 1)
        assert [flow.copy_row(n).copy_text for n in range(len(flow.rows))] == before[1:]
    finally:
        flow.close()
