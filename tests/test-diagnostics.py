import gzip
import json
import os

from slipagent.diagnostics import RequestDiagnostics


def test_exact_request_deduplication_and_failed_attempt_retention(tmp_path):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path / "requests")
    request = json.dumps({"messages": [{"role": "user", "content": "exact 日本語"}]}, ensure_ascii=False)
    first = archive.begin(request, step=1, post=1)
    archive.finish(first, "request_error", detail="Disconnected", response="partial thoughts")
    second = archive.begin(request, step=1, post=1)
    archive.finish(second, "accepted", response="done")
    files = list(archive.directory.glob("*.gz"))
    assert len(files) == 1 and gzip.decompress(files[0].read_bytes()).decode() == request
    assert [row["outcome"] for row in archive.listing()] == ["request_error", "accepted"]
    assert "partial thoughts" in archive.read(first)
    if os.name == "posix":
        assert files[0].stat().st_mode & 0o777 == 0o600
    directory = archive.directory
    archive.clear()
    assert files[0].exists() and directory.exists()


def test_quota_preserves_existing_requests_and_reports_failure(tmp_path):
    archive = RequestDiagnostics(1000)
    archive.use_directory(tmp_path)
    first = archive.begin("first", step=1, post=1)
    assert first == 1
    archive.finish(first, "accepted", response="x" * 4000)
    assert "quota" in archive.error
    assert archive.begin("later", step=2, post=2) is None
    assert "first" in archive.read(first)
