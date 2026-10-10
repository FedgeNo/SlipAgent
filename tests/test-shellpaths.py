"""Shell tracking distinguishes literal file operands from other arguments."""

from pathlib import Path

import pytest

from slipagent.shellpaths import shell_file_references


@pytest.mark.parametrize(("command", "expected"), [
    ('cp -rv -- "source file" dest', [("source file", "read"), ("dest", "write")]),
    ('env X=1 timeout 3 cp -t dest a b', [("dest", "write"), ("a", "read"), ("b", "read")]),
    ('mv -- a b', [("a", "read/write"), ("b", "write")]),
    ('rg -n -g "*.py" -e "foo bar" src', [("src", "read")]),
    ('grep -f patterns -C 3 input', [("patterns", "read"), ("input", "read")]),
    ("sed -i.bak -e 's/a/b/' file", [("file", "read/write"), ("file.bak", "write")]),
    ('sudo -u someone nice -n 5 cat file', [("file", "read")]),
    ('head -n 10 file', [("file", "read")]),
    ('cc -I include -DNAME=1 -o build/app source.c', [("build/app", "write"), ("source.c", "read")]),
    ('dd if=input of=output bs=4096', [("input", "read"), ("output", "write")]),
    ('cat < input | tee output 2> error', [("input", "read"), ("error", "write"), ("output", "write")]),
    ('echo hello > output 2>&1', [("output", "write")]),
    ('cd sub && cat file', [("sub/file", "read")]),
    ('(cd sub; cat file); cat root', [("sub/file", "read"), ("root", "read")]),
    ('cat -- -filename', [("-filename", "read")]),
    ('printf "%s" "not a filename"', []),
    ('rm -rf -- old; mkdir -p new', [("old", "delete"), ("new", "write")]),
    ('find src tests -name "*.py" -type f', [("src", "read"), ("tests", "read")]),
    ('cat < "input>name"', [("input>name", "read")]),
    ('touch -r reference target', [("reference", "read"), ("target", "write")]),
    ('stat -c %s file', [("file", "read")]),
    ('diff -L label a b', [("a", "read"), ("b", "read")]),
    ('sort -S 1M -k 2 -o result data', [("result", "write"), ("data", "read")]),
    ("awk -v key=value '{print $0}' input extra=value", [("input", "read")]),
    ('find -L src -name "*.py"', [("src", "read")]),
])
def test_literal_command_references(tmp_path: Path, command, expected):
    result = shell_file_references(command, tmp_path)
    actual = [(str(Path(item["path"]).relative_to(tmp_path)), item["access"]) for item in result["references"]]
    assert sorted(actual) == sorted(expected)
    assert not result["incomplete"]


@pytest.mark.parametrize("command", [
    'cp "$SOURCE" dest', 'cp *.py dest', 'cat "prefix"*.py',
    'python -c "open(\'hidden\', \'w\')"', 'cat $(echo file)',
    'for file in a b; do cat "$file"; done', 'cat "unterminated',
    'grep --unknown value file',
    'cat <> data',  # This grammar version does not recognize read/write redirects.
])
def test_dynamic_or_unsupported_commands_are_explicitly_incomplete(tmp_path: Path, command):
    result = shell_file_references(command, tmp_path)
    assert result["incomplete"]
    assert result["limitations"]
    assert not result["references"]


def test_redirections_survive_unknown_command_and_do_not_execute(tmp_path: Path):
    result = shell_file_references('unknown > result', tmp_path)
    assert result["references"] == [{"path": str(tmp_path / "result"), "access": "write", "confidence": "inferred"}]
    assert result["incomplete"]
    assert not (tmp_path / "result").exists()


def test_unresolved_cd_suppresses_relative_paths_but_keeps_absolute(tmp_path: Path):
    result = shell_file_references('cd "$DIR"; cat relative /absolute', tmp_path)
    assert [item["path"] for item in result["references"]] == ["/absolute"]
    assert result["incomplete"]


def test_heredoc_is_data(tmp_path: Path):
    result = shell_file_references('cat <<\'EOF\' > output\nrm secret\nEOF', tmp_path)
    assert [item["path"] for item in result["references"]] == [str(tmp_path / "output")]


def test_executable_heredoc_is_incomplete(tmp_path: Path):
    result = shell_file_references('cat <<EOF\n$(touch hidden)\nEOF', tmp_path)
    assert result["incomplete"]
    assert not (tmp_path / "hidden").exists()


def test_uncertain_cd_branch_does_not_invent_subsequent_relative_paths(tmp_path: Path):
    result = shell_file_references('cd sub || cat fallback; cat later', tmp_path)
    assert not result["references"]
    assert result["incomplete"]


def test_source_can_change_working_directory(tmp_path: Path):
    result = shell_file_references('source script; cat later', tmp_path)
    assert not result["references"]
    assert result["incomplete"]


@pytest.mark.parametrize("command", ['VALUE=$(touch hidden)', 'VALUE=$(touch hidden) cat visible'])
def test_assignment_expansion_reports_untracked_effects(tmp_path: Path, command):
    result = shell_file_references(command, tmp_path)
    assert result["incomplete"]
    assert any("assignment" in reason for reason in result["limitations"])
    assert not (tmp_path / "hidden").exists()
