"""Git uses explicit workspace-checked worktrees, metadata, and literal paths.

Hooks, signing, external diff helpers, and active staging filters cannot run
through these tools; they would bypass the filesystem boundary.
"""

from __future__ import annotations

from ..prompts import load_prompt

import asyncio
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from ..workspace import Workspace, WorkspaceError
from .base import Tool, ToolResult
from .output import CommandArchive, CommandLog
from .shell import DEFAULT_TIMEOUT, MAX_OUTPUT_CHARS, MAX_TIMEOUT, capture_process, _truncate

REPO_PARAMETER = {
    "type": "string",
    "description": "",
}
PATHS_PARAMETER = {
    "type": "array",
    "items": {"type": "string"},
    "description": "",
}
TIMEOUT_PARAMETER = {
    "type": "number",
    "minimum": 0.1,
    "description": "",
}



@dataclass(slots=True)
class GitRepository:
    root: Path
    git_dir: Path


@dataclass(slots=True)
class GitOutput:
    command: list[str]
    code: int
    stdout: str
    stderr: str
    log: CommandLog | None = None

    def result(self) -> ToolResult:
        parts = [f"$ {shlex.join(self.command)}", f"exit code: {self.code}"]
        for label, value in (("stdout", self.stdout), ("stderr", self.stderr)):
            if value.strip():
                if len(value) > MAX_OUTPUT_CHARS:
                    value, _ = _truncate(value, MAX_OUTPUT_CHARS, notice="… [output truncated]", archived=self.log is not None)
                parts.append(f"--- {label} ---\n{value.rstrip()}")
        if not self.stdout.strip() and not self.stderr.strip() and self.code == 0:
            parts.append("No output (operation completed successfully).")
        content = "\n".join(parts)
        if self.log is not None:
            content += "\n" + self.log.notice()
        return ToolResult.error(content) if self.code else ToolResult.ok(content)


class GitTool(Tool):
    def __init__(self, workspace: Workspace, archive: CommandArchive | None = None) -> None:
        self.workspace = workspace
        self.archive = archive

    def _repository(self, repo: str) -> GitRepository:
        root = self.workspace.resolve(repo)
        if not root.is_dir():
            raise WorkspaceError(f"Repository directory does not exist: {repo!r}")
        while not (root / ".git").exists():
            if root == root.parent or (not self.workspace.access.danger and root == self.workspace.root):
                raise WorkspaceError(
                    "No Git repository found within the allowed directories at this path. "
                    "Parent repositories require danger mode; bare repositories are not supported."
                )
            root = root.parent
        marker = self.workspace.resolve(root / ".git")
        git_dir = marker
        if marker.is_file():
            value = marker.read_text(encoding="utf-8").strip()
            if not value.startswith("gitdir: "):
                raise WorkspaceError("Invalid .git file: expected a gitdir pointer.")
            git_dir = self.workspace.resolve(root / value[8:])
        if not git_dir.is_dir():
            raise WorkspaceError("Git metadata directory does not exist.")
        common = git_dir
        common_file = self.workspace.resolve(git_dir / "commondir")
        if common_file.is_file():
            common = self.workspace.resolve(git_dir / common_file.read_text().strip())
        for directory in {git_dir, common}:
            self._metadata(directory)
        return GitRepository(root, git_dir)

    def _metadata(self, directory: Path) -> None:
        # Git writes refs, objects, logs, and the index independently of its cwd.
        # Check metadata links too, including links nested beneath refs/objects.
        def walk_error(error: OSError) -> None:
            raise error

        pending = [directory]
        visited: set[Path] = set()
        while pending:
            for current, dirs, files in os.walk(pending.pop(), onerror=walk_error, followlinks=True):
                resolved = self.workspace.resolve(current)
                if resolved in visited:
                    dirs.clear()
                    continue
                visited.add(resolved)
                for name in dirs + files:
                    entry = Path(current) / name
                    if entry.is_symlink():
                        self.workspace.resolve(entry)
                # Alternate object stores can themselves link to further stores.
                if resolved.name == "info" and "alternates" in files:
                    alternates = self.workspace.resolve(resolved / "alternates")
                    for value in alternates.read_text().splitlines():
                        if value.strip():
                            pending.append(self.workspace.resolve(resolved.parent / value))

    def _paths(self, repository: GitRepository, paths: list[str]) -> list[str]:
        if not paths:
            raise WorkspaceError("paths must contain at least one file or directory.")
        result = []
        for value in paths:
            self.workspace.resolve(value)
            # Preserve the link/deleted file's lexical name instead of staging its target.
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = self.workspace.root / path
            path = Path(os.path.abspath(path))
            try:
                relative = path.relative_to(repository.root).as_posix()
            except ValueError as exc:
                raise WorkspaceError(f"Path is outside this repository: {value!r}") from exc
            if relative == ".git" or relative.startswith(".git/"):
                raise WorkspaceError("Git metadata cannot be used as a file path.")
            result.append(relative)
        return result

    async def _execute(
        self, repository: GitRepository, args: list[str], timeout: float,
    ) -> GitOutput:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        # Author/committer identity overrides do not redirect repository operations.
        for role in ("AUTHOR", "COMMITTER"):
            for field in ("NAME", "EMAIL", "DATE"):
                key = f"GIT_{role}_{field}"
                if key in os.environ:
                    env[key] = os.environ[key]
        env["GIT_OPTIONAL_LOCKS"] = "0"
        # Partial clones otherwise fetch missing objects during local reads.
        env["GIT_NO_LAZY_FETCH"] = "1"
        env["GIT_ALLOW_PROTOCOL"] = ""
        command = [
            "git", "--no-pager", "--literal-pathspecs",
            f"--git-dir={repository.git_dir}", f"--work-tree={repository.root}",
        ]
        for setting in (
            f"core.hooksPath={os.devnull}", "core.fsmonitor=false",
            "core.quotePath=false", "color.ui=false", "gc.auto=0",
            "maintenance.auto=false", "commit.gpgSign=false", "log.showSignature=false",
        ):
            command.extend(["-c", setting])
        command.extend(args)
        log = self.archive.start(shlex.join(["git", *args])) if self.archive is not None else None
        try:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=repository.root, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            if log is not None:
                log.finish(None, False)
            raise WorkspaceError("Git is not installed or is not available on PATH.") from exc
        except OSError as exc:
            if log is not None:
                log.finish(None, False)
            raise WorkspaceError(f"Could not start Git: {exc}") from exc
        limit = min(timeout, MAX_TIMEOUT)
        stdout, stderr, timed_out = await capture_process(process, limit, log=log)
        output = GitOutput(["git", *args], process.returncode or 0, stdout, stderr, log)
        if timed_out:
            raise WorkspaceError(
                f"Git timed out after {limit:g}s and was killed. Effects may be partial. "
                "Check git_status before retrying.\n" + output.result().content
            )
        return output

    @staticmethod
    def _complete(output: GitOutput) -> str:
        if output.code:
            raise WorkspaceError(output.result().content)
        if len(output.stdout) > MAX_OUTPUT_CHARS:
            raise WorkspaceError("Git validation output is too large; select fewer paths.")
        return output.stdout


class GitStatusTool(GitTool):
    parameter_prompts = 'tools/git-status-parameters.json'
    name = "git_status"
    description_prompt = 'tools/git-status.txt'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {"repo": REPO_PARAMETER, "timeout": TIMEOUT_PARAMETER},
    }

    async def run(self, repo: str = ".", timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        repository = self._repository(repo)
        output = await self._execute(
            repository, ["status", "--short", "--branch", "--ignore-submodules=all"], timeout,
        )
        return output.result()


class GitDiffTool(GitTool):
    parameter_prompts = 'tools/git-diff-parameters.json'
    name = "git_diff"
    description_prompt = 'tools/git-diff.txt'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "repo": REPO_PARAMETER, "timeout": TIMEOUT_PARAMETER,
            "staged": {"type": "boolean", "description": ""},
            "paths": PATHS_PARAMETER,
        },
    }

    async def run(
        self, repo: str = ".", staged: bool = False, paths: list[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> ToolResult:
        repository = self._repository(repo)
        args = ["diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules=all"]
        if staged:
            args.append("--cached")
        args.append("--")
        if paths is not None:
            args.extend(self._paths(repository, paths))
        return (await self._execute(repository, args, timeout)).result()


class GitLogTool(GitTool):
    parameter_prompts = 'tools/git-log-parameters.json'
    name = "git_log"
    description_prompt = 'tools/git-log.txt'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "repo": REPO_PARAMETER, "timeout": TIMEOUT_PARAMETER,
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    }

    async def run(self, repo: str = ".", limit: int = 10, timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        repository = self._repository(repo)
        output = await self._execute(
            repository,
            ["log", f"-n{limit}", "--format=%h %s", "--no-ext-diff", "--no-textconv", "--no-show-signature"],
            timeout,
        )
        return output.result()


class GitAddTool(GitTool):
    parameter_prompts = 'tools/git-add-parameters.json'
    name = "git_add"
    description_prompt = 'tools/git-add.txt'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "repo": REPO_PARAMETER, "paths": PATHS_PARAMETER, "timeout": TIMEOUT_PARAMETER,
        },
        "required": ["paths"],
    }

    async def run(self, paths: list[str], repo: str = ".", timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        repository = self._repository(repo)
        relative = self._paths(repository, paths)
        listing = await self._execute(
            repository,
            ["ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *relative],
            timeout,
        )
        files = list(dict.fromkeys(self._complete(listing).rstrip("\0").split("\0")))
        files = [path for path in files if path]
        for path in files:
            candidate = self.workspace.resolve(repository.root / path)
            if candidate.is_dir() and (candidate / ".git").exists():
                self._repository(str(candidate))
        if files:
            attributes = await self._execute(
                repository, ["check-attr", "-z", "filter", "--", *files], timeout,
            )
            fields = self._complete(attributes).rstrip("\0").split("\0")
            for driver in set(fields[2::3]):
                for kind in ("clean", "process"):
                    config = await self._execute(
                        repository, ["config", "--get", f"filter.{driver}.{kind}"], timeout,
                    )
                    if config.code not in (0, 1):
                        return config.result()
                    if config.stdout.strip():
                        return ToolResult.error(
                            f"Cannot stage files using filter {driver!r}: its {kind} helper "
                            "can access paths outside the workspace."
                        )
        return (await self._execute(repository, ["add", "--", *relative], timeout)).result()


class GitCommitTool(GitTool):
    parameter_prompts = 'tools/git-commit-parameters.json'
    name = "git_commit"
    description_prompt = 'tools/git-commit.txt'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "repo": REPO_PARAMETER, "timeout": TIMEOUT_PARAMETER,
            "message": {"type": "string", "description": ""},
        },
        "required": ["message"],
    }

    async def run(self, message: str, repo: str = ".", timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        if not message.strip():
            return ToolResult.error("Commit message must not be blank.")
        repository = self._repository(repo)
        return (await self._execute(repository, ["commit", "-m", message], timeout)).result()


def git_tools(workspace: Workspace, archive: CommandArchive | None = None) -> list[Tool]:
    return [
        GitStatusTool(workspace, archive), GitDiffTool(workspace, archive), GitLogTool(workspace, archive),
        GitAddTool(workspace, archive), GitCommitTool(workspace, archive),
    ]
