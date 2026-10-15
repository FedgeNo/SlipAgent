"""Typed session-resource ownership with a raw mapping compatibility adapter.

This contract stays in the stable frame. Resource implementations can reload
while tools and session code retain their owners.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

if TYPE_CHECKING:
    from .artifacts import ArtifactStore
    from .budget import ContextBudget
    from .checkpoints import FileCheckpoints
    from .compaction import StepCompactor
    from .diagnostics import RequestDiagnostics
    from .environment import ProjectEnvironment
    from .instructions import ProjectInstructions
    from .jobs import CommandJobs
    from .lsp import LanguageServers
    from .progress import LoopGuard
    from .repomap import RepositoryMap
    from .sessions import SessionJournal
    from .tools.output import CommandArchive
    from .useredits import UserEdits
    from .workspace import Workspace
    from .turnstore import TurnStore

T = TypeVar("T")


class ResourceRef(Generic[T]):
    """Expose a typed owner without duplicating the compatibility-map value."""

    def __init__(self, key: str) -> None:
        self.key = key

    @overload
    def __get__(self, instance: None, owner: type[ResourceOwner]) -> ResourceRef[T]: ...
    @overload
    def __get__(self, instance: ResourceOwner, owner: type[ResourceOwner] | None = None) -> T | None: ...
    def __get__(self, instance: ResourceOwner | None, owner: type[ResourceOwner] | None = None) -> T | None | ResourceRef[T]:
        return self if instance is None else cast(T | None, instance.values.get(self.key))

    def __set__(self, instance: ResourceOwner, value: T | None) -> None:
        if value is None:
            instance.values.pop(self.key, None)
        else:
            instance.values[self.key] = value


class ResourceOwner:
    """One authoritative map; named references adapt existing integrations."""

    shutdown_keys: tuple[str, ...] = ()

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def close_order(self, tools: Iterable[object]) -> list[object]:
        known = set(self.shutdown_keys)
        extras = [value for key, value in reversed(list(self.values.items())) if key not in known]
        return [*tools, *extras, *(self.values[key] for key in self.shutdown_keys if key in self.values)]


class SessionResources(ResourceOwner):
    workspace = ResourceRef["Workspace"]("workspace")
    environment = ResourceRef["ProjectEnvironment"]("project_environment")
    instructions = ResourceRef["ProjectInstructions"]("project_instructions")
    archive = ResourceRef["CommandArchive"]("command_archive")
    artifacts = ResourceRef["ArtifactStore"]("artifact_store")
    jobs = ResourceRef["CommandJobs"]("command_jobs")
    diagnostics = ResourceRef["RequestDiagnostics"]("request_diagnostics")
    servers = ResourceRef["LanguageServers"]("language_servers")
    journal = ResourceRef["SessionJournal"]("session_journal")
    checkpoints = ResourceRef["FileCheckpoints"]("file_checkpoints")
    compactor = ResourceRef["StepCompactor"]("step_compactor")
    loop_guard = ResourceRef["LoopGuard"]("loop_guard")
    budget = ResourceRef["ContextBudget"]("context_budget")
    repository_map = ResourceRef["RepositoryMap"]("repository_map")
    user_edits = ResourceRef["UserEdits"]("user_edits")
    turns = ResourceRef["TurnStore"]("turn_store")

    # Editing and producers settle before durable stores they can still write.
    shutdown_keys = (
        "user_edits", "step_compactor", "command_jobs", "language_servers",
        "request_diagnostics", "session_journal", "file_checkpoints",
        "artifact_store", "command_archive", "turn_store", "project_instructions",
        "project_environment", "workspace",
    )
