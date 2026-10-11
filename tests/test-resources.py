"""Typed adapters preserve identity; shutdown follows resource dependencies."""

from types import SimpleNamespace

from slipagent.resources import SessionResources
from slipagent.tools.base import ToolRegistry
from slipagent.work import SessionWork, WorkCoordinator


def test_typed_references_and_compatibility_mapping_share_one_value():
    values = {}
    resources = SessionResources(values)
    jobs = object()
    resources.jobs = jobs
    assert values["command_jobs"] is jobs
    replacement = object()
    values["command_jobs"] = replacement
    assert resources.jobs is replacement
    resources.jobs = None
    assert "command_jobs" not in values
    assert resources.jobs is None
    work = SessionWork(values)
    work.deferred_commands = ["/planning off"]
    values["deferred_commands"].append("/planning on")
    assert work.deferred_commands == ["/planning off", "/planning on"]


async def test_shutdown_dependency_order_is_independent_of_map_insertion():
    closed = []
    class Owner:
        def __init__(self, name):
            self.name = name
        async def aclose(self):
            closed.append(self.name)
    services = {name: Owner(name) for name in (
        "command_archive", "command_jobs", "artifact_store", "user_edits",
    )}
    registry = ToolRegistry(services=services)
    await registry.aclose()
    await registry.aclose()
    assert closed == ["user_edits", "command_jobs", "artifact_store", "command_archive"]


async def test_borrowed_registry_does_not_close_session_owners():
    closed = []
    class Owner:
        async def aclose(self):
            closed.append(True)
    registry = ToolRegistry(services={"command_jobs": Owner()}, owns_services=False)
    await registry.aclose()
    assert closed == []


def test_work_hold_releases_the_frame_it_acquired_once():
    first, second = SimpleNamespace(busy=0), SimpleNamespace(busy=0)
    current = [first]
    work = WorkCoordinator(lambda: current[0])
    work.hold()
    work.hold()
    assert first.busy == 1
    assert not work.released.is_set()
    current[0] = second
    work.release()
    work.release()
    assert first.busy == second.busy == 0
    assert work.released.is_set()
