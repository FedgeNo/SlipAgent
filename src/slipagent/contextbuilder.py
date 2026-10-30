"""Build model context from explicit settings, history, and owned session resources."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .budget import ContextBudget
from .capabilities import RequestProfile, model_context_limit
from .context import ContextError, ConversationHistory, tool_history_as_text
from .data_text import render_data
from .diffcomments import current_references
from .environment import harness_python_environment, session_environment
from .mcp import MCPTool
from .prompts import PromptSections, load_prompt, model_prompt_profile
from .protocol import model_response_format
from .repomap import RepositoryMap
from .tools.base import ToolRegistry
from .tools.blocking import run_blocking
from .sources import materialize
from .tools.discovery import CORE_TOOLS
from .turnstore import resident_content
from .types import Message, ToolSpec


@dataclass(frozen=True)
class ContextSettings:
    model: str
    planning: bool
    context_steps: int
    overthinking: bool
    reasoning_history_steps: int
    filtered_thoughts: bool
    max_tokens: int | None


@dataclass
class ContextBuilder:
    registry: ToolRegistry
    history: ConversationHistory
    messages: list[Message]
    budget: ContextBudget
    settings: ContextSettings

    async def build(
        self, specs: list[ToolSpec], length: int, native_tools: bool, *,
        capabilities: RequestProfile | None = None, preview: bool = False,
        repair: str = "", budget_fraction: float | None = None,
    ) -> list[Message]:
        model_profile = model_prompt_profile(self.settings.model)
        length = model_context_limit(self.settings.model, capabilities, length)
        sections = PromptSections()
        if self.settings.planning:
            # Read archived envelopes only; the matching plan alone is loaded.
            saved_plan = next(((prior.id, materialize(observation['content']))
                         for prior in reversed(self.history.steps)
                         for message in reversed(prior.messages)
                         if message.role == 'tool'
                         and isinstance(observation := resident_content(message), dict)
                         and observation.get('tool') == 'update_plan'
                         and observation.get('status') == 'success'), None)
            if saved_plan is not None:
                plan_record = load_prompt('working-plan-record.md', step_id=saved_plan[0],
                                          plan=render_data(saved_plan[1]))
                sections.add('plan', 'Working Plan', load_prompt('working-plan.md',
                             plan_record=plan_record.strip()), 45)
        workspace = self.registry.resources.workspace
        if workspace is not None:
            access = load_prompt('workspace-access-off.md')
            if workspace.access.danger:
                access = (load_prompt('workspace-access-on.md'))
                package = Path(__file__).resolve().parent
                harness_location = package
                if package.parent.name == "src" and (package.parent.parent / "pyproject.toml").is_file():
                    harness_location = package.parent.parent
                sections.add("installation", "SlipAgent Installation",
                             load_prompt('harness-installation.md', harness_location=harness_location),
                             16, owner="workspace")
            sections.add("access", "Workspace Access",
                         load_prompt('workspace-access.md', access=access.strip(), workspace=workspace.root,
                                     temporary_directory=workspace.temporary_directory),
                         15, owner="workspace")
        instructions = self.registry.resources.instructions
        if instructions is not None:
            snapshot = instructions.snapshot()
            sections.add("project", "Project Instructions (Current)", instructions.render(snapshot), 20, owner="instructions")
            if not preview:
                self.registry.services["instruction_snapshot"] = snapshot
        environment = self.registry.resources.environment
        session_info = await run_blocking(session_environment)
        session_info["model"] = self.settings.model
        if workspace is not None:
            session_info["workspace"] = str(workspace.root)
            session_info["command_working_directory"] = str(workspace.root)
        journal = self.registry.resources.journal
        if journal is not None:
            session_info["session_id"] = journal.session_id
        sections.add("session", "Session Environment", load_prompt(
            "session-environment.md", environment=render_data(session_info)), 28, owner="environment")
        sections.add("harness_python", "Harness Python and Mathematics",
                     load_prompt("harness-python.md", environment=render_data(
                         await run_blocking(harness_python_environment))), 29, owner="environment")
        if environment is not None:
            sections.add("environment", "Project Python Environment", load_prompt('environment-snapshot.md', environment=render_data(await environment.snapshot())), 30, owner="environment")
        response_format = model_response_format(self.settings.model, capabilities, native_tools=native_tools)
        schema = response_format if response_format is not None and response_format["type"] == "json_schema" else None
        glm_xml = model_profile.get("tool_channel") == "glm_xml"
        # GLM's reply and call format heads the context instead; see ConversationHistory.view.
        if (not native_tools or schema is not None) and not glm_xml:
            sections.add("schema", "Response Schema",
                load_prompt('response-json.md') + '\n\n'
                + (load_prompt('response-native-json.md')
                   if native_tools else load_prompt('response-embedded-json.md')),
                0, owner="protocol", dynamic=False,
            )
        elif native_tools:
            sections.add("schema", "Reply Format",
                         load_prompt('response-native-text.md'),
                         0, owner="protocol", dynamic=False)
        for index, filename in enumerate(model_profile.get("sections", [])):
            sections.add(f"model_{index}", filename, load_prompt(filename),
                         5, owner="model", dynamic=False)
        if not glm_xml:  # Suggestions need a JSON reply object, which GLM's text replies avoid.
            sections.add("suggestions", "Optional Suggested Next Steps", load_prompt('response-suggestions.md'),
                         1, owner="protocol", dynamic=False)
            journal = self.registry.resources.journal
            # Dynamic, so the cached static prefix stays intact once the session is named.
            if journal is not None and not journal.named:
                sections.add("session_title", "Session Title", load_prompt('response-session-title.md'),
                             1, owner="protocol")
        if glm_xml:
            # GLM's own template lists one JSON signature per line inside <tools>.
            sections.add("tools", "Tools", load_prompt('tool-definitions-glm-xml.md', definitions="\n".join(
                json.dumps({"name": spec.name, "description": spec.description, "parameters": spec.parameters},
                           ensure_ascii=False) for spec in specs)), 10, owner="tools", dynamic=False)
        else:
            sections.add("tools", "Available Tool Definitions", load_prompt('tool-definitions.md',
                         core_definitions=json.dumps({spec.name: {"description": spec.description, "parameters": spec.parameters}
                                                      for spec in specs if spec.name in CORE_TOOLS}, ensure_ascii=False, indent=2),
                         loaded_definitions=json.dumps({spec.name: {"description": spec.description, "parameters": spec.parameters}
                                                        for spec in specs if spec.name not in CORE_TOOLS}, ensure_ascii=False, indent=2)),
                         10, owner="tools", dynamic=False)
        catalog = self.registry.services.get("tool_catalog")
        if catalog is not None and catalog.deferred():
            sections.add("tool_catalog", "Additional Tool Catalog",
                         load_prompt('tool-discovery.md', catalog="\n".join(
                             "| `" + entry["name"].replace("|", "\\|").replace("`", "\\`") + "` | "
                             + entry["purpose"].replace("|", "\\|") + " |" for entry in catalog.deferred())),
                         11, owner="tools", dynamic=False)
        available_names = {spec.name for spec in specs}
        if catalog is not None:
            # Server guidance must precede tool loading; the catalog retains the allowlist.
            available_names.update(catalog.available)
        servers = {tool.client.spec.name: tool.client.instructions for tool in self.registry.tools
                   if isinstance(tool, MCPTool) and tool.name in available_names
                   and tool.client.connected and tool.client.instructions}
        if servers:
            sections.add("mcp", "Connected MCP Server Guidance",
                load_prompt('mcp-guidance.md', servers=render_data(servers)),
                40, owner="mcp", dynamic=False,
            )
        references = current_references(self.messages)
        if references:
            sections.add("diff_comments", "Current Diff Comments", load_prompt("diff-comment.md",
                         references=json.dumps(references, ensure_ascii=False, indent=2)), 46, owner="user")
        jobs = self.registry.resources.jobs
        if "monitor_command" in available_names and jobs is not None:
            monitors = jobs.monitors.snapshots()
            if not monitors:
                sections.add("monitor_setup", "Command Monitoring Setup", load_prompt("monitor-setup.md"), 47)
            active = [monitor for monitor in monitors if monitor["state"] == "active"]
            ready = [monitor for monitor in monitors if monitor["state"] == "ready"]
            if active:
                sections.add("monitor_active", "Active Command Monitors", load_prompt("monitor-active.md",
                             monitors=render_data(active)), 48)
            if ready:
                sections.add("monitor_results", "Command Monitor Results", load_prompt("monitor-results.md",
                             monitors=render_data(ready)), 49)
        state_notes = {key: value for key, value in self.registry.context_notes.items() if key not in {"user_edits", "progress"}}
        if not preview:
            self.registry.services["background_snapshot"] = copy.deepcopy(
                state_notes.get("background_commands", {}).get("completed", []))
        if state_notes:
            sections.add("state", "Current Harness State", load_prompt('harness-state.md', state=render_data(state_notes)), 50)
        extra_instructions = sections.render()
        budget = copy.copy(self.budget) if preview else self.budget
        token_scale = budget.select(self.settings.model, capabilities)
        if budget_fraction is None:
            budget_fraction = budget.fraction
        history = replace(self.history, steps=list(self.history.steps), task=copy.deepcopy(self.history.task)) if preview else self.history
        repository_map = ""
        if environment is not None:
            if self.registry.resources.repository_map is None:
                self.registry.resources.repository_map = RepositoryMap(environment.workspace)
            query = next((m.content or "" for m in reversed(self.messages) if m.role == "user"), "")
            repository_map = await self.registry.resources.repository_map.snapshot(query, min(128_000, int(length * budget_fraction * .03 * 3)))
        view_options: dict[str, Any] = dict(
            user_edits=[{**entry, "edit_id": key} for key, entry in self.registry.context_notes.get("user_edits", {}).items()
                        if not entry.get("delivered")],
            user_corrections="\n\n".join(part.strip() for part in (
                self.registry.context_notes.get("progress", ""), repair,
            ) if part.strip()),
            keep_steps=self.settings.context_steps,
            overthinking=self.settings.overthinking,
            reasoning_history_steps=self.settings.reasoning_history_steps,
            filtered_thoughts=self.settings.filtered_thoughts,
            context_length=int(length * budget_fraction),
            max_output=self.settings.max_tokens or min(8192, max(256, length // 8)),
            native_tools=native_tools,
            text_tool_history=not native_tools,
            glm_xml=glm_xml,
            schema=schema,
            token_scale=token_scale,
        )
        try:
            supplement = "\n\n".join(part.strip() for part in (extra_instructions, repository_map) if part.strip())
            context = await history.view(self.messages, specs, extra_instructions=supplement, **view_options)
        except ContextError:
            if not repository_map:
                raise
            # Orientation is optional; it must never displace the current task.
            context = await history.view(self.messages, specs, extra_instructions=extra_instructions, **view_options)
        if not native_tools:
            context = tool_history_as_text(context, glm_xml=glm_xml)
        return context
