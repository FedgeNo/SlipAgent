"""Discover and activate portable skill instructions through session-owned state."""

from __future__ import annotations

from typing import Any

from .. import skills
from ..workspace import Workspace
from .base import Tool, ToolResult
from .blocking import run_blocking


class UseSkillTool(Tool):
    name = 'use_skill'
    description_prompt = 'tools/use-skill.md'
    parameter_prompts = 'tools/use-skill-parameters.json'
    parameters = {
        'type': 'object',
        'properties': {
            'action': {'type': 'string', 'enum': ['list', 'load', 'unload']},
            'selector': {'type': 'string'}, 'arguments': {'type': 'string'},
            'query': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
        },
        'required': ['action'],
        'additionalProperties': False,
    }

    def __init__(self, workspace: Workspace, services: dict[str, Any]) -> None:
        self.workspace = workspace
        self.services = services

    async def run(self, action: str, selector: str = '', arguments: str = '', query: str = '',
                  offset: int = 0, limit: int = 20) -> ToolResult:
        try:
            if action == 'list':
                packages, errors = await run_blocking(skills.catalog, self.workspace)
                packages = [s for s in packages if s.enabled and not s.metadata.get('disable-model-invocation', False)
                            and query.casefold() in (s.name + ' ' + s.description).casefold()]
                rows = [{'name': s.name, 'description': s.description, 'path': str(s.path), 'scope': s.scope}
                        for s in packages[offset:offset + limit]]
                return ToolResult.ok({'skills': rows, 'next_offset': offset + limit if offset + limit < len(packages) else None,
                                      'errors': errors})
            if action == 'unload':
                snapshots = [s for s in skills.active(self.services) if selector and selector not in (s['name'], s['path'])]
                skills.restore_active(self.services, snapshots)
            else:
                if not selector:
                    return ToolResult.error('Choose a skill name or SKILL.md path.')
                journal = self.services.get('session_journal')
                snapshot = await run_blocking(skills.select, self.workspace, selector, arguments=arguments,
                                              implicit=True, session_id=journal.session_id if journal is not None else '')
                catalog = self.services.get('tool_catalog')
                available = list(catalog.available) if catalog is not None else []
                skills.required_tools(snapshot, available)
                skills.set_active(self.services, snapshot)
            return ToolResult.ok({'active': [{'name': s['name'], 'path': s['path']} for s in skills.active(self.services)]})
        except (OSError, ValueError) as exc:
            return ToolResult.error(str(exc))
