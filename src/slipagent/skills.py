"""Explicit skill discovery and pinned instruction snapshots; no code execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import yaml

from .config import data_directory
from .fileio import open_regular, read_regular
from .prompts import load_prompt
from .workspace import Workspace

MAX_SKILL_BYTES = 256 * 1024
MAX_METADATA_BYTES = 16 * 1024


class SkillError(ValueError):
    """A package cannot be discovered or explicitly selected."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    enabled: bool
    scope: str


def _header(source: BinaryIO) -> dict[str, Any]:
    if source.readline(5) not in (b'---\n', b'---\r\n'):
        raise SkillError('SKILL.md must begin with YAML front matter.')
    lines: list[bytes] = []
    size = 0
    while size <= MAX_METADATA_BYTES:
        line = source.readline(MAX_METADATA_BYTES + 1)
        size += len(line)
        if size > MAX_METADATA_BYTES:
            break
        if line.strip() == b'---':
            try:
                metadata = yaml.safe_load(b''.join(lines).decode('utf-8'))
            except (yaml.YAMLError, UnicodeError) as exc:
                raise SkillError(f'Invalid skill metadata: {exc}') from exc
            if not isinstance(metadata, dict):
                raise SkillError('Skill metadata must be a mapping.')
            for key, limit in (('name', 128), ('description', 2048)):
                value = metadata.get(key)
                if (not isinstance(value, str) or not value.strip() or len(value) > limit
                        or any(ord(char) < 32 and char not in '\n\r\t' for char in value)):
                    raise SkillError(f'Skill {key} must be nonempty text, at most {limit} characters.')
            if '\n' in metadata['name'] or '\r' in metadata['name']:
                raise SkillError('Skill name must fit on one line.')
            if type(metadata.get('enabled', True)) is not bool:
                raise SkillError('Skill enabled must be a boolean.')
            return metadata
        if not line:
            break
        lines.append(line)
    raise SkillError('Skill front matter is missing its closing delimiter or exceeds 16384 bytes.')


def _skill(path: Path, scope: str) -> Skill:
    with open_regular(path) as source:
        metadata = _header(source)
    return Skill(metadata['name'].strip(), ' '.join(metadata['description'].split()),
                 path, metadata.get('enabled', True), scope)


def skill_roots(workspace: Workspace) -> list[tuple[str, Path]]:
    return [('project', workspace.root / '.slipagent' / 'skills'), ('user', data_directory() / 'skills')]


def catalog(workspace: Workspace) -> tuple[list[Skill], list[str]]:
    found = []
    errors = []
    for scope, root in skill_roots(workspace):
        try:
            children = sorted(root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f'{root}: {exc}')
            continue
        for child in children:
            # Discovery never follows linked packages into unrelated locations.
            path = child / 'SKILL.md'
            try:
                if child.is_symlink() or not child.is_dir():
                    continue
                if not path.exists() and not path.is_symlink():
                    continue
                found.append(_skill(path, scope))
            except (OSError, ValueError) as exc:
                errors.append(f'{path}: {exc}')
    return found, errors


def select(workspace: Workspace, selector: str) -> dict[str, str]:
    packages, _ = catalog(workspace)
    matches = [skill for skill in packages if skill.name == selector]
    if matches:
        project = [skill for skill in matches if skill.scope == 'project']
        matches = project or matches
        if len(matches) != 1:
            raise SkillError('Ambiguous skill name; select a SKILL.md path: ' + ', '.join(str(s.path) for s in matches))
        selected = matches[0]
    else:
        path = Path(selector).expanduser()
        if not path.is_absolute():
            path = workspace.root / path
        if path.is_dir():
            path = path / 'SKILL.md'
        # Catalogued user packages are explicitly trusted instruction sources.
        # Other paths use the ordinary workspace access policy.
        known = next((skill for skill in packages if skill.path == path.absolute()), None)
        path = known.path if known is not None else workspace.resolve(path)
        if path.name != 'SKILL.md':
            raise SkillError('Select a package directory or its SKILL.md file.')
        selected = _skill(path, known.scope if known is not None else 'path')
    if not selected.enabled:
        raise SkillError(f'Skill is disabled: {selected.path}')
    raw = read_regular(selected.path, MAX_SKILL_BYTES)
    # Parse the same bytes saved in the snapshot, including during file edits.
    import io
    source = io.BytesIO(raw)
    metadata = _header(source)
    if not metadata.get('enabled', True):
        raise SkillError(f'Skill is disabled: {selected.path}')
    body = source.read().decode('utf-8').strip()
    if not body:
        raise SkillError('Skill instructions must not be empty.')
    return {'name': metadata['name'].strip(), 'description': ' '.join(metadata['description'].split()),
            'path': str(selected.path), 'root': str(selected.path.parent), 'instructions': body,
            'sha256': hashlib.sha256(raw).hexdigest()}


def validate_snapshot(value: Any) -> None:
    if value is None:
        return
    keys = {'name', 'description', 'path', 'root', 'instructions', 'sha256'}
    if (not isinstance(value, dict) or set(value) != keys
            or any(not isinstance(item, str) or not item for item in value.values())
            or len(value['instructions'].encode('utf-8')) > MAX_SKILL_BYTES
            or len(value['sha256']) != 64
            or any(char not in '0123456789abcdef' for char in value['sha256'])):
        raise SkillError('Invalid saved skill selection.')


def render(snapshot: dict[str, str]) -> str:
    return load_prompt('selected-skill.md', name=snapshot['name'], path=snapshot['path'],
                       root=snapshot['root'], sha256=snapshot['sha256'], instructions=snapshot['instructions'])
