"""Shared skill discovery and pinned instruction snapshots; no code execution."""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import io
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import yaml

from .config import data_directory
from .fileio import open_regular, read_regular
from .prompts import load_prompt, prompt_directory
from .workspace import Workspace

MAX_SKILL_BYTES = 256 * 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_ACTIVE_SKILLS = 16
SKILL_DIRECTORIES = ('.slipagent', '.agents', '.claude', '.github', '.copilot', '.codex')
NAME = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*\Z')
CODE_SUBSTITUTION = re.compile(r'!`[^`]+`')


def split_invocation(text: str) -> tuple[str, str]:
    text = text.lstrip()
    if not text.startswith(('"', "'")):
        # This is command input, not a shell: keep literal path backslashes.
        parts = text.split(None, 1)
        return (parts[0] if parts else '', parts[1] if len(parts) > 1 else '')
    source = io.StringIO(text)
    parser = shlex.shlex(source, posix=True)
    parser.whitespace_split = True
    parser.commenters = ''
    selector = parser.get_token()
    return selector or '', source.read().lstrip()


class SkillError(ValueError):
    """A package cannot be discovered or explicitly selected."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    enabled: bool
    scope: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _metadata(value: Any, folder: Path, source: BinaryIO) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SkillError('Skill metadata must be a mapping with string keys.')
    metadata = dict(value)
    metadata.setdefault('name', folder.name)
    if 'description' not in metadata:
        position = source.tell()
        preview = source.read(MAX_METADATA_BYTES).decode('utf-8', errors='replace')
        source.seek(position)
        metadata['description'] = next((line.lstrip('# ').strip() for line in preview.splitlines() if line.strip()), '')
    for key, limit in (('name', 64), ('description', 1024)):
        text = metadata.get(key)
        if (not isinstance(text, str) or not text.strip() or len(text) > limit
                or any(ord(char) < 32 and char not in '\n\r\t' for char in text)):
            raise SkillError(f'Skill {key} must be nonempty text, at most {limit} characters.')
    if NAME.fullmatch(metadata['name']) is None:
        raise SkillError('Skill name must use lowercase letters, digits and single hyphens.')
    for key in ('enabled', 'disable-model-invocation', 'user-invocable'):
        if key in metadata:
            flag = metadata[key]
            if type(flag) is int and flag in (0, 1):
                metadata[key] = bool(flag)
            elif type(flag) is not bool:
                raise SkillError(f'Skill {key} must be a boolean.')
    for key in ('allowed-tools', 'disallowed-tools'):
        if key in metadata and not (isinstance(metadata[key], str) or isinstance(metadata[key], list)
                                    and all(isinstance(rule, str) for rule in metadata[key])):
            raise SkillError(f'Skill {key} must be text or a list of tool rules.')
    if 'compatibility' in metadata and (not isinstance(metadata['compatibility'], str)
                                      or not 1 <= len(metadata['compatibility']) <= 500):
        raise SkillError('Skill compatibility must be text, at most 500 characters.')
    _openai_metadata(metadata.get('openai', {}))
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise SkillError(f'Skill metadata must contain JSON-compatible values: {exc}') from exc
    return metadata


def _openai_metadata(options: Any) -> None:
    if not isinstance(options, dict):
        raise SkillError('OpenAI skill metadata must contain a mapping.')
    for key in ('interface', 'policy', 'dependencies'):
        if not isinstance(options.get(key, {}), dict):
            raise SkillError(f'OpenAI skill {key} must contain a mapping.')
    if type(options.get('policy', {}).get('allow_implicit_invocation', True)) is not bool:
        raise SkillError('allow_implicit_invocation must be a boolean.')
    entries = options.get('dependencies', {}).get('tools', [])
    if not isinstance(entries, list):
        raise SkillError('Skill tool dependencies must be a list.')
    for entry in entries:
        if (not isinstance(entry, dict) or entry.get('type') != 'mcp'
                or not isinstance(entry.get('value'), str) or not entry['value'].strip()):
            raise SkillError('Only named MCP tool dependencies are supported.')


def _header(source: BinaryIO, folder: Path) -> dict[str, Any]:
    if source.readline(5) not in (b'---\n', b'---\r\n'):
        source.seek(0)
        return _metadata({}, folder, source)
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
            except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
                raise SkillError(f'Invalid skill metadata: {exc}') from exc
            return _metadata(metadata, folder, source)
        if not line:
            break
        lines.append(line)
    raise SkillError('Skill front matter is missing its closing delimiter or exceeds 16384 bytes.')


def _skill(path: Path, scope: str) -> Skill:
    path = path.resolve()
    with open_regular(path) as source:
        metadata = _header(source, path.parent)
    return Skill(metadata['name'].strip(), ' '.join(metadata['description'].split()),
                 path, metadata.get('enabled', True), scope, metadata)


def skill_roots(workspace: Workspace) -> list[tuple[str, Path]]:
    ancestors = [workspace.root]
    for parent in workspace.root.parents:
        if (ancestors[-1] / '.git').exists():
            break
        ancestors.append(parent)
    if not any((parent / '.git').exists() for parent in ancestors):
        ancestors = [workspace.root]
    roots = [('project', parent / directory / 'skills') for parent in ancestors for directory in SKILL_DIRECTORIES]
    roots.append(('user', data_directory() / 'skills'))
    roots.extend(('user', Path.home() / directory / 'skills') for directory in SKILL_DIRECTORIES)
    roots.append(('system', Path('/etc/codex/skills')))
    roots.append(('bundled', prompt_directory() / 'skills'))
    return roots


def catalog(workspace: Workspace) -> tuple[list[Skill], list[str]]:
    found = []
    errors = []
    seen: set[Path] = set()
    for scope, root in skill_roots(workspace):
        try:
            children = sorted(root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f'{root}: {exc}')
            continue
        for child in children:
            path = child / 'SKILL.md'
            try:
                if not child.is_dir():
                    continue
                if not path.exists() and not path.is_symlink():
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    found.append(_skill(resolved, scope))
                    seen.add(resolved)
            except (OSError, ValueError) as exc:
                errors.append(f'{path}: {exc}')
    return found, errors


def invocation(workspace: Workspace, text: str) -> tuple[str, str] | None:
    match = re.match(r'^[/\$]([a-z0-9]+(?:-[a-z0-9]+)*)(?:\s+([\s\S]*))?$', text.strip())
    if match is not None and any(skill.name == match[1] for skill in catalog(workspace)[0]):
        return match[1], match[2] or ''
    return None


def select(workspace: Workspace, selector: str, *, arguments: str = '', implicit: bool = False,
           session_id: str = '') -> dict[str, Any]:
    packages, _ = catalog(workspace)
    matches = [skill for skill in packages if skill.name == selector]
    if matches:
        # Catalog roots are ordered by precedence; duplicates within a scope
        # remain ambiguous instead of choosing an arbitrary package.
        scope = matches[0].scope
        matches = [skill for skill in matches if skill.scope == scope]
        if len(matches) != 1:
            raise SkillError('Ambiguous skill name; select a SKILL.md path: ' + ', '.join(str(s.path) for s in matches))
        selected = matches[0]
    else:
        path = Path(selector).expanduser()
        if not path.is_absolute():
            path = workspace.root / path
        if path.is_dir():
            path = path / 'SKILL.md'
        # User-selected packages may live outside the workspace. Implicit
        # selection remains limited to discovered packages and workspace paths.
        known = next((skill for skill in packages if skill.path == path.resolve()), None)
        if known is not None:
            path = known.path
        elif implicit:
            path = workspace.resolve(path)
        else:
            path = path.resolve()
        if path.name != 'SKILL.md':
            raise SkillError('Select a package directory or its SKILL.md file.')
        selected = _skill(path, known.scope if known is not None else 'path')
    if not selected.enabled:
        raise SkillError(f'Skill is disabled: {selected.path}')
    raw = read_regular(selected.path, MAX_SKILL_BYTES)
    # Parse the same bytes saved in the snapshot, including during file edits.
    source = io.BytesIO(raw)
    metadata = _header(source, selected.path.parent)
    if not metadata.get('enabled', True):
        raise SkillError(f'Skill is disabled: {selected.path}')
    body = source.read().decode('utf-8').strip()
    if not body:
        raise SkillError('Skill instructions must not be empty.')
    sidecar = selected.path.parent / 'agents' / 'openai.yaml'
    extra = b''
    if sidecar.exists():
        extra = read_regular(sidecar, MAX_METADATA_BYTES)
        try:
            options = yaml.safe_load(extra.decode('utf-8'))
        except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
            raise SkillError(f'Invalid agents/openai.yaml: {exc}') from exc
        _openai_metadata(options)
        metadata['openai'] = options
        # Sidecar values share the same safe, serializable snapshot contract.
        metadata = _metadata(metadata, selected.path.parent, source)
    openai = metadata.get('openai', {})
    policy = openai.get('policy', {})
    if implicit and (metadata.get('disable-model-invocation', False)
                     or policy.get('allow_implicit_invocation', True) is False):
        raise SkillError(f'Skill {selected.name} requires explicit invocation; implicit loading is disabled.')
    if not implicit and metadata.get('user-invocable', True) is False:
        raise SkillError(f'Skill {selected.name} is model-invocable only.')
    unsupported = [key for key in ('hooks', 'agent') if metadata.get(key)]
    if metadata.get('context') not in (None, 'inline'):
        unsupported.append('context')
    if metadata.get('model') not in (None, 'inherit'):
        unsupported.append('model')
    if unsupported:
        raise SkillError('Unsupported host execution settings: ' + ', '.join(unsupported)
                         + '. Use a portable inline skill or remove these host-specific settings.')
    if CODE_SUBSTITUTION.search(body):
        raise SkillError('Dynamic !`command` preprocessing is not supported; put the command in the workflow instructions.')
    if re.search(r'\$\{CLAUDE_PLUGIN_(?:ROOT|DATA)\}', body):
        raise SkillError('Claude plugin paths require a plugin host; install this workflow as a standalone skill using CLAUDE_SKILL_DIR.')
    warnings = []
    if metadata['name'] != selected.path.parent.name:
        warnings.append('The name differs from the package directory; this Claude-style alias is not portable to strict Agent Skills validators.')
    body = expand(body, arguments, metadata.get('arguments', []), selected.path.parent, workspace.root, session_id)
    digest = hashlib.sha256(raw)
    if extra:
        digest.update(b'\0agents/openai.yaml\0' + extra)
    return {'name': metadata['name'].strip(), 'description': ' '.join(metadata['description'].split()),
            'path': str(selected.path), 'root': str(selected.path.parent), 'instructions': body,
            'sha256': digest.hexdigest(), 'metadata': metadata, 'arguments': arguments, 'warnings': warnings}


def expand(body: str, arguments: str, named: Any, root: Path, project: Path, session_id: str) -> str:
    words = shlex.split(arguments)
    names = shlex.split(named) if isinstance(named, str) else named
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise SkillError('Skill arguments must be a list or space-separated argument names.')
    if any(re.fullmatch(r'[A-Za-z_]\w*', name) is None for name in names):
        raise SkillError('Named skill arguments must be identifiers.')
    values = {name: words[index] if index < len(words) else '' for index, name in enumerate(names)}
    values.update(CLAUDE_SKILL_DIR=str(root), CLAUDE_PROJECT_DIR=str(project), CLAUDE_SESSION_ID=session_id)
    received = False

    def substitute(match: re.Match[str]) -> str:
        nonlocal received
        if match[1]:
            return match[0][1:]
        token = match[2]
        if token == 'ARGUMENTS':
            received = True
            return arguments
        if token.startswith('ARGUMENTS[') or token.isdigit():
            index = int(token[10:-1]) if token.startswith('ARGUMENTS[') else int(token)
            received = True
            return words[index] if index < len(words) else ''
        else:
            key = token[1:-1] if token.startswith('{') else token
            if key in values:
                received = received or key in names
                return values[key]
        return match[0]

    pattern = r'(\\)?\$(ARGUMENTS\[\d+\]|ARGUMENTS(?!\w)|\d+|\{CLAUDE_(?:SKILL_DIR|PROJECT_DIR|SESSION_ID)\}|[A-Za-z_]\w*)'
    pieces, position, size = [], 0, 0
    for match in re.finditer(pattern, body):
        for piece in (body[position:match.start()], substitute(match)):
            size += len(piece.encode('utf-8'))
            if size > MAX_SKILL_BYTES:
                raise SkillError('Expanded skill instructions exceed the size limit.')
            pieces.append(piece)
        position = match.end()
    pieces.append(body[position:])
    rendered = ''.join(pieces)
    if arguments and not received:
        rendered += '\n\nARGUMENTS: ' + arguments
    if len(rendered.encode('utf-8')) > MAX_SKILL_BYTES:
        raise SkillError('Expanded skill instructions exceed the size limit.')
    return rendered


def validate_snapshot(value: Any) -> None:
    if value is None:
        return
    keys = {'name', 'description', 'path', 'root', 'instructions', 'sha256'}
    if (not isinstance(value, dict) or not keys <= set(value)
            or set(value) - keys - {'metadata', 'arguments', 'warnings'}
            or any(not isinstance(value[key], str) or not value[key] for key in keys)
            or len(value['instructions'].encode('utf-8')) > MAX_SKILL_BYTES
            or len(value['sha256']) != 64
            or any(char not in '0123456789abcdef' for char in value['sha256'])):
        raise SkillError('Invalid saved skill selection.')
    if not Path(value['root']).is_absolute() or Path(value['path']).parent != Path(value['root']):
        raise SkillError('Invalid saved skill package directory.')
    if not isinstance(value.get('metadata', {}), dict) or not isinstance(value.get('arguments', ''), str):
        raise SkillError('Invalid saved skill metadata or arguments.')
    if 'metadata' in value:
        _metadata(value['metadata'], Path(value['root']), io.BytesIO(value['instructions'].encode('utf-8')))
    if not isinstance(value.get('warnings', []), list) or any(not isinstance(w, str) for w in value.get('warnings', [])):
        raise SkillError('Invalid saved skill compatibility warnings.')


def active(services: dict[str, Any]) -> list[dict[str, Any]]:
    primary = services.get('selected_skill')
    if primary is None:
        return []
    snapshots = services.get('active_skills', [])
    return snapshots if snapshots and snapshots[-1] == primary else [primary]


def sync_access(services: dict[str, Any]) -> None:
    workspace = services.get('workspace')
    if workspace is not None:
        workspace.access.read_roots = tuple(Path(snapshot['root']) for snapshot in active(services))


def restore_active(services: dict[str, Any], snapshots: list[dict[str, Any]]) -> None:
    validate_active(snapshots)
    if snapshots:
        services['active_skills'] = copy.deepcopy(snapshots)
        services['selected_skill'] = services['active_skills'][-1]
    else:
        services.pop('active_skills', None)
        services.pop('selected_skill', None)
    sync_access(services)


def validate_active(snapshots: Any) -> None:
    if not isinstance(snapshots, list) or any(not isinstance(snapshot, dict) for snapshot in snapshots):
        raise SkillError('Invalid saved active skills.')
    if len(snapshots) > MAX_ACTIVE_SKILLS:
        raise SkillError(f'At most {MAX_ACTIVE_SKILLS} skills can be active.')
    for snapshot in snapshots:
        validate_snapshot(snapshot)
    if len({snapshot['path'] for snapshot in snapshots}) != len(snapshots):
        raise SkillError('Duplicate saved active skills.')


def set_active(services: dict[str, Any], snapshot: dict[str, Any] | None, *, append: bool = True) -> None:
    snapshots = []
    if snapshot is not None:
        if append:
            snapshots = [saved for saved in active(services) if saved['path'] != snapshot['path']]
        snapshots.append(snapshot)
    restore_active(services, snapshots)


def required_tools(snapshot: dict[str, Any], available: list[str]) -> set[str]:
    dependencies = snapshot.get('metadata', {}).get('openai', {}).get('dependencies', {})
    entries = dependencies.get('tools', [])
    found: set[str] = set()
    for dependency in entries:
        server = dependency['value']
        matching = {name for name in available if name.startswith((server + '__', 'mcp__' + server + '__'))}
        if not matching:
            raise SkillError(f'Skill {snapshot["name"]} requires the configured MCP server {server}.')
        found.update(matching)
    return found


def catalog_hint(workspace: Workspace) -> str:
    packages, errors = catalog(workspace)
    lines = []
    remaining = 8192
    hidden = 0
    for skill in packages:
        if not skill.enabled or skill.metadata.get('disable-model-invocation', False):
            continue
        line = f'{skill.name}: {skill.description[:180]} ({skill.path})'
        if len(line) + 1 > remaining:
            hidden += 1
            continue
        lines.append(line)
        remaining -= len(line) + 1
    if hidden:
        lines.append(f'{hidden} more skills: search with use_skill action=list and query.')
    if errors:
        lines.append(f'{len(errors)} packages could not be read; use_skill action=list reports their errors.')
    return '\n'.join(lines)


TOOL_ALIASES = {'Read': 'read_file', 'Write': 'write_file', 'Edit': 'edit_file', 'Glob': 'glob',
                'Grep': 'grep', 'Bash': 'run_command', 'WebFetch': 'fetch_page', 'WebSearch': 'web_search',
                'Skill': 'use_skill'}


def denied_tool(services: dict[str, Any], name: str, arguments: Any = None) -> str | None:
    for snapshot in active(services):
        rules = snapshot.get('metadata', {}).get('disallowed-tools', [])
        rules = [rules] if isinstance(rules, str) else rules
        for rule in rules:
            for pattern in re.findall(r'[^,\s()]+(?:\([^)]*\))?', rule):
                tool, _, condition = pattern.partition('(')
                target = TOOL_ALIASES.get(tool, tool)
                if not fnmatch.fnmatchcase(name, target):
                    continue
                if condition:
                    if arguments is None:
                        continue
                    data = json.loads(arguments) if isinstance(arguments, str) else arguments
                    text = data.get('command', data.get('path', '')) if isinstance(data, dict) else ''
                    condition = condition[:-1]
                    prefix = condition[:-2] if condition.endswith(':*') else None
                    if not (text == prefix or fnmatch.fnmatchcase(text, condition.replace(':*', ' *'))):
                        continue
                return f'Skill {snapshot["name"]} disallows tool {name}.'
    return None


def render(snapshot: dict[str, Any]) -> str:
    return load_prompt('selected-skill-v2.md', name=snapshot['name'], path=snapshot['path'],
                       root=snapshot['root'], sha256=snapshot['sha256'], instructions=snapshot['instructions'],
                       metadata=json.dumps(snapshot.get('metadata', {}), ensure_ascii=False, indent=2),
                       arguments=snapshot.get('arguments', ''), warnings='\n'.join(snapshot.get('warnings', [])))
