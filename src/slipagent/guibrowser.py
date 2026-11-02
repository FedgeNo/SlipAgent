"""Paged browsing of originals and structured actions on the shared application."""

from __future__ import annotations

import heapq
import itertools
import re
from dataclasses import asdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from .application import application
from .context import CompletedStep, HistoryStep
from .fileio import open_regular
from .historyquestions import QUESTION_LIMIT, QUESTION_PREFIX, question_preview
from .records import response_source
from .sessionstate import Session
from .sources import TextSource, json_chunks, text_chunks
from .turnstore import resident_content

PAGE_SIZE = 100
TEXT_PAGE = 16384


def text_page(value: Any, offset: int = 0, limit: int = TEXT_PAGE) -> dict[str, Any]:
    """Slice a stored original without materializing its large text fields."""
    chunks = text_chunks(value) if isinstance(value, (str, TextSource)) else json_chunks(value)
    return slice_text(chunks, offset, limit)


def slice_text(chunks: Iterator[str], offset: int, limit: int = TEXT_PAGE) -> dict[str, Any]:
    pieces = []
    remaining = limit + 1
    for chunk in chunks:
        if offset >= len(chunk):
            offset -= len(chunk)
            continue
        chunk = chunk[offset:offset + remaining]
        offset = 0
        pieces.append(chunk)
        remaining -= len(chunk)
        if remaining == 0:
            break
    result = ''.join(pieces)
    return {'text': result[:limit], 'more': len(result) > limit}


def child_id(parent: str, key: str | int) -> str:
    return parent.rstrip('/') + '/' + quote(str(key), safe='')


class GuiBrowser:
    """Read original owners, independently of model context selection or summaries."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.controller = application(session)

    def settings(self) -> dict[str, Any]:
        return {'planning': self.session.agent.planning,
                'overthinking': self.session.agent.overthinking,
                'danger': self.session.workspace.access.danger,
                'hashline': bool(self.session.registry.services.get('hashline_enabled', True))}

    def plan(self) -> Any:
        for step in reversed(self.session.agent.history.steps):
            for message in reversed(step.messages):
                if message.role != 'tool':
                    continue
                observation = resident_content(message)
                if (isinstance(observation, dict) and observation.get('tool') == 'update_plan'
                        and observation.get('status') == 'success'):
                    return observation['content']
        return {'status': 'No working plan recorded yet.'}

    def turn(self, step: HistoryStep) -> dict[str, Any]:
        return step.parts(stream=True) if isinstance(step, CompletedStep) else step.record()

    def value(self, node: str) -> Any:
        node = self.session_node(node)
        segments = [unquote(segment) for segment in node.strip('/').split('/')]
        category = segments.pop(0)
        if category == 'turns' and segments:
            turn_id = int(segments.pop(0))
            steps = self.session.agent.history.steps
            if not 1 <= turn_id <= len(steps) or steps[turn_id - 1].id != turn_id:
                raise ValueError('This turn is no longer in the current session.')
            step = steps[turn_id - 1]
            value: Any = self.turn(step)
        elif category == 'messages' and segments:
            message = self.session.agent.messages[int(segments.pop(0))]
            stored = getattr(message, '_stored', None)
            value = {'role': message.role, 'content': resident_content(message),
                     'response_source': response_source([message]),
                     'tool_calls': stored.get('tool_calls') if stored is not None else
                     [call.to_record() for call in message.tool_calls or []],
                     'reasoning': stored.get('reasoning') if stored is not None else message.reasoning,
                     'reasoning_details': stored.get('reasoning_details') if stored is not None else message.reasoning_details}
        elif category == 'settings':
            value = self.settings()
        elif category == 'tools' and segments:
            name = segments.pop(0)
            tool = self.session.registry.get(name)
            if tool is None:
                raise ValueError('Tool is no longer available.')
            spec = tool.spec
            value = {'name': spec.name, 'description': spec.description, 'parameters': spec.parameters}
        elif category == 'models' and len(segments) >= 2:
            provider, model_id = segments.pop(0), segments.pop(0)
            model = next((model for model in self.session._catalog or []
                          if model.provider == provider and model.id == model_id), None)
            if model is None:
                raise ValueError('This model is no longer in the catalog.')
            value = asdict(model)
        elif category == 'models' and segments == ['optimal']:
            value = {'mode': 'Optimal', 'description': 'Automatically select eligible ranked free models.'}
            segments = []
        elif category == 'skills' and segments:
            from .skills import catalog
            path = segments.pop(0)
            skill = next((skill for skill in catalog(self.session.workspace)[0] if str(skill.path) == path), None)
            if skill is None:
                raise ValueError('This skill is no longer available.')
            value = {**asdict(skill), 'path': str(skill.path)}
        elif category == 'artifacts' and segments:
            record_id = int(segments.pop(0))
            store = self.session.registry.resources.artifacts
            if store is None or record_id not in store.records:
                raise ValueError('This output record is no longer available.')
            value = {'metadata': store.records[record_id], 'output': '', 'error': ''}
        elif category == 'checkpoints' and segments:
            checkpoint_id = segments.pop(0)
            checkpoints = self.session.registry.resources.checkpoints
            value = next((batch for batch in checkpoints.listing() if batch['id'] == checkpoint_id), None) if checkpoints else None
            if value is None:
                raise ValueError('This checkpoint is no longer available.')
        elif category == 'plan':
            value = self.plan()
        elif category == 'queue' and segments:
            queue_id = segments.pop(0)
            value = next((text for key, text in self.controller.queued_input() if key == queue_id), None)
            if value is None:
                raise ValueError('This prompt has already been submitted or discarded.')
        else:
            raise ValueError('Choose an object to inspect.')
        for key in segments:
            value = value[int(key)] if isinstance(value, (list, tuple)) else value[key]
        return value

    def describe(self, identifier: str, label: str, value: Any, *, kind: str = '') -> dict[str, Any]:
        collection = isinstance(value, (dict, list, tuple))
        preview = f'{len(value)} items' if collection else text_page(value, limit=160)['text']
        if kind in ('sessions', 'turns', 'folder', 'tools', 'models', 'skills', 'artifacts', 'provider', 'checkpoints', 'queue', 'messages', 'plan'):
            preview = 'Browse collection'
        elif kind == 'tool':
            preview = 'Tool definition'
        return {'id': identifier, 'label': label, 'kind': kind or ('collection' if collection else 'value'),
                'collection': collection, 'preview': preview}

    def children(self, node: str, cursor: str = '', limit: int = PAGE_SIZE) -> dict[str, Any]:
        canonical = self.session_node(node)
        if canonical != node:
            alias_page = self.children(canonical, cursor, limit)
            alias_page['node'], alias_page['parent'] = node, self.parent(node)
            for item in alias_page['items']:
                if item['id'].startswith(canonical + '/'):
                    item['id'] = node + item['id'][len(canonical):]
            return alias_page
        rows: Iterator[tuple[str, str, Any, str]]
        offset = int(cursor or '0') if not node.startswith('/files') and node != '/sessions' else 0
        if node == '/':
            rows = iter((('/sessions', 'Sessions', [], 'sessions'), ('/turns', 'Turns', [], 'turns'), ('/messages', 'All Messages', [], 'messages'),
                         ('/files', 'Files', [], 'folder'), ('/plan', 'Plan', [], 'plan'),
                         ('/tools', 'Tools', [], 'tools'), ('/models', 'Models', [], 'models'),
                         ('/skills', 'Skills', [], 'skills'), ('/artifacts', 'Outputs', [], 'artifacts'),
                         ('/checkpoints', 'Checkpoints', [], 'checkpoints'),
                         ('/queue', 'Queued Input', [], 'queue'),
                         ('/settings', 'Settings', [], 'settings')))
        elif node == '/turns':
            rows = ((f'/turns/{step.id}', f'Turn {step.id}', self.turn(step), 'turn')
                    for step in self.session.agent.history.steps)
        elif node == '/messages':
            rows = ((f'/messages/{index}', f'{message.role.title()} · {index}', {}, 'message')
                    for index, message in enumerate(self.session.agent.messages))
        elif node == '/tools':
            rows = ((child_id(node, name), name, {}, 'tool') for name in self.session.registry.names)
        elif node == '/models':
            providers = sorted({model.provider for model in self.session._catalog or []})
            rows = iter([('/models/optimal', 'Optimal Free Selection', {}, 'optimal'),
                         *((child_id(node, provider), provider.title(), [], 'provider') for provider in providers)])
        elif node.startswith('/models/') and node.count('/') == 2 and node != '/models/optimal':
            from .modelpricing import price_label
            provider = unquote(node.rsplit('/', 1)[1])
            models = (model for model in self.session._catalog or [] if model.provider == provider)
            page_models = list(itertools.islice(models, offset, offset + limit + 1))
            model_items = []
            for model in page_models[:limit]:
                price = 'free' if model.provider == 'groq' and not model.pricing else price_label(model.pricing)
                if price.startswith('$'):
                    price += '/Mtok'
                model_items.append({'id': child_id(node, model.id), 'label': model.name or model.id,
                                    'kind': 'model', 'collection': True, 'actionValue': model.selector,
                                    'preview': price + (f' · {model.context_length:,} context' if model.context_length else '')})
            return {'node': node, 'parent': '/models', 'items': model_items,
                'nextCursor': str(offset + limit) if len(page_models) > limit else None}
        elif node == '/skills':
            from .skills import catalog
            skills, errors = catalog(self.session.workspace)
            page_skills = skills[offset:offset + limit + 1]
            return {'node': node, 'parent': '/', 'items': [
                {'id': child_id(node, str(skill.path)), 'label': skill.name, 'kind': 'skill', 'collection': True,
                 'preview': f'{skill.scope} · {skill.description}', 'actionValue': str(skill.path),
                 'enabled': skill.enabled} for skill in page_skills[:limit]],
                'nextCursor': str(offset + limit) if len(page_skills) > limit else None, 'errors': errors}
        elif node == '/artifacts':
            store = self.session.registry.resources.artifacts
            records = store.listing_page(offset, limit + 1)['records'] if store is not None else []
            return {'node': node, 'parent': '/', 'items': [
                {'id': child_id(node, record['record_id']), 'label': record['command'], 'kind': 'artifact',
                 'collection': True, 'preview': record['kind']} for record in records[:limit]],
                'nextCursor': str(offset + limit) if len(records) > limit else None}
        elif node == '/checkpoints':
            checkpoints = self.session.registry.resources.checkpoints
            batches = checkpoints.listing()[offset:offset + limit + 1] if checkpoints else []
            return {'node': node, 'parent': '/', 'items': [
                {'id': child_id(node, batch['id']), 'label': f"Turn {batch['step_id']} · {len(batch['files'])} files",
                 'kind': 'checkpoint', 'collection': True, 'preview': batch['created'],
                 'actionValue': batch['id']} for batch in batches[:limit]],
                'nextCursor': str(offset + limit) if len(batches) > limit else None}
        elif node == '/sessions':
            journal = self.session.registry.resources.journal
            page_sessions = journal.listing_page(cursor, limit) if journal is not None else {'items': [], 'nextCursor': None}
            return {'node': node, 'parent': '/', 'items': [
                {'id': child_id(node, item['id']), 'label': item['title'], 'kind': 'session', 'collection': True,
                 'preview': item.get('error') or f"{item['created']} · {item['id'][:8]}" + (' · current' if item['current'] else '')}
                for item in page_sessions['items']], 'nextCursor': page_sessions['nextCursor']}
        elif node == '/queue':
            rows = ((child_id(node, identifier), f'Queued Prompt {index + 1}', text, 'queued')
                    for index, (identifier, text) in enumerate(self.controller.queued_input()))
        elif node.startswith('/sessions/'):
            identifier = unquote(node.split('/', 2)[2])
            current = identifier == self.session.agent.session_id
            items: list[tuple[str, str, Any, str]] = [
                (node + '/turns', 'Turns', [], 'turns'),
                (node + '/messages', 'All Messages', [], 'messages'),
                (node + '/artifacts', 'Outputs', [], 'artifacts'),
                (node + '/checkpoints', 'Checkpoints', [], 'checkpoints'),
            ] if current else []
            rows = iter(items)
        elif node == '/files' or node.startswith('/files/'):
            return self.files(node, cursor, limit)
        else:
            value = self.value(node)
            pairs = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
            rows = ((child_id(node, key), str(key), child, '') for key, child in pairs)
        page = list(itertools.islice(rows, offset, offset + limit + 1))
        return {'node': node, 'parent': self.parent(node),
                'items': [self.describe(*row[:3], kind=row[3]) for row in page[:limit]],
                'nextCursor': str(offset + limit) if len(page) > limit else None}

    def parent(self, node: str) -> str | None:
        if node == '/':
            return None
        return node.rsplit('/', 1)[0] or '/'

    def session_node(self, node: str) -> str:
        parts = node.strip('/').split('/')
        if len(parts) > 2 and parts[0] == 'sessions':
            if unquote(parts[1]) != self.session.agent.session_id:
                raise ValueError('Resume this saved session to browse its original history.')
            return '/' + '/'.join(parts[2:])
        return node

    def path(self, node: str) -> Path:
        return self.session.workspace.resolve(unquote(node[len('/files/'):]) if node.startswith('/files/') else '.')

    def files(self, node: str, cursor: str, limit: int) -> dict[str, Any]:
        import os
        path = self.path(node)
        if not path.is_dir():
            return {'node': node, 'parent': self.parent(node), 'items': [], 'nextCursor': None}
        with os.scandir(path) as entries:
            selected = heapq.nsmallest(limit + 1, (entry.name for entry in entries if entry.name > cursor))
        items = []
        for name in selected[:limit]:
            target = path / name
            folder = target.is_dir()
            items.append({'id': child_id(node, name), 'label': name, 'kind': 'folder' if folder else 'file',
                          'collection': folder, 'preview': 'Folder' if folder else 'View or edit file'})
        return {'node': node, 'parent': self.parent(node), 'items': items,
                'nextCursor': selected[limit - 1] if len(selected) > limit else None}

    def detail(self, node: str, offset: int = 0) -> dict[str, Any]:
        node = self.session_node(node)
        if node.startswith('/artifacts/') and node.rsplit('/', 1)[-1] in ('output', 'error'):
            store = self.session.registry.resources.artifacts
            if store is None:
                raise ValueError('Output storage is unavailable.')
            record_id, field = node.split('/')[2:]
            page = store.read_page(int(record_id), field, offset, TEXT_PAGE)
            return {'text': page[field], 'more': not page['eof']}
        if node.startswith('/files/'):
            import io
            path = self.path(node)
            with io.TextIOWrapper(open_regular(path), encoding='utf-8', errors='replace') as source:
                def chunks() -> Iterator[str]:
                    while chunk := source.read(16384):
                        if '\x00' in chunk:
                            raise ValueError('This file is binary; text viewing is unavailable.')
                        yield chunk
                return slice_text(chunks(), offset)
        return text_page(self.value(node), offset)

    def search(self, query: str, cursor: str = '0:0', limit: int = 100) -> dict[str, Any]:
        """Search originals in bounded scans, with one located excerpt per message."""
        index, offset = map(int, cursor.split(':'))
        if index < 0 or offset < 0:
            raise ValueError('Invalid search position.')
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results = []
        scanned = visited = 0
        messages = self.session.agent.messages
        while index < len(messages) and visited < 512:
            value = resident_content(messages[index])
            chunks = text_chunks(value) if isinstance(value, (str, TextSource)) else json_chunks(value)
            skipped = 0
            tail = ''
            for chunk in chunks:
                if skipped + len(chunk) <= offset:
                    skipped += len(chunk)
                    continue
                chunk = chunk[max(0, offset - skipped):]
                skipped = max(skipped, offset) + len(chunk)
                scanned += len(chunk)
                buffer = tail + chunk
                match = pattern.search(buffer)
                if match is not None:
                    before = re.sub(r'\s+', ' ', buffer[max(0, match.start() - 140):match.start()])
                    after = re.sub(r'\s+', ' ', buffer[match.end():match.end() + 140])
                    results.append({'index': index, 'id': f'/messages/{index}',
                                    'before': before, 'match': match.group(), 'after': after})
                    break
                tail = buffer[-max(140, len(query) - 1):]
                if scanned >= 4 * 1024 * 1024:
                    return {'items': results, 'nextCursor': f'{index}:{max(offset, skipped - len(tail))}'}
            index += 1
            offset = 0
            visited += 1
            if len(results) >= limit or scanned >= 4 * 1024 * 1024:
                break
        return {'items': results, 'nextCursor': f'{index}:0' if index < len(messages) else None}

    def feed(self, offset: int = -1, limit: int = 40) -> dict[str, Any]:
        messages = self.session.agent.messages
        if offset < 0:
            offset = max(0, len(messages) - limit)
        items = []
        for index in range(offset, min(len(messages), offset + limit)):
            message = messages[index]
            stored = getattr(message, '_stored', None)
            if stored is not None:
                calls = stored.get('tool_calls') or []
            else:
                calls = message.tool_calls or []
            tool_calls = []
            for call_index, call in enumerate(calls[:16]):
                name = call['function']['name'] if isinstance(call, dict) else call.name
                tool_calls.append({'index': call_index, 'name': text_page(name, limit=128)['text']})
            value = resident_content(message)
            tool_result = None
            preview: dict[str, Any]
            if message.role == 'tool':
                envelope = value if isinstance(value, dict) else {}
                name = envelope.get('tool') or message.name or 'Tool'
                status = envelope.get('status')
                tool_result = {'name': text_page(name, limit=128)['text'],
                               'status': status if status in ('success', 'error') else 'unknown'}
                preview = {'text': '', 'more': False}
            else:
                preview = text_page('' if value is None and message.role == 'assistant' else value, limit=4096)
            history_context = None
            if message.role == 'user' and preview['text'].startswith(QUESTION_PREFIX):
                original = text_page(value, limit=QUESTION_LIMIT)
                question = question_preview(original['text']) if not original['more'] else None
                if question is not None:
                    preview = text_page(question['question'], limit=4096)
                    history_context = question['history_reference']['label']
            items.append({'index': index, 'id': f'/messages/{index}', 'role': message.role,
                          'source': response_source([message]),
                          'toolCalls': tool_calls, 'toolCount': len(calls), 'toolResult': tool_result,
                          'historyContext': history_context,
                          **preview})
        return {'items': items, 'offset': offset, 'total': len(messages),
                'sessionId': self.session.agent.session_id}
