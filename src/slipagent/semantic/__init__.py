"""Local semantic retrieval; heavy dependencies load only when requested."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DEFAULT_MODEL": "embed", "Embedder": "embed", "ONNXEmbedder": "embed",
    "Chunk": "indexer", "chunk_file": "indexer", "walk_files": "indexer",
    "Store": "storage", "Searcher": "search", "IndexStats": "search",
    "IndexWatcher": "watcher",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
