"""Embedding model wrappers.

The :class:`Embedder` protocol defines the minimal interface every backend must
implement so that :class:`~slipagent.semantic.search.Searcher` and
:class:`~slipagent.semantic.storage.Store` stay backend-agnostic. The default
implementation, :class:`ONNXEmbedder`, runs quantized BGE on CPU, but users
can plug in any other backend (API client, ONNX model, hash-based embedder, etc.).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Protocol, runtime_checkable

from .cpu import default_threads

DEFAULT_MODEL = "Xenova/bge-small-en-v1.5"
MODEL_FILES = ("config.json", "tokenizer.json", "onnx/model_quantized.onnx")


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding models.

    Implementations need only provide :meth:`embed`, :meth:`embed_batch`, and
    :meth:`dim`. The protocol is runtime-checkable so callers can duck-type
    validate an arbitrary object.
    """

    def embed(self, text: str) -> List[float]:
        """Embed a single text and return a list of floats."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts and return a list of float lists."""
        ...

    def dim(self) -> int:
        """Return the dimensionality of produced embeddings."""
        ...


class ONNXEmbedder:
    """Lazy CPU embeddings matching Conventions MCP's quantized BGE pipeline.

    Reuses the Conventions model cache when available, otherwise uses the
    Hugging Face cache and downloads missing files on first use. An explicit
    model_dir always loads locally, without downloading anything.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *,
                 model_dir: str | Path | None = None, batch_size: int = 1,
                 threads: int | None = None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive")
        self.threads = default_threads() if threads is None else threads
        self.model_name = model_name
        self.model_dir = Path(model_dir).expanduser() if model_dir is not None else None
        self.batch_size = batch_size
        self._session: Any = None
        self._tokenizer: Any = None
        self._dimension: int | None = None
        self._identity: str | None = None
        self._max_tokens: int | None = None

    def _files(self) -> dict[str, Path]:
        if self.model_dir is not None:
            files = {name: self.model_dir / name for name in MODEL_FILES}
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(f"Missing embedding model file: {path}")
            return files

        cache = Path(os.environ.get("MEMORY_MODEL_CACHE_PATH", str(Path.home() / ".conventions-mcp" / "models"))).expanduser()
        files = {name: cache / self.model_name / name for name in MODEL_FILES}
        if all(path.is_file() for path in files.values()):
            return files

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        files = {}
        for name in MODEL_FILES:
            try:
                downloaded = hf_hub_download(self.model_name, name, local_files_only=True)
            except LocalEntryNotFoundError:
                downloaded = hf_hub_download(self.model_name, name)
            files[name] = Path(downloaded)
        return files

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        files = self._files()
        config = json.loads(files["config.json"].read_text(encoding="utf-8"))
        tokenizer = Tokenizer.from_file(str(files["tokenizer.json"]))
        tokenizer.no_truncation()
        pad_id = config["pad_token_id"]
        tokenizer.enable_padding(pad_id=pad_id, pad_token=tokenizer.id_to_token(pad_id))
        options = ort.SessionOptions()
        # Bound CPU workers so a small model does not occupy every core.
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(files["onnx/model_quantized.onnx"]),
                                       sess_options=options, providers=["CPUExecutionProvider"])
        fingerprint = hashlib.sha256()
        for name in MODEL_FILES:
            fingerprint.update(name.encode())
            with files[name].open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    fingerprint.update(block)
        self._identity = f"onnx-q8-mean-l2-complete-v1:{fingerprint.hexdigest()}"
        self._dimension = int(config["hidden_size"])
        self._max_tokens = int(config["max_position_embeddings"])
        self._tokenizer = tokenizer
        self._session = session

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import numpy as np

        self._load()
        assert self._max_tokens is not None
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self._tokenizer.encode_batch(texts[start:start + self.batch_size])
            if any(len(token.ids) > self._max_tokens for token in tokens):
                raise ValueError(f"Embedding input exceeds the {self._max_tokens}-token limit; "
                                 "split source into token-sized chunks or shorten the query.")
            inputs = {
                "input_ids": np.asarray([token.ids for token in tokens], dtype=np.int64),
                "attention_mask": np.asarray([token.attention_mask for token in tokens], dtype=np.int64),
                "token_type_ids": np.asarray([token.type_ids for token in tokens], dtype=np.int64),
            }
            feed = {item.name: inputs[item.name] for item in self._session.get_inputs()}
            hidden = self._session.run(["last_hidden_state"], feed)[0]
            mask = inputs["attention_mask"][..., None].astype(hidden.dtype)
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            normalized = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            vectors.extend(normalized.tolist())
        return vectors

    def dim(self) -> int:
        self._load()
        assert self._dimension is not None
        return self._dimension

    @property
    def max_tokens(self) -> int:
        """Maximum input tokens, including the tokenizer's special tokens."""
        self._load()
        assert self._max_tokens is not None
        return self._max_tokens

    def token_count(self, text: str) -> int:
        """Count the complete input without truncating it."""
        self._load()
        return len(self._tokenizer.encode(text).ids)

    @property
    def identity(self) -> str:
        """Identify the actual weights, tokenizer, and pooling used by an index."""
        self._load()
        assert self._identity is not None
        return self._identity

    def __repr__(self) -> str:
        return f"ONNXEmbedder(model_name={self.model_name!r})"
