"""The measured code-word tokenizer and normalized hybrid ranking contract."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


STOP_WORDS = frozenset("a an and are as at be before besides by can could do does for from get gets "
                       "how i if in into is it its me of on or our that the their them these this "
                       "to use used using was we what when where which why will with without would "
                       "find show explain does not than then too any all have has".split())
RETRIEVAL = {"source_context": "relative-path-v1", "ranker": "hybrid-25-v1"}


def words(text: str) -> list[str]:
    """Split Unicode prose, snake_case, camelCase, and acronyms consistently."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)
    return [word.casefold() for word in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
            if word.casefold() not in STOP_WORDS]


def source_header(path: str) -> str:
    return f"File: {path}\n\n"


def hybrid_scores(semantic: np.ndarray, lexical: np.ndarray) -> np.ndarray:
    """Give BM25 25% and cosine 75%, normalizing each query independently."""
    import numpy as np

    dense = (semantic - semantic.min()) / max(float(semantic.max() - semantic.min()), 1e-12)
    sparse = lexical / max(float(lexical.max()), 1e-12)
    return np.asarray(0.75 * dense + 0.25 * sparse)
