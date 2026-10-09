"""Curated exclusions from coding model listings; unknown models remain visible."""

from .types import ModelInfo


# Exact IDs avoid excluding future models based on names or incomplete metadata.
MODEL_BLACKLIST = {
    "google/deplot": "Chart-to-table extraction",
    "meta/llama-guard-4-12b": "Content moderation",
    "nvidia/ai-synthetic-video-detector": "Synthetic video detection",
    "nvidia/embed-qa-4": "Retrieval embeddings",
    "nvidia/llama-3.1-nemoguard-8b-content-safety": "Content moderation",
    "nvidia/llama-3.1-nemoguard-8b-topic-control": "Topic classification",
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3": "Content moderation",
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1": "Retrieval embeddings",
    "nvidia/llama-3.2-nv-embedqa-1b-v1": "Retrieval embeddings",
    "nvidia/llama-nemotron-embed-vl-1b-v2": "Retrieval embeddings",
    "nvidia/nemotron-3-embed-1b": "Retrieval embeddings",
    "nvidia/nemotron-3.5-content-safety": "Content moderation",
    "nvidia/nemotron-4-340b-reward": "Response scoring",
    "nvidia/nemotron-parse": "Document extraction",
    "nvidia/nemotron-parse-2.0": "Document extraction",
    "nvidia/nv-embedqa-mistral-7b-v2": "Retrieval embeddings",
    "nvidia/nvclip": "Image and text embeddings",
    "nvidia/riva-translate-4b-instruct": "Translation",
    "nvidia/riva-translate-4b-instruct-v2": "Translation",
    "snowflake/arctic-embed-l": "Retrieval embeddings",
}


def coding_models(models: list[ModelInfo]) -> list[ModelInfo]:
    """Hide only explicitly curated IDs, across providers."""
    return [model for model in models if model.id not in MODEL_BLACKLIST]
