"""
Embedding model integrations for Perquire.
"""

from .base import BaseEmbeddingProvider, EmbeddingResult, embedding_registry
from .gemini_embeddings import GeminiEmbeddingProvider
from .utils import cosine_similarity, normalize_embedding

__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingResult",
    "GeminiEmbeddingProvider",
    "embedding_registry",
    "cosine_similarity",
    "normalize_embedding",
]