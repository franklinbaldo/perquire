"""
Embedding model integrations for Perquire.

This module automatically registers available embedding providers with the central registry.
"""

from .base import BaseEmbeddingProvider, EmbeddingResult, embedding_registry, EmbeddingError
from .gemini_embeddings import GeminiEmbeddingProvider
from .openai_embeddings import OpenAIEmbeddingProvider # Added new provider
from .utils import cosine_similarity, normalize_embedding

# Default configuration for providers
DEFAULT_EMBEDDING_PROVIDER_CONFIGS = {
    "openai": {"model": "text-embedding-ada-002"},
    "gemini": {"model": "models/embedding-001"},
}

# Register providers
try:
    embedding_registry.register_provider(
        "openai",
        OpenAIEmbeddingProvider(config=DEFAULT_EMBEDDING_PROVIDER_CONFIGS["openai"]),
        set_as_default=True # Example: Set OpenAI as default
    )
except EmbeddingError as e:
    print(f"Note: OpenAI Embedding provider not fully configured: {e}") # Or use logger

try:
    embedding_registry.register_provider(
        "gemini",
        GeminiEmbeddingProvider(config=DEFAULT_EMBEDDING_PROVIDER_CONFIGS["gemini"])
    )
except EmbeddingError as e:
    print(f"Note: Gemini Embedding provider not fully configured: {e}")


__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingResult",
    "GeminiEmbeddingProvider",
    "OpenAIEmbeddingProvider", # Added new provider
    "embedding_registry",
    "cosine_similarity",
    "normalize_embedding",
]