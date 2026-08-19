"""Embedding registry with optional provider integrations."""

from importlib import import_module
from typing import Any

from .base import BaseEmbeddingProvider, EmbeddingError, EmbeddingResult, embedding_registry
from .utils import cosine_similarity, normalize_embedding
from ..exceptions import ConfigurationError

_PROVIDER_SPECS = {
    "openai": (".openai_embeddings", "OpenAIEmbeddingProvider", {"model": "text-embedding-ada-002"}),
    "gemini": (".gemini_embeddings", "GeminiEmbeddingProvider", {"model": "models/embedding-001"}),
    "openrouter": (
        ".openrouter_embeddings",
        "OpenRouterEmbeddingProvider",
        {"model": "nvidia/nemotron-3-embed-1b:free"},
    ),
}


def _load_provider(name: str) -> type[BaseEmbeddingProvider] | None:
    module_name, class_name, _ = _PROVIDER_SPECS[name]
    try:
        module = import_module(module_name, package=__name__)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None


def register_available_providers() -> list[str]:
    registered: list[str] = []
    for name, (_, _, config) in _PROVIDER_SPECS.items():
        provider_class = _load_provider(name)
        if provider_class is None:
            continue
        try:
            embedding_registry.register_provider(
                name,
                provider_class(config=dict(config)),
                set_as_default=not registered,
            )
            registered.append(name)
        except (EmbeddingError, ConfigurationError):
            continue
    return registered


register_available_providers()


def __getattr__(name: str) -> Any:
    for provider_name, (_, class_name, _) in _PROVIDER_SPECS.items():
        if name == class_name:
            provider_class = _load_provider(provider_name)
            if provider_class is None:
                raise ImportError(f"Optional dependencies for {provider_name!r} are not installed")
            return provider_class
    raise AttributeError(name)


__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingResult",
    "embedding_registry",
    "register_available_providers",
    "cosine_similarity",
    "normalize_embedding",
]
