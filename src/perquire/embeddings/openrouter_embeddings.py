"""OpenRouter embedding provider.

litellm routes ``openrouter/<vendor>/<model>`` to OpenRouter's embeddings
endpoint, so the benchmark can score candidates with the same credential it uses
for generation.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from litellm import embedding

from ..exceptions import ConfigurationError
from ..providers.embedding_cache import EmbeddingCache, default_embedding_cache_path
from ..providers.rate_limit import get_shared_pacer
from ..providers.retry import call_with_transient_retries
from .base import BaseEmbeddingProvider, EmbeddingError, EmbeddingResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/nemotron-3-embed-1b:free"
DEFAULT_REQUESTS_PER_MINUTE = 20
DEFAULT_MAX_RETRIES = 2


def _served_upstream_provider(response: Any) -> str | None:
    """Extract OpenRouter's served-provider metadata without inventing it."""
    candidates: list[Any] = []
    if isinstance(response, dict):
        candidates.append(response.get("provider"))
    else:
        candidates.append(getattr(response, "provider", None))
        model_extra = getattr(response, "model_extra", None)
        if isinstance(model_extra, dict):
            candidates.append(model_extra.get("provider"))
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            candidates.append(hidden.get("provider"))
            headers = hidden.get("additional_headers")
            if isinstance(headers, dict):
                lowered = {str(key).lower(): value for key, value in headers.items()}
                candidates.extend([lowered.get("x-openrouter-provider"), lowered.get("x-provider")])
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


class OpenRouterEmbeddingProvider(BaseEmbeddingProvider):
    """Embed text through any OpenRouter-hosted embedding model."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        rpm = int(self.config.get("requests_per_minute", DEFAULT_REQUESTS_PER_MINUTE))
        self._pacer = get_shared_pacer("openrouter", rpm)
        self._max_retries = int(self.config.get("max_retries", DEFAULT_MAX_RETRIES))
        cache_path = Path(self.config.get("cache_path") or default_embedding_cache_path())
        self._cache = EmbeddingCache(cache_path)
        self.transport_attempts = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self._dimensions: int | None = None
        self._served_upstream_provider_counts: Counter[str] = Counter()
        self._successful_responses_without_provider = 0
        self._last_served_upstream_provider: str | None = None

    def validate_config(self) -> None:
        if not self._api_key():
            raise ConfigurationError(
                "OpenRouter API key not found. Provide 'api_key' in config or set OPENROUTER_API_KEY."
            )

    def _api_key(self) -> str | None:
        return self.config.get("api_key") or os.getenv("OPENROUTER_API_KEY")

    @property
    def model(self) -> str:
        return f"openrouter/{self.config.get('model', DEFAULT_MODEL)}"

    def _embed_uncached(self, texts: list[str]) -> tuple[list[np.ndarray], str | None]:
        def request():
            self.transport_attempts += 1
            return embedding(model=self.model, input=texts, api_key=self._api_key())

        try:
            response = call_with_transient_retries(
                request,
                before_attempt=self._pacer.wait,
                max_retries=self._max_retries,
            )
        except Exception as error:
            logger.exception("OpenRouter embedding failed")
            raise EmbeddingError(f"OpenRouter embedding failed: {error}") from error

        served_provider = _served_upstream_provider(response)
        self._last_served_upstream_provider = served_provider
        if served_provider is None:
            self._successful_responses_without_provider += 1
        else:
            self._served_upstream_provider_counts[served_provider] += 1

        data = response["data"] if isinstance(response, dict) else response.data
        vectors = [np.asarray(item["embedding"], dtype=np.float64) for item in data]
        if vectors:
            self._dimensions = int(vectors[0].size)
        return vectors, served_provider

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self._cache.get(self.model, text)
            if cached is None:
                self.cache_misses += 1
                missing_indexes.append(index)
                missing_texts.append(text)
            else:
                self.cache_hits += 1
                self._dimensions = int(cached.size)
                vectors[index] = cached
        if missing_texts:
            fetched, _ = self._embed_uncached(missing_texts)
            if len(fetched) != len(missing_texts):
                raise EmbeddingError(
                    f"OpenRouter returned {len(fetched)} embeddings for {len(missing_texts)} inputs"
                )
            for index, text, vector in zip(missing_indexes, missing_texts, fetched, strict=True):
                self._cache.put(self.model, text, vector)
                self.cache_writes += 1
                vectors[index] = vector
        if any(vector is None for vector in vectors):
            raise EmbeddingError("Embedding cache/fetch pipeline left an input unresolved")
        return [vector for vector in vectors if vector is not None]

    def _result(
        self,
        text: str,
        vector: np.ndarray,
        *,
        upstream_provider: str | None = None,
        cache_status: str | None = None,
    ) -> EmbeddingResult:
        metadata: dict[str, Any] = {
            "provider": "openrouter",
            "model": self.model,
            "text_length": len(text),
        }
        if upstream_provider is not None:
            metadata["upstream_provider"] = upstream_provider
        if cache_status is not None:
            metadata["cache_status"] = cache_status
        return EmbeddingResult(
            embedding=vector,
            metadata=metadata,
            model=self.model,
            dimensions=int(vector.size),
        )

    def cached_embedding(self, text: str) -> np.ndarray | None:
        """Read an existing vector without silently paying for a provider call."""
        return self._cache.get(self.model, text)

    def embed_text_uncached(self, text: str) -> EmbeddingResult:
        """Embed through the provider boundary while bypassing cache lookup and write."""
        vectors, served_provider = self._embed_uncached([text])
        if not vectors:
            raise EmbeddingError("OpenRouter returned no embedding for the requested text")
        return self._result(
            text,
            vectors[0],
            upstream_provider=served_provider,
            cache_status="bypass",
        )

    def _execute_embed_text(self, text: str, **kwargs: Any) -> EmbeddingResult:
        vectors = self._embed([text])
        if not vectors:
            raise EmbeddingError("OpenRouter returned no embedding for the requested text")
        return self._result(text, vectors[0])

    def _execute_embed_batch(self, texts: list[str], **kwargs: Any) -> list[EmbeddingResult]:
        vectors = self._embed(list(texts))
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"OpenRouter returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
        return [self._result(text, vector) for text, vector in zip(texts, vectors, strict=True)]

    def get_embedding_dimensions(self) -> int:
        if self._dimensions is None:
            self._embed(["dimension probe"])
        return int(self._dimensions or 0)

    def is_available(self) -> bool:
        return bool(self._api_key())

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "model": self.model,
            "dimensions": self._dimensions,
            "requests_per_minute": self._pacer.requests_per_minute,
            "max_retries": self._max_retries,
            "transport_attempts": self.transport_attempts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_writes": self.cache_writes,
            "cache_path": str(self._cache.path),
            "served_upstream_providers": sorted(self._served_upstream_provider_counts),
            "served_upstream_provider_counts": dict(sorted(self._served_upstream_provider_counts.items())),
            "successful_responses_without_upstream_provider": self._successful_responses_without_provider,
            "last_served_upstream_provider": self._last_served_upstream_provider,
        }
