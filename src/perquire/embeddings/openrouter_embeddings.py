"""OpenRouter embedding provider.

litellm routes ``openrouter/<vendor>/<model>`` to OpenRouter's embeddings
endpoint, so the benchmark can score candidates with the same credential it uses
for generation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from litellm import embedding

from ..exceptions import ConfigurationError
from ..providers.rate_limit import get_shared_pacer
from .base import BaseEmbeddingProvider, EmbeddingError, EmbeddingResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/nemotron-3-embed-1b:free"
DEFAULT_REQUESTS_PER_MINUTE = 20


class OpenRouterEmbeddingProvider(BaseEmbeddingProvider):
    """Embed text through any OpenRouter-hosted embedding model."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        rpm = int(self.config.get("requests_per_minute", DEFAULT_REQUESTS_PER_MINUTE))
        self._pacer = get_shared_pacer("openrouter", rpm)
        self._dimensions: int | None = None

    def validate_config(self) -> None:
        if not self._api_key():
            raise ConfigurationError(
                "OpenRouter API key not found. Provide 'api_key' in config or set "
                "OPENROUTER_API_KEY."
            )

    def _api_key(self) -> str | None:
        return self.config.get("api_key") or os.getenv("OPENROUTER_API_KEY")

    @property
    def model(self) -> str:
        return f"openrouter/{self.config.get('model', DEFAULT_MODEL)}"

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        self._pacer.wait()
        try:
            response = embedding(model=self.model, input=texts, api_key=self._api_key())
        except Exception as error:
            logger.exception("OpenRouter embedding failed")
            raise EmbeddingError(f"OpenRouter embedding failed: {error}") from error

        data = response["data"] if isinstance(response, dict) else response.data
        vectors = [np.asarray(item["embedding"], dtype=np.float64) for item in data]
        if vectors:
            self._dimensions = int(vectors[0].size)
        return vectors

    def _result(self, text: str, vector: np.ndarray) -> EmbeddingResult:
        return EmbeddingResult(
            embedding=vector,
            metadata={"provider": "openrouter", "model": self.model, "text_length": len(text)},
            model=self.model,
            dimensions=int(vector.size),
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
        }
