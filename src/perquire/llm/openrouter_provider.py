"""OpenRouter LLM provider.

OpenRouter is reached through litellm, which namespaces models as
``openrouter/<vendor>/<model>`` and reads ``OPENROUTER_API_KEY`` from the
environment.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from litellm import completion

from ..exceptions import ConfigurationError
from ..providers.llm_cache import (
    LLMResponseCache,
    default_llm_cache_path,
    normalize_llm_cache_mode,
)
from ..providers.rate_limit import get_shared_pacer
from ..providers.retry import EmptyProviderResponse, call_with_transient_retries
from .base import BaseLLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_REQUESTS_PER_MINUTE = 20
DEFAULT_MAX_RETRIES = 2
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def _strip_bullet(line: str) -> str:
    return _LIST_MARKER.sub("", line.strip()).strip()


def parse_lines(content: str) -> list[str]:
    """Split model output into clean, de-duplicated, non-empty lines."""
    lines: list[str] = []
    for raw in content.splitlines():
        text = _strip_bullet(raw)
        if text and text not in lines:
            lines.append(text)
    return lines


class OpenRouterProvider(BaseLLMProvider):
    """Generate candidate text through any OpenRouter-hosted chat model."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        rpm = int(self.config.get("requests_per_minute", DEFAULT_REQUESTS_PER_MINUTE))
        self._pacer = get_shared_pacer("openrouter", rpm)
        self._max_retries = int(self.config.get("max_retries", DEFAULT_MAX_RETRIES))
        self.transport_attempts = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self._cache_mode = normalize_llm_cache_mode(self.config.get("cache_mode"))
        cache_path = self.config.get("cache_path") or default_llm_cache_path()
        self._cache = None if self._cache_mode == "off" else LLMResponseCache(cache_path)

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

    def _complete(self, prompt: str, **kwargs: Any) -> str:
        logical_request = kwargs.pop("cache_request_id", None)
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 256))
        parameters = {"temperature": temperature, "max_tokens": max_tokens}

        if self._cache is not None and self._cache_mode == "replay" and logical_request:
            cached = self._cache.get_first(
                model=self.model,
                prompt=prompt,
                logical_request=str(logical_request),
                parameters=parameters,
            )
            if cached is not None:
                self.cache_hits += 1
                return cached
            self.cache_misses += 1

        def request():
            self.transport_attempts += 1
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._api_key(),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise EmptyProviderResponse("OpenRouter returned an empty completion")
            return content

        try:
            content = call_with_transient_retries(
                request,
                before_attempt=self._pacer.wait,
                max_retries=self._max_retries,
            )
        except Exception as error:
            logger.exception("OpenRouter completion failed")
            raise LLMProviderError(f"OpenRouter completion failed: {error}") from error

        if self._cache is not None and logical_request:
            self._cache.put(
                model=self.model,
                prompt=prompt,
                logical_request=str(logical_request),
                parameters=parameters,
                content=content,
            )
            self.cache_writes += 1
        return content

    def generate_response(self, prompt: str, **kwargs: Any) -> LLMResponse:
        content = self._complete(prompt, **kwargs)
        return LLMResponse(
            content=content,
            metadata={"provider": "openrouter", "model": self.model},
            model=self.model,
        )

    def generate_questions(
        self,
        current_description: str,
        target_similarity: float,
        phase: str,
        previous_questions: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        previous = previous_questions or []
        avoid = "\n".join(f"- {item}" for item in previous[-10:])
        prompt = (
            "You are recovering the meaning of a hidden text using only similarity feedback.\n"
            f"Investigation phase: {phase}.\n"
            f"Best description so far: {current_description or '(none yet)'}\n"
            f"Its similarity to the target: {target_similarity:.4f}\n"
            f"{'Already tried, do not repeat:' + chr(10) + avoid if avoid else ''}\n"
            "Propose one concise new description that should score higher. "
            "Return the description only, with no commentary."
        )
        return parse_lines(self._complete(prompt, **kwargs))

    def synthesize_description(
        self,
        questions_and_scores: list[dict[str, Any]],
        final_similarity: float,
        **kwargs: Any,
    ) -> str:
        ranked = sorted(
            questions_and_scores, key=lambda item: item.get("similarity", 0.0), reverse=True
        )
        evidence = "\n".join(
            f"- {item.get('question', '')} (similarity {item.get('similarity', 0.0):.4f})"
            for item in ranked[:10]
        )
        prompt = (
            "These candidate descriptions were scored against a hidden text:\n"
            f"{evidence}\n"
            f"Best similarity reached: {final_similarity:.4f}\n"
            "Write one concise description of the hidden text. Return the description only."
        )
        lines = parse_lines(self._complete(prompt, **kwargs))
        return lines[0] if lines else ""

    def is_available(self) -> bool:
        return bool(self._api_key())

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "model": self.model,
            "requests_per_minute": self._pacer.requests_per_minute,
            "max_retries": self._max_retries,
            "transport_attempts": self.transport_attempts,
            "cache_mode": self._cache_mode,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_writes": self.cache_writes,
        }
