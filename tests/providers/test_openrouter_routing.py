from dataclasses import dataclass

import numpy as np
import pytest

from perquire.embeddings.openrouter_embeddings import OpenRouterEmbeddingProvider
from perquire.llm.openrouter_provider import OpenRouterProvider
from perquire.providers.openrouter_routing import provider_routing


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list


def test_provider_order_defaults_to_no_fallback_when_explicit():
    assert provider_routing({"provider_order": ["provider-a"]}) == {
        "order": ["provider-a"],
        "allow_fallbacks": False,
    }


def test_unconfigured_provider_routing_preserves_legacy_dynamic_behavior():
    assert provider_routing({}) is None


def test_duplicate_provider_order_is_rejected():
    with pytest.raises(ValueError):
        provider_routing({"provider_order": ["same", "same"]})


def test_llm_sends_openrouter_provider_object_through_litellm(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _Completion(choices=[_Choice(message=_Message(content="candidate"))])

    monkeypatch.setattr("perquire.llm.openrouter_provider.completion", completion)
    provider = OpenRouterProvider(
        config={
            "model": "example/model",
            "provider_order": ["provider-a"],
            "allow_fallbacks": False,
            "requests_per_minute": 0,
            "cache_mode": "off",
            "cache_path": tmp_path / "llm.sqlite3",
        }
    )
    result = provider.generate_response("prompt")

    assert result.content == "candidate"
    assert calls[0]["extra_body"] == {
        "provider": {"order": ["provider-a"], "allow_fallbacks": False}
    }
    assert provider.get_model_info()["provider_routing"] == {
        "order": ["provider-a"],
        "allow_fallbacks": False,
    }


def test_embedding_sends_same_provider_object_and_scopes_cache_by_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    calls = []

    def embedding(**kwargs):
        calls.append(kwargs)
        return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr("perquire.embeddings.openrouter_embeddings.embedding", embedding)
    path = tmp_path / "embeddings.sqlite3"
    pinned = OpenRouterEmbeddingProvider(
        config={
            "model": "example/embed",
            "provider_order": ["provider-a"],
            "allow_fallbacks": False,
            "requests_per_minute": 0,
            "cache_path": path,
        }
    )
    result = pinned.embed_text("same text")

    assert calls[0]["extra_body"] == {
        "provider": {"order": ["provider-a"], "allow_fallbacks": False}
    }
    assert "provider_order=provider-a;allow_fallbacks=false" in pinned.cache_space
    assert result.metadata["provider_routing"]["order"] == ["provider-a"]

    # The legacy unpinned cache space must not see the vector created through the
    # explicitly pinned provider path, even though the model slug and text match.
    unpinned = OpenRouterEmbeddingProvider(
        config={
            "model": "example/embed",
            "requests_per_minute": 0,
            "cache_path": path,
        }
    )
    assert unpinned.cached_embedding("same text") is None
    np.testing.assert_allclose(pinned.cached_embedding("same text"), [1.0, 0.0])
