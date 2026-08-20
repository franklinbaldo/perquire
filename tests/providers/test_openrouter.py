"""Contract tests for the OpenRouter providers.

Both provider roles reach OpenRouter through litellm, so the transport is stubbed
here: these tests fix the adapter contract, not the network.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from perquire.embeddings.openrouter_embeddings import OpenRouterEmbeddingProvider
from perquire.exceptions import ConfigurationError
from perquire.llm.openrouter_provider import OpenRouterProvider
from perquire.providers.rate_limit import RequestPacer


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list
    model: str = "openrouter/test-model"
    usage: object = None


def _completion_returning(content: str):
    def _call(**kwargs):
        _call.kwargs = kwargs
        return _Completion(choices=[_Choice(message=_Message(content=content))])

    return _call


def _embedding_returning(vector: list[float], *, provider: str | None = None):
    def _call(**kwargs):
        _call.kwargs = kwargs
        response = {"data": [{"embedding": vector}], "model": "openrouter/test-embed"}
        if provider is not None:
            response["provider"] = provider
        return response

    return _call


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    return OpenRouterProvider(config={"model": "openai/gpt-oss-20b:free", "requests_per_minute": 0})


@pytest.fixture
def embedder(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    return OpenRouterEmbeddingProvider(
        config={
            "model": "nvidia/nemotron-3-embed-1b:free",
            "requests_per_minute": 0,
            "cache_path": tmp_path / "embeddings.sqlite3",
        }
    )


def test_missing_credential_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        OpenRouterProvider(config={"model": "openai/gpt-oss-20b:free"})


def test_model_is_namespaced_to_openrouter(llm, monkeypatch):
    call = _completion_returning("hello")
    monkeypatch.setattr("perquire.llm.openrouter_provider.completion", call)
    llm.generate_response("prompt")
    assert call.kwargs["model"] == "openrouter/openai/gpt-oss-20b:free"


def test_generate_response_returns_content(llm, monkeypatch):
    monkeypatch.setattr(
        "perquire.llm.openrouter_provider.completion", _completion_returning("a description")
    )
    response = llm.generate_response("prompt")
    assert response.content == "a description"
    assert response.metadata["provider"] == "openrouter"


def test_generate_questions_splits_lines_and_drops_bullets(llm, monkeypatch):
    monkeypatch.setattr(
        "perquire.llm.openrouter_provider.completion",
        _completion_returning("- first candidate\n2. second candidate\n\n"),
    )
    questions = llm.generate_questions(
        current_description="", target_similarity=0.0, phase="exploration"
    )
    assert questions == ["first candidate", "second candidate"]


def test_embedding_returns_a_float_vector(embedder, monkeypatch):
    monkeypatch.setattr(
        "perquire.embeddings.openrouter_embeddings.embedding",
        _embedding_returning([0.5, 0.5, 0.0]),
    )
    result = embedder.embed_text("some text")
    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.dtype == np.float64
    assert result.dimensions == 3


def test_embedding_model_is_namespaced_to_openrouter(embedder, monkeypatch):
    call = _embedding_returning([1.0, 0.0])
    monkeypatch.setattr("perquire.embeddings.openrouter_embeddings.embedding", call)
    embedder.embed_text("some text")
    assert call.kwargs["model"] == "openrouter/nvidia/nemotron-3-embed-1b:free"


def test_embedding_records_served_upstream_when_response_exposes_it(embedder, monkeypatch):
    monkeypatch.setattr(
        "perquire.embeddings.openrouter_embeddings.embedding",
        _embedding_returning([1.0, 0.0], provider="ExampleBackend"),
    )
    embedder.embed_text("served upstream")
    info = embedder.get_model_info()
    assert info["served_upstream_providers"] == ["ExampleBackend"]
    assert info["served_upstream_provider_counts"] == {"ExampleBackend": 1}
    assert info["successful_responses_without_upstream_provider"] == 0


def test_embedding_marks_unobservable_upstream_without_guessing(embedder, monkeypatch):
    monkeypatch.setattr(
        "perquire.embeddings.openrouter_embeddings.embedding",
        _embedding_returning([1.0, 0.0]),
    )
    embedder.embed_text("unknown upstream")
    info = embedder.get_model_info()
    assert info["served_upstream_providers"] == []
    assert info["successful_responses_without_upstream_provider"] == 1
    assert info["last_served_upstream_provider"] is None


def test_embedding_cache_skips_repeat_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    path = tmp_path / "persisted.sqlite3"
    call = _embedding_returning([0.25, 0.75])
    monkeypatch.setattr("perquire.embeddings.openrouter_embeddings.embedding", call)

    first = OpenRouterEmbeddingProvider(
        config={"requests_per_minute": 0, "cache_path": path}
    )
    first_result = first.embed_text("reuse me")
    assert first.transport_attempts == 1
    assert first.cache_misses == 1
    assert first.cache_writes == 1

    def forbidden(**kwargs):
        raise AssertionError("transport must not run on a persisted cache hit")

    monkeypatch.setattr("perquire.embeddings.openrouter_embeddings.embedding", forbidden)
    second = OpenRouterEmbeddingProvider(
        config={"requests_per_minute": 0, "cache_path": path}
    )
    second_result = second.embed_text("reuse me")

    np.testing.assert_allclose(second_result.embedding, first_result.embedding)
    assert second.transport_attempts == 0
    assert second.cache_hits == 1
    assert second.cache_misses == 0


def test_uncached_embedding_bypasses_cache_without_overwriting_it(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    path = tmp_path / "persisted.sqlite3"
    responses = iter(
        [
            {"data": [{"embedding": [1.0, 0.0]}], "provider": "OldBackend"},
            {"data": [{"embedding": [0.0, 1.0]}], "provider": "NewBackend"},
        ]
    )

    def call(**kwargs):
        return next(responses)

    monkeypatch.setattr("perquire.embeddings.openrouter_embeddings.embedding", call)
    provider = OpenRouterEmbeddingProvider(config={"requests_per_minute": 0, "cache_path": path})
    cached = provider.embed_text("same text")
    fresh = provider.embed_text_uncached("same text")
    still_cached = provider.cached_embedding("same text")

    np.testing.assert_allclose(cached.embedding, [1.0, 0.0])
    np.testing.assert_allclose(fresh.embedding, [0.0, 1.0])
    np.testing.assert_allclose(still_cached, [1.0, 0.0])
    assert fresh.metadata["cache_status"] == "bypass"
    assert fresh.metadata["upstream_provider"] == "NewBackend"


def test_pacer_waits_to_respect_the_configured_rate():
    now = [0.0]
    slept: list[float] = []
    pacer = RequestPacer(
        requests_per_minute=20, clock=lambda: now[0], sleep=lambda s: slept.append(s)
    )
    pacer.wait()
    pacer.wait()
    assert slept == [pytest.approx(3.0)]


def test_pacer_disabled_never_sleeps():
    slept: list[float] = []
    pacer = RequestPacer(requests_per_minute=0, clock=lambda: 0.0, sleep=lambda s: slept.append(s))
    pacer.wait()
    pacer.wait()
    assert slept == []


def test_llm_and_embeddings_share_the_same_openrouter_quota(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm_provider = OpenRouterProvider(
        config={"model": "openai/gpt-oss-20b:free", "requests_per_minute": 20}
    )
    embedding_provider = OpenRouterEmbeddingProvider(
        config={
            "model": "nvidia/nemotron-3-embed-1b:free",
            "requests_per_minute": 20,
            "cache_path": tmp_path / "embeddings.sqlite3",
        }
    )

    assert llm_provider._pacer is embedding_provider._pacer

    now = [0.0]
    slept: list[float] = []
    shared = llm_provider._pacer
    shared.clock = lambda: now[0]
    shared.sleep = lambda seconds: slept.append(seconds)
    shared._last_request_at = None

    llm_provider._pacer.wait()
    embedding_provider._pacer.wait()
    llm_provider._pacer.wait()

    assert slept == [pytest.approx(3.0), pytest.approx(3.0)]


def test_parse_lines_strips_list_markers_but_keeps_leading_digits():
    from perquire.llm.openrouter_provider import parse_lines

    assert parse_lines("- 3D printing techniques") == ["3D printing techniques"]
    assert parse_lines("2. 1980s synthpop revival") == ["1980s synthpop revival"]
    assert parse_lines("1980s synthpop revival") == ["1980s synthpop revival"]
    assert parse_lines("* a plain bullet") == ["a plain bullet"]


def test_pacer_does_not_inflate_waits_when_sleep_returns_early():
    slept: list[float] = []
    pacer = RequestPacer(
        requests_per_minute=20, clock=lambda: 0.0, sleep=lambda seconds: slept.append(seconds)
    )

    for _ in range(4):
        pacer.wait()

    assert slept == [pytest.approx(3.0)] * 3
