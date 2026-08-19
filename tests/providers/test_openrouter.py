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


def _embedding_returning(vector: list[float]):
    def _call(**kwargs):
        _call.kwargs = kwargs
        return {"data": [{"embedding": vector}], "model": "openrouter/test-embed"}

    return _call


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    return OpenRouterProvider(config={"model": "openai/gpt-oss-20b:free", "requests_per_minute": 0})


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    return OpenRouterEmbeddingProvider(
        config={"model": "nvidia/nemotron-3-embed-1b:free", "requests_per_minute": 0}
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


def test_llm_and_embeddings_share_the_same_openrouter_quota(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm_provider = OpenRouterProvider(
        config={"model": "openai/gpt-oss-20b:free", "requests_per_minute": 20}
    )
    embedding_provider = OpenRouterEmbeddingProvider(
        config={"model": "nvidia/nemotron-3-embed-1b:free", "requests_per_minute": 20}
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
    """A short sleep must not compound into ever-longer waits.

    time.sleep can return early, so the pacer cannot assume the clock advanced
    by the amount it asked for. Reading a clock that appears to run backwards
    would otherwise grow every subsequent wait: 3s, 6s, 9s, ...
    """
    slept: list[float] = []
    pacer = RequestPacer(
        requests_per_minute=20, clock=lambda: 0.0, sleep=lambda seconds: slept.append(seconds)
    )

    for _ in range(4):
        pacer.wait()

    assert slept == [pytest.approx(3.0)] * 3
