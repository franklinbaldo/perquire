"""Persistent LLM cache must save requests without confusing replay with sampling."""

from dataclasses import dataclass

from perquire.llm.openrouter_provider import OpenRouterProvider
from perquire.providers.llm_cache import LLMResponseCache


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list


def _provider(monkeypatch, tmp_path, *, mode="replay"):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    return OpenRouterProvider(
        config={
            "model": "openai/gpt-oss-20b:free",
            "requests_per_minute": 0,
            "cache_mode": mode,
            "cache_path": tmp_path / "llm.sqlite3",
        }
    )


def test_cache_key_changes_with_logical_request_even_for_same_prompt(tmp_path):
    cache = LLMResponseCache(tmp_path / "llm.sqlite3")
    common = {
        "model": "openrouter/test",
        "prompt": "same prompt",
        "parameters": {"temperature": 0.7, "max_tokens": 256},
    }
    first = cache.key(logical_request="case-a:independent:0", **common)
    second = cache.key(logical_request="case-b:independent:0", **common)
    assert first != second


def test_replay_persists_across_provider_instances_and_skips_transport(monkeypatch, tmp_path):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _Completion(choices=[_Choice(message=_Message(content="cached candidate"))])

    monkeypatch.setattr("perquire.llm.openrouter_provider.completion", completion)
    first = _provider(monkeypatch, tmp_path)
    response = first.generate_response("prompt", cache_request_id="case-a:independent:0")
    assert response.content == "cached candidate"
    assert first.transport_attempts == 1
    assert first.cache_writes == 1

    second = _provider(monkeypatch, tmp_path)
    replayed = second.generate_response("prompt", cache_request_id="case-a:independent:0")
    assert replayed.content == "cached candidate"
    assert second.transport_attempts == 0
    assert second.cache_hits == 1
    assert len(calls) == 1


def test_same_prompt_in_different_cases_does_not_replay(monkeypatch, tmp_path):
    outputs = iter(["case a", "case b"])

    def completion(**kwargs):
        return _Completion(choices=[_Choice(message=_Message(content=next(outputs)))])

    monkeypatch.setattr("perquire.llm.openrouter_provider.completion", completion)
    provider = _provider(monkeypatch, tmp_path)
    a = provider.generate_response("same prompt", cache_request_id="case-a:independent:0")
    b = provider.generate_response("same prompt", cache_request_id="case-b:independent:0")
    assert a.content == "case a"
    assert b.content == "case b"
    assert provider.transport_attempts == 2
    assert provider.cache_hits == 0


def test_fresh_mode_never_replays_and_keeps_new_samples(monkeypatch, tmp_path):
    outputs = iter(["sample one", "sample two"])

    def completion(**kwargs):
        return _Completion(choices=[_Choice(message=_Message(content=next(outputs)))])

    monkeypatch.setattr("perquire.llm.openrouter_provider.completion", completion)
    provider = _provider(monkeypatch, tmp_path, mode="fresh")
    first = provider.generate_response("prompt", cache_request_id="case-a:adaptive:0")
    second = provider.generate_response("prompt", cache_request_id="case-a:adaptive:0")
    assert first.content == "sample one"
    assert second.content == "sample two"
    assert provider.transport_attempts == 2
    assert provider.cache_hits == 0
    assert provider.cache_writes == 2

    replay = _provider(monkeypatch, tmp_path, mode="replay")
    observed = replay.generate_response("prompt", cache_request_id="case-a:adaptive:0")
    assert observed.content == "sample one"
    assert replay.transport_attempts == 0
    assert replay.cache_hits == 1


def test_parameter_change_invalidates_replay(monkeypatch, tmp_path):
    outputs = iter(["cold", "hot"])

    def completion(**kwargs):
        return _Completion(choices=[_Choice(message=_Message(content=next(outputs)))])

    monkeypatch.setattr("perquire.llm.openrouter_provider.completion", completion)
    provider = _provider(monkeypatch, tmp_path)
    first = provider.generate_response(
        "prompt", cache_request_id="case-a:independent:0", temperature=0.1
    )
    second = provider.generate_response(
        "prompt", cache_request_id="case-a:independent:0", temperature=0.9
    )
    assert first.content == "cold"
    assert second.content == "hot"
    assert provider.transport_attempts == 2
