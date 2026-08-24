"""Regression tests for the Perquire OpenRouter generation default."""

from perquire.llm import _PROVIDER_SPECS
from perquire.llm.openrouter_provider import DEFAULT_MODEL, OpenRouterProvider


EXPECTED_OPENROUTER_MODEL = "stealth/ox-alpha"


def test_openrouter_default_is_ox_alpha(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    provider = OpenRouterProvider(config={"requests_per_minute": 0})

    assert DEFAULT_MODEL == EXPECTED_OPENROUTER_MODEL
    assert provider.model == EXPECTED_OPENROUTER_MODEL


def test_registry_and_provider_defaults_cannot_drift():
    registry_default = _PROVIDER_SPECS["openrouter"][2]["model"]

    assert registry_default == DEFAULT_MODEL == EXPECTED_OPENROUTER_MODEL
