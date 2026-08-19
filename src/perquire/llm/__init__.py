"""LLM provider registry with optional integrations."""

from importlib import import_module
from typing import Any

from .base import BaseLLMProvider, LLMProviderError, LLMResponse, provider_registry
from ..exceptions import ConfigurationError

_PROVIDER_SPECS = {
    "openai": (".openai_provider", "OpenAIProvider", {"model": "gpt-3.5-turbo"}),
    "gemini": (".gemini_provider", "GeminiProvider", {"model": "gemini-2.5-flash-lite-preview-06-17"}),
    "anthropic": (".anthropic_provider", "AnthropicProvider", {"model": "claude-3-sonnet-20240229"}),
    "ollama": (".ollama_provider", "OllamaProvider", {"model": "llama2", "base_url": "http://localhost:11434"}),
}


def _load_provider(name: str) -> type[BaseLLMProvider] | None:
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
            provider_registry.register_provider(
                name,
                provider_class(config=dict(config)),
                set_as_default=not registered,
            )
            registered.append(name)
        except (LLMProviderError, ConfigurationError):
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


__all__ = ["BaseLLMProvider", "LLMResponse", "provider_registry", "register_available_providers"]
