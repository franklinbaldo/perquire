"""
LLM provider integrations for Perquire.

This module automatically registers available LLM providers with the central registry.
"""

from .base import BaseLLMProvider, provider_registry, LLMProviderError
from ..exceptions import ConfigurationError
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .ollama_provider import OllamaProvider

# Default configuration for providers if not specified elsewhere
# These are minimal configs; users should provide API keys etc.
DEFAULT_PROVIDER_CONFIGS = {
    "openai": {"model": "gpt-3.5-turbo"},
    "gemini": {"model": "gemini-2.5-flash-lite-preview-06-17"},
    "anthropic": {"model": "claude-3-sonnet-20240229"},
    "ollama": {"model": "llama2", "base_url": "http://localhost:11434"},
}

# Register providers
# Users can override these registrations or add new ones.
# The Investigator will use these by default.
try:
    provider_registry.register_provider(
        "openai",
        OpenAIProvider(config=DEFAULT_PROVIDER_CONFIGS["openai"]),
        set_as_default=True # Example: set OpenAI as default if available
    )
except (LLMProviderError, ConfigurationError) as e:
    print(f"Note: OpenAI LLM provider not fully configured: {e}")

try:
    provider_registry.register_provider(
        "gemini",
        GeminiProvider(config=DEFAULT_PROVIDER_CONFIGS["gemini"])
    )
except (LLMProviderError, ConfigurationError) as e:
    print(f"Note: Gemini LLM provider not fully configured: {e}")

try:
    provider_registry.register_provider(
        "anthropic",
        AnthropicProvider(config=DEFAULT_PROVIDER_CONFIGS["anthropic"])
    )
except (LLMProviderError, ConfigurationError) as e:
    print(f"Note: Anthropic LLM provider not fully configured: {e}")

try:
    provider_registry.register_provider(
        "ollama",
        OllamaProvider(config=DEFAULT_PROVIDER_CONFIGS["ollama"])
    )
except (LLMProviderError, ConfigurationError) as e:
    print(f"Note: Ollama LLM provider not fully configured: {e}")


__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "provider_registry", # Expose the registry
]