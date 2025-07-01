"""
Provider factory for LLM and embedding services.
Lazy-loads dependencies to avoid ImportError for unused providers.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass

class ProviderNotInstalledError(ProviderError):
    """Raised when required dependencies for a provider are not installed."""
    pass

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        pass

def get_embedding_provider(name: str, **config) -> EmbeddingProvider:
    """Get embedding provider by name with lazy loading."""
    
    if name == "openai":
        try:
            from .openai_provider import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(**config)
        except ImportError:
            raise ProviderNotInstalledError(
                f"OpenAI provider requires 'api-openai' extra.\n"
                f"Install with: pip install perquire[api-openai]"
            )
    
    elif name == "gemini":
        try:
            from .gemini_provider import GeminiEmbeddingProvider
            return GeminiEmbeddingProvider(**config)
        except ImportError:
            raise ProviderNotInstalledError(
                f"Gemini provider requires 'api-gemini' extra.\n"
                f"Install with: pip install perquire[api-gemini]"
            )
    
    elif name == "sentence-transformers" or name == "sbert":
        try:
            from .local_provider import SentenceTransformerProvider
            return SentenceTransformerProvider(**config)
        except ImportError:
            raise ProviderNotInstalledError(
                f"Local embeddings require 'local-embeddings' extra.\n"
                f"Install with: pip install perquire[local-embeddings]"
            )
    
    else:
        raise ProviderError(f"Unknown embedding provider: {name}")

def get_llm_provider(name: str, **config) -> LLMProvider:
    """Get LLM provider by name with lazy loading."""
    
    if name == "openai":
        try:
            from .openai_provider import OpenAILLMProvider
            return OpenAILLMProvider(**config)
        except ImportError:
            raise ProviderNotInstalledError(
                f"OpenAI provider requires 'api-openai' extra.\n"
                f"Install with: pip install perquire[api-openai]"
            )
    
    elif name == "gemini":
        try:
            from .gemini_provider import GeminiLLMProvider
            return GeminiLLMProvider(**config)
        except ImportError:
            raise ProviderNotInstalledError(
                f"Gemini provider requires 'api-gemini' extra.\n"
                f"Install with: pip install perquire[api-gemini]"
            )
    
    elif name == "anthropic":
        try:
            from .anthropic_provider import AnthropicLLMProvider
            return AnthropicLLMProvider(**config)
        except ImportError:
            raise ProviderNotInstalledError(
                f"Anthropic provider requires 'api-anthropic' extra.\n"
                f"Install with: pip install perquire[api-anthropic]"
            )
    
    elif name == "ollama":
        try:
            from .ollama_provider import OllamaLLMProvider
            return OllamaLLMProvider(**config)
        except ImportError:
            raise ProviderNotInstalledError(
                f"Ollama provider requires 'api-ollama' extra.\n"
                f"Install with: pip install perquire[api-ollama]"
            )
    
    else:
        raise ProviderError(f"Unknown LLM provider: {name}")

def list_available_providers() -> Dict[str, Dict[str, Any]]:
    """List all available providers and their installation status."""
    
    providers = {
        "embedding": {
            "openai": {"extra": "api-openai", "installed": False},
            "gemini": {"extra": "api-gemini", "installed": False},
            "sentence-transformers": {"extra": "local-embeddings", "installed": False},
        },
        "llm": {
            "openai": {"extra": "api-openai", "installed": False},
            "gemini": {"extra": "api-gemini", "installed": False},
            "anthropic": {"extra": "api-anthropic", "installed": False},
            "ollama": {"extra": "api-ollama", "installed": False},
        }
    }
    
    # Check installation status by trying imports directly
    for provider_type, provider_list in providers.items():
        for provider_name, info in provider_list.items():
            try:
                if provider_name == "openai":
                    import openai
                elif provider_name == "gemini":
                    import google.generativeai
                elif provider_name == "anthropic":
                    import anthropic
                elif provider_name == "ollama":
                    import ollama
                elif provider_name == "sentence-transformers":
                    import sentence_transformers
                    import torch
                info["installed"] = True
            except ImportError:
                info["installed"] = False
            except Exception:
                info["installed"] = False
    
    return providers