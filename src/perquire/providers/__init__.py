"""
Utility functions related to provider dependencies and availability.

This module helps in checking if the necessary dependencies for various
LLM and embedding providers are installed. It's primarily intended for
user feedback, e.g., in a CLI.
"""

from typing import Dict, Any

# Consolidate custom exceptions here or ensure they are in perquire.exceptions
class ProviderError(Exception):
    """Base exception for provider-related errors, typically dependency issues."""
    pass

class ProviderNotInstalledError(ProviderError):
    """Raised when required dependencies for a provider are not installed."""
    pass


# This function can be used by a CLI to inform users about optional dependencies.
def list_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    List all supported providers and their installation status for optional extras.
    
    This checks if the core libraries for each provider are importable.
    Actual functionality also depends on API keys and correct configuration.
    """
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