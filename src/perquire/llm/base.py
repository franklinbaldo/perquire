"""
Base LLM provider interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..exceptions import LLMProviderError, ConfigurationError


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM providers must implement
    to work with Perquire's investigation system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The main prompt/question
            context: Optional context information
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse object containing the generated content
            
        Raises:
            LLMProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_questions(
        self,
        current_description: str,
        target_similarity: float,
        phase: str,
        previous_questions: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate investigation questions based on current progress.
        
        Args:
            current_description: Current best description of the embedding
            target_similarity: Current similarity score
            phase: Investigation phase (exploration, refinement, convergence)
            previous_questions: Previously asked questions to avoid repetition
            **kwargs: Provider-specific parameters
            
        Returns:
            List of generated questions
            
        Raises:
            LLMProviderError: If question generation fails
        """
        pass
    
    @abstractmethod
    def synthesize_description(
        self,
        questions_and_scores: List[Dict[str, Any]],
        final_similarity: float,
        **kwargs
    ) -> str:
        """
        Synthesize a final description from investigation results.
        
        Args:
            questions_and_scores: List of question/similarity pairs
            final_similarity: Final similarity score achieved
            **kwargs: Provider-specific parameters
            
        Returns:
            Synthesized description string
            
        Raises:
            LLMProviderError: If synthesis fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available/configured properly.
        
        Returns:
            True if provider is ready to use, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this provider.
        
        Returns:
            Dictionary with default configuration values
        """
        return {
            "temperature": 0.7,
            "max_tokens": 150,
            "timeout": 30
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        Returns:
            Dictionary with health check results
        """
        try:
            if not self.is_available():
                return {
                    "status": "unhealthy",
                    "reason": "Provider not available"
                }
            
            # Try a simple test request
            test_response = self.generate_response(
                "Test prompt for health check",
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "model_info": self.get_model_info(),
                "test_response_length": len(test_response.content)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e)
            }


class LLMProviderRegistry:
    """Registry for managing multiple LLM providers."""
    
    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._default_provider: Optional[str] = None
    
    def register_provider(self, name: str, provider: BaseLLMProvider, set_as_default: bool = False):
        """
        Register an LLM provider.
        
        Args:
            name: Provider name
            provider: Provider instance
            set_as_default: Whether to set as default provider
        """
        self._providers[name] = provider
        
        if set_as_default or self._default_provider is None:
            self._default_provider = name
    
    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        """
        Get a provider by name or return default.
        
        Args:
            name: Provider name, or None for default
            
        Returns:
            LLM provider instance
            
        Raises:
            LLMProviderError: If provider not found
        """
        if name is None:
            name = self._default_provider
        
        if name is None:
            raise LLMProviderError("No default provider set")
        
        if name not in self._providers:
            raise LLMProviderError(f"Provider '{name}' not found")
        
        return self._providers[name]
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of providers that pass health checks."""
        healthy = []
        for name, provider in self._providers.items():
            health = provider.health_check()
            if health["status"] == "healthy":
                healthy.append(name)
        return healthy
    
    def set_default_provider(self, name: str):
        """
        Set the default provider.
        
        Args:
            name: Provider name
            
        Raises:
            LLMProviderError: If provider not found
        """
        if name not in self._providers:
            raise LLMProviderError(f"Provider '{name}' not found")
        
        self._default_provider = name


# Global provider registry instance
provider_registry = LLMProviderRegistry()