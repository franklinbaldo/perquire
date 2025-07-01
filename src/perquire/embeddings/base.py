"""
Base embedding provider interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

from ..exceptions import EmbeddingError, ConfigurationError


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: np.ndarray
    metadata: Dict[str, Any]
    model: Optional[str] = None
    dimensions: Optional[int] = None


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.
    
    This class defines the interface that all embedding providers must implement
    to work with Perquire's investigation system.
    It includes LRU caching for embedding calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding provider with configuration.
        
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

    # --- Cached public methods ---
    
    def embed_text(self, text: str, **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text with caching.
        
        Args:
            text: Text to embed
            **kwargs: Provider-specific parameters (must be hashable for caching)
            
        Returns:
            EmbeddingResult containing the embedding and metadata
        """
        kwargs_tuple = tuple(sorted(kwargs.items()))
        return self._cached_execute_embed_text(text, kwargs_tuple=kwargs_tuple)

    def embed_batch(self, texts: List[str], **kwargs) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts with caching.
        
        Args:
            texts: List of texts to embed
            **kwargs: Provider-specific parameters (must be hashable for caching)
            
        Returns:
            List of EmbeddingResult objects
        """
        texts_tuple = tuple(texts) # Convert list to tuple for hashability
        kwargs_tuple = tuple(sorted(kwargs.items()))
        return self._cached_execute_embed_batch(texts_tuple, kwargs_tuple=kwargs_tuple)

    # --- Internal cached execution methods ---

    @lru_cache(maxsize=1024) # Adjust maxsize as needed
    def _cached_execute_embed_text(self, text: str, kwargs_tuple: Tuple) -> EmbeddingResult:
        """Internal cached method for single text embedding."""
        kwargs_dict = dict(kwargs_tuple)
        return self._execute_embed_text(text, **kwargs_dict)

    @lru_cache(maxsize=128) # Adjust maxsize as needed for batches
    def _cached_execute_embed_batch(self, texts_tuple: Tuple[str, ...], kwargs_tuple: Tuple) -> List[EmbeddingResult]:
        """Internal cached method for batch text embedding."""
        texts_list = list(texts_tuple)
        kwargs_dict = dict(kwargs_tuple)
        return self._execute_embed_batch(texts_list, **kwargs_dict)

    # --- Abstract methods for concrete providers to implement ---

    @abstractmethod
    def _execute_embed_text(self, text: str, **kwargs) -> EmbeddingResult:
        """
        Core logic for generating embedding for a single text.
        To be implemented by concrete providers.
        """
        pass

    @abstractmethod
    def _execute_embed_batch(self, texts: List[str], **kwargs) -> List[EmbeddingResult]:
        """
        Core logic for generating embeddings for a batch of texts.
        To be implemented by concrete providers.
        """
        pass
    
    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider.
        
        Returns:
            Number of dimensions in the embedding vectors
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
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    def similarity(
        self, 
        embedding1: Union[np.ndarray, EmbeddingResult], 
        embedding2: Union[np.ndarray, EmbeddingResult]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Extract numpy arrays if EmbeddingResult objects
        if isinstance(embedding1, EmbeddingResult):
            embedding1 = embedding1.embedding
        if isinstance(embedding2, EmbeddingResult):
            embedding2 = embedding2.embedding
        
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        return float(np.dot(embedding1_norm, embedding2_norm))
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding has the correct format and dimensions.
        
        Args:
            embedding: Embedding to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(embedding, np.ndarray):
                return False
            
            if len(embedding.shape) != 1:
                return False
            
            expected_dims = self.get_embedding_dimensions()
            if embedding.shape[0] != expected_dims:
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(embedding).all():
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this provider.
        
        Returns:
            Dictionary with default configuration values
        """
        return {}
    
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
            
            # Try a simple test embedding
            test_result = self.embed_text("Test text for health check")
            
            if not self.validate_embedding(test_result.embedding):
                return {
                    "status": "unhealthy",
                    "reason": "Invalid embedding produced"
                }
            
            return {
                "status": "healthy",
                "model_info": self.get_model_info(),
                "embedding_dimensions": self.get_embedding_dimensions(),
                "test_embedding_norm": float(np.linalg.norm(test_result.embedding))
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e)
            }


class EmbeddingProviderRegistry:
    """Registry for managing multiple embedding providers."""
    
    def __init__(self):
        self._providers: Dict[str, BaseEmbeddingProvider] = {}
        self._default_provider: Optional[str] = None
    
    def register_provider(self, name: str, provider: BaseEmbeddingProvider, set_as_default: bool = False):
        """
        Register an embedding provider.
        
        Args:
            name: Provider name
            provider: Provider instance
            set_as_default: Whether to set as default provider
        """
        self._providers[name] = provider
        
        if set_as_default or self._default_provider is None:
            self._default_provider = name
    
    def get_provider(self, name: Optional[str] = None) -> BaseEmbeddingProvider:
        """
        Get a provider by name or return default.
        
        Args:
            name: Provider name, or None for default
            
        Returns:
            Embedding provider instance
            
        Raises:
            EmbeddingError: If provider not found
        """
        if name is None:
            name = self._default_provider
        
        if name is None:
            raise EmbeddingError("No default provider set")
        
        if name not in self._providers:
            raise EmbeddingError(f"Provider '{name}' not found")
        
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
            EmbeddingError: If provider not found
        """
        if name not in self._providers:
            raise EmbeddingError(f"Provider '{name}' not found")
        
        self._default_provider = name


# Global provider registry instance
embedding_registry = EmbeddingProviderRegistry()