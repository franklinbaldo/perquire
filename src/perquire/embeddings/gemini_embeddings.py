"""
Google Gemini embeddings provider using LlamaIndex.
"""

import os
from typing import Dict, Any, Optional, List
import logging
import numpy as np

from llama_index.embeddings.gemini import GeminiEmbedding

from .base import BaseEmbeddingProvider, EmbeddingResult
from ..exceptions import EmbeddingError, ConfigurationError, RateLimitError

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Google Gemini embedding provider implementation using LlamaIndex.
    
    This provider uses Google's text-embedding-004 and models-embedding-001 models
    through LlamaIndex integration for high-quality text embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemini embedding provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: Google API key (optional, can use env var)
                - model: Model name (default: "models/embedding-001")
                - timeout: Request timeout in seconds (default: 30)
                - task_type: Task type for embedding (default: "SEMANTIC_SIMILARITY")
        """
        super().__init__(config)
        self._embedding_model = None
        self._initialize_model()
    
    def validate_config(self) -> None:
        """Validate Gemini embedding configuration."""
        # Check for API key in config or environment
        api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Gemini API key not found. Please provide 'api_key' in config or set GOOGLE_API_KEY/GEMINI_API_KEY environment variable"
            )
        
        # Validate model name
        model = self.config.get("model", "models/embedding-001")
        valid_models = [
            "models/embedding-001",
            "models/text-embedding-004"
        ]
        if model not in valid_models:
            logger.warning(f"Model '{model}' not in known models {valid_models}, but will attempt to use it")
        
        # Validate task type
        task_type = self.config.get("task_type", "SEMANTIC_SIMILARITY")
        valid_task_types = [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT", 
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING"
        ]
        if task_type not in valid_task_types:
            raise ConfigurationError(f"Invalid task_type '{task_type}'. Must be one of {valid_task_types}")
    
    def _initialize_model(self):
        """Initialize the Gemini embedding model."""
        try:
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            model = self.config.get("model", "models/embedding-001")
            
            self._embedding_model = GeminiEmbedding(
                model_name=model,
                api_key=api_key,
                task_type=self.config.get("task_type", "SEMANTIC_SIMILARITY")
            )
            
            logger.info(f"Initialized Gemini embedding provider with model: {model}")
            
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Gemini embedding model: {str(e)}")
    
    def _execute_embed_text(self, text: str, **kwargs) -> EmbeddingResult:
        """
        Generate embedding for a single text using Gemini.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters (task_type override, etc.)
            
        Returns:
            EmbeddingResult containing the embedding and metadata
        """
        try:
            if not text or not text.strip():
                raise EmbeddingError("Cannot embed empty or whitespace-only text")
            
            # Generate embedding
            embedding_vector = self._embedding_model.get_text_embedding(text.strip())
            
            # Convert to numpy array if not already
            if not isinstance(embedding_vector, np.ndarray):
                embedding_vector = np.array(embedding_vector)
            
            # Validate embedding
            if not self.validate_embedding(embedding_vector):
                raise EmbeddingError("Generated embedding failed validation")
            
            return EmbeddingResult(
                embedding=embedding_vector,
                metadata={
                    "model": self.config.get("model", "models/embedding-001"),
                    "provider": "gemini",
                    "task_type": kwargs.get("task_type", self.config.get("task_type", "SEMANTIC_SIMILARITY")),
                    "text_length": len(text),
                    "embedding_norm": float(np.linalg.norm(embedding_vector))
                },
                model=self.config.get("model", "models/embedding-001"),
                dimensions=len(embedding_vector)
            )
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise RateLimitError(f"Gemini embedding rate limit exceeded: {str(e)}")
            raise EmbeddingError(f"Gemini embedding generation failed: {str(e)}")
    
    def _execute_embed_batch(self, texts: List[str], **kwargs) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts using Gemini.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            List of EmbeddingResult objects
        """
        try:
            if not texts:
                return []
            
            # Filter out empty texts and track indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
            
            if not valid_texts:
                raise EmbeddingError("No valid (non-empty) texts to embed")
            
            # Generate batch embeddings
            embedding_vectors = self._embedding_model.get_text_embedding_batch(valid_texts)
            
            # Convert to numpy arrays and create results
            results = []
            for i, text in enumerate(texts):
                if i in valid_indices:
                    # Get corresponding embedding
                    valid_idx = valid_indices.index(i)
                    embedding_vector = embedding_vectors[valid_idx]
                    
                    if not isinstance(embedding_vector, np.ndarray):
                        embedding_vector = np.array(embedding_vector)
                    
                    if not self.validate_embedding(embedding_vector):
                        raise EmbeddingError(f"Generated embedding for text {i} failed validation")
                    
                    result = EmbeddingResult(
                        embedding=embedding_vector,
                        metadata={
                            "model": self.config.get("model", "models/embedding-001"),
                            "provider": "gemini",
                            "task_type": kwargs.get("task_type", self.config.get("task_type", "SEMANTIC_SIMILARITY")),
                            "text_length": len(text),
                            "embedding_norm": float(np.linalg.norm(embedding_vector)),
                            "batch_index": i
                        },
                        model=self.config.get("model", "models/embedding-001"),
                        dimensions=len(embedding_vector)
                    )
                else:
                    # Create dummy result for empty text
                    result = EmbeddingResult(
                        embedding=np.zeros(self.get_embedding_dimensions()),
                        metadata={
                            "model": self.config.get("model", "models/embedding-001"),
                            "provider": "gemini",
                            "error": "empty_text",
                            "batch_index": i
                        },
                        model=self.config.get("model", "models/embedding-001"),
                        dimensions=self.get_embedding_dimensions()
                    )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise RateLimitError(f"Gemini batch embedding rate limit exceeded: {str(e)}")
            raise EmbeddingError(f"Gemini batch embedding generation failed: {str(e)}")
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of Gemini embeddings.
        
        Returns:
            Number of dimensions (768 for embedding-001)
        """
        model = self.config.get("model", "models/embedding-001")
        
        # Known dimensions for Gemini models
        model_dimensions = {
            "models/embedding-001": 768,
            "models/text-embedding-004": 768
        }
        
        return model_dimensions.get(model, 768)  # Default to 768
    
    def is_available(self) -> bool:
        """Check if Gemini embedding is available."""
        try:
            # Check if API key is available
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return False
            
            # Check if model is initialized
            if self._embedding_model is None:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini embedding model."""
        return {
            "provider": "gemini",
            "model": self.config.get("model", "models/embedding-001"),
            "dimensions": self.get_embedding_dimensions(),
            "task_type": self.config.get("task_type", "SEMANTIC_SIMILARITY"),
            "supports_batch": True,
            "max_batch_size": 100,  # Gemini typical batch limit
            "max_text_length": 2048,  # Typical token limit
            "supports_multilingual": True
        }
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Gemini embeddings."""
        return {
            "model": "models/embedding-001",
            "task_type": "SEMANTIC_SIMILARITY",
            "timeout": 30
        }
    
    def set_task_type(self, task_type: str):
        """
        Change the task type for embeddings.
        
        Args:
            task_type: New task type (SEMANTIC_SIMILARITY, CLASSIFICATION, etc.)
        """
        valid_task_types = [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT", 
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING"
        ]
        
        if task_type not in valid_task_types:
            raise ConfigurationError(f"Invalid task_type '{task_type}'. Must be one of {valid_task_types}")
        
        self.config["task_type"] = task_type
        
        # Reinitialize model with new task type
        self._initialize_model()
        
        logger.info(f"Changed Gemini embedding task type to: {task_type}")
    
    def embed_for_similarity(self, text: str) -> EmbeddingResult:
        """
        Generate embedding optimized for similarity comparison.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult optimized for similarity tasks
        """
        return self.embed_text(text, task_type="SEMANTIC_SIMILARITY")
    
    def embed_for_classification(self, text: str) -> EmbeddingResult:
        """
        Generate embedding optimized for classification tasks.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult optimized for classification
        """
        return self.embed_text(text, task_type="CLASSIFICATION")
    
    def embed_for_clustering(self, text: str) -> EmbeddingResult:
        """
        Generate embedding optimized for clustering tasks.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult optimized for clustering
        """
        return self.embed_text(text, task_type="CLUSTERING")