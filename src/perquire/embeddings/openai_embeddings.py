import os
from typing import Dict, Any, Optional, List
import logging
import numpy as np

from llama_index.embeddings.openai import OpenAIEmbedding
from .base import BaseEmbeddingProvider, EmbeddingResult
from ..exceptions import EmbeddingError, ConfigurationError, RateLimitError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider implementation using LlamaIndex.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._embedding_model = None
        self._initialize_model()

    def validate_config(self) -> None:
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not found. Provide 'api_key' in config or set OPENAI_API_KEY."
            )
        # model = self.config.get("model", "text-embedding-ada-002")
        # Add any other model specific validations if needed

    def _initialize_model(self):
        try:
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            model_name = self.config.get("model", "text-embedding-ada-002") # Default model

            # LlamaIndex's OpenAIEmbedding takes 'model_name' or 'model'
            # Let's be explicit with 'model' as used by their class
            self._embedding_model = OpenAIEmbedding(
                model=model_name,
                api_key=api_key,
                # dimensions=self.config.get("dimensions") # Pass if specified in config
            )
            logger.info(f"Initialized OpenAI embedding provider with model: {model_name}")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI embedding model: {str(e)}")

    def _execute_embed_text(self, text: str, **kwargs) -> EmbeddingResult:
        try:
            if not text or not text.strip():
                # Return a zero vector for empty text, consistent with some behaviors
                logger.warn("Attempted to embed empty or whitespace-only text. Returning zero vector.")
                zero_embedding = np.zeros(self.get_embedding_dimensions())
                return EmbeddingResult(
                    embedding=zero_embedding,
                    metadata={
                        "model": self.config.get("model", "text-embedding-ada-002"),
                        "provider": "openai",
                        "text_length": len(text),
                        "error": "empty_text"
                    },
                    model=self.config.get("model", "text-embedding-ada-002"),
                    dimensions=self.get_embedding_dimensions()
                )

            embedding_vector = self._embedding_model.get_text_embedding(text.strip())

            if not isinstance(embedding_vector, np.ndarray):
                embedding_vector = np.array(embedding_vector)

            if not self.validate_embedding(embedding_vector):
                raise EmbeddingError("Generated embedding failed validation")

            return EmbeddingResult(
                embedding=embedding_vector,
                metadata={
                    "model": self.config.get("model", "text-embedding-ada-002"),
                    "provider": "openai",
                    "text_length": len(text),
                    "embedding_norm": float(np.linalg.norm(embedding_vector))
                },
                model=self.config.get("model", "text-embedding-ada-002"),
                dimensions=len(embedding_vector)
            )
        except Exception as e:
            if hasattr(e, 'code') and e.code == 'rate_limit_exceeded': # Check for OpenAI specific error
                raise RateLimitError(f"OpenAI embedding rate limit exceeded: {str(e)}")
            raise EmbeddingError(f"OpenAI embedding generation failed: {str(e)}")

    def _execute_embed_batch(self, texts: List[str], **kwargs) -> List[EmbeddingResult]:
        try:
            if not texts: return []

            # LlamaIndex OpenAIEmbedding handles batching internally.
            # It also handles empty strings in the list by returning zero vectors.
            embedding_vectors = self._embedding_model.get_text_embedding_batch(texts, **kwargs) # Pass kwargs like show_progress

            final_results = []
            for i, original_text in enumerate(texts):
                embedding_vector = embedding_vectors[i]
                if not isinstance(embedding_vector, np.ndarray):
                    embedding_vector = np.array(embedding_vector)

                # Check if it's a zero vector which LlamaIndex might return for empty strings
                is_zero_vector = not np.any(embedding_vector)
                metadata_error = {}
                if (not original_text or not original_text.strip()) and is_zero_vector:
                     metadata_error = {"error": "empty_text_in_batch", "original_text": original_text}


                if not self.validate_embedding(embedding_vector):
                    # Allow zero vectors for empty inputs to pass validation if dimensions match
                    if not (is_zero_vector and len(embedding_vector) == self.get_embedding_dimensions()):
                        raise EmbeddingError(f"Generated batch embedding for text '{original_text}' failed validation")

                final_results.append(EmbeddingResult(
                    embedding=embedding_vector,
                    metadata={
                        "model": self.config.get("model", "text-embedding-ada-002"),
                        "provider": "openai",
                        "text_length": len(original_text),
                        "embedding_norm": float(np.linalg.norm(embedding_vector)),
                        **metadata_error
                    },
                    model=self.config.get("model", "text-embedding-ada-002"),
                    dimensions=len(embedding_vector)
                ))
            return final_results
        except Exception as e:
            if hasattr(e, 'code') and e.code == 'rate_limit_exceeded':
                raise RateLimitError(f"OpenAI batch embedding rate limit exceeded: {str(e)}")
            raise EmbeddingError(f"OpenAI batch embedding generation failed: {str(e)}")

    def get_embedding_dimensions(self) -> int:
        # Try to get from LlamaIndex model object first
        if self._embedding_model and hasattr(self._embedding_model, 'embed_dim') and self._embedding_model.embed_dim is not None:
             return self._embedding_model.embed_dim

        # Fallback to config or known defaults
        model_name = self.config.get("model", "text-embedding-ada-002")
        dimensions_config = self.config.get("dimensions") # User might specify this for newer models

        if dimensions_config:
            return int(dimensions_config)

        model_dimensions_defaults = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return model_dimensions_defaults.get(model_name, 1536)


    def is_available(self) -> bool:
        try:
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key: return False
            if self._embedding_model is None:
                # Try to initialize if not already
                self._initialize_model()
            return self._embedding_model is not None
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.config.get("model", "text-embedding-ada-002"),
            "dimensions": self.get_embedding_dimensions(),
            "supports_batch": True,
            "max_batch_size": self.config.get("max_batch_size", 2048),
            "max_tokens_per_input": self.config.get("max_tokens", 8191)
        }

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": "text-embedding-ada-002",
            # "dimensions": 1536, # Optional: specify for newer models if not default
            "timeout": 60.0 # Default timeout for OpenAI API calls in LlamaIndex
        }
