"""Gemini API provider implementation."""

import os
from typing import List, Optional

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai is required for Gemini provider")

from . import EmbeddingProvider, LLMProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider using google-generativeai."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "models/text-embedding-004"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = model
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Gemini API."""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="semantic_similarity",
                title="Perquire Investigation"
            )
            return result['embedding']
        except Exception as e:
            raise Exception(f"Gemini embedding failed: {e}")


class GeminiLLMProvider(LLMProvider):
    """Gemini LLM provider using google-generativeai."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini API."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.7),
                    max_output_tokens=kwargs.get("max_tokens", 2048),
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini generation failed: {e}")