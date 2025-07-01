"""OpenAI API provider implementation."""

import os
from typing import List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai is required for OpenAI provider")

from . import EmbeddingProvider, LLMProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using openai library."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large"):
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embedding failed: {e}")


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider using openai library."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2048),
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {e}")