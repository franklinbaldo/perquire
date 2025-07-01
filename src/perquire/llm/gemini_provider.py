"""
Google Gemini provider using LlamaIndex.
"""

import os
from typing import Dict, Any, Optional, List
import logging

from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, MessageRole

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import LLMProviderError, ConfigurationError, RateLimitError

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider implementation using LlamaIndex.
    
    This provider uses Google's Gemini models through LlamaIndex integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemini provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: Google API key (optional, can use env var)
                - model: Model name (default: "gemini-pro")
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 150)
                - timeout: Request timeout in seconds (default: 30)
        """
        super().__init__(config)
        self._llm = None
        self._initialize_llm()
    
    def validate_config(self) -> None:
        """Validate Gemini configuration."""
        # Check for API key in config or environment
        api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Gemini API key not found. Please provide 'api_key' in config or set GOOGLE_API_KEY/GEMINI_API_KEY environment variable"
            )
        
        # Validate temperature
        temperature = self.config.get("temperature", 0.7)
        if not 0 <= temperature <= 1:
            raise ConfigurationError(f"Temperature must be between 0 and 1, got {temperature}")
        
        # Validate max_tokens
        max_tokens = self.config.get("max_tokens", 150)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ConfigurationError(f"max_tokens must be a positive integer, got {max_tokens}")
    
    def _initialize_llm(self):
        """Initialize the Gemini LLM instance."""
        try:
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            model = self.config.get("model", "gemini-pro")
            
            self._llm = Gemini(
                model=model,
                api_key=api_key,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 150)
            )
            
            logger.info(f"Initialized Gemini provider with model: {model}")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Gemini LLM: {str(e)}")
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini."""
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            
            # Create chat message
            messages = [ChatMessage(role=MessageRole.USER, content=full_prompt)]
            
            # Generate response
            response = self._llm.chat(messages)
            
            return LLMResponse(
                content=response.message.content,
                metadata={
                    "model": self.config.get("model", "gemini-pro"),
                    "provider": "gemini"
                },
                usage=getattr(response, 'usage', None),
                model=self.config.get("model", "gemini-pro")
            )
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {str(e)}")
            raise LLMProviderError(f"Gemini generation failed: {str(e)}")
    
    def generate_questions(
        self,
        current_description: str,
        target_similarity: float,
        phase: str,
        previous_questions: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Generate investigation questions using Gemini."""
        try:
            # Build prompt for question generation
            prompt = self._build_question_generation_prompt(
                current_description, target_similarity, phase, previous_questions
            )
            
            response = self.generate_response(prompt, **kwargs)
            
            # Parse questions from response
            questions = self._parse_questions_from_response(response.content)
            
            return questions
            
        except Exception as e:
            raise LLMProviderError(f"Question generation failed: {str(e)}")
    
    def synthesize_description(
        self,
        questions_and_scores: List[Dict[str, Any]],
        final_similarity: float,
        **kwargs
    ) -> str:
        """Synthesize final description using Gemini."""
        try:
            prompt = self._build_synthesis_prompt(questions_and_scores, final_similarity)
            
            response = self.generate_response(prompt, **kwargs)
            
            return response.content.strip()
            
        except Exception as e:
            raise LLMProviderError(f"Description synthesis failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        try:
            # Check if API key is available
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return False
            
            # Check if LLM is initialized
            if self._llm is None:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model."""
        return {
            "provider": "gemini",
            "model": self.config.get("model", "gemini-pro"),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 150),
            "supports_streaming": True,
            "supports_function_calling": True
        }
    
    def _build_question_generation_prompt(
        self,
        current_description: str,
        target_similarity: float,
        phase: str,
        previous_questions: Optional[List[str]] = None
    ) -> str:
        """Build prompt for generating investigation questions."""
        previous_questions = previous_questions or []
        
        prompt = f"""You are helping investigate an unknown embedding by generating strategic questions.

Current best description: "{current_description}"
Current similarity score: {target_similarity:.3f}
Investigation phase: {phase}

Your task is to generate 3-5 questions that will help refine our understanding of what this embedding represents.

Phase guidelines:
- exploration: Ask broad, categorical questions to map general semantic territory
- refinement: Ask focused questions to narrow down specific aspects
- convergence: Ask precise, nuanced questions to capture subtle details

"""
        
        if previous_questions:
            prompt += f"Previously asked questions (avoid repeating):\n"
            for i, q in enumerate(previous_questions[-10:], 1):  # Show last 10
                prompt += f"{i}. {q}\n"
            prompt += "\n"
        
        prompt += """Generate questions that are:
1. Distinct from previous questions
2. Appropriate for the current phase
3. Likely to improve similarity score
4. Clear and specific

Return only the questions, one per line, without numbering."""
        
        return prompt
    
    def _build_synthesis_prompt(
        self,
        questions_and_scores: List[Dict[str, Any]],
        final_similarity: float
    ) -> str:
        """Build prompt for synthesizing final description."""
        prompt = f"""Based on an investigation of an unknown embedding, synthesize a final description.

Investigation results (question -> similarity score):
"""
        
        # Sort by similarity score (highest first)
        sorted_results = sorted(questions_and_scores, key=lambda x: x["similarity"], reverse=True)
        
        for result in sorted_results[:15]:  # Show top 15 results
            prompt += f"• {result['question']} -> {result['similarity']:.3f}\n"
        
        prompt += f"""
Final similarity achieved: {final_similarity:.3f}

Synthesize a concise, accurate description (2-3 sentences) that captures what this embedding most likely represents based on the highest-scoring questions. Focus on the most confident findings."""
        
        return prompt
    
    def _parse_questions_from_response(self, response: str) -> List[str]:
        """Parse questions from LLM response."""
        lines = response.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering if present
            if line and line[0].isdigit() and '.' in line:
                line = line.split('.', 1)[1].strip()
            
            # Remove bullet points
            if line.startswith('- ') or line.startswith('• '):
                line = line[2:].strip()
            
            if line and line.endswith('?'):
                questions.append(line)
        
        return questions[:5]  # Limit to 5 questions
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Gemini."""
        return {
            "model": "gemini-pro",
            "temperature": 0.7,
            "max_tokens": 150,
            "timeout": 30
        }