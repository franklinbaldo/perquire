"""
Ollama provider using LlamaIndex.
"""

import os
from typing import Dict, Any, Optional, List
import logging

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import LLMProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider implementation using LlamaIndex.
    
    This provider uses local Ollama models through LlamaIndex integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama provider.
        
        Args:
            config: Configuration dictionary containing:
                - model: Model name (default: "llama2")
                - base_url: Ollama server URL (default: "http://localhost:11434")
                - temperature: Sampling temperature (default: 0.7)
                - timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(config)
        self._llm = None
        self._initialize_llm()
    
    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        # Validate temperature
        temperature = self.config.get("temperature", 0.7)
        if not 0 <= temperature <= 2:
            raise ConfigurationError(f"Temperature must be between 0 and 2, got {temperature}")
        
        # Validate base_url
        base_url = self.config.get("base_url", "http://localhost:11434")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError(f"Invalid base_url format: {base_url}")
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM instance."""
        try:
            model = self.config.get("model", "llama2")
            base_url = self.config.get("base_url", "http://localhost:11434")
            
            self._llm = Ollama(
                model=model,
                base_url=base_url,
                temperature=self.config.get("temperature", 0.7),
                request_timeout=self.config.get("timeout", 60)
            )
            
            logger.info(f"Initialized Ollama provider with model: {model}")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Ollama LLM: {str(e)}")
    
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama."""
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
                    "model": self.config.get("model", "llama2"),
                    "provider": "ollama",
                    "base_url": self.config.get("base_url", "http://localhost:11434")
                },
                usage=getattr(response, 'usage', None),
                model=self.config.get("model", "llama2")
            )
            
        except Exception as e:
            raise LLMProviderError(f"Ollama generation failed: {str(e)}")
    
    def generate_questions(
        self,
        current_description: str,
        target_similarity: float,
        phase: str,
        previous_questions: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Generate investigation questions using Ollama."""
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
        """Synthesize final description using Ollama."""
        try:
            prompt = self._build_synthesis_prompt(questions_and_scores, final_similarity)
            
            response = self.generate_response(prompt, **kwargs)
            
            return response.content.strip()
            
        except Exception as e:
            raise LLMProviderError(f"Description synthesis failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            # Try to ping the Ollama server
            import requests
            base_url = self.config.get("base_url", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model."""
        return {
            "provider": "ollama",
            "model": self.config.get("model", "llama2"),
            "base_url": self.config.get("base_url", "http://localhost:11434"),
            "temperature": self.config.get("temperature", 0.7),
            "supports_streaming": True,
            "supports_function_calling": False,
            "local_model": True
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
        """Get default configuration for Ollama."""
        return {
            "model": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "timeout": 60
        }