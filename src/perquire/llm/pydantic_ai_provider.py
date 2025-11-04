"""
Pydantic AI-based LLM provider that properly inherits from BaseLLMProvider.

This module provides a bridge between Pydantic AI and PERQUIRE's existing
provider architecture, maintaining backward compatibility while adding
type safety and validation internally.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

from .base import BaseLLMProvider, LLMResponse
from .models import (
    QuestionBatch,
    InvestigationQuestion,
    SynthesizedDescription,
    InvestigationContext,
)
from ..exceptions import LLMProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class PydanticAIProvider(BaseLLMProvider):
    """
    Pydantic AI-based provider that inherits from BaseLLMProvider.

    This provider bridges the gap between Pydantic AI's async interface
    and PERQUIRE's synchronous provider contract, enabling:
    - Full backward compatibility with existing system
    - Internal type safety via Pydantic models
    - Automatic validation of LLM outputs
    - Support for multiple models (Gemini, OpenAI, Anthropic, Ollama)

    The provider uses Pydantic AI internally but exposes the standard
    BaseLLMProvider interface for seamless integration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pydantic AI provider.

        Args:
            config: Configuration dictionary containing:
                - model: Model name (e.g., 'gemini-1.5-pro', 'openai:gpt-4')
                - api_key: Optional API key (uses env vars if not provided)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 1000)
        """
        super().__init__(config)  # Calls validate_config()

        self.model = self.config.get('model', 'gemini-1.5-flash')
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 1000)

        # Set API key if provided
        if 'api_key' in self.config:
            self._set_api_key(self.model, self.config['api_key'])

        # Initialize Pydantic AI agents
        self._init_agents()

        logger.info(f"Initialized PydanticAIProvider with model: {self.model}")

    def validate_config(self) -> None:
        """Validate provider configuration."""
        model = self.config.get('model', 'gemini-1.5-flash')

        # Validate temperature
        temperature = self.config.get('temperature', 0.7)
        if not 0 <= temperature <= 2.0:
            raise ConfigurationError(f"Temperature must be 0-2.0, got {temperature}")

        # Validate max_tokens
        max_tokens = self.config.get('max_tokens', 1000)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ConfigurationError(f"max_tokens must be positive integer, got {max_tokens}")

        # Check for API key if needed
        if 'gemini' in model.lower() or 'google' in model.lower():
            if not (self.config.get('api_key') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')):
                raise ConfigurationError("Gemini API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY")
        elif 'openai' in model.lower() or 'gpt' in model.lower():
            if not (self.config.get('api_key') or os.getenv('OPENAI_API_KEY')):
                raise ConfigurationError("OpenAI API key required. Set OPENAI_API_KEY")
        elif 'anthropic' in model.lower() or 'claude' in model.lower():
            if not (self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')):
                raise ConfigurationError("Anthropic API key required. Set ANTHROPIC_API_KEY")

    def _set_api_key(self, model: str, api_key: str) -> None:
        """Set the appropriate API key environment variable."""
        if 'gemini' in model.lower() or 'google' in model.lower():
            os.environ['GOOGLE_API_KEY'] = api_key
        elif 'openai' in model.lower() or 'gpt' in model.lower():
            os.environ['OPENAI_API_KEY'] = api_key
        elif 'anthropic' in model.lower() or 'claude' in model.lower():
            os.environ['ANTHROPIC_API_KEY'] = api_key

    def _init_agents(self) -> None:
        """Initialize Pydantic AI agents for different tasks."""
        try:
            # Agent for question generation
            self._question_agent = Agent(
                self.model,
                result_type=QuestionBatch,
                system_prompt=self._get_question_generation_prompt()
            )

            # Agent for description synthesis
            self._synthesis_agent = Agent(
                self.model,
                result_type=SynthesizedDescription,
                system_prompt=self._get_synthesis_prompt()
            )

        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Pydantic AI agents: {str(e)}")

    def _get_question_generation_prompt(self) -> str:
        """Get the system prompt for question generation."""
        return """You are an expert investigator helping to understand unknown embeddings.

Your task is to generate strategic questions that will help discover what an embedding represents.

Guidelines:
- EXPLORATION phase: Ask broad, categorical questions to map general semantic territory
- REFINEMENT phase: Ask focused questions to narrow down specific aspects
- CONVERGENCE phase: Ask precise, nuanced questions to capture subtle details

Generate questions that:
1. Are distinct from previously asked questions
2. Are appropriate for the current investigation phase
3. Are likely to improve the similarity score
4. Are clear, specific, and end with a question mark

Always provide a rationale for why each question will be useful."""

    def _get_synthesis_prompt(self) -> str:
        """Get the system prompt for description synthesis."""
        return """You are synthesizing the final results of an embedding investigation.

Based on the investigation history (questions and their similarity scores),
create a concise, accurate description of what the embedding represents.

Focus on:
- The highest-scoring questions (these are most similar to the target)
- Patterns across multiple high-scoring questions
- Confident findings vs uncertain areas

Provide:
- A clear 2-3 sentence description
- Key findings that emerged from the investigation
- Areas where uncertainty remains
- Confidence level in your synthesis"""

    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        This method provides a generic response generation capability
        using Pydantic AI's text generation.
        """
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"

            # Use asyncio.run to bridge async Pydantic AI with sync interface
            async def _generate():
                # For simple text generation, use the question agent
                result = await self._question_agent.run(
                    full_prompt,
                    model_settings={
                        'temperature': self.temperature,
                        'max_tokens': self.max_tokens
                    }
                )
                return result

            result = asyncio.run(_generate())

            return LLMResponse(
                content=str(result.data),
                metadata={
                    "model": self.model,
                    "provider": "pydantic-ai"
                },
                model=self.model
            )

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise LLMProviderError(f"Failed to generate response: {str(e)}")

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

        This method matches the BaseLLMProvider signature exactly,
        using Pydantic AI internally for type-safe generation.

        Returns:
            List[str]: List of generated questions (as required by base class)
        """
        try:
            # Create context for Pydantic AI
            context = InvestigationContext(
                current_description=current_description,
                current_similarity=target_similarity,
                phase=phase,  # type: ignore
                previous_questions=previous_questions or [],
                iteration=kwargs.get('iteration', 1)
            )

            # Build prompt
            user_prompt = f"""Current investigation status:

Description so far: "{context.current_description}"
Current similarity: {context.current_similarity:.3f}
Phase: {context.phase}
Iteration: {context.iteration}

Previously asked questions (avoid repeating):
{self._format_previous_questions(context.get_recent_questions())}

Generate 3-5 new questions for the {context.phase} phase that will help
improve our understanding and increase the similarity score."""

            # Run async generation synchronously
            async def _generate():
                result = await self._question_agent.run(
                    user_prompt,
                    model_settings={
                        'temperature': self.temperature,
                        'max_tokens': self.max_tokens
                    }
                )
                return result.data  # Returns validated QuestionBatch

            batch: QuestionBatch = asyncio.run(_generate())

            # Extract just the question strings for backward compatibility
            questions = [q.question for q in batch.questions]

            # Store structured data for internal use if needed
            self._last_question_batch = batch

            return questions

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            raise LLMProviderError(f"Failed to generate questions: {str(e)}")

    def synthesize_description(
        self,
        questions_and_scores: List[Dict[str, Any]],
        final_similarity: float,
        **kwargs
    ) -> str:
        """
        Synthesize a final description from investigation results.

        This method matches the BaseLLMProvider signature exactly,
        using Pydantic AI internally for validated synthesis.

        Returns:
            str: Synthesized description (as required by base class)
        """
        try:
            # Sort by similarity score (highest first)
            sorted_results = sorted(
                questions_and_scores,
                key=lambda x: x.get('similarity', 0),
                reverse=True
            )

            # Build user prompt
            user_prompt = f"""Investigation complete. Synthesize the results.

Top investigation results (question → similarity score):
{self._format_results(sorted_results[:15])}

Final similarity achieved: {final_similarity:.3f}

Based on these results, provide a comprehensive synthesis."""

            # Run async synthesis synchronously
            async def _synthesize():
                result = await self._synthesis_agent.run(
                    user_prompt,
                    model_settings={
                        'temperature': self.temperature,
                        'max_tokens': self.max_tokens
                    }
                )
                return result.data  # Returns validated SynthesizedDescription

            synthesis: SynthesizedDescription = asyncio.run(_synthesize())

            # Store structured data for internal use if needed
            self._last_synthesis = synthesis

            # Return just the description string for backward compatibility
            return synthesis.description

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise LLMProviderError(f"Failed to synthesize description: {str(e)}")

    def is_available(self) -> bool:
        """Check if the provider is available."""
        try:
            # Check if model is configured
            if not self.model:
                return False

            # Check for API keys based on model type
            if 'gemini' in self.model.lower() or 'google' in self.model.lower():
                return bool(os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'))
            elif 'openai' in self.model.lower() or 'gpt' in self.model.lower():
                return bool(os.getenv('OPENAI_API_KEY'))
            elif 'anthropic' in self.model.lower() or 'claude' in self.model.lower():
                return bool(os.getenv('ANTHROPIC_API_KEY'))
            elif 'ollama' in self.model.lower():
                return True  # Local, no API key needed

            return True

        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "pydantic-ai",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "supports_streaming": True,
            "supports_function_calling": True,
            "type_safe": True,
            "validated_outputs": True
        }

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this provider."""
        return {
            "model": "gemini-1.5-flash",
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30
        }

    @staticmethod
    def _format_previous_questions(questions: List[str]) -> str:
        """Format previous questions for prompt."""
        if not questions:
            return "(None yet)"
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    @staticmethod
    def _format_results(results: List[Dict[str, Any]]) -> str:
        """Format investigation results for prompt."""
        return "\n".join(
            f"• {r['question']} → {r['similarity']:.3f}"
            for r in results
        )

    # Optional: Expose structured outputs for advanced use cases
    def get_last_question_batch(self) -> Optional[QuestionBatch]:
        """Get the last generated question batch with full structure."""
        return getattr(self, '_last_question_batch', None)

    def get_last_synthesis(self) -> Optional[SynthesizedDescription]:
        """Get the last synthesis with full structure."""
        return getattr(self, '_last_synthesis', None)


# Factory functions for easy provider creation
def create_pydantic_gemini_provider(
    model: str = "gemini-1.5-flash",
    api_key: Optional[str] = None,
    **kwargs
) -> PydanticAIProvider:
    """
    Create a Gemini provider using Pydantic AI.

    Args:
        model: Gemini model name (default: gemini-1.5-flash)
        api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
        **kwargs: Additional config (temperature, max_tokens, etc.)

    Returns:
        PydanticAIProvider configured for Gemini
    """
    config = {
        'model': f'google-gla:{model}',
        **kwargs
    }
    if api_key:
        config['api_key'] = api_key
    return PydanticAIProvider(config)


def create_pydantic_openai_provider(
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    **kwargs
) -> PydanticAIProvider:
    """
    Create an OpenAI provider using Pydantic AI.

    Args:
        model: OpenAI model name (default: gpt-4)
        api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
        **kwargs: Additional config (temperature, max_tokens, etc.)

    Returns:
        PydanticAIProvider configured for OpenAI
    """
    config = {
        'model': f'openai:{model}',
        **kwargs
    }
    if api_key:
        config['api_key'] = api_key
    return PydanticAIProvider(config)


def create_pydantic_anthropic_provider(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    **kwargs
) -> PydanticAIProvider:
    """
    Create an Anthropic provider using Pydantic AI.

    Args:
        model: Anthropic model name (default: claude-3-5-sonnet)
        api_key: Optional API key (uses ANTHROPIC_API_KEY env var if not provided)
        **kwargs: Additional config (temperature, max_tokens, etc.)

    Returns:
        PydanticAIProvider configured for Anthropic
    """
    config = {
        'model': f'anthropic:{model}',
        **kwargs
    }
    if api_key:
        config['api_key'] = api_key
    return PydanticAIProvider(config)


def create_pydantic_ollama_provider(
    model: str = "llama2",
    **kwargs
) -> PydanticAIProvider:
    """
    Create an Ollama provider using Pydantic AI (local).

    Args:
        model: Ollama model name (default: llama2)
        **kwargs: Additional config (temperature, max_tokens, etc.)

    Returns:
        PydanticAIProvider configured for Ollama
    """
    config = {
        'model': f'ollama:{model}',
        **kwargs
    }
    return PydanticAIProvider(config)
