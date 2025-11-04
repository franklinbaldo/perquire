"""
Pydantic AI-based LLM provider.

This module provides a simplified, type-safe LLM provider using Pydantic AI
to replace the complex manual provider implementations.
"""

import os
import logging
from typing import Any, Literal, Optional
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

from .models import (
    QuestionBatch,
    InvestigationQuestion,
    SynthesizedDescription,
    InvestigationContext,
    LLMProviderInfo,
    HealthCheckResult
)
from ..exceptions import LLMProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class PydanticAIProvider:
    """
    Simplified LLM provider using Pydantic AI.

    This replaces 277+ lines of manual provider code with ~150 lines
    of type-safe, validated LLM interactions.
    """

    def __init__(
        self,
        model: KnownModelName | str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize Pydantic AI provider.

        Args:
            model: Model name (e.g., 'gemini-1.5-pro', 'openai:gpt-4', 'anthropic:claude-3-sonnet')
            api_key: API key (optional, uses environment variables if not provided)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set API key if provided
        if api_key:
            self._set_api_key(model, api_key)

        # Initialize agents for different tasks
        self._question_agent = Agent(
            model,
            result_type=QuestionBatch,
            system_prompt=self._get_question_generation_prompt()
        )

        self._synthesis_agent = Agent(
            model,
            result_type=SynthesizedDescription,
            system_prompt=self._get_synthesis_prompt()
        )

        logger.info(f"Initialized PydanticAIProvider with model: {model}")

    def _set_api_key(self, model: str, api_key: str) -> None:
        """Set the appropriate API key environment variable."""
        if 'gemini' in model.lower() or 'google' in model.lower():
            os.environ['GOOGLE_API_KEY'] = api_key
        elif 'openai' in model.lower() or 'gpt' in model.lower():
            os.environ['OPENAI_API_KEY'] = api_key
        elif 'anthropic' in model.lower() or 'claude' in model.lower():
            os.environ['ANTHROPIC_API_KEY'] = api_key

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

    async def generate_questions(
        self,
        context: InvestigationContext,
        num_questions: int = 3
    ) -> QuestionBatch:
        """
        Generate investigation questions based on current context.

        Args:
            context: Current investigation context
            num_questions: Number of questions to generate (1-5)

        Returns:
            QuestionBatch with validated questions

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            # Build the user prompt from context
            user_prompt = f"""Current investigation status:

Description so far: "{context.current_description}"
Current similarity: {context.current_similarity:.3f}
Phase: {context.phase}
Iteration: {context.iteration}

Previously asked questions (avoid repeating):
{self._format_previous_questions(context.get_recent_questions())}

Generate {num_questions} new questions for the {context.phase} phase that will help
improve our understanding and increase the similarity score."""

            result = await self._question_agent.run(
                user_prompt,
                model_settings={
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens
                }
            )

            # Result is automatically a validated QuestionBatch!
            return result.data

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            raise LLMProviderError(f"Failed to generate questions: {str(e)}")

    async def synthesize_description(
        self,
        questions_and_scores: list[dict[str, Any]],
        final_similarity: float
    ) -> SynthesizedDescription:
        """
        Synthesize final description from investigation results.

        Args:
            questions_and_scores: List of dicts with 'question' and 'similarity'
            final_similarity: Final similarity score achieved

        Returns:
            SynthesizedDescription with validated output

        Raises:
            LLMProviderError: If synthesis fails
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

            result = await self._synthesis_agent.run(
                user_prompt,
                model_settings={
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens
                }
            )

            # Result is automatically a validated SynthesizedDescription!
            return result.data

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise LLMProviderError(f"Failed to synthesize description: {str(e)}")

    async def health_check(self) -> HealthCheckResult:
        """
        Perform a health check on the provider.

        Returns:
            HealthCheckResult with status and details
        """
        start_time = datetime.now()

        try:
            # Simple test with question agent
            test_context = InvestigationContext(
                current_description="test description",
                current_similarity=0.5,
                phase="exploration",
                iteration=1
            )

            await self.generate_questions(test_context, num_questions=1)

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                status="healthy",
                provider_info=self.get_provider_info(),
                response_time_ms=response_time
            )

        except Exception as e:
            return HealthCheckResult(
                status="unhealthy",
                error_message=str(e)
            )

    def get_provider_info(self) -> LLMProviderInfo:
        """Get information about this provider."""
        return LLMProviderInfo(
            provider_name="pydantic-ai",
            model_name=str(self.model),
            supports_streaming=True,
            supports_function_calling=True,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            is_available=True
        )

    @staticmethod
    def _format_previous_questions(questions: list[str]) -> str:
        """Format previous questions for prompt."""
        if not questions:
            return "(None yet)"
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    @staticmethod
    def _format_results(results: list[dict[str, Any]]) -> str:
        """Format investigation results for prompt."""
        return "\n".join(
            f"• {r['question']} → {r['similarity']:.3f}"
            for r in results
        )


# Convenience factory functions
def create_gemini_provider(
    model: str = "gemini-1.5-pro",
    api_key: Optional[str] = None,
    **kwargs
) -> PydanticAIProvider:
    """Create a Gemini provider."""
    return PydanticAIProvider(f"google-gla:{model}", api_key=api_key, **kwargs)


def create_openai_provider(
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    **kwargs
) -> PydanticAIProvider:
    """Create an OpenAI provider."""
    return PydanticAIProvider(f"openai:{model}", api_key=api_key, **kwargs)


def create_anthropic_provider(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    **kwargs
) -> PydanticAIProvider:
    """Create an Anthropic provider."""
    return PydanticAIProvider(f"anthropic:{model}", api_key=api_key, **kwargs)


def create_ollama_provider(
    model: str = "llama2",
    **kwargs
) -> PydanticAIProvider:
    """Create an Ollama provider (local)."""
    return PydanticAIProvider(f"ollama:{model}", **kwargs)
