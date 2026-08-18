import numpy as np
import pytest

from perquire.convergence.algorithms import ConvergenceReason, ConvergenceResult
from perquire.core.investigator import PerquireInvestigator
from perquire.core.strategy import InvestigationPhase, QuestioningStrategy
from perquire.embeddings.base import BaseEmbeddingProvider, EmbeddingResult
from perquire.exceptions import InvestigationError
from perquire.llm.base import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.available = True
        self.generated: list[dict] = []
        super().__init__({})

    def validate_config(self) -> None:
        return None

    def generate_response(self, prompt, context=None, **kwargs) -> LLMResponse:
        return LLMResponse(content="mock response", metadata={}, model="mock")

    def generate_questions(
        self,
        current_description,
        target_similarity,
        phase,
        previous_questions=None,
        **kwargs,
    ) -> list[str]:
        self.generated.append(
            {
                "current_description": current_description,
                "target_similarity": target_similarity,
                "phase": phase,
                "previous_questions": list(previous_questions or []),
            }
        )
        return [f"probe-{len(self.generated)}"]

    def synthesize_description(self, questions_and_scores, final_similarity, **kwargs) -> str:
        return "mock synthesized description"

    def is_available(self) -> bool:
        return self.available

    def get_model_info(self) -> dict:
        return {"provider": "mock", "model": "mock"}


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.available = True
        super().__init__({})

    def validate_config(self) -> None:
        return None

    def _vector(self, text: str) -> np.ndarray:
        # Deterministic, non-zero vectors keep the tests independent of model SDKs.
        if text == "probe-1":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if text == "probe-2":
            return np.array([0.8, 0.2, 0.0], dtype=np.float32)
        return np.array([0.5, 0.5, 0.0], dtype=np.float32)

    def _execute_embed_text(self, text: str, **kwargs) -> EmbeddingResult:
        vector = self._vector(text)
        return EmbeddingResult(
            embedding=vector,
            metadata={},
            model="mock",
            dimensions=len(vector),
        )

    def _execute_embed_batch(self, texts: list[str], **kwargs) -> list[EmbeddingResult]:
        return [self._execute_embed_text(text, **kwargs) for text in texts]

    def get_embedding_dimensions(self) -> int:
        return 3

    def is_available(self) -> bool:
        return self.available

    def get_model_info(self) -> dict:
        return {"provider": "mock", "model": "mock", "dimensions": 3}


class TwoStepConvergence:
    def __init__(self) -> None:
        self.calls = 0

    def should_continue(self, similarity_scores, current_iteration, phase="exploration"):
        self.calls += 1
        return ConvergenceResult(
            converged=self.calls >= 2,
            reason=(
                ConvergenceReason.DIMINISHING_RETURNS
                if self.calls >= 2
                else ConvergenceReason.MANUAL_STOP
            ),
            confidence=1.0 if self.calls >= 2 else 0.0,
            iteration_reached=current_iteration,
            final_similarity=max(similarity_scores),
            similarity_improvement=0.0,
            plateau_length=0,
            statistical_metrics={},
            recommendation="stop" if self.calls >= 2 else "continue",
        )


@pytest.fixture
def llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()


def test_investigator_initializes_current_dependencies(llm, embedder) -> None:
    investigator = PerquireInvestigator(llm_provider=llm, embedding_provider=embedder)
    assert investigator.llm_provider is llm
    assert investigator.embedding_provider is embedder
    assert isinstance(investigator.questioning_strategy, QuestioningStrategy)


def test_investigation_uses_similarity_feedback_without_hidden_text(llm, embedder) -> None:
    strategy = QuestioningStrategy(max_iterations=2, exploration_depth=1)
    convergence = TwoStepConvergence()
    investigator = PerquireInvestigator(
        llm_provider=llm,
        embedding_provider=embedder,
        questioning_strategy=strategy,
        convergence_detector=convergence,
    )

    result = investigator.investigate(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        save_to_database=False,
    )

    assert result.iterations == 2
    assert result.description == "mock synthesized description"
    assert result.final_similarity == pytest.approx(1.0)
    assert len(result.question_history) == 2
    assert llm.generated[0]["current_description"] == ""
    assert llm.generated[0]["target_similarity"] == 0.0
    assert llm.generated[1]["current_description"] == "probe-1"
    assert llm.generated[1]["target_similarity"] == pytest.approx(1.0)


def test_question_generation_receives_phase_from_strategy(llm, embedder) -> None:
    strategy = QuestioningStrategy(max_iterations=1, exploration_depth=1)
    investigator = PerquireInvestigator(
        llm_provider=llm,
        embedding_provider=embedder,
        questioning_strategy=strategy,
        convergence_detector=TwoStepConvergence(),
    )
    investigator.investigate(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        save_to_database=False,
    )
    assert llm.generated[0]["phase"] == InvestigationPhase.EXPLORATION.value


def test_investigation_rejects_unavailable_provider(llm, embedder) -> None:
    llm.available = False
    investigator = PerquireInvestigator(llm_provider=llm, embedding_provider=embedder)
    with pytest.raises(InvestigationError, match="LLM provider is not available"):
        investigator.investigate(np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_investigation_rejects_zero_target(llm, embedder) -> None:
    investigator = PerquireInvestigator(llm_provider=llm, embedding_provider=embedder)
    with pytest.raises(InvestigationError, match="zero vector"):
        investigator.investigate(np.zeros(3, dtype=np.float32))
