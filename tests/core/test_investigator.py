import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from perquire.core.investigator import PerquireInvestigator
from perquire.llm.base import BaseLLMProvider, LLMResponse
from perquire.embeddings.base import BaseEmbeddingProvider, EmbeddingResult
from perquire.core.strategy import QuestioningStrategy, InvestigationPhase
from perquire.convergence.algorithms import ConvergenceDetector, ConvergenceResult, ConvergenceReason
from perquire.exceptions import InvestigationError, ConfigurationError, ValidationError

# Mock LLM Provider
class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config=None):
        super().__init__(config or {})
        self.is_available_flag = True
        self.name = "mock_llm"
    def validate_config(self): pass
    def generate_response(self, prompt, context=None, **kwargs):
        return LLMResponse(content="mock llm response", metadata={}, model=self.name)
    def generate_questions(self, current_description, target_similarity, phase, previous_questions=None, **kwargs):
        return [f"mock question for {phase} based on '{current_description}'?"]
    def synthesize_description(self, questions_and_scores, final_similarity, **kwargs):
        if questions_and_scores:
            return f"Synthesized: {questions_and_scores[0]['question']}"
        return "mock synthesized description"
    def is_available(self): return self.is_available_flag
    def get_model_info(self): return {"provider": self.name, "model": "test_model"}

# Mock Embedding Provider
class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config=None, dim=10):
        super().__init__(config or {})
        self.dim = dim
        self.is_available_flag = True
        self.name = "mock_embedding"
    def validate_config(self): pass
    def _execute_embed_text(self, text, **kwargs):
        return EmbeddingResult(embedding=np.random.rand(self.dim).astype(np.float32), metadata={}, model=self.name, dimensions=self.dim)
    def _execute_embed_batch(self, texts, **kwargs):
        return [EmbeddingResult(embedding=np.random.rand(self.dim).astype(np.float32), metadata={}, model=self.name, dimensions=self.dim) for _ in texts]
    def get_embedding_dimensions(self): return self.dim
    def is_available(self): return self.is_available_flag
    def get_model_info(self): return {"provider": self.name, "model": "test_model", "dimensions": self.dim}

@pytest.fixture
def mock_llm_provider_instance(): # Renamed to avoid conflict with type
    return MockLLMProvider()

@pytest.fixture
def mock_embedding_provider_instance(): # Renamed
    return MockEmbeddingProvider(dim=10) # Example dimension

@pytest.fixture
def mock_strategy_instance(): # Renamed
    strategy = MagicMock(spec=QuestioningStrategy)
    strategy.name = "mock_strategy"
    strategy.convergence_threshold = 0.95
    strategy.min_improvement = 0.0001
    strategy.convergence_window = 2 # Smaller for tests
    strategy.max_iterations = 3 # Test with few iterations

    # Mock determine_phase to cycle through phases for testing
    strategy.determine_phase = MagicMock(side_effect=[
        InvestigationPhase.EXPLORATION,
        InvestigationPhase.REFINEMENT,
        InvestigationPhase.CONVERGENCE
    ])
    # Mock generate_question if specific question patterns per phase are not critical for this test unit
    strategy.generate_question = MagicMock(return_value="Strategy question?")
    strategy.get_strategy_info = MagicMock(return_value={"name": "mock_strategy_info"})
    return strategy

@pytest.fixture
def mock_convergence_detector_instance(mock_strategy_instance): # Renamed
    # Use a real ConvergenceDetector but with strategy's params for consistency in tests
    # Or mock it if very specific non-converging/converging behavior is needed for *every* call
    detector = ConvergenceDetector(
        similarity_threshold=mock_strategy_instance.convergence_threshold,
        min_improvement=mock_strategy_instance.min_improvement,
        convergence_window=mock_strategy_instance.convergence_window,
        max_iterations=mock_strategy_instance.max_iterations
    )
    return detector


def test_perquire_investigator_initialization(mock_llm_provider_instance, mock_embedding_provider_instance, mock_strategy_instance, mock_convergence_detector_instance):
    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
        questioning_strategy=mock_strategy_instance,
        convergence_detector=mock_convergence_detector_instance
    )
    assert investigator.llm_provider == mock_llm_provider_instance
    assert investigator.embedding_provider == mock_embedding_provider_instance
    assert investigator.questioning_strategy == mock_strategy_instance
    assert investigator.convergence_detector == mock_convergence_detector_instance

def test_perquire_investigator_init_with_provider_names(mock_llm_provider_instance, mock_embedding_provider_instance):
    with patch('perquire.llm.base.provider_registry.get_provider', return_value=mock_llm_provider_instance) as mock_get_llm, \
         patch('perquire.embeddings.base.embedding_registry.get_provider', return_value=mock_embedding_provider_instance) as mock_get_emb:

        investigator = PerquireInvestigator(
            llm_provider="mock_llm_name",
            embedding_provider="mock_emb_name"
        )
        mock_get_llm.assert_called_once_with("mock_llm_name")
        mock_get_emb.assert_called_once_with("mock_emb_name")
        assert investigator.llm_provider == mock_llm_provider_instance
        assert investigator.embedding_provider == mock_embedding_provider_instance

def test_investigation_fails_if_llm_unavailable(mock_llm_provider_instance, mock_embedding_provider_instance):
    mock_llm_provider_instance.is_available_flag = False
    target_embedding = np.random.rand(10).astype(np.float32)
    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
    )
    with pytest.raises(InvestigationError, match="LLM provider is not available"):
        investigator.investigate(target_embedding)

def test_investigation_fails_if_embedding_provider_unavailable(mock_llm_provider_instance, mock_embedding_provider_instance):
    mock_embedding_provider_instance.is_available_flag = False
    target_embedding = np.random.rand(mock_embedding_provider_instance.dim).astype(np.float32)
    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
    )
    with pytest.raises(InvestigationError, match="Embedding provider is not available"):
        investigator.investigate(target_embedding)

def test_investigation_fails_with_invalid_embedding(mock_llm_provider_instance, mock_embedding_provider_instance):
    investigator = PerquireInvestigator(llm_provider=mock_llm_provider_instance, embedding_provider=mock_embedding_provider_instance)
    with pytest.raises(ValidationError, match="Target embedding must be a numpy array"):
        investigator.investigate(target_embedding=[0.1, 0.2]) # Not a numpy array
    with pytest.raises(ValidationError, match="Target embedding must be 1-dimensional"):
        investigator.investigate(target_embedding=np.array([[0.1],[0.2]]))
    with pytest.raises(ValidationError, match="Target embedding contains invalid values"):
        investigator.investigate(target_embedding=np.array([0.1, np.nan, 0.2]))

def test_investigation_loop_runs_and_completes(mock_llm_provider_instance, mock_embedding_provider_instance, mock_strategy_instance, mock_convergence_detector_instance):
    """Test that the investigation loop runs, calls mocks, and completes."""
    target_embedding = np.random.rand(mock_embedding_provider_instance.dim).astype(np.float32)

    # Ensure the convergence detector will eventually say "converged"
    # For example, after 2 iterations, then converge on the 3rd. Max iterations is 3 in mock_strategy.
    mock_convergence_detector_instance.should_continue = MagicMock(side_effect=[
        ConvergenceResult(converged=False, reason=ConvergenceReason.MANUAL_STOP, confidence=0.0, iteration_reached=1, final_similarity=0.1, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="continue"),
        ConvergenceResult(converged=False, reason=ConvergenceReason.MANUAL_STOP, confidence=0.0, iteration_reached=2, final_similarity=0.2, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="continue"),
        ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=3, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop")
    ])

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
        questioning_strategy=mock_strategy_instance,
        convergence_detector=mock_convergence_detector_instance
    )

    result = investigator.investigate(target_embedding, verbose=False)

    assert result is not None
    assert result.iterations == 3 # Based on side_effect of should_continue and max_iterations
    assert result.final_similarity > 0 # Some similarity should be calculated
    assert result.description == "mock synthesized description" # From mock LLM

    # Check if providers were called
    # LLM's generate_questions or strategy's generate_question should be called
    # If LLM has generate_questions, it's preferred by investigator
    assert mock_llm_provider_instance.generate_questions.call_count == 3
    # Embedding provider's embed_text should be called for each question
    assert mock_embedding_provider_instance.embed_text.call_count == 3
    # LLM's synthesize_description should be called once at the end
    mock_llm_provider_instance.synthesize_description.assert_called_once()
    # Convergence detector should be called multiple times
    assert mock_convergence_detector_instance.should_continue.call_count == 3
    # Strategy's determine_phase should be called
    assert mock_strategy_instance.determine_phase.call_count == 3
```
