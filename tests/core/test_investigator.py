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


def test_investigation_uses_strategy_question_gen_fallback(mock_llm_provider_instance, mock_embedding_provider_instance, mock_strategy_instance, mock_convergence_detector_instance):
    """Test that QuestioningStrategy.generate_question is used if LLM provider lacks generate_questions."""
    # Remove generate_questions from the mock LLM provider for this test
    del mock_llm_provider_instance.generate_questions

    target_embedding = np.random.rand(mock_embedding_provider_instance.dim).astype(np.float32)

    mock_convergence_detector_instance.should_continue = MagicMock(return_value=ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=1, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop"))
    mock_strategy_instance.max_iterations = 1 # Ensure only one iteration

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
        questioning_strategy=mock_strategy_instance,
        convergence_detector=mock_convergence_detector_instance
    )

    investigator.investigate(target_embedding, verbose=False)

    # generate_question on the strategy mock should be called
    mock_strategy_instance.generate_question.assert_called_once()


@pytest.fixture
def mock_db_provider_instance():
    db_provider = MagicMock()
    db_provider.save_investigation = MagicMock()
    # Mock other DB methods if they are called directly by PerquireInvestigator during a typical flow
    db_provider.get_cached_llm_question_gen = MagicMock(return_value=None)
    db_provider.set_cached_llm_question_gen = MagicMock()
    db_provider.get_cached_embedding = MagicMock(return_value=None)
    db_provider.set_cached_embedding = MagicMock()
    db_provider.get_cached_similarity = MagicMock(return_value=None)
    db_provider.set_cached_similarity = MagicMock()
    db_provider.get_cached_llm_synthesis = MagicMock(return_value=None)
    db_provider.set_cached_llm_synthesis = MagicMock()
    return db_provider

def test_investigation_saves_to_database(mock_llm_provider_instance, mock_embedding_provider_instance, mock_strategy_instance, mock_convergence_detector_instance, mock_db_provider_instance):
    target_embedding = np.random.rand(mock_embedding_provider_instance.dim).astype(np.float32)
    mock_convergence_detector_instance.should_continue = MagicMock(return_value=ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=1, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop"))
    mock_strategy_instance.max_iterations = 1


    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
        questioning_strategy=mock_strategy_instance,
        convergence_detector=mock_convergence_detector_instance,
        database_provider=mock_db_provider_instance
    )

    # Test saving enabled (default)
    investigator.investigate(target_embedding, save_to_database=True, verbose=False)
    mock_db_provider_instance.save_investigation.assert_called_once()

    # Test saving disabled
    mock_db_provider_instance.save_investigation.reset_mock() # Reset mock for the next call
    investigator.investigate(target_embedding, save_to_database=False, verbose=False)
    mock_db_provider_instance.save_investigation.assert_not_called()


def test_investigate_batch_processes_multiple_embeddings(mock_llm_provider_instance, mock_embedding_provider_instance, mock_strategy_instance, mock_convergence_detector_instance):
    num_embeddings = 3
    embeddings_dim = mock_embedding_provider_instance.dim
    target_embeddings = [np.random.rand(embeddings_dim).astype(np.float32) for _ in range(num_embeddings)]

    mock_convergence_detector_instance.should_continue = MagicMock(return_value=ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=1, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop"))
    mock_strategy_instance.max_iterations = 1

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
        questioning_strategy=mock_strategy_instance,
        convergence_detector=mock_convergence_detector_instance
    )

    # Patch the investigate method to track calls without running full logic
    with patch.object(investigator, 'investigate', wraps=investigator.investigate) as mock_single_investigate:
        results = investigator.investigate_batch(target_embeddings, verbose=False)

        assert len(results) == num_embeddings
        assert mock_single_investigate.call_count == num_embeddings
        for i in range(num_embeddings):
            # Check that the correct embedding was passed to each call
            passed_embedding_arg = mock_single_investigate.call_args_list[i][1]['target_embedding']
            np.testing.assert_array_equal(passed_embedding_arg, target_embeddings[i])


def test_health_check_structure(mock_llm_provider_instance, mock_embedding_provider_instance, mock_db_provider_instance):
    # Case 1: All healthy
    mock_llm_provider_instance.health_check = MagicMock(return_value={"status": "healthy"})
    mock_embedding_provider_instance.health_check = MagicMock(return_value={"status": "healthy"})
    mock_db_provider_instance.health_check = MagicMock(return_value={"status": "healthy"})

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
        database_provider=mock_db_provider_instance
    )
    health = investigator.health_check()
    assert health["overall_status"] == "healthy"
    assert health["components"]["llm_provider"]["status"] == "healthy"
    assert health["components"]["embedding_provider"]["status"] == "healthy"
    assert health["components"]["database_provider"]["status"] == "healthy"

    # Case 2: One unhealthy
    mock_llm_provider_instance.health_check = MagicMock(return_value={"status": "unhealthy", "reason": "LLM down"})
    health = investigator.health_check()
    assert health["overall_status"] == "unhealthy"
    assert health["components"]["llm_provider"]["status"] == "unhealthy"

    # Case 3: DB not configured
    investigator_no_db = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance, # Still unhealthy from above
        embedding_provider=mock_embedding_provider_instance,
        database_provider=None
    )
    mock_llm_provider_instance.health_check = MagicMock(return_value={"status": "healthy"}) # Make LLM healthy again for this sub-test
    mock_embedding_provider_instance.health_check = MagicMock(return_value={"status": "healthy"})

    health_no_db = investigator_no_db.health_check()
    assert health_no_db["overall_status"] == "degraded" # degraded because DB is not configured
    assert health_no_db["components"]["database_provider"]["status"] == "not_configured"


def test_investigation_uses_db_cache_for_question_generation(
    mock_llm_provider_instance,
    mock_embedding_provider_instance,
    mock_strategy_instance,
    mock_convergence_detector_instance,
    mock_db_provider_instance):

    target_embedding = np.random.rand(mock_embedding_provider_instance.dim).astype(np.float32)
    # Setup convergence to run for one iteration
    mock_convergence_detector_instance.should_continue = MagicMock(return_value=ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=1, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop"))
    mock_strategy_instance.max_iterations = 1

    # LLM's generate_questions is a MagicMock by default in mock_llm_provider_instance
    # We can add a side_effect to it to track calls
    mock_llm_provider_instance.generate_questions = MagicMock(return_value=["LLM generated question?"])

    # Simulate cache hit for question generation
    cached_questions = ["Cached question?"]
    mock_db_provider_instance.get_cached_llm_question_gen.return_value = cached_questions

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider_instance,
        embedding_provider=mock_embedding_provider_instance,
        questioning_strategy=mock_strategy_instance,
        convergence_detector=mock_convergence_detector_instance,
        database_provider=mock_db_provider_instance
    )

    result = investigator.investigate(target_embedding, verbose=False)

    # Assert that DB cache was checked
    mock_db_provider_instance.get_cached_llm_question_gen.assert_called_once()
    # Assert that LLM was NOT called because of cache hit
    mock_llm_provider_instance.generate_questions.assert_not_called()
    # Assert that the cached question was used (can be inferred if the result.question_history contains it)
    assert len(result.question_history) == 1
    assert result.question_history[0].question == "Cached question?"

    # Reset mocks and simulate cache miss
    mock_db_provider_instance.reset_mock()
    mock_llm_provider_instance.reset_mock() # Reset the entire mock to clear call counts
    mock_llm_provider_instance.generate_questions = MagicMock(return_value=["Fresh LLM question?"]) # Re-attach generate_questions
    mock_db_provider_instance.get_cached_llm_question_gen.return_value = None # Cache miss

    result_cache_miss = investigator.investigate(target_embedding, verbose=False)

    mock_db_provider_instance.get_cached_llm_question_gen.assert_called_once()
    mock_llm_provider_instance.generate_questions.assert_called_once()
    mock_db_provider_instance.set_cached_llm_question_gen.assert_called_once()
    assert len(result_cache_miss.question_history) == 1
    assert result_cache_miss.question_history[0].question == "Fresh LLM question?"
