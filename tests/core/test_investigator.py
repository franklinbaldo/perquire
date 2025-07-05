import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY

from perquire.core.investigator import PerquireInvestigator
from perquire.llm.base import BaseLLMProvider, LLMResponse
from perquire.embeddings.base import BaseEmbeddingProvider, EmbeddingResult
from perquire.core.strategy import QuestioningStrategy, InvestigationPhase, InterrogatorStrategy, DefaultInterrogatorStrategy # Added
from perquire.core.result import QuestionAnswer, InvestigationResult # Added
from perquire.convergence.algorithms import ConvergenceDetector, ConvergenceResult, ConvergenceReason
from perquire.exceptions import InvestigationError, ConfigurationError, ValidationError, QuestionGenerationError # Added

# Mock LLM Provider (simplified for these tests)
class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config=None):
        super().__init__(config or {})
        self.is_available_flag = True
        self.name = "mock_llm"
    def validate_config(self): pass
    def generate_response(self, prompt, context=None, **kwargs): return LLMResponse(content="mock llm response", metadata={}, model=self.name)
    def synthesize_description(self, questions_and_scores, final_similarity, **kwargs):
        return "mock synthesized description"
    def is_available(self): return self.is_available_flag
    def get_model_info(self): return {"provider": self.name, "model": "test_model"}
    # generate_questions is NOT part of BaseLLMProvider, it was specific to the old investigator logic

# Mock Embedding Provider (simplified)
class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config=None, dim=10):
        super().__init__(config or {})
        self.dim = dim; self.is_available_flag = True; self.name = "mock_embedding"
    def validate_config(self): pass
    def _execute_embed_text(self, text, **kwargs): return EmbeddingResult(embedding=np.random.rand(self.dim).astype(np.float32), metadata={}, model=self.name, dimensions=self.dim)
    def get_embedding_dimensions(self): return self.dim
    def is_available(self): return self.is_available_flag
    def get_model_info(self): return {"provider": self.name, "model": "test_model", "dimensions": self.dim}

@pytest.fixture
def mock_llm_provider(): # Renamed for clarity
    return MockLLMProvider()

@pytest.fixture
def mock_embedding_provider(): # Renamed
    return MockEmbeddingProvider(dim=10)

@pytest.fixture
def mock_questioning_strategy(): # This is the overall strategy for phases/convergence
    strategy = MagicMock(spec=QuestioningStrategy)
    strategy.name = "mock_questioning_strategy"
    strategy.convergence_threshold = 0.95; strategy.min_improvement = 0.0001
    strategy.convergence_window = 2; strategy.max_iterations = 3
    strategy.determine_phase = MagicMock(side_effect=[InvestigationPhase.EXPLORATION, InvestigationPhase.REFINEMENT, InvestigationPhase.CONVERGENCE])
    strategy.generate_question = MagicMock(return_value="QuestioningStrategy default question?") # Used by DefaultInterrogatorStrategy
    strategy.get_strategy_info = MagicMock(return_value={"name": "mock_qs_info"})
    return strategy

@pytest.fixture
def mock_interrogator_strategy(): # This is the pluggable strategy for question generation
    is_mock = MagicMock(spec=InterrogatorStrategy)
    is_mock.generate_question = MagicMock(return_value="Custom Interrogator Question?")
    is_mock.reset_state = MagicMock() # For testing if it's called
    return is_mock

@pytest.fixture
def mock_convergence_detector(mock_questioning_strategy): # Renamed
    return ConvergenceDetector(
        similarity_threshold=mock_questioning_strategy.convergence_threshold,
        min_improvement=mock_questioning_strategy.min_improvement,
        convergence_window=mock_questioning_strategy.convergence_window,
        max_iterations=mock_questioning_strategy.max_iterations
    )

@pytest.fixture
def mock_db_provider():
    db_mock = MagicMock()
    db_mock.save_investigation = MagicMock()
    db_mock.get_cached_similarity = MagicMock(return_value=None) # Default to cache miss
    db_mock.set_cached_similarity = MagicMock()
    db_mock.get_cached_embedding = MagicMock(return_value=None) # Default to cache miss
    db_mock.set_cached_embedding = MagicMock()
    db_mock.get_cached_llm_synthesis = MagicMock(return_value=None) # Default to cache miss
    db_mock.set_cached_llm_synthesis = MagicMock()
    # Note: Caching for LLM question generation is now internal to DefaultInterrogatorStrategy or custom ones
    return db_mock

def test_perquire_investigator_initialization_default_interrogator(mock_llm_provider, mock_embedding_provider, mock_questioning_strategy, mock_convergence_detector):
    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider,
        questioning_strategy=mock_questioning_strategy, # Passed for DefaultInterrogator and phase logic
        convergence_detector=mock_convergence_detector
    )
    assert isinstance(investigator.interrogator_strategy, DefaultInterrogatorStrategy)
    assert investigator.interrogator_strategy.questioning_strategy == mock_questioning_strategy
    assert investigator.interrogator_strategy.llm_provider == mock_llm_provider

def test_perquire_investigator_init_with_custom_interrogator_strategy(mock_llm_provider, mock_embedding_provider, mock_interrogator_strategy):
    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider,
        interrogator_strategy=mock_interrogator_strategy # Pass the custom one
    )
    assert investigator.interrogator_strategy == mock_interrogator_strategy

def test_investigation_loop_with_custom_interrogator(
    mock_llm_provider, mock_embedding_provider, mock_questioning_strategy,
    mock_interrogator_strategy, mock_convergence_detector, mock_db_provider
):
    target_embedding = np.random.rand(mock_embedding_provider.dim).astype(np.float32)

    mock_convergence_detector.should_continue = MagicMock(side_effect=[
        ConvergenceResult(converged=False, reason=ConvergenceReason.MANUAL_STOP, confidence=0.0, iteration_reached=1, final_similarity=0.1, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="continue"),
        ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=2, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop")
    ])
    mock_questioning_strategy.max_iterations = 2 # Align with convergence mock

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider,
        questioning_strategy=mock_questioning_strategy, # For phase/convergence
        interrogator_strategy=mock_interrogator_strategy, # Custom interrogator
        convergence_detector=mock_convergence_detector,
        database_provider=mock_db_provider
    )

    result = investigator.investigate(target_embedding, verbose=False)

    assert result is not None
    assert result.iterations == 2

    # Check custom interrogator was called correctly
    mock_interrogator_strategy.generate_question.assert_called()
    assert mock_interrogator_strategy.generate_question.call_count == 2

    # Check arguments of the first call to generate_question
    first_call_args = mock_interrogator_strategy.generate_question.call_args_list[0]
    assert isinstance(first_call_args.kwargs['history'], list)
    assert len(first_call_args.kwargs['history']) == 0 # Initially empty
    assert first_call_args.kwargs['current_similarity'] == 0.0
    np.testing.assert_array_equal(first_call_args.kwargs['target_embedding'], target_embedding)
    assert first_call_args.kwargs['investigator_context']['phase'] == InvestigationPhase.EXPLORATION

    # Check arguments of the second call (history should have one item)
    second_call_args = mock_interrogator_strategy.generate_question.call_args_list[1]
    assert len(second_call_args.kwargs['history']) == 1
    assert isinstance(second_call_args.kwargs['history'][0], QuestionAnswer)

    mock_embedding_provider._execute_embed_text.assert_called() # embed_text on provider
    mock_llm_provider.synthesize_description.assert_called_once()
    mock_convergence_detector.should_continue.assert_called()
    mock_questioning_strategy.determine_phase.assert_called()


def test_investigation_with_default_interrogator_calls_reset_state(
    mock_llm_provider, mock_embedding_provider, mock_questioning_strategy, mock_convergence_detector
):
    target_embedding = np.random.rand(mock_embedding_provider.dim).astype(np.float32)
    mock_convergence_detector.should_continue = MagicMock(return_value=ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=1, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop"))
    mock_questioning_strategy.max_iterations = 1

    # Patch DefaultInterrogatorStrategy to spy on its instance methods
    with patch('perquire.core.investigator.DefaultInterrogatorStrategy', autospec=True) as MockDefaultInterrogator:
        mock_default_instance = MockDefaultInterrogator.return_value
        mock_default_instance.generate_question.return_value = "Default strategy question"

        investigator = PerquireInvestigator(
            llm_provider=mock_llm_provider,
            embedding_provider=mock_embedding_provider,
            questioning_strategy=mock_questioning_strategy, # Will be passed to DefaultInterrogator
            convergence_detector=mock_convergence_detector
            # No interrogator_strategy, so DefaultInterrogatorStrategy is created
        )

        investigator.investigate(target_embedding, verbose=False)

        mock_default_instance.reset_state.assert_called_once()
        mock_default_instance.generate_question.assert_called_once()

# Test for failure when LLM/Embedding provider is unavailable (largely unchanged)
def test_investigation_fails_if_providers_unavailable(mock_llm_provider, mock_embedding_provider):
    target_embedding = np.random.rand(mock_embedding_provider.dim).astype(np.float32)
    investigator_no_llm = PerquireInvestigator(llm_provider=None, embedding_provider=mock_embedding_provider) # Will fail on init

    with patch('perquire.llm.base.provider_registry.get_provider', side_effect=Exception("No default LLM")):
        with pytest.raises(InvestigationError, match="No LLM provider specified or default available"):
             PerquireInvestigator(embedding_provider=mock_embedding_provider)

    with patch('perquire.embeddings.base.embedding_registry.get_provider', side_effect=Exception("No default Emb")):
        with pytest.raises(InvestigationError, match="No embedding provider specified or default available"):
             PerquireInvestigator(llm_provider=mock_llm_provider)

    mock_llm_provider.is_available_flag = False
    investigator_bad_llm = PerquireInvestigator(llm_provider=mock_llm_provider, embedding_provider=mock_embedding_provider)
    with pytest.raises(InvestigationError, match="LLM provider not available/configured."):
        investigator_bad_llm.investigate(target_embedding)
    mock_llm_provider.is_available_flag = True # Reset

    mock_embedding_provider.is_available_flag = False
    investigator_bad_emb = PerquireInvestigator(llm_provider=mock_llm_provider, embedding_provider=mock_embedding_provider)
    with pytest.raises(InvestigationError, match="Embedding provider not available/configured."):
        investigator_bad_emb.investigate(target_embedding)


# Simplified DB caching test - focus is on similarity and embedding caching by Investigator
# Question generation caching is now an internal concern of the InterrogatorStrategy (e.g. DefaultInterrogatorStrategy)
def test_investigation_uses_db_cache_for_similarity_and_embeddings(
    mock_llm_provider, mock_embedding_provider, mock_questioning_strategy,
    mock_interrogator_strategy, mock_convergence_detector, mock_db_provider):

    target_embedding = np.random.rand(mock_embedding_provider.dim).astype(np.float32)
    mock_convergence_detector.should_continue = MagicMock(return_value=ConvergenceResult(converged=True, reason=ConvergenceReason.MAX_ITERATIONS, confidence=1.0, iteration_reached=1, final_similarity=0.3, similarity_improvement=0.1, plateau_length=0, statistical_metrics={}, recommendation="stop"))
    mock_questioning_strategy.max_iterations = 1

    # Mock interrogator to return a consistent question
    fixed_question = "Fixed question for caching test?"
    mock_interrogator_strategy.generate_question.return_value = fixed_question

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider, embedding_provider=mock_embedding_provider,
        questioning_strategy=mock_questioning_strategy, interrogator_strategy=mock_interrogator_strategy,
        convergence_detector=mock_convergence_detector, database_provider=mock_db_provider
    )

    # 1. First run (cache miss for everything)
    investigator.investigate(target_embedding, verbose=False)
    mock_db_provider.get_cached_similarity.assert_called_once()
    mock_db_provider.set_cached_similarity.assert_called_once()
    mock_db_provider.get_cached_embedding.assert_called_once() # For the question embedding
    mock_embedding_provider._execute_embed_text.assert_called_once_with(fixed_question) # Embedding generated
    mock_db_provider.set_cached_embedding.assert_called_once()

    # Reset mocks for second run
    mock_db_provider.reset_mock()
    mock_embedding_provider._execute_embed_text.reset_mock() # Important: reset call count on this specific method

    # 2. Second run (should hit caches)
    # Simulate cache hit for similarity
    mock_db_provider.get_cached_similarity.return_value = 0.75
    # Simulate cache hit for question embedding (though similarity hit should prevent this call)
    # mock_db_provider.get_cached_embedding.return_value = np.random.rand(mock_embedding_provider.dim).astype(np.float32)

    investigator.investigate(target_embedding, verbose=False)
    mock_db_provider.get_cached_similarity.assert_called_once() # Checked cache
    mock_db_provider.set_cached_similarity.assert_not_called() # Not set, because it was a hit

    # Because similarity was a hit, embedding for the question should not be re-calculated or re-fetched from DB cache
    mock_db_provider.get_cached_embedding.assert_not_called()
    mock_embedding_provider._execute_embed_text.assert_not_called()
    mock_db_provider.set_cached_embedding.assert_not_called()


def test_investigation_handles_question_generation_error(
    mock_llm_provider, mock_embedding_provider, mock_questioning_strategy,
    mock_interrogator_strategy, mock_convergence_detector
):
    target_embedding = np.random.rand(mock_embedding_provider.dim).astype(np.float32)
    mock_interrogator_strategy.generate_question.side_effect = QuestionGenerationError("Test QGE")

    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider, embedding_provider=mock_embedding_provider,
        questioning_strategy=mock_questioning_strategy, interrogator_strategy=mock_interrogator_strategy,
        convergence_detector=mock_convergence_detector
    )
    result = investigator.investigate(target_embedding, verbose=False)
    assert result.convergence_reason == ConvergenceReason.ERROR.value
    assert "Test QGE" in result.description or "Synthesis unavailable" in result.description or "Synthesis failed" in result.description # Description might be fallback
    assert result.iterations == 0 # Loop breaks on first iteration due to QGE

# Health check should now also report interrogator_strategy
def test_health_check_includes_interrogator_strategy(mock_llm_provider, mock_embedding_provider, mock_interrogator_strategy):
    mock_interrogator_strategy.health_check = MagicMock(return_value={"status": "healthy_interrogator"})
    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider,
        interrogator_strategy=mock_interrogator_strategy
    )
    health = investigator.health_check()
    assert "interrogator_strategy" in health["components"]
    assert health["components"]["interrogator_strategy"]["status"] == "healthy_interrogator"

    # Test with default interrogator strategy (no specific health_check method)
    investigator_default_is = PerquireInvestigator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider
    )
    health_default = investigator_default_is.health_check()
    assert health_default["components"]["interrogator_strategy"]["status"] == "configured"
    assert health_default["components"]["interrogator_strategy"]["type"] == "DefaultInterrogatorStrategy"

# Ensure _get_model_config includes interrogator_strategy
def test_get_model_config_includes_interrogator_strategy(mock_llm_provider, mock_embedding_provider, mock_interrogator_strategy):
    investigator = PerquireInvestigator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider,
        interrogator_strategy=mock_interrogator_strategy
    )
    config = investigator._get_model_config() # Accessing protected member for test
    assert "interrogator_strategy" in config
    assert config["interrogator_strategy"]["name"] == type(mock_interrogator_strategy).__name__

    # Test with default
    investigator_default = PerquireInvestigator(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_embedding_provider
    )
    config_default = investigator_default._get_model_config()
    assert config_default["interrogator_strategy"]["name"] == "DefaultInterrogatorStrategy"
```
