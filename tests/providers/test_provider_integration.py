import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Test LLM Provider Registry
from perquire.llm.base import LLMProviderRegistry, BaseLLMProvider, LLMResponse, LLMProviderError
from perquire.exceptions import ConfigurationError # For provider init

class DummyLLMProvider(BaseLLMProvider):
    def __init__(self, config=None):
        self.name = "dummy_llm"
        super().__init__(config or {})
    def validate_config(self): pass # Assume valid
    def generate_response(self, prompt, context=None, **kwargs):
        return LLMResponse(content=f"response to {prompt}", metadata={}, model=self.name)
    def generate_questions(self, current_description, target_similarity, phase, previous_questions=None, **kwargs):
        return [f"q for {phase}?"]
    def synthesize_description(self, questions_and_scores, final_similarity, **kwargs):
        return "dummy description"
    def is_available(self): return True
    def get_model_info(self): return {"provider": self.name, "model": "test_model"}

def test_llm_registry_register_get():
    registry = LLMProviderRegistry()
    provider = DummyLLMProvider()
    registry.register_provider("dummy", provider, set_as_default=True)
    assert registry.get_provider("dummy") == provider
    assert registry.get_provider() == provider # Test default
    assert "dummy" in registry.list_providers()
    # Clean up global registry if it's being modified by other tests/imports
    registry._providers.clear()
    registry._default_provider = None


def test_llm_registry_get_unknown_raises_error():
    registry = LLMProviderRegistry()
    with pytest.raises(LLMProviderError, match="No default provider set"):
        registry.get_provider()
    with pytest.raises(LLMProviderError, match="Provider 'unknown' not found"):
        registry.get_provider("unknown")

# Test Embedding Provider Registry
from perquire.embeddings.base import EmbeddingProviderRegistry, BaseEmbeddingProvider, EmbeddingResult, EmbeddingError

class DummyEmbeddingProvider(BaseEmbeddingProvider):
    _execute_embed_text_call_count = 0 # Class variable to track calls across instances if needed, or instance var
    _execute_embed_batch_call_count = 0

    def __init__(self, config=None, dim=5):
        self.name = "dummy_embedding"
        self.dim = dim
        super().__init__(config or {})
        self._execute_embed_text_call_count = 0 # Instance variable
        self._execute_embed_batch_call_count = 0


    def validate_config(self): pass
    def _execute_embed_text(self, text, **kwargs):
        self._execute_embed_text_call_count += 1
        return EmbeddingResult(embedding=np.random.rand(self.dim).astype(np.float32), metadata={}, model=self.name, dimensions=self.dim)

    def _execute_embed_batch(self, texts, **kwargs):
        self._execute_embed_batch_call_count +=1
        return [EmbeddingResult(embedding=np.random.rand(self.dim).astype(np.float32), metadata={}, model=self.name, dimensions=self.dim) for _ in texts]

    def get_embedding_dimensions(self): return self.dim
    def is_available(self): return True
    def get_model_info(self): return {"provider": self.name, "model": "test_model", "dimensions": self.dim}

@pytest.fixture
def fresh_embedding_registry(): # Fixture to ensure clean registry for each test
    registry = EmbeddingProviderRegistry()
    # Preserve and restore original global registry if necessary, or work on a copy
    # For simplicity, this creates a new one. Tests should ideally not pollute global state.
    return registry

def test_embedding_registry_register_get(fresh_embedding_registry):
    registry = fresh_embedding_registry
    provider = DummyEmbeddingProvider()
    registry.register_provider("dummy_emb", provider, set_as_default=True)
    assert registry.get_provider("dummy_emb") == provider
    assert registry.get_provider() == provider
    assert "dummy_emb" in registry.list_providers()

def test_embedding_registry_get_unknown_raises_error(fresh_embedding_registry):
    registry = fresh_embedding_registry
    with pytest.raises(EmbeddingError, match="No default provider set"):
        registry.get_provider()
    with pytest.raises(EmbeddingError, match="Provider 'unknown_emb' not found"):
        registry.get_provider("unknown_emb")

# Test BaseEmbeddingProvider Caching
def test_base_embedding_provider_text_caching():
    provider = DummyEmbeddingProvider() # Uses instance call counts

    res1 = provider.embed_text("hello world")
    assert provider._execute_embed_text_call_count == 1

    res2 = provider.embed_text("hello world")
    assert provider._execute_embed_text_call_count == 1 # Should be cached
    assert np.array_equal(res1.embedding, res2.embedding)

    res3 = provider.embed_text("different text")
    assert provider._execute_embed_text_call_count == 2
    assert not np.array_equal(res1.embedding, res3.embedding)

    # Test with kwargs
    res4 = provider.embed_text("hello world", foo="bar") # Different kwargs tuple
    assert provider._execute_embed_text_call_count == 3

    res5 = provider.embed_text("hello world", foo="bar") # Same kwargs tuple
    assert provider._execute_embed_text_call_count == 3 # Cached
    assert np.array_equal(res4.embedding, res5.embedding)

    # Test cache clearing if needed (e.g. provider._cached_execute_embed_text.cache_clear())
    # For this test, assume cache persists for the provider instance.

def test_base_embedding_provider_batch_caching():
    provider = DummyEmbeddingProvider()

    texts1 = ["text a", "text b"]
    batch_res1 = provider.embed_batch(texts1)
    assert provider._execute_embed_batch_call_count == 1

    batch_res2 = provider.embed_batch(texts1) # Same batch texts
    assert provider._execute_embed_batch_call_count == 1 # Cached
    assert len(batch_res1) == len(batch_res2)
    for r1, r2 in zip(batch_res1, batch_res2):
        assert np.array_equal(r1.embedding, r2.embedding)

    texts2 = ["text c", "text d"] # Different batch
    provider.embed_batch(texts2)
    assert provider._execute_embed_batch_call_count == 2

    # Test batch with kwargs
    batch_res3 = provider.embed_batch(texts1, foo="baz")
    assert provider._execute_embed_batch_call_count == 3

    batch_res4 = provider.embed_batch(texts1, foo="baz")
    assert provider._execute_embed_batch_call_count == 3 # Cached
    assert len(batch_res3) == len(batch_res4)


def test_llm_registry_set_default_provider(fresh_llm_registry): # Assuming fresh_llm_registry fixture
    registry = fresh_llm_registry
    provider1 = DummyLLMProvider()
    provider2 = DummyLLMProvider()
    registry.register_provider("llm1", provider1)
    registry.register_provider("llm2", provider2) # Default is llm1 initially

    assert registry.get_provider() == provider1
    registry.set_default_provider("llm2")
    assert registry.get_provider() == provider2

    with pytest.raises(LLMProviderError, match="Provider 'unknown_llm' not found"):
        registry.set_default_provider("unknown_llm")


def test_llm_registry_get_healthy_providers(fresh_llm_registry):
    registry = fresh_llm_registry
    healthy_provider = DummyLLMProvider()
    unhealthy_provider = DummyLLMProvider()
    unhealthy_provider.is_available = MagicMock(return_value=False) # Make it unhealthy by is_available
    # Or mock health_check directly for more control if needed:
    healthy_provider.health_check = MagicMock(return_value={"status": "healthy"})
    unhealthy_provider.health_check = MagicMock(return_value={"status": "unhealthy", "reason": "test"})


    registry.register_provider("healthy_llm", healthy_provider)
    registry.register_provider("unhealthy_llm", unhealthy_provider)

    healthy_list = registry.get_healthy_providers()
    assert "healthy_llm" in healthy_list
    assert "unhealthy_llm" not in healthy_list
    healthy_provider.health_check.assert_called_once()
    unhealthy_provider.health_check.assert_called_once()


def test_embedding_registry_set_default_provider(fresh_embedding_registry):
    registry = fresh_embedding_registry
    provider1 = DummyEmbeddingProvider()
    provider2 = DummyEmbeddingProvider()
    registry.register_provider("emb1", provider1)
    registry.register_provider("emb2", provider2)

    assert registry.get_provider() == provider1 # First registered becomes default
    registry.set_default_provider("emb2")
    assert registry.get_provider() == provider2

    with pytest.raises(EmbeddingError, match="Provider 'unknown_emb' not found"):
        registry.set_default_provider("unknown_emb")


def test_embedding_registry_get_healthy_providers(fresh_embedding_registry):
    registry = fresh_embedding_registry
    healthy_provider = DummyEmbeddingProvider()
    unhealthy_provider = DummyEmbeddingProvider()

    healthy_provider.health_check = MagicMock(return_value={"status": "healthy"})
    unhealthy_provider.health_check = MagicMock(return_value={"status": "unhealthy", "reason": "test fail"})

    registry.register_provider("healthy_emb", healthy_provider)
    registry.register_provider("unhealthy_emb", unhealthy_provider)

    healthy_list = registry.get_healthy_providers()
    assert "healthy_emb" in healthy_list
    assert "unhealthy_emb" not in healthy_list
    healthy_provider.health_check.assert_called_once()
    unhealthy_provider.health_check.assert_called_once()

# Need a fixture for LLM Registry as well for consistency
@pytest.fixture
def fresh_llm_registry():
    registry = LLMProviderRegistry()
    return registry
