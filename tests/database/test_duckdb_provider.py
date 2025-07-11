import pytest
import duckdb
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from perquire.database.duckdb_provider import DuckDBProvider
from perquire.database.base import DatabaseConfig
from perquire.core.result import InvestigationResult, QuestionAnswer # For creating dummy data

# Use a consistent temporary directory for all tests in this module
@pytest.fixture(scope="module")
def temp_db_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("test_dbs")

@pytest.fixture
def in_memory_provider():
    """Provides a DuckDBProvider with an in-memory database."""
    config = DatabaseConfig(connection_string=":memory:")
    provider = DuckDBProvider(config)
    provider._ensure_schema() # Ensure tables are created
    return provider

@pytest.fixture
def file_based_provider(temp_db_dir):
    """Provides a DuckDBProvider with a file-based database."""
    db_file = temp_db_dir / "test_file.db"
    config = DatabaseConfig(connection_string=str(db_file))
    provider = DuckDBProvider(config)
    provider._ensure_schema() # Ensure tables are created
    yield provider
    # Teardown: close connection and remove db file if necessary
    # DuckDBProvider's __del__ should handle connection closing.
    # For file removal, it can be tricky if not closed properly.
    # Rely on tmp_path_factory for cleanup where possible or add explicit close/delete.
    if db_file.exists():
        try:
            # Ensure connection is closed before attempting to delete
            # This might require adding an explicit close method to DuckDBProvider
            # For now, assume DuckDB handles this gracefully or tests might show issues.
            if hasattr(provider, 'con') and provider.con:
                provider.con.close()
            # db_file.unlink() # Can cause issues if not fully closed
        except Exception as e:
            print(f"Error during test db file cleanup: {e}")


# Test both in-memory and file-based providers
@pytest.mark.parametrize("provider_fixture", ["in_memory_provider", "file_based_provider"])
def test_provider_initialization(provider_fixture, request):
    provider = request.getfixturevalue(provider_fixture)
    assert provider is not None
    assert provider.con is not None
    # Check if tables were created by _ensure_schema
    tables = provider.con.execute("SHOW TABLES;").fetchall()
    table_names = [table[0] for table in tables]
    assert "embeddings_cache" in table_names
    assert "similarity_cache" in table_names
    assert "llm_generations_cache" in table_names
    assert "investigations" in table_names
    assert "question_history" in table_names


@pytest.mark.parametrize("provider_fixture", ["in_memory_provider", "file_based_provider"])
class TestDuckDBCaching:

    def test_store_and_get_embedding(self, provider_fixture, request):
        provider: DuckDBProvider = request.getfixturevalue(provider_fixture)
        text_content = "Hello, world!"
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        model_name = "test_model"
        provider_name = "test_emb_provider"
        dimensions = 3
        embedding_norm = float(np.linalg.norm(embedding))

        # Store
        provider.set_cached_embedding(
            text_content=text_content,
            embedding=embedding,
            model_name=model_name,
            provider_name=provider_name,
            dimensions=dimensions,
            embedding_norm=embedding_norm,
            metadata={"source": "test"}
        )

        # Get by text_hash (derived from text_content)
        text_hash = provider._generate_hash(text_content)
        cached_embedding_data = provider.get_cached_embedding(text_hash)

        assert cached_embedding_data is not None
        np.testing.assert_array_almost_equal(cached_embedding_data, embedding, decimal=5)

        # Test cache miss
        non_existent_hash = provider._generate_hash("non_existent_text")
        assert provider.get_cached_embedding(non_existent_hash) is None

        # Test storing and retrieving with different metadata
        text_content_2 = "Another text"
        embedding_2 = np.array([0.4,0.5,0.6])
        provider.set_cached_embedding(text_content_2, embedding_2, "model2", "prov2", 3, float(np.linalg.norm(embedding_2)), {"other":"data"})
        text_hash_2 = provider._generate_hash(text_content_2)
        cached_emb_2 = provider.get_cached_embedding(text_hash_2)
        assert cached_emb_2 is not None
        np.testing.assert_array_almost_equal(cached_emb_2, embedding_2, decimal=5)


    def test_store_and_get_similarity(self, provider_fixture, request):
        provider: DuckDBProvider = request.getfixturevalue(provider_fixture)
        question_hash = provider._generate_hash("Is it blue?")
        target_embedding_hash = provider._generate_hash(np.array([0.1,0.2,0.3]).tobytes()) # Example target
        similarity_score = 0.85

        provider.set_cached_similarity(question_hash, target_embedding_hash, similarity_score)

        cached_score = provider.get_cached_similarity(question_hash, target_embedding_hash)
        assert cached_score is not None
        assert abs(cached_score - similarity_score) < 1e-6

        # Test cache miss
        other_q_hash = provider._generate_hash("Is it red?")
        assert provider.get_cached_similarity(other_q_hash, target_embedding_hash) is None


    def test_store_and_get_llm_question_gen(self, provider_fixture, request):
        provider: DuckDBProvider = request.getfixturevalue(provider_fixture)
        input_context_hash = provider._generate_hash({"desc": "animal", "sim": 0.5})
        generated_questions = ["Is it a cat?", "Does it have fur?"]

        provider.set_cached_llm_question_gen(input_context_hash, generated_questions)
        cached_questions = provider.get_cached_llm_question_gen(input_context_hash)

        assert cached_questions is not None
        assert isinstance(cached_questions, list)
        assert cached_questions == generated_questions

        # Test cache miss
        other_ctx_hash = provider._generate_hash({"desc": "vehicle", "sim": 0.6})
        assert provider.get_cached_llm_question_gen(other_ctx_hash) is None

        # Test storing empty list
        provider.set_cached_llm_question_gen(other_ctx_hash, [])
        assert provider.get_cached_llm_question_gen(other_ctx_hash) == []


    def test_store_and_get_llm_synthesis(self, provider_fixture, request):
        provider: DuckDBProvider = request.getfixturevalue(provider_fixture)
        input_context_hash = provider._generate_hash({"qas": [{"q":"Is it blue?","s":0.8}], "final_sim": 0.8})
        synthesized_description = "It is likely a blue object."

        provider.set_cached_llm_synthesis(input_context_hash, synthesized_description)
        cached_desc = provider.get_cached_llm_synthesis(input_context_hash)

        assert cached_desc is not None
        assert cached_desc == synthesized_description

        # Test cache miss
        other_ctx_hash = provider._generate_hash({"qas": [{"q":"Is it red?","s":0.7}], "final_sim": 0.7})
        assert provider.get_cached_llm_synthesis(other_ctx_hash) is None

@pytest.mark.parametrize("provider_fixture", ["in_memory_provider", "file_based_provider"])
class TestDuckDBInvestigationStorage:

    def _create_dummy_investigation_result(self, inv_id="test_inv_123") -> InvestigationResult:
        # Create a more complete InvestigationResult
        start_time = datetime.now(timezone.utc)
        # Ensure end_time is after start_time for valid duration
        end_time = datetime.fromtimestamp(start_time.timestamp() + 10, tz=timezone.utc)

        res = InvestigationResult(
            investigation_id=inv_id,
            description="A mock investigation about a blue sphere.",
            final_similarity=0.92,
            iterations=5,
            start_time=start_time,
            end_time=end_time, # Make sure this is set
            strategy_name="mock_strategy",
            model_config={"llm": "mock_llm", "embedding": "mock_embedding"},
            convergence_reason="threshold_met",
            phase_reached="convergence"
        )
        res.add_question_answer(QuestionAnswer(question="Is it blue?", similarity=0.8, phase="exploration", iteration=1))
        res.add_question_answer(QuestionAnswer(question="Is it a sphere?", similarity=0.92, phase="refinement", iteration=2))
        return res

    def test_save_and_load_investigation(self, provider_fixture, request):
        provider: DuckDBProvider = request.getfixturevalue(provider_fixture)
        original_result = self._create_dummy_investigation_result()

        provider.save_investigation(original_result.to_dict())

        loaded_data = provider.load_investigation(original_result.investigation_id)
        assert loaded_data is not None

        # Reconstruct InvestigationResult from loaded_data to compare easily
        # Note: Datetime precision issues might occur if not handled carefully.
        # DuckDB stores timestamps, Python datetime objects might have microseconds.
        # We need to ensure InvestigationResult.from_dict handles this.
        loaded_result = InvestigationResult.from_dict(loaded_data)

        assert loaded_result.investigation_id == original_result.investigation_id
        assert loaded_result.description == original_result.description
        assert abs(loaded_result.final_similarity - original_result.final_similarity) < 1e-6
        assert loaded_result.iterations == original_result.iterations
        # Compare timestamps with some tolerance or by converting to a common string format / int timestamp
        assert abs(loaded_result.start_time.timestamp() - original_result.start_time.timestamp()) < 1
        assert abs(loaded_result.end_time.timestamp() - original_result.end_time.timestamp()) < 1
        assert loaded_result.strategy_name == original_result.strategy_name
        assert loaded_result.model_config == original_result.model_config # Relies on JSON serialization being consistent
        assert loaded_result.convergence_reason == original_result.convergence_reason
        assert loaded_result.phase_reached == original_result.phase_reached

        # Check question history
        assert len(loaded_result.question_history) == len(original_result.question_history)
        for i, qa_loaded in enumerate(loaded_result.question_history):
            qa_orig = original_result.question_history[i]
            assert qa_loaded.question == qa_orig.question
            assert abs(qa_loaded.similarity - qa_orig.similarity) < 1e-6
            assert qa_loaded.phase == qa_orig.phase
            assert qa_loaded.iteration == qa_orig.iteration

        # Test loading non-existent investigation
        assert provider.load_investigation("non_existent_id") is None

    def test_list_investigations(self, provider_fixture, request):
        provider: DuckDBProvider = request.getfixturevalue(provider_fixture)
        res1 = self._create_dummy_investigation_result("inv1")
        res2 = self._create_dummy_investigation_result("inv2")
        res2.strategy_name = "other_strategy" # For filtering test

        provider.save_investigation(res1.to_dict())
        provider.save_investigation(res2.to_dict())

        all_invs = provider.list_investigations()
        assert len(all_invs) >= 2 # Could be more if file DB persists across test runs

        limited_invs = provider.list_investigations(limit=1)
        assert len(limited_invs) == 1

        # Test filtering (example: by strategy_name)
        # Note: SQL LIKE might be case-sensitive depending on DB collation.
        # For robust filtering, ensure data and filter values have consistent casing or use LOWER().
        filtered_invs = provider.list_investigations(filters={"strategy_name": "mock_strategy"})
        assert len(filtered_invs) >= 1
        assert all(inv['strategy_name'] == "mock_strategy" for inv in filtered_invs if inv['investigation_id'] in ["inv1", "inv2"])

        filtered_invs_other = provider.list_investigations(filters={"strategy_name": "other_strategy"})
        assert len(filtered_invs_other) >= 1
        assert all(inv['strategy_name'] == "other_strategy" for inv in filtered_invs_other if inv['investigation_id'] in ["inv1", "inv2"])

    def test_get_investigation_stats(self, provider_fixture, request):
        provider: DuckDBProvider = request.getfixturevalue(provider_fixture)
        # Clear tables or use a fresh DB for precise stats, or check for non-zero if DB persists
        # For this test, let's assume it's relatively fresh or we check basic structure

        # Add some data if the DB might be empty
        if provider.get_statistics().get('total_investigations', 0) == 0:
            res1 = self._create_dummy_investigation_result("stat_inv1")
            res1.final_similarity = 0.8
            res1.iterations = 3 # from 2 qas + 1 in dummy
            provider.save_investigation(res1.to_dict())

            res2 = self._create_dummy_investigation_result("stat_inv2")
            res2.final_similarity = 0.9
            res2.iterations = 5 # from 2 qas + 3 in dummy
            provider.save_investigation(res2.to_dict())

        stats = provider.get_statistics()

        assert "total_investigations" in stats
        assert "total_questions" in stats # sum of iterations from question_history
        assert "avg_similarity" in stats
        assert "avg_iterations" in stats

        if stats["total_investigations"] > 0:
            assert stats["total_questions"] > 0
            assert 0 <= stats["avg_similarity"] <= 1
            assert stats["avg_iterations"] > 0
        else: # If DB was empty and no data added
            assert stats["total_questions"] == 0
            assert stats["avg_similarity"] == 0
            assert stats["avg_iterations"] == 0


def test_hash_consistency(in_memory_provider):
    """Test that hashing is consistent for same inputs."""
    provider = in_memory_provider
    text1 = "Hello"
    text2 = "Hello"
    text3 = "Goodbye"

    hash1 = provider._generate_hash(text1)
    hash2 = provider._generate_hash(text2)
    hash3 = provider._generate_hash(text3)

    assert hash1 == hash2
    assert hash1 != hash3

    arr1 = np.array([1,2,3]).tobytes()
    arr2 = np.array([1,2,3]).tobytes()
    arr3 = np.array([4,5,6]).tobytes()

    hash_arr1 = provider._generate_hash(arr1)
    hash_arr2 = provider._generate_hash(arr2)
    hash_arr3 = provider._generate_hash(arr3)

    assert hash_arr1 == hash_arr2
    assert hash_arr1 != hash_arr3

    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 2, "a": 1} # Order shouldn't matter for dicts if using sort_keys=True in json.dumps
    dict3 = {"a": 1, "c": 3}

    hash_dict1 = provider._generate_hash(dict1)
    hash_dict2 = provider._generate_hash(dict2) # This relies on json.dumps(sort_keys=True)
    hash_dict3 = provider._generate_hash(dict3)

    assert hash_dict1 == hash_dict2
    assert hash_dict1 != hash_dict3

# TODO: Add tests for schema migration if that feature is added.
# TODO: Add tests for error handling, e.g., DB connection errors (might require more mocking).
# TODO: Test date/time storage and retrieval precision more rigorously.
# TODO: Test handling of very large embeddings or large numbers of cache entries.
