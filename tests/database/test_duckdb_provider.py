from datetime import datetime, timedelta

import numpy as np
import pytest

from perquire.core.result import InvestigationResult, QuestionAnswer
from perquire.database.base import DatabaseConfig
from perquire.database.duckdb_provider import DuckDBProvider


@pytest.fixture
def provider() -> DuckDBProvider:
    database = DuckDBProvider(DatabaseConfig(connection_string=":memory:"))
    database.connect()
    yield database
    database.disconnect()


def make_result(investigation_id: str = "inv-test", strategy: str = "default") -> InvestigationResult:
    start = datetime(2026, 1, 1, 12, 0, 0)
    result = InvestigationResult(
        investigation_id=investigation_id,
        description="A semantic candidate",
        final_similarity=0.82,
        iterations=2,
        start_time=start,
        end_time=start + timedelta(seconds=1),
        strategy_name=strategy,
        model_config={"embedding": "mock", "llm": "mock"},
    )
    result.add_question_answer(
        QuestionAnswer(
            question="candidate one",
            similarity=0.61,
            phase="exploration",
            iteration=1,
        )
    )
    result.add_question_answer(
        QuestionAnswer(
            question="candidate two",
            similarity=0.82,
            phase="refinement",
            iteration=2,
        )
    )
    return result


def test_provider_connects_and_creates_public_tables(provider) -> None:
    assert provider.is_connected()
    tables = {row[0] for row in provider._connection.execute("SHOW TABLES").fetchall()}
    assert {"investigations", "questions", "embeddings"}.issubset(tables)


def test_investigation_round_trip(provider) -> None:
    original = make_result()
    provider.save_investigation(original.to_dict())

    loaded = provider.load_investigation(original.investigation_id)
    assert loaded is not None
    restored = InvestigationResult.from_dict(loaded)
    assert restored.investigation_id == original.investigation_id
    assert restored.description == original.description
    assert restored.final_similarity == pytest.approx(original.final_similarity)
    assert restored.model_config == original.model_config
    assert [item.question for item in restored.question_history] == [
        "candidate one",
        "candidate two",
    ]


def test_list_and_stats_use_documented_interface(provider) -> None:
    provider.save_investigation(make_result("a", "default").to_dict())
    provider.save_investigation(make_result("b", "other").to_dict())

    assert len(provider.list_investigations(limit=1)) == 1
    filtered = provider.list_investigations(filters={"strategy": "other"})
    assert [item["investigation_id"] for item in filtered] == ["b"]

    stats = provider.get_investigation_stats()
    assert stats["total_investigations"] == 2
    assert stats["total_questions"] == 4
    assert stats["average_similarity"] == pytest.approx(0.82)
    assert stats["average_iterations"] == pytest.approx(2.0)


def test_embedding_round_trip_and_similarity_search(provider) -> None:
    embedding = np.zeros(768, dtype=np.float32)
    embedding[0] = 1.0
    embedding_id = provider.save_embedding(
        "known text",
        embedding,
        {"model": "mock", "provider": "mock"},
    )

    loaded = provider.load_embedding(embedding_id)
    assert loaded is not None
    assert loaded["text"] == "known text"
    assert loaded["dimensions"] == 768

    matches = provider.search_embeddings(embedding, limit=1, similarity_threshold=0.9)
    assert matches
    assert matches[0]["embedding_id"] == embedding_id
    assert matches[0]["similarity_score"] == pytest.approx(1.0)


def test_missing_records_return_empty(provider) -> None:
    assert provider.load_investigation("missing") is None
    assert provider.load_embedding("missing") is None
