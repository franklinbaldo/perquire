import json
import sqlite3
from pathlib import Path

import numpy as np

from perquire.providers.embedding_cache import EmbeddingCache
from perquire.providers.semantic_store import SemanticStore


def test_cache_persists_across_instances(tmp_path: Path):
    path = tmp_path / "embeddings.sqlite3"
    first = EmbeddingCache(path)
    first.put("openrouter/model-a", "hidden text", np.array([0.1, 0.2, 0.3]))
    first.close()

    second = EmbeddingCache(path)
    restored = second.get("openrouter/model-a", "hidden text")

    assert restored is not None
    np.testing.assert_allclose(restored, [0.1, 0.2, 0.3])


def test_cache_key_changes_with_model_or_text():
    key = EmbeddingCache.key("model-a", "same text")
    assert key != EmbeddingCache.key("model-b", "same text")
    assert key != EmbeddingCache.key("model-a", "different text")


def test_one_semantic_text_accumulates_multiple_unaligned_embedding_spaces(tmp_path: Path):
    store = SemanticStore(tmp_path / "embeddings.sqlite3")
    text = "a red dress near the river"

    text_id_a = store.put_embedding("space/model-a", text, np.array([1.0, 0.0]))
    text_id_b = store.put_embedding("space/model-b", text, np.array([0.1, 0.2, 0.3]))

    assert text_id_a == text_id_b
    assert store.embedding_models(text) == ["space/model-a", "space/model-b"]
    np.testing.assert_allclose(store.get_embedding("space/model-a", text), [1.0, 0.0])
    np.testing.assert_allclose(store.get_embedding("space/model-b", text), [0.1, 0.2, 0.3])


def test_embedding_spaces_do_not_cross_hit(tmp_path: Path):
    store = SemanticStore(tmp_path / "embeddings.sqlite3")
    store.put_embedding("space/model-a", "same text", np.array([1.0, 0.0]))

    assert store.get_embedding("space/model-b", "same text") is None


def test_legacy_embedding_is_promoted_without_recomputing(tmp_path: Path):
    path = tmp_path / "embeddings.sqlite3"
    model = "openrouter/legacy-model"
    text = "already paid for"
    vector = np.array([0.4, 0.5])

    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE embeddings (
            cache_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO embeddings(cache_key, model, dimensions, vector_json) VALUES (?, ?, ?, ?)",
        (
            EmbeddingCache.key(model, text),
            model,
            2,
            json.dumps(vector.tolist(), separators=(",", ":")),
        ),
    )
    connection.commit()
    connection.close()

    cache = EmbeddingCache(path)
    restored = cache.get(model, text)

    assert restored is not None
    np.testing.assert_allclose(restored, vector)
    assert cache.legacy_promotions == 1
    assert cache.embedding_models(text) == [model]

    # The second read comes from the semantic schema, not another legacy promotion.
    cache.get(model, text)
    assert cache.legacy_promotions == 1
