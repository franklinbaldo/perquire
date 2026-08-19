from pathlib import Path

import numpy as np

from perquire.providers.embedding_cache import EmbeddingCache


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
