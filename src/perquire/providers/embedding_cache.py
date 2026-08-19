"""Compatibility wrapper over the multi-space semantic store.

The legacy ``embeddings`` table remains readable so previously paid-for vectors
can be promoted lazily when their source text appears again.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from .semantic_store import SemanticStore

_CACHE_SCHEMA_VERSION = "embedding-v1"
_DEFAULT_CACHE_DIR = ".cache/perquire"


class EmbeddingCache:
    """Backward-compatible embedding cache with lazy semantic-store promotion."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        # Keep the original schema intact for old Actions cache snapshots.
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self.semantic_store = SemanticStore(self.path)
        self.legacy_promotions = 0

    @staticmethod
    def key(model: str, text: str) -> str:
        payload = json.dumps(
            {"schema": _CACHE_SCHEMA_VERSION, "model": model, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _get_legacy(self, model: str, text: str) -> np.ndarray | None:
        row = self._connection.execute(
            "SELECT dimensions, vector_json FROM embeddings WHERE cache_key = ? AND model = ?",
            (self.key(model, text), model),
        ).fetchone()
        if row is None:
            return None
        dimensions, raw = row
        vector = np.asarray(json.loads(raw), dtype=np.float64)
        if vector.size != int(dimensions):
            return None
        return vector

    def get(self, model: str, text: str) -> np.ndarray | None:
        vector = self.semantic_store.get_embedding(model, text)
        if vector is not None:
            return vector

        vector = self._get_legacy(model, text)
        if vector is None:
            return None

        # We can only recover text identity when the text appears again. At that
        # point promote the old vector without another provider request.
        self.semantic_store.put_embedding(model, text, vector)
        self.legacy_promotions += 1
        return vector

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        value = np.asarray(vector, dtype=np.float64)
        # Write both during the compatibility window. This means older code can
        # still consume a cache produced by newer runs.
        self._connection.execute(
            """
            INSERT OR REPLACE INTO embeddings(cache_key, model, dimensions, vector_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                self.key(model, text),
                model,
                int(value.size),
                json.dumps(value.tolist(), separators=(",", ":")),
            ),
        )
        self._connection.commit()
        self.semantic_store.put_embedding(model, text, value)

    def embedding_models(self, text: str) -> list[str]:
        """Return all model spaces already materialized for this semantic text."""
        return self.semantic_store.embedding_models(text)

    def close(self) -> None:
        self.semantic_store.close()
        self._connection.close()


def default_embedding_cache_path() -> Path:
    root = Path(os.getenv("PERQUIRE_CACHE_DIR", _DEFAULT_CACHE_DIR))
    return root / "embeddings.sqlite3"
