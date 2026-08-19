"""Persistent content-addressed cache for deterministic embedding responses."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

_CACHE_SCHEMA_VERSION = "embedding-v1"
_DEFAULT_CACHE_DIR = ".cache/perquire"


class EmbeddingCache:
    """Store embedding vectors by model and input text without persisting plaintext.

    Embeddings are deterministic enough for benchmark replay only when the exact
    provider/model identifier and input text match. The cache key therefore hashes
    the schema version, model id, and text. SQLite stores only the digest and vector.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
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

    @staticmethod
    def key(model: str, text: str) -> str:
        payload = json.dumps(
            {"schema": _CACHE_SCHEMA_VERSION, "model": model, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model: str, text: str) -> np.ndarray | None:
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

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        value = np.asarray(vector, dtype=np.float64)
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

    def close(self) -> None:
        self._connection.close()


def default_embedding_cache_path() -> Path:
    root = Path(os.getenv("PERQUIRE_CACHE_DIR", _DEFAULT_CACHE_DIR))
    return root / "embeddings.sqlite3"
