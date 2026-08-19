"""Content-addressed semantic store for texts and embeddings across spaces.

The store is intentionally agnostic to the search algorithm. A semantic text is
identified by content hash, while each embedding is identified by the exact
provider/model space that produced it. This lets one text accumulate multiple
unaligned representations without pretending the vectors are directly
comparable across spaces.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

_SCHEMA_VERSION = "semantic-store-v1"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SemanticStore:
    """Persist semantic text identities and 1:N model-specific embeddings."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL UNIQUE,
                schema_version TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_embeddings (
                semantic_text_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY (semantic_text_id, model),
                FOREIGN KEY (semantic_text_id) REFERENCES semantic_texts(id)
            )
            """
        )
        self._connection.commit()

    def text_id(self, text: str, *, create: bool = False) -> int | None:
        digest = content_hash(text)
        row = self._connection.execute(
            "SELECT id FROM semantic_texts WHERE content_hash = ?",
            (digest,),
        ).fetchone()
        if row is not None:
            return int(row[0])
        if not create:
            return None
        cursor = self._connection.execute(
            "INSERT INTO semantic_texts(content_hash, schema_version) VALUES (?, ?)",
            (digest, _SCHEMA_VERSION),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def get_embedding(self, model: str, text: str) -> np.ndarray | None:
        semantic_text_id = self.text_id(text)
        if semantic_text_id is None:
            return None
        row = self._connection.execute(
            """
            SELECT dimensions, vector_json
            FROM semantic_embeddings
            WHERE semantic_text_id = ? AND model = ?
            """,
            (semantic_text_id, model),
        ).fetchone()
        if row is None:
            return None
        dimensions, raw = row
        vector = np.asarray(json.loads(raw), dtype=np.float64)
        if vector.size != int(dimensions):
            return None
        return vector

    def put_embedding(self, model: str, text: str, vector: np.ndarray) -> int:
        semantic_text_id = self.text_id(text, create=True)
        assert semantic_text_id is not None
        value = np.asarray(vector, dtype=np.float64)
        self._connection.execute(
            """
            INSERT OR REPLACE INTO semantic_embeddings(
                semantic_text_id, model, dimensions, vector_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                semantic_text_id,
                model,
                int(value.size),
                json.dumps(value.tolist(), separators=(",", ":")),
            ),
        )
        self._connection.commit()
        return semantic_text_id

    def embedding_models(self, text: str) -> list[str]:
        semantic_text_id = self.text_id(text)
        if semantic_text_id is None:
            return []
        rows = self._connection.execute(
            "SELECT model FROM semantic_embeddings WHERE semantic_text_id = ? ORDER BY model",
            (semantic_text_id,),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def close(self) -> None:
        self._connection.close()
