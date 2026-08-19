"""Persistent replay cache for exact logical LLM requests.

Unlike embeddings, LLM generations may be stochastic. A cache hit is therefore
explicitly a replay of a previously observed sample, not a fresh model sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

_CACHE_SCHEMA_VERSION = "llm-v1"
_DEFAULT_CACHE_DIR = ".cache/perquire"
_VALID_MODES = {"replay", "fresh", "off"}


class LLMResponseCache:
    """Store generated text by exact model/request identity without prompt plaintext."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL,
                model TEXT NOT NULL,
                logical_request TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_responses_key ON llm_responses(cache_key, id)"
        )
        self._connection.commit()

    @staticmethod
    def key(
        *,
        model: str,
        prompt: str,
        logical_request: str,
        parameters: dict[str, Any],
    ) -> str:
        payload = json.dumps(
            {
                "schema": _CACHE_SCHEMA_VERSION,
                "model": model,
                "prompt": prompt,
                "logical_request": logical_request,
                "parameters": parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_first(
        self,
        *,
        model: str,
        prompt: str,
        logical_request: str,
        parameters: dict[str, Any],
    ) -> str | None:
        cache_key = self.key(
            model=model,
            prompt=prompt,
            logical_request=logical_request,
            parameters=parameters,
        )
        row = self._connection.execute(
            """
            SELECT content FROM llm_responses
            WHERE cache_key = ? AND model = ? AND logical_request = ?
            ORDER BY id ASC LIMIT 1
            """,
            (cache_key, model, logical_request),
        ).fetchone()
        return None if row is None else str(row[0])

    def put(
        self,
        *,
        model: str,
        prompt: str,
        logical_request: str,
        parameters: dict[str, Any],
        content: str,
    ) -> None:
        cache_key = self.key(
            model=model,
            prompt=prompt,
            logical_request=logical_request,
            parameters=parameters,
        )
        self._connection.execute(
            """
            INSERT INTO llm_responses(cache_key, model, logical_request, content)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, model, logical_request, content),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


def normalize_llm_cache_mode(value: str | None) -> str:
    mode = (value or os.getenv("PERQUIRE_LLM_CACHE_MODE", "replay")).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid LLM cache mode {mode!r}; expected replay, fresh, or off")
    return mode


def default_llm_cache_path() -> Path:
    root = Path(os.getenv("PERQUIRE_CACHE_DIR", _DEFAULT_CACHE_DIR))
    return root / "llm_responses.sqlite3"
