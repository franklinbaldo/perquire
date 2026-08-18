"""Persistence interfaces for Perquire.

DuckDB is an optional integration and is imported only when explicitly used.
"""

from typing import Any

from .base import BaseDatabaseProvider, DatabaseConfig, DatabaseError


def __getattr__(name: str) -> Any:
    if name == "DuckDBProvider":
        try:
            from .duckdb_provider import DuckDBProvider
        except ImportError as exc:
            raise ImportError(
                "DuckDB persistence requires the optional 'persistence' extra"
            ) from exc
        return DuckDBProvider
    raise AttributeError(name)


__all__ = ["BaseDatabaseProvider", "DatabaseConfig", "DatabaseError"]
