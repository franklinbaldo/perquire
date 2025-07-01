"""
Database components for Perquire.
"""

from .base import BaseDatabaseProvider
from .duckdb_provider import DuckDBProvider
from .models import InvestigationRecord, EmbeddingRecord, QuestionRecord

__all__ = [
    "BaseDatabaseProvider",
    "DuckDBProvider",
    "InvestigationRecord",
    "EmbeddingRecord", 
    "QuestionRecord",
]