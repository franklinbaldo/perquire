"""Perquire: experimental similarity-guided semantic inversion."""

from __future__ import annotations

from typing import Optional, Union

from .core.investigator import PerquireInvestigator
from .core.result import InvestigationResult
from .core.strategy import InvestigationPhase, QuestioningStrategy
from .embeddings.base import BaseEmbeddingProvider
from .exceptions import InvestigationError
from .llm.base import BaseLLMProvider
from .providers import list_available_providers

__version__ = "0.2.0"
__author__ = "Franklin Baldo"


def create_investigator(
    llm_provider: Optional[Union[str, BaseLLMProvider]] = None,
    embedding_provider: Optional[Union[str, BaseEmbeddingProvider]] = None,
    strategy: Optional[QuestioningStrategy] = None,
    database_path: Optional[str] = None,
    config: Optional[dict] = None,
) -> PerquireInvestigator:
    """Create an investigator with persistence only when explicitly requested."""
    database_provider = None
    if database_path is not None:
        from .database.base import DatabaseConfig
        from .database.duckdb_provider import DuckDBProvider

        database_provider = DuckDBProvider(DatabaseConfig(connection_string=database_path))
        database_provider.connect()

    return PerquireInvestigator(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        questioning_strategy=strategy,
        database_provider=database_provider,
        config=config,
    )


__all__ = [
    "PerquireInvestigator",
    "create_investigator",
    "QuestioningStrategy",
    "InvestigationResult",
    "InvestigationPhase",
    "InvestigationError",
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "list_available_providers",
]
