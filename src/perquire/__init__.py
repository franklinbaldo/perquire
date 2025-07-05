"""
Perquire: Reverse Embedding Search Through Systematic Questioning

A revolutionary AI system that reverses the traditional embedding search process.
Instead of finding embeddings that match a known query, Perquire investigates 
mysterious embeddings through systematic questioning, gradually uncovering what they represent.
"""

__version__ = "0.1.0"
__author__ = "Franklin Baldo"
__email__ = "franklinbaldo@gmail.com"

from typing import List, Optional, Union, Dict, Any
import logging

# Core components
from .core import (
    PerquireInvestigator,
    EnsembleInvestigator,
    QuestioningStrategy,
    InvestigationResult,
    InvestigationPhase,
)
from .exceptions import InvestigationError
from .database.base import BaseDatabaseProvider, DatabaseConfig
from .database.duckdb_provider import DuckDBProvider # Default DB
from .llm.base import BaseLLMProvider, provider_registry as llm_registry
from .embeddings.base import BaseEmbeddingProvider, embedding_registry

logger = logging.getLogger(__name__)

# Factory functions for web/CLI use
def create_investigator(
    llm_provider: Optional[Union[str, BaseLLMProvider]] = None,
    embedding_provider: Optional[Union[str, BaseEmbeddingProvider]] = "gemini", # Default from app.py
    strategy: Optional[Union[str, QuestioningStrategy]] = "default",
    database_path: Optional[str] = "perquire.db",
    config: Optional[Dict[str, Any]] = None
) -> PerquireInvestigator:
    """
    Factory function to create a PerquireInvestigator instance.
    Handles string inputs for strategy and database provider creation.
    """
    q_strategy: Optional[QuestioningStrategy] = None
    if isinstance(strategy, str):
        # Simple strategy mapping; can be expanded
        q_strategy = QuestioningStrategy(name=strategy)
        if strategy == "artistic":
            q_strategy.convergence_threshold = 0.92
        elif strategy == "scientific":
            q_strategy.convergence_threshold = 0.88
        # Add more predefined strategies as needed
    elif isinstance(strategy, QuestioningStrategy):
        q_strategy = strategy

    db_provider: Optional[BaseDatabaseProvider] = None
    if database_path:
        try:
            db_config = DatabaseConfig(connection_string=database_path)
            db_provider = DuckDBProvider(db_config)
        except Exception as e:
            logger.warning(f"Could not initialize DuckDBProvider for path '{database_path}': {e}. Proceeding without DB.")
            db_provider = None

    return PerquireInvestigator(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        questioning_strategy=q_strategy,
        database_provider=db_provider,
        config=config
    )

def create_ensemble_investigator(
    llm_providers: Optional[List[Union[str, BaseLLMProvider]]] = None,
    embedding_providers: Optional[List[Union[str, BaseEmbeddingProvider]]] = None,
    strategies: Optional[List[Union[str, QuestioningStrategy]]] = None,
    database_path: Optional[str] = "perquire.db",
    config: Optional[Dict[str, Any]] = None
) -> EnsembleInvestigator:
    """
    Factory function to create an EnsembleInvestigator instance.
    Handles string inputs for strategies and database provider creation.
    """
    processed_strategies: Optional[List[QuestioningStrategy]] = None
    if strategies:
        processed_strategies = []
        for strat_input in strategies:
            if isinstance(strat_input, str):
                q_strat = QuestioningStrategy(name=strat_input)
                if strat_input == "artistic": q_strat.convergence_threshold = 0.92
                elif strat_input == "scientific": q_strat.convergence_threshold = 0.88
                processed_strategies.append(q_strat)
            elif isinstance(strat_input, QuestioningStrategy):
                processed_strategies.append(strat_input)

    db_provider: Optional[BaseDatabaseProvider] = None
    if database_path:
        try:
            db_config = DatabaseConfig(connection_string=database_path)
            db_provider = DuckDBProvider(db_config)
        except Exception as e:
            logger.warning(f"Could not initialize DuckDBProvider for path '{database_path}' in ensemble: {e}. Proceeding without DB.")
            db_provider = None

    # EnsembleInvestigator's _create_investigators handles default providers if None
    return EnsembleInvestigator(
        llm_providers=llm_providers,
        embedding_providers=embedding_providers,
        strategies=processed_strategies,
        database_provider=db_provider,
        # Add other EnsembleInvestigator params from config if needed
    )


# Previously existing exports - keep them if they are still relevant
from .providers import list_available_providers
from .akinator import (
    WikipediaDatasetBuilder,
    BatchEmbeddingGenerator,
    MassiveQuestionGenerator,
    OptimizedBootstrapInvestigator,
    DimensionalAnalyzer,
)


__all__ = [
    "PerquireInvestigator",
    "EnsembleInvestigator",
    "create_investigator",
    "create_ensemble_investigator",
    "QuestioningStrategy",
    "InvestigationResult",
    "InvestigationPhase",
    "InvestigationError",
    "DatabaseConfig", # Exporting for potential direct use
    "DuckDBProvider",   # Exporting for potential direct use
    # Keep existing exports
    "list_available_providers",
    "WikipediaDatasetBuilder",
    "BatchEmbeddingGenerator",
    "MassiveQuestionGenerator",
    "OptimizedBootstrapInvestigator",
    "DimensionalAnalyzer",
]
