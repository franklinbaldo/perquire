"""
Perquire: Reverse Embedding Search Through Systematic Questioning

A revolutionary AI system that reverses the traditional embedding search process.
Instead of finding embeddings that match a known query, Perquire investigates 
mysterious embeddings through systematic questioning, gradually uncovering what they represent.
"""

__version__ = "0.1.0"
__author__ = "Franklin Baldo"
__email__ = "franklinbaldo@gmail.com"

# Core imports for easy access
from .core import (
    PerquireInvestigator,
    EnsembleInvestigator,
    InvestigationResult,
    QuestionResult,
    QuestioningStrategy,
    InvestigationPhase
)

from .llm import (
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider
)

from .embeddings import (
    GeminiEmbeddingProvider,
    cosine_similarity,
    normalize_embedding
)

from .database import (
    DuckDBProvider,
    DatabaseConfig
)

from .convergence import (
    ConvergenceDetector,
    ConvergenceResult
)

# Pre-configured strategies
from .core.strategy import (
    create_artistic_strategy,
    create_scientific_strategy,
    create_emotional_strategy
)

__all__ = [
    # Core classes
    "PerquireInvestigator",
    "EnsembleInvestigator",
    "InvestigationResult",
    "QuestionResult",
    "QuestioningStrategy",
    "InvestigationPhase",
    
    # LLM providers
    "GeminiProvider",
    "OpenAIProvider", 
    "AnthropicProvider",
    "OllamaProvider",
    
    # Embedding providers
    "GeminiEmbeddingProvider",
    "cosine_similarity",
    "normalize_embedding",
    
    # Database
    "DuckDBProvider",
    "DatabaseConfig",
    
    # Convergence
    "ConvergenceDetector",
    "ConvergenceResult",
    
    # Predefined strategies
    "create_artistic_strategy",
    "create_scientific_strategy",
    "create_emotional_strategy",
]


def create_investigator(
    llm_provider: str = "gemini",
    embedding_provider: str = "gemini",
    strategy: str = "default",
    database_path: str = "perquire.db",
    **kwargs
) -> PerquireInvestigator:
    """
    Create a pre-configured PerquireInvestigator with sensible defaults.
    
    Args:
        llm_provider: LLM provider to use ("gemini", "openai", "anthropic", "ollama")
        embedding_provider: Embedding provider to use ("gemini", "openai", "huggingface")
        strategy: Questioning strategy ("default", "artistic", "scientific", "emotional")
        database_path: Path to DuckDB database file
        **kwargs: Additional configuration options
        
    Returns:
        Configured PerquireInvestigator instance
        
    Example:
        >>> investigator = create_investigator(
        ...     llm_provider="gemini",
        ...     embedding_provider="gemini",
        ...     strategy="artistic"
        ... )
        >>> result = investigator.investigate(target_embedding)
    """
    from .llm.base import provider_registry as llm_registry
    from .embeddings.base import embedding_registry
    
    # Configure LLM provider
    if llm_provider == "gemini":
        llm_prov = GeminiProvider(kwargs.get("llm_config", {}))
        llm_registry.register_provider("gemini", llm_prov, set_as_default=True)
    elif llm_provider == "openai":
        llm_prov = OpenAIProvider(kwargs.get("llm_config", {}))
        llm_registry.register_provider("openai", llm_prov, set_as_default=True)
    elif llm_provider == "anthropic":
        llm_prov = AnthropicProvider(kwargs.get("llm_config", {}))
        llm_registry.register_provider("anthropic", llm_prov, set_as_default=True)
    elif llm_provider == "ollama":
        llm_prov = OllamaProvider(kwargs.get("llm_config", {}))
        llm_registry.register_provider("ollama", llm_prov, set_as_default=True)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")
    
    # Configure embedding provider
    if embedding_provider == "gemini":
        embed_prov = GeminiEmbeddingProvider(kwargs.get("embedding_config", {}))
        embedding_registry.register_provider("gemini", embed_prov, set_as_default=True)
    else:
        raise ValueError(f"Unknown embedding provider: {embedding_provider}")
    
    # Configure strategy
    if strategy == "default":
        questioning_strategy = QuestioningStrategy()
    elif strategy == "artistic":
        questioning_strategy = create_artistic_strategy()
    elif strategy == "scientific":
        questioning_strategy = create_scientific_strategy()
    elif strategy == "emotional":
        questioning_strategy = create_emotional_strategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Configure database
    from .database.base import DatabaseConfig
    db_config = DatabaseConfig(
        connection_string=database_path,
        **kwargs.get("database_config", {})
    )
    database_provider = DuckDBProvider(db_config)
    
    # Create investigator
    return PerquireInvestigator(
        llm_provider=llm_prov,
        embedding_provider=embed_prov,
        questioning_strategy=questioning_strategy,
        database_provider=database_provider,
        config=kwargs
    )


def create_ensemble_investigator(
    strategies: list = None,
    database_path: str = "perquire.db",
    **kwargs
) -> EnsembleInvestigator:
    """
    Create a pre-configured EnsembleInvestigator with multiple strategies.
    
    Args:
        strategies: List of strategy names to use
        database_path: Path to DuckDB database file
        **kwargs: Additional configuration options
        
    Returns:
        Configured EnsembleInvestigator instance
        
    Example:
        >>> ensemble = create_ensemble_investigator(
        ...     strategies=["default", "artistic", "scientific"]
        ... )
        >>> result = ensemble.investigate(target_embedding)
    """
    if strategies is None:
        strategies = ["default", "artistic", "scientific"]
    
    # Create individual investigators
    investigators = []
    for strategy_name in strategies:
        investigator = create_investigator(
            strategy=strategy_name,
            database_path=database_path,
            **kwargs
        )
        investigators.append(investigator)
    
    return EnsembleInvestigator(
        investigators=investigators,
        voting_method=kwargs.get("voting_method", "weighted_similarity"),
        min_agreement=kwargs.get("min_agreement", 0.6)
    )


# Quick start function
def investigate_embedding(
    embedding,
    llm_provider: str = "gemini",
    strategy: str = "default",
    verbose: bool = True,
    **kwargs
) -> InvestigationResult:
    """
    Quick investigation of an embedding with minimal setup.
    
    Args:
        embedding: Numpy array or list representing the embedding
        llm_provider: LLM provider to use
        strategy: Questioning strategy to use
        verbose: Whether to print progress
        **kwargs: Additional options
        
    Returns:
        InvestigationResult
        
    Example:
        >>> import numpy as np
        >>> embedding = np.random.rand(768)  # Example embedding
        >>> result = investigate_embedding(embedding, verbose=True)
        >>> print(f"Description: {result.description}")
    """
    import numpy as np
    
    # Convert to numpy array if needed
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    # Create investigator
    investigator = create_investigator(
        llm_provider=llm_provider,
        strategy=strategy,
        database_path=":memory:",  # Use in-memory DB for quick investigations
        **kwargs
    )
    
    # Run investigation
    return investigator.investigate(
        target_embedding=embedding,
        verbose=verbose,
        save_to_database=False  # Don't save for quick investigations
    )