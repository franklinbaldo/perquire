"""
Custom exceptions for the Perquire library.
"""


class PerquireException(Exception):
    """Base exception class for all Perquire-related errors."""
    pass


class InvestigationError(PerquireException):
    """Raised when an investigation fails or encounters an error."""
    pass


class ConvergenceError(PerquireException):
    """Raised when convergence detection fails or takes too long."""
    pass


class EmbeddingError(PerquireException):
    """Raised when embedding operations fail."""
    pass


class LLMProviderError(PerquireException):
    """Raised when LLM provider operations fail."""
    pass


class QuestionGenerationError(PerquireException):
    """Raised when question generation fails."""
    pass


class ConfigurationError(PerquireException):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(PerquireException):
    """Raised when input validation fails."""
    pass


class ModelNotFoundError(PerquireException):
    """Raised when a requested model is not found or unavailable."""
    pass


class RateLimitError(PerquireException):
    """Raised when API rate limits are exceeded."""
    pass


class TimeoutError(PerquireException):
    """Raised when operations timeout."""
    pass