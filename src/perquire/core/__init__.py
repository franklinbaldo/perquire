"""Minimal executable core for Perquire."""

from .investigator import PerquireInvestigator
from .result import InvestigationResult, QuestionResult
from .strategy import InvestigationPhase, QuestioningStrategy

__all__ = [
    "PerquireInvestigator",
    "InvestigationResult",
    "QuestionResult",
    "QuestioningStrategy",
    "InvestigationPhase",
]
