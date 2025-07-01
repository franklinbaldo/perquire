"""
Convergence detection algorithms for Perquire.
"""

from .algorithms import ConvergenceDetector, ConvergenceResult
from .statistical_methods import (
    plateau_detection,
    moving_average_convergence,
    statistical_significance_test,
    change_point_detection
)

__all__ = [
    "ConvergenceDetector",
    "ConvergenceResult",
    "plateau_detection",
    "moving_average_convergence", 
    "statistical_significance_test",
    "change_point_detection",
]