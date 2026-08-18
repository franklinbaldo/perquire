"""Convergence detection for iterative semantic search."""

from dataclasses import dataclass
from enum import Enum
from math import erfc, sqrt
from typing import Any, Dict, List

import numpy as np

from ..exceptions import ConvergenceError


class ConvergenceReason(Enum):
    SIMILARITY_THRESHOLD = "similarity_threshold"
    PLATEAU_DETECTED = "plateau_detected"
    MAX_ITERATIONS = "max_iterations"
    DIMINISHING_RETURNS = "diminishing_returns"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    CHANGE_POINT = "change_point"
    MANUAL_STOP = "manual_stop"


@dataclass
class ConvergenceResult:
    converged: bool
    reason: ConvergenceReason
    confidence: float
    iteration_reached: int
    final_similarity: float
    similarity_improvement: float
    plateau_length: int
    statistical_metrics: Dict[str, float]
    recommendation: str


class ConvergenceDetector:
    """Small, deterministic stopping policy for iterative search.

    The detector uses only the observed similarity trajectory. It deliberately
    avoids heavyweight statistical dependencies; the optional trend check uses
    the normal distribution formula from the Python standard library.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        min_improvement: float = 0.001,
        convergence_window: int = 3,
        plateau_threshold: float = 0.0005,
        max_iterations: int = 25,
        statistical_confidence: float = 0.95,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_improvement = min_improvement
        self.convergence_window = convergence_window
        self.plateau_threshold = plateau_threshold
        self.max_iterations = max_iterations
        self.statistical_confidence = statistical_confidence

    def should_continue(
        self,
        similarity_scores: List[float],
        current_iteration: int,
        phase: str = "exploration",
    ) -> ConvergenceResult:
        try:
            if not similarity_scores:
                return self._result(
                    False,
                    ConvergenceReason.MANUAL_STOP,
                    current_iteration,
                    0.0,
                    recommendation="Need more data points to analyze convergence",
                )

            current_similarity = max(similarity_scores)

            if current_iteration >= self.max_iterations:
                return self._result(
                    True,
                    ConvergenceReason.MAX_ITERATIONS,
                    current_iteration,
                    current_similarity,
                    similarity_scores,
                    confidence=1.0,
                    recommendation=f"Reached maximum iterations ({self.max_iterations})",
                )

            if len(similarity_scores) < 2:
                return self._result(
                    False,
                    ConvergenceReason.MANUAL_STOP,
                    current_iteration,
                    current_similarity,
                    similarity_scores,
                    recommendation="Need more iterations for convergence analysis",
                )

            plateau = self._detect_plateau(similarity_scores)

            if current_similarity >= self.similarity_threshold and plateau["is_plateau"]:
                return self._result(
                    True,
                    ConvergenceReason.SIMILARITY_THRESHOLD,
                    current_iteration,
                    current_similarity,
                    similarity_scores,
                    confidence=0.95,
                    plateau_length=plateau["plateau_length"],
                    recommendation=f"High similarity ({current_similarity:.3f}) with plateau detected",
                )

            if plateau["is_plateau"] and plateau["plateau_length"] >= self.convergence_window:
                confidence = min(0.9, 0.5 + plateau["plateau_length"] / self.max_iterations)
                return self._result(
                    True,
                    ConvergenceReason.PLATEAU_DETECTED,
                    current_iteration,
                    current_similarity,
                    similarity_scores,
                    confidence=confidence,
                    plateau_length=plateau["plateau_length"],
                    recommendation=f"Plateau detected for {plateau['plateau_length']} iterations",
                )

            if len(similarity_scores) >= self.convergence_window:
                recent_improvement = self._calculate_recent_improvement(similarity_scores)
                if recent_improvement < self.min_improvement:
                    return self._result(
                        True,
                        ConvergenceReason.DIMINISHING_RETURNS,
                        current_iteration,
                        current_similarity,
                        similarity_scores,
                        confidence=0.8,
                        improvement=recent_improvement,
                        recommendation=(
                            f"Diminishing returns: {recent_improvement:.4f} "
                            f"< {self.min_improvement}"
                        ),
                    )

            if len(similarity_scores) >= 5:
                statistical = self._statistical_convergence_test(similarity_scores)
                if statistical["converged"]:
                    return ConvergenceResult(
                        converged=True,
                        reason=ConvergenceReason.STATISTICAL_SIGNIFICANCE,
                        confidence=statistical["confidence"],
                        iteration_reached=current_iteration,
                        final_similarity=current_similarity,
                        similarity_improvement=self._calculate_improvement(similarity_scores),
                        plateau_length=0,
                        statistical_metrics=statistical["metrics"],
                        recommendation=(
                            "Statistical convergence detected "
                            f"(p-value: {statistical['p_value']:.4f})"
                        ),
                    )

            return self._result(
                False,
                ConvergenceReason.MANUAL_STOP,
                current_iteration,
                current_similarity,
                similarity_scores,
                plateau_length=plateau.get("plateau_length", 0),
                recommendation="Continue investigation - no convergence criteria met",
            )
        except Exception as exc:
            raise ConvergenceError(f"Failed to analyze convergence: {exc}") from exc

    def _result(
        self,
        converged: bool,
        reason: ConvergenceReason,
        iteration: int,
        similarity: float,
        scores: List[float] | None = None,
        *,
        confidence: float = 0.0,
        plateau_length: int = 0,
        improvement: float | None = None,
        recommendation: str,
    ) -> ConvergenceResult:
        scores = scores or []
        return ConvergenceResult(
            converged=converged,
            reason=reason,
            confidence=confidence,
            iteration_reached=iteration,
            final_similarity=similarity,
            similarity_improvement=(
                self._calculate_improvement(scores) if improvement is None else improvement
            ),
            plateau_length=plateau_length,
            statistical_metrics=self._calculate_statistics(scores) if scores else {},
            recommendation=recommendation,
        )

    def _detect_plateau(self, scores: List[float]) -> Dict[str, Any]:
        if len(scores) < self.convergence_window:
            return {"is_plateau": False, "plateau_length": 0}

        recent = scores[-self.convergence_window :]
        variance = float(np.var(recent))
        is_plateau = variance <= self.plateau_threshold

        plateau_length = 0
        if is_plateau:
            # Count how many consecutive observations at the tail belong to
            # overlapping low-variance windows. One qualifying window spans
            # `convergence_window` observations; each preceding qualifying
            # overlapping window extends the plateau by one observation.
            qualifying_windows = 0
            for end in range(len(scores), self.convergence_window - 1, -1):
                window = scores[end - self.convergence_window : end]
                if float(np.var(window)) <= self.plateau_threshold:
                    qualifying_windows += 1
                else:
                    break
            plateau_length = self.convergence_window + qualifying_windows - 1

        return {
            "is_plateau": is_plateau,
            "plateau_length": plateau_length,
            "variance": variance,
            "threshold": self.plateau_threshold,
        }

    def _calculate_improvement(self, scores: List[float]) -> float:
        if len(scores) < 2:
            return 0.0
        return float(max(scores) - scores[0])

    def _calculate_recent_improvement(self, scores: List[float]) -> float:
        if len(scores) < self.convergence_window:
            return float("inf")
        recent = scores[-self.convergence_window :]
        improvements = [max(0.0, recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return float(np.mean(improvements)) if improvements else 0.0

    def _statistical_convergence_test(self, scores: List[float]) -> Dict[str, Any]:
        """Mann-Kendall-style trend check without SciPy."""
        n = len(scores)
        if n < 5:
            return {"converged": False, "confidence": 0.0, "p_value": 1.0, "metrics": {}}

        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += (scores[j] > scores[i]) - (scores[j] < scores[i])

        var_s = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / sqrt(var_s)
        elif s < 0:
            z = (s + 1) / sqrt(var_s)
        else:
            z = 0.0

        # Two-tailed normal probability: erfc(|z| / sqrt(2)).
        p_value = erfc(abs(z) / sqrt(2))
        converged = p_value > (1 - self.statistical_confidence)
        confidence = max(0.0, 1 - p_value) if converged else 0.0
        return {
            "converged": converged,
            "confidence": confidence,
            "p_value": p_value,
            "metrics": {
                "mann_kendall_s": float(s),
                "z_statistic": float(z),
                "trend_direction": (
                    "increasing" if s > 0 else "decreasing" if s < 0 else "no_trend"
                ),
            },
        }

    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {}
        array = np.asarray(scores, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "variance": float(np.var(array)),
        }
