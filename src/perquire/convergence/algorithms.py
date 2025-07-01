"""
Core convergence detection algorithms.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum

from ..exceptions import ConvergenceError


class ConvergenceReason(Enum):
    """Reasons for convergence detection."""
    SIMILARITY_THRESHOLD = "similarity_threshold"
    PLATEAU_DETECTED = "plateau_detected"
    MAX_ITERATIONS = "max_iterations"
    DIMINISHING_RETURNS = "diminishing_returns"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    CHANGE_POINT = "change_point"
    MANUAL_STOP = "manual_stop"


@dataclass
class ConvergenceResult:
    """Result of convergence detection analysis."""
    
    converged: bool
    reason: ConvergenceReason
    confidence: float  # 0-1 confidence in convergence decision
    iteration_reached: int
    final_similarity: float
    similarity_improvement: float
    plateau_length: int
    statistical_metrics: Dict[str, float]
    recommendation: str  # Human-readable recommendation


class ConvergenceDetector:
    """
    Advanced convergence detection for investigation processes.
    
    This class implements multiple statistical methods to detect when
    an investigation has converged and should stop.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.90,
        min_improvement: float = 0.001,
        convergence_window: int = 3,
        plateau_threshold: float = 0.0005,
        max_iterations: int = 25,
        statistical_confidence: float = 0.95
    ):
        """
        Initialize convergence detector.
        
        Args:
            similarity_threshold: Minimum similarity to consider converged
            min_improvement: Minimum improvement per iteration to continue
            convergence_window: Number of iterations to analyze for convergence
            plateau_threshold: Maximum variance to consider a plateau
            max_iterations: Maximum iterations before forced convergence
            statistical_confidence: Confidence level for statistical tests
        """
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
        phase: str = "exploration"
    ) -> ConvergenceResult:
        """
        Determine if investigation should continue based on similarity scores.
        
        Args:
            similarity_scores: List of similarity scores from investigation
            current_iteration: Current iteration number
            phase: Current investigation phase
            
        Returns:
            ConvergenceResult with decision and analysis
        """
        try:
            if not similarity_scores:
                return ConvergenceResult(
                    converged=False,
                    reason=ConvergenceReason.MANUAL_STOP,
                    confidence=0.0,
                    iteration_reached=current_iteration,
                    final_similarity=0.0,
                    similarity_improvement=0.0,
                    plateau_length=0,
                    statistical_metrics={},
                    recommendation="Need more data points to analyze convergence"
                )
            
            current_similarity = max(similarity_scores)
            
            # Check max iterations first
            if current_iteration >= self.max_iterations:
                return ConvergenceResult(
                    converged=True,
                    reason=ConvergenceReason.MAX_ITERATIONS,
                    confidence=1.0,
                    iteration_reached=current_iteration,
                    final_similarity=current_similarity,
                    similarity_improvement=self._calculate_improvement(similarity_scores),
                    plateau_length=0,
                    statistical_metrics=self._calculate_statistics(similarity_scores),
                    recommendation=f"Reached maximum iterations ({self.max_iterations})"
                )
            
            # Need minimum data points for analysis
            if len(similarity_scores) < 2:
                return ConvergenceResult(
                    converged=False,
                    reason=ConvergenceReason.MANUAL_STOP,
                    confidence=0.0,
                    iteration_reached=current_iteration,
                    final_similarity=current_similarity,
                    similarity_improvement=0.0,
                    plateau_length=0,
                    statistical_metrics={},
                    recommendation="Need more iterations for convergence analysis"
                )
            
            # Check similarity threshold
            if current_similarity >= self.similarity_threshold:
                # Additional checks for high similarity
                plateau_result = self._detect_plateau(similarity_scores)
                
                if plateau_result["is_plateau"]:
                    return ConvergenceResult(
                        converged=True,
                        reason=ConvergenceReason.SIMILARITY_THRESHOLD,
                        confidence=0.95,
                        iteration_reached=current_iteration,
                        final_similarity=current_similarity,
                        similarity_improvement=self._calculate_improvement(similarity_scores),
                        plateau_length=plateau_result["plateau_length"],
                        statistical_metrics=self._calculate_statistics(similarity_scores),
                        recommendation=f"High similarity ({current_similarity:.3f}) with plateau detected"
                    )
            
            # Check for plateau detection
            plateau_result = self._detect_plateau(similarity_scores)
            if plateau_result["is_plateau"] and plateau_result["plateau_length"] >= self.convergence_window:
                confidence = min(0.9, 0.5 + (plateau_result["plateau_length"] / self.max_iterations))
                
                return ConvergenceResult(
                    converged=True,
                    reason=ConvergenceReason.PLATEAU_DETECTED,
                    confidence=confidence,
                    iteration_reached=current_iteration,
                    final_similarity=current_similarity,
                    similarity_improvement=self._calculate_improvement(similarity_scores),
                    plateau_length=plateau_result["plateau_length"],
                    statistical_metrics=self._calculate_statistics(similarity_scores),
                    recommendation=f"Plateau detected for {plateau_result['plateau_length']} iterations"
                )
            
            # Check for diminishing returns
            if len(similarity_scores) >= self.convergence_window:
                recent_improvement = self._calculate_recent_improvement(similarity_scores)
                
                if recent_improvement < self.min_improvement:
                    return ConvergenceResult(
                        converged=True,
                        reason=ConvergenceReason.DIMINISHING_RETURNS,
                        confidence=0.8,
                        iteration_reached=current_iteration,
                        final_similarity=current_similarity,
                        similarity_improvement=recent_improvement,
                        plateau_length=0,
                        statistical_metrics=self._calculate_statistics(similarity_scores),
                        recommendation=f"Diminishing returns: {recent_improvement:.4f} < {self.min_improvement}"
                    )
            
            # Check statistical significance
            if len(similarity_scores) >= 5:  # Need minimum samples
                stat_result = self._statistical_convergence_test(similarity_scores)
                
                if stat_result["converged"]:
                    return ConvergenceResult(
                        converged=True,
                        reason=ConvergenceReason.STATISTICAL_SIGNIFICANCE,
                        confidence=stat_result["confidence"],
                        iteration_reached=current_iteration,
                        final_similarity=current_similarity,
                        similarity_improvement=self._calculate_improvement(similarity_scores),
                        plateau_length=0,
                        statistical_metrics=stat_result["metrics"],
                        recommendation=f"Statistical convergence detected (p-value: {stat_result['p_value']:.4f})"
                    )
            
            # Continue investigation
            return ConvergenceResult(
                converged=False,
                reason=ConvergenceReason.MANUAL_STOP,
                confidence=0.0,
                iteration_reached=current_iteration,
                final_similarity=current_similarity,
                similarity_improvement=self._calculate_improvement(similarity_scores),
                plateau_length=plateau_result.get("plateau_length", 0),
                statistical_metrics=self._calculate_statistics(similarity_scores),
                recommendation="Continue investigation - no convergence criteria met"
            )
            
        except Exception as e:
            raise ConvergenceError(f"Failed to analyze convergence: {str(e)}")
    
    def _detect_plateau(self, scores: List[float]) -> Dict[str, Any]:
        """Detect if similarity scores have plateaued."""
        if len(scores) < self.convergence_window:
            return {"is_plateau": False, "plateau_length": 0}
        
        # Check recent scores for low variance
        recent_scores = scores[-self.convergence_window:]
        variance = np.var(recent_scores)
        
        is_plateau = variance <= self.plateau_threshold
        
        # Count consecutive plateau iterations
        plateau_length = 0
        if is_plateau:
            for i in range(len(scores) - 1, -1, -1):
                window_start = max(0, i - self.convergence_window + 1)
                window_scores = scores[window_start:i + 1]
                
                if len(window_scores) >= self.convergence_window:
                    window_var = np.var(window_scores)
                    if window_var <= self.plateau_threshold:
                        plateau_length += 1
                    else:
                        break
                else:
                    break
        
        return {
            "is_plateau": is_plateau,
            "plateau_length": plateau_length,
            "variance": variance,
            "threshold": self.plateau_threshold
        }
    
    def _calculate_improvement(self, scores: List[float]) -> float:
        """Calculate overall similarity improvement."""
        if len(scores) < 2:
            return 0.0
        
        return max(scores) - scores[0]
    
    def _calculate_recent_improvement(self, scores: List[float]) -> float:
        """Calculate recent improvement rate."""
        if len(scores) < self.convergence_window:
            return float('inf')  # Not enough data, assume improvement
        
        recent_scores = scores[-self.convergence_window:]
        improvements = []
        
        for i in range(1, len(recent_scores)):
            improvement = recent_scores[i] - recent_scores[i-1]
            improvements.append(max(0, improvement))  # Only positive improvements
        
        return np.mean(improvements) if improvements else 0.0
    
    def _statistical_convergence_test(self, scores: List[float]) -> Dict[str, Any]:
        """Perform statistical test for convergence."""
        try:
            # Use Mann-Kendall trend test for trend detection
            n = len(scores)
            if n < 5:
                return {"converged": False, "confidence": 0.0, "p_value": 1.0, "metrics": {}}
            
            # Calculate trend statistic
            s = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if scores[j] > scores[i]:
                        s += 1
                    elif scores[j] < scores[i]:
                        s -= 1
            
            # Calculate variance
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            # Calculate z-statistic
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            # Calculate p-value (two-tailed test)
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Check if trend is statistically insignificant (converged)
            converged = p_value > (1 - self.statistical_confidence)
            confidence = 1 - p_value if converged else 0.0
            
            return {
                "converged": converged,
                "confidence": confidence,
                "p_value": p_value,
                "metrics": {
                    "mann_kendall_s": s,
                    "z_statistic": z,
                    "trend_direction": "increasing" if s > 0 else "decreasing" if s < 0 else "no_trend"
                }
            }
            
        except ImportError:
            # Fallback if scipy not available
            # Simple variance-based test
            if len(scores) >= self.convergence_window:
                recent_var = np.var(scores[-self.convergence_window:])
                converged = recent_var < self.plateau_threshold
                confidence = max(0.5, 1 - (recent_var / self.plateau_threshold)) if converged else 0.0
                
                return {
                    "converged": converged,
                    "confidence": confidence,
                    "p_value": recent_var,
                    "metrics": {"variance": recent_var, "method": "fallback_variance"}
                }
            
            return {"converged": False, "confidence": 0.0, "p_value": 1.0, "metrics": {}}
        
        except Exception:
            return {"converged": False, "confidence": 0.0, "p_value": 1.0, "metrics": {}}
    
    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for the score series."""
        if not scores:
            return {}
        
        stats = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "range": np.max(scores) - np.min(scores),
            "total_improvement": self._calculate_improvement(scores)
        }
        
        if len(scores) >= 2:
            # Calculate trend
            x = np.arange(len(scores))
            slope, intercept = np.polyfit(x, scores, 1)
            stats["trend_slope"] = slope
            stats["trend_intercept"] = intercept
            
            # Calculate recent statistics
            if len(scores) >= self.convergence_window:
                recent_scores = scores[-self.convergence_window:]
                stats["recent_mean"] = np.mean(recent_scores)
                stats["recent_std"] = np.std(recent_scores)
                stats["recent_improvement"] = self._calculate_recent_improvement(scores)
        
        return stats
    
    def analyze_convergence_quality(self, result: ConvergenceResult) -> Dict[str, Any]:
        """
        Analyze the quality of convergence detection.
        
        Args:
            result: ConvergenceResult from should_continue()
            
        Returns:
            Quality analysis dictionary
        """
        quality = {
            "overall_score": 0.0,
            "confidence_assessment": "low",
            "reliability": "poor",
            "recommendations": []
        }
        
        # Base score from confidence
        quality["overall_score"] = result.confidence
        
        # Assess confidence level
        if result.confidence >= 0.9:
            quality["confidence_assessment"] = "very_high"
        elif result.confidence >= 0.8:
            quality["confidence_assessment"] = "high"
        elif result.confidence >= 0.6:
            quality["confidence_assessment"] = "medium"
        elif result.confidence >= 0.4:
            quality["confidence_assessment"] = "low"
        else:
            quality["confidence_assessment"] = "very_low"
        
        # Assess reliability based on reason and metrics
        if result.reason == ConvergenceReason.SIMILARITY_THRESHOLD:
            quality["reliability"] = "excellent"
            quality["recommendations"].append("High similarity achieved with strong convergence")
        elif result.reason == ConvergenceReason.PLATEAU_DETECTED:
            if result.plateau_length >= self.convergence_window * 2:
                quality["reliability"] = "good"
                quality["recommendations"].append("Strong plateau detected")
            else:
                quality["reliability"] = "fair"
                quality["recommendations"].append("Short plateau - consider more iterations")
        elif result.reason == ConvergenceReason.STATISTICAL_SIGNIFICANCE:
            quality["reliability"] = "good"
            quality["recommendations"].append("Statistical convergence confirmed")
        elif result.reason == ConvergenceReason.DIMINISHING_RETURNS:
            quality["reliability"] = "fair"
            quality["recommendations"].append("Diminishing returns - consider alternative strategies")
        elif result.reason == ConvergenceReason.MAX_ITERATIONS:
            quality["reliability"] = "poor"
            quality["recommendations"].append("Forced stop - increase max iterations or adjust strategy")
        
        # Additional recommendations based on final similarity
        if result.final_similarity < 0.7:
            quality["recommendations"].append("Low similarity score - consider different questioning strategy")
        elif result.final_similarity < 0.8:
            quality["recommendations"].append("Moderate similarity - results may be acceptable")
        else:
            quality["recommendations"].append("Good similarity achieved")
        
        return quality