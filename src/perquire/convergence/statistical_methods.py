"""
Statistical methods for convergence detection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import warnings

from ..exceptions import ConvergenceError


def plateau_detection(
    scores: List[float],
    window_size: int = 3,
    variance_threshold: float = 0.0005
) -> Dict[str, Any]:
    """
    Detect if scores have plateaued using variance analysis.
    
    Args:
        scores: List of similarity scores
        window_size: Size of moving window for analysis
        variance_threshold: Maximum variance to consider a plateau
        
    Returns:
        Dictionary with plateau detection results
    """
    if len(scores) < window_size:
        return {
            "is_plateau": False,
            "plateau_length": 0,
            "variance": float('inf'),
            "confidence": 0.0
        }
    
    # Calculate variance in sliding windows
    variances = []
    for i in range(len(scores) - window_size + 1):
        window = scores[i:i + window_size]
        variance = np.var(window)
        variances.append(variance)
    
    # Check recent variance
    recent_variance = variances[-1] if variances else float('inf')
    is_plateau = recent_variance <= variance_threshold
    
    # Count consecutive plateau windows
    plateau_length = 0
    for variance in reversed(variances):
        if variance <= variance_threshold:
            plateau_length += 1
        else:
            break
    
    # Calculate confidence based on plateau consistency
    confidence = 0.0
    if is_plateau:
        confidence = min(0.95, 0.5 + (plateau_length / len(variances)) * 0.45)
    
    return {
        "is_plateau": is_plateau,
        "plateau_length": plateau_length,
        "variance": recent_variance,
        "confidence": confidence,
        "variance_threshold": variance_threshold,
        "all_variances": variances
    }


def moving_average_convergence(
    scores: List[float],
    short_window: int = 3,
    long_window: int = 6,
    convergence_threshold: float = 0.001
) -> Dict[str, Any]:
    """
    Detect convergence using moving average crossover analysis.
    
    Args:
        scores: List of similarity scores
        short_window: Size of short-term moving average
        long_window: Size of long-term moving average
        convergence_threshold: Threshold for moving average convergence
        
    Returns:
        Dictionary with moving average convergence results
    """
    if len(scores) < long_window:
        return {
            "converged": False,
            "confidence": 0.0,
            "short_ma": None,
            "long_ma": None,
            "difference": float('inf')
        }
    
    # Calculate moving averages
    short_ma = []
    long_ma = []
    
    for i in range(len(scores)):
        if i >= short_window - 1:
            short_avg = np.mean(scores[i - short_window + 1:i + 1])
            short_ma.append(short_avg)
        
        if i >= long_window - 1:
            long_avg = np.mean(scores[i - long_window + 1:i + 1])
            long_ma.append(long_avg)
    
    if not short_ma or not long_ma:
        return {
            "converged": False,
            "confidence": 0.0,
            "short_ma": short_ma,
            "long_ma": long_ma,
            "difference": float('inf')
        }
    
    # Get recent values
    recent_short = short_ma[-1]
    recent_long = long_ma[-1]
    difference = abs(recent_short - recent_long)
    
    # Check convergence
    converged = difference <= convergence_threshold
    
    # Calculate confidence
    confidence = 0.0
    if converged:
        # Higher confidence for smaller differences
        confidence = max(0.5, 1.0 - (difference / convergence_threshold) * 0.5)
    
    return {
        "converged": converged,
        "confidence": confidence,
        "short_ma": short_ma,
        "long_ma": long_ma,
        "difference": difference,
        "convergence_threshold": convergence_threshold
    }


def statistical_significance_test(
    scores: List[float],
    significance_level: float = 0.05,
    min_samples: int = 5
) -> Dict[str, Any]:
    """
    Test for statistical significance of trend in scores.
    
    Args:
        scores: List of similarity scores
        significance_level: P-value threshold for significance
        min_samples: Minimum number of samples required
        
    Returns:
        Dictionary with statistical test results
    """
    if len(scores) < min_samples:
        return {
            "significant_trend": False,
            "p_value": 1.0,
            "test_statistic": 0.0,
            "confidence": 0.0,
            "trend_direction": "insufficient_data"
        }
    
    try:
        # Try to use scipy for robust testing
        try:
            from scipy import stats
            
            # Mann-Kendall trend test
            n = len(scores)
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
            
            # Calculate p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            significant_trend = p_value < significance_level
            trend_direction = "increasing" if s > 0 else "decreasing" if s < 0 else "no_trend"
            
        except ImportError:
            # Fallback to simple linear regression
            x = np.arange(len(scores))
            slope, intercept = np.polyfit(x, scores, 1)
            
            # Simple t-test for slope significance
            residuals = scores - (slope * x + intercept)
            mse = np.mean(residuals ** 2)
            se_slope = np.sqrt(mse / np.sum((x - np.mean(x)) ** 2))
            
            t_stat = slope / se_slope if se_slope > 0 else 0
            
            # Approximate p-value using normal distribution
            p_value = 2 * (1 - np.exp(-0.717 * abs(t_stat) - 0.416 * abs(t_stat) ** 2))
            p_value = min(1.0, max(0.0, p_value))
            
            significant_trend = p_value < significance_level
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "no_trend"
            z = t_stat
        
        # Calculate confidence in convergence
        # High p-value (no significant trend) suggests convergence
        confidence = min(0.95, p_value) if p_value > significance_level else 0.0
        
        return {
            "significant_trend": significant_trend,
            "p_value": p_value,
            "test_statistic": z,
            "confidence": confidence,
            "trend_direction": trend_direction,
            "converged": not significant_trend  # No trend = converged
        }
        
    except Exception as e:
        return {
            "significant_trend": False,
            "p_value": 1.0,
            "test_statistic": 0.0,
            "confidence": 0.0,
            "trend_direction": "error",
            "error": str(e)
        }


def change_point_detection(
    scores: List[float],
    min_segment_length: int = 3,
    penalty: float = 1.0
) -> Dict[str, Any]:
    """
    Detect change points in similarity score series.
    
    Args:
        scores: List of similarity scores
        min_segment_length: Minimum length of segments
        penalty: Penalty for additional change points
        
    Returns:
        Dictionary with change point detection results
    """
    if len(scores) < min_segment_length * 2:
        return {
            "change_points": [],
            "segments": [],
            "converged": False,
            "confidence": 0.0
        }
    
    try:
        # Simple change point detection using variance
        change_points = []
        scores_array = np.array(scores)
        
        # Look for points where variance changes significantly
        for i in range(min_segment_length, len(scores) - min_segment_length):
            left_segment = scores_array[:i]
            right_segment = scores_array[i:]
            
            if len(left_segment) >= min_segment_length and len(right_segment) >= min_segment_length:
                left_var = np.var(left_segment)
                right_var = np.var(right_segment)
                total_var = np.var(scores_array)
                
                # Calculate variance reduction
                weighted_var = (len(left_segment) * left_var + len(right_segment) * right_var) / len(scores)
                variance_reduction = total_var - weighted_var
                
                # Use penalty to avoid too many change points
                if variance_reduction > penalty * 0.001:  # Threshold based on penalty
                    change_points.append(i)
        
        # Create segments
        segments = []
        if change_points:
            # Add segments based on change points
            start = 0
            for cp in change_points:
                if cp > start:
                    segments.append((start, cp))
                start = cp
            if start < len(scores):
                segments.append((start, len(scores)))
        else:
            segments = [(0, len(scores))]
        
        # Determine convergence based on last segment
        converged = False
        confidence = 0.0
        
        if segments:
            last_segment_start, last_segment_end = segments[-1]
            last_segment_scores = scores[last_segment_start:last_segment_end]
            
            if len(last_segment_scores) >= min_segment_length:
                # Check if last segment has low variance (converged)
                last_var = np.var(last_segment_scores)
                if last_var < 0.001:  # Low variance threshold
                    converged = True
                    confidence = min(0.9, 0.5 + (len(last_segment_scores) / len(scores)) * 0.4)
        
        return {
            "change_points": change_points,
            "segments": segments,
            "converged": converged,
            "confidence": confidence,
            "num_change_points": len(change_points)
        }
        
    except Exception as e:
        return {
            "change_points": [],
            "segments": [(0, len(scores))],
            "converged": False,
            "confidence": 0.0,
            "error": str(e)
        }


def adaptive_threshold_convergence(
    scores: List[float],
    initial_threshold: float = 0.90,
    adaptation_rate: float = 0.1,
    min_improvement_ratio: float = 0.01
) -> Dict[str, Any]:
    """
    Adaptive threshold convergence detection.
    
    Adjusts convergence threshold based on investigation progress.
    
    Args:
        scores: List of similarity scores
        initial_threshold: Starting convergence threshold
        adaptation_rate: Rate of threshold adaptation
        min_improvement_ratio: Minimum improvement ratio to continue
        
    Returns:
        Dictionary with adaptive convergence results
    """
    if len(scores) < 2:
        return {
            "converged": False,
            "adaptive_threshold": initial_threshold,
            "confidence": 0.0,
            "final_similarity": scores[-1] if scores else 0.0
        }
    
    # Calculate current improvement rate
    total_improvement = max(scores) - scores[0] if len(scores) > 1 else 0.0
    iterations = len(scores)
    improvement_rate = total_improvement / iterations if iterations > 0 else 0.0
    
    # Adapt threshold based on progress
    if improvement_rate > 0:
        # Lower threshold if improvement is slow
        threshold_adjustment = -adaptation_rate * (1.0 / (improvement_rate + 0.01))
        adaptive_threshold = max(0.5, initial_threshold + threshold_adjustment)
    else:
        adaptive_threshold = initial_threshold
    
    current_similarity = max(scores)
    
    # Check convergence against adaptive threshold
    threshold_met = current_similarity >= adaptive_threshold
    
    # Check improvement ratio
    recent_improvement = 0.0
    if len(scores) >= 3:
        recent_scores = scores[-3:]
        recent_improvement = (max(recent_scores) - min(recent_scores))
    
    improvement_sufficient = recent_improvement >= min_improvement_ratio
    
    converged = threshold_met and not improvement_sufficient
    
    # Calculate confidence
    confidence = 0.0
    if converged:
        threshold_confidence = min(1.0, current_similarity / adaptive_threshold)
        improvement_confidence = max(0.0, 1.0 - (recent_improvement / min_improvement_ratio))
        confidence = (threshold_confidence + improvement_confidence) / 2
    
    return {
        "converged": converged,
        "adaptive_threshold": adaptive_threshold,
        "confidence": confidence,
        "final_similarity": current_similarity,
        "improvement_rate": improvement_rate,
        "recent_improvement": recent_improvement,
        "threshold_met": threshold_met,
        "improvement_sufficient": improvement_sufficient
    }


def ensemble_convergence_detection(
    scores: List[float],
    methods: Optional[List[str]] = None,
    weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Ensemble convergence detection using multiple methods.
    
    Args:
        scores: List of similarity scores
        methods: List of method names to use
        weights: Weights for each method (must sum to 1)
        
    Returns:
        Dictionary with ensemble convergence results
    """
    if methods is None:
        methods = ["plateau", "moving_average", "statistical", "adaptive"]
    
    if weights is None:
        weights = [1.0 / len(methods)] * len(methods)
    
    if len(weights) != len(methods):
        raise ConvergenceError("Number of weights must match number of methods")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ConvergenceError("Weights must sum to 1.0")
    
    results = {}
    confidences = []
    convergence_votes = []
    
    # Run each method
    for method in methods:
        try:
            if method == "plateau":
                result = plateau_detection(scores)
                results[method] = result
                confidences.append(result["confidence"])
                convergence_votes.append(result["is_plateau"])
                
            elif method == "moving_average":
                result = moving_average_convergence(scores)
                results[method] = result
                confidences.append(result["confidence"])
                convergence_votes.append(result["converged"])
                
            elif method == "statistical":
                result = statistical_significance_test(scores)
                results[method] = result
                confidences.append(result["confidence"])
                convergence_votes.append(result.get("converged", False))
                
            elif method == "adaptive":
                result = adaptive_threshold_convergence(scores)
                results[method] = result
                confidences.append(result["confidence"])
                convergence_votes.append(result["converged"])
                
            else:
                # Unknown method, add zero weight
                results[method] = {"error": f"Unknown method: {method}"}
                confidences.append(0.0)
                convergence_votes.append(False)
                
        except Exception as e:
            results[method] = {"error": str(e)}
            confidences.append(0.0)
            convergence_votes.append(False)
    
    # Calculate weighted ensemble decision
    weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
    weighted_votes = sum(int(v) * w for v, w in zip(convergence_votes, weights))
    
    # Ensemble converged if weighted votes > 0.5
    ensemble_converged = weighted_votes > 0.5
    
    return {
        "converged": ensemble_converged,
        "confidence": weighted_confidence,
        "weighted_votes": weighted_votes,
        "individual_results": results,
        "individual_confidences": confidences,
        "individual_votes": convergence_votes,
        "methods": methods,
        "weights": weights
    }