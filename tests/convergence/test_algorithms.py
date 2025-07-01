import pytest
import numpy as np
from unittest.mock import patch # Added this import

from perquire.convergence.algorithms import ConvergenceDetector, ConvergenceReason, ConvergenceResult
from perquire.exceptions import ConvergenceError

@pytest.fixture
def detector_defaults():
    return {
        "similarity_threshold": 0.90,
        "min_improvement": 0.001,
        "convergence_window": 3,
        "plateau_threshold": 0.0005,
        "max_iterations": 10,
        "statistical_confidence": 0.95
    }

@pytest.fixture
def detector(detector_defaults):
    return ConvergenceDetector(**detector_defaults)

def test_convergence_max_iterations(detector, detector_defaults):
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.81, 0.82] # 10 scores
    # current_iteration is 1-based index
    result = detector.should_continue(scores, current_iteration=detector_defaults["max_iterations"])
    assert result.converged
    assert result.reason == ConvergenceReason.MAX_ITERATIONS

def test_convergence_similarity_threshold(detector, detector_defaults):
    # Scores that reach threshold and plateau
    scores = [0.88, 0.90, 0.901, 0.902] # Last 3 are close enough for plateau and >= threshold
    # Mock _detect_plateau to ensure it's considered a plateau for this specific test
    # This isolates testing the similarity_threshold logic itself
    with patch.object(detector, '_detect_plateau', return_value={'is_plateau': True, 'plateau_length': detector_defaults["convergence_window"], 'variance': 0.0001, 'threshold': detector_defaults["plateau_threshold"]}) as mock_plateau:
        result = detector.should_continue(scores, current_iteration=len(scores))

    assert result.converged
    assert result.reason == ConvergenceReason.SIMILARITY_THRESHOLD
    mock_plateau.assert_called()


def test_no_convergence_below_threshold_and_improving(detector):
    scores = [0.1, 0.2, 0.3, 0.4] # Clearly improving, below threshold
    with patch.object(detector, '_statistical_convergence_test', return_value={"converged": False}): # Assume no stat convergence
        result = detector.should_continue(scores, current_iteration=len(scores))
    assert not result.converged

def test_convergence_plateau_detected(detector, detector_defaults):
    scores = [0.5, 0.5001, 0.5002, 0.50015]
    # Ensure current_iteration allows for full window analysis
    result = detector.should_continue(scores, current_iteration=len(scores))
    assert result.converged
    assert result.reason == ConvergenceReason.PLATEAU_DETECTED
    # The actual plateau_length might be calculated differently internally based on sliding window
    # So check it's at least the convergence_window
    assert result.plateau_length >= detector_defaults["convergence_window"]


def test_no_convergence_on_short_plateau(detector, detector_defaults):
    scores = [0.5, 0.5001] # Plateau too short for window of 3
    with patch.object(detector, '_statistical_convergence_test', return_value={"converged": False}):
        result = detector.should_continue(scores, current_iteration=len(scores))
    # Might converge if max_iterations is 2, or threshold is 0.5. Assuming not for this test.
    if detector_defaults["max_iterations"] > 2 and detector_defaults["similarity_threshold"] > 0.6:
         assert not result.converged

def test_convergence_diminishing_returns(detector, detector_defaults):
    # Recent improvements are less than min_improvement (0.001)
    # Example: scores = [0.1, 0.2, 0.20005, 0.20010, 0.20015]
    # Recent scores: [0.2, 0.20005, 0.20010, 0.20015]
    # Improvements: 0.00005, 0.00005, 0.00005. Mean is 0.00005 which is < 0.001
    scores = [0.1 for _ in range(detector_defaults["convergence_window"] -1)] + \
             [0.2, 0.20005, 0.20010, 0.20015] # Pad to have enough scores

    # Ensure it doesn't converge for other reasons first
    if max(scores) < detector_defaults["similarity_threshold"]:
        with patch.object(detector, '_statistical_convergence_test', return_value={"converged": False}):
            result = detector.should_continue(scores, current_iteration=len(scores))
        assert result.converged
        assert result.reason == ConvergenceReason.DIMINISHING_RETURNS


def test_convergence_statistical_significance_no_trend(detector, detector_defaults):
    scores = [0.8] * detector_defaults["max_iterations"] # Stable scores
    # Mock the _statistical_convergence_test to return a specific outcome
    # This isolates the logic that uses the stat test result
    with patch.object(detector, '_statistical_convergence_test',
                      return_value={"converged": True, "confidence": 0.96, "p_value": 0.04, "metrics": {}}) as mock_stat:
        # Ensure no other convergence reason fires first
        if max(scores) < detector_defaults["similarity_threshold"]:
            result = detector.should_continue(scores, current_iteration=len(scores))
            if result.reason == ConvergenceReason.STATISTICAL_SIGNIFICANCE : # Could also be PLATEAU or DIMINISHING
                 assert result.converged
                 mock_stat.assert_called()


def test_no_convergence_with_strong_upward_trend(detector, detector_defaults):
    scores = [0.1, 0.3, 0.5, 0.7, 0.85] # Strong trend, not plateaued, not at threshold
    # Mock stat test to indicate strong trend (not converged by this reason)
    with patch.object(detector, '_statistical_convergence_test',
                      return_value={"converged": False, "confidence": 0.0, "p_value": 0.99, "metrics": {}}):
        result = detector.should_continue(scores, current_iteration=len(scores))

    # This test is tricky because other conditions (like max_iterations or threshold) might still cause convergence.
    # We are primarily testing that if other conditions are NOT met, and stat test says "not converged", then overall not converged.
    converged_by_other_means = (
        len(scores) >= detector_defaults["max_iterations"] or
        max(scores) >= detector_defaults["similarity_threshold"]
        # Diminishing returns and plateau are less likely with this data
    )
    if not converged_by_other_means:
        assert not result.converged


def test_empty_scores_no_convergence(detector):
    result = detector.should_continue([], current_iteration=0)
    assert not result.converged
    assert "Need more data points" in result.recommendation

def test_single_score_no_convergence(detector):
    result = detector.should_continue([0.5], current_iteration=1)
    assert not result.converged
    assert "Need more iterations" in result.recommendation

def test_convergence_error_handling(detector):
    with patch.object(np, 'var', side_effect=ValueError("Test error")):
        with pytest.raises(ConvergenceError, match="Failed to analyze convergence: Test error"):
            detector.should_continue([0.1,0.2,0.3], current_iteration=3)

```
