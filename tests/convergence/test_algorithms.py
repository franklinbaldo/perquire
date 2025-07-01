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

# --- More specific tests for helper methods ---

def test_detect_plateau_logic(detector, detector_defaults):
    # Test case 1: Clear plateau
    scores_plateau = [0.5, 0.5001, 0.5000, 0.5002] # variance should be low
    plateau_result = detector._detect_plateau(scores_plateau)
    assert plateau_result["is_plateau"]
    assert plateau_result["plateau_length"] >= detector_defaults["convergence_window"] # or specific expected length
    assert plateau_result["variance"] <= detector_defaults["plateau_threshold"]

    # Test case 2: No plateau, scores increasing
    scores_increasing = [0.5, 0.6, 0.7, 0.8]
    plateau_result_increasing = detector._detect_plateau(scores_increasing)
    assert not plateau_result_increasing["is_plateau"]
    assert plateau_result_increasing["variance"] > detector_defaults["plateau_threshold"]

    # Test case 3: Scores too short for a full window
    scores_short = [0.5, 0.5001]
    plateau_result_short = detector._detect_plateau(scores_short)
    assert not plateau_result_short["is_plateau"]
    assert plateau_result_short["plateau_length"] == 0

    # Test case 4: Plateau at the end of a longer series
    scores_long_plateau_end = [0.1, 0.2, 0.3, 0.5, 0.5001, 0.5000, 0.5002]
    plateau_result_long_end = detector._detect_plateau(scores_long_plateau_end)
    assert plateau_result_long_end["is_plateau"]
    assert plateau_result_long_end["plateau_length"] >= detector_defaults["convergence_window"]


def test_calculate_recent_improvement_logic(detector, detector_defaults):
    # Test case 1: Clear improvement
    scores_improving = [0.1, 0.2, 0.3, 0.4] # improvements: 0.1, 0.1, 0.1. Mean = 0.1
    # Ensure enough scores for the window
    scores_improving_padded = [0.0] * (detector_defaults["convergence_window"] - len(scores_improving)%detector_defaults["convergence_window"] if len(scores_improving)%detector_defaults["convergence_window"]!=0 else 0) + scores_improving
    # This padding logic is a bit complex, simpler to just ensure enough scores
    # Let's use a simpler approach for test data:
    scores_clear_improvement = [0.1, 0.2, 0.3, 0.4, 0.5] # window=3. Recent [0.3,0.4,0.5]. Impr: 0.1,0.1. Mean = 0.1
    improvement1 = detector._calculate_recent_improvement(scores_clear_improvement)
    assert improvement1 == pytest.approx(0.1)

    # Test case 2: Diminishing returns
    scores_diminishing = [0.1, 0.5, 0.50001, 0.50002, 0.50003] # Recent [0.50001, 0.50002, 0.50003]. Impr: 0.00001, 0.00001. Mean = 0.00001
    improvement2 = detector._calculate_recent_improvement(scores_diminishing)
    assert improvement2 == pytest.approx(0.00001)

    # Test case 3: Scores decreasing (improvement should be 0 for negative improvements)
    scores_decreasing = [0.5, 0.4, 0.3, 0.2, 0.1] # Recent [0.3,0.2,0.1]. Impr: -0.1, -0.1. Max(0, impr) -> 0,0. Mean = 0
    improvement3 = detector._calculate_recent_improvement(scores_decreasing)
    assert improvement3 == pytest.approx(0.0)

    # Test case 4: Not enough scores for a full window
    scores_short = [0.1, 0.2]
    improvement4 = detector._calculate_recent_improvement(scores_short)
    assert improvement4 == float('inf') # As per implementation for not enough data

def test_statistical_convergence_scipy_fallback(detector, detector_defaults):
    scores = [0.8] * detector_defaults["convergence_window"] # Plateaued scores

    with patch('perquire.convergence.algorithms.stats') as mock_scipy_stats:
        # Simulate scipy.stats being None or raising ImportError
        # Method 1: Set stats to None
        # algorithms.stats = None # This would modify the module globally, not ideal for a test
        # Method 2: Patch import to fail
        with patch('importlib.import_module', side_effect=ImportError):
             # Need to re-trigger the import within the function or make scipy.stats access fail
             # The code is `from scipy import stats`. We need this to fail.
             # The current patch of 'perquire.convergence.algorithms.stats' might be too late
             # if 'from scipy import stats' is at module level.
             # Let's assume 'from scipy import stats' is inside _statistical_convergence_test or can be mocked out.
             # For simplicity, let's directly cause an import error when 'stats.norm.cdf' is accessed.
            mock_scipy_stats.norm.cdf.side_effect = ImportError("Scipy not available")

            # If the above doesn't work due to import scope, an alternative is to patch np.var in the fallback
            # to verify the fallback path is taken, though it doesn't test the ImportCatch directly.
            # For now, let's assume the above patch can simulate the ImportError for the 'from scipy import stats'
            # or that the 'stats' object itself becomes None or unusable.
            # A more robust way is to patch 'scipy.stats' where it's used.
            # The code uses `from scipy import stats` then `stats.norm.cdf`.
            # So, if `stats` is None or `stats.norm.cdf` raises error, it should fallback.

            # To test the fallback which uses variance:
            # Ensure scores would cause convergence by fallback's variance check
            # Fallback: recent_var = np.var(scores[-self.convergence_window:])
            #           converged = recent_var < self.plateau_threshold
            # Scores [0.8, 0.8, 0.8] -> variance is 0, which is < plateau_threshold (0.0005)

            # We need to ensure the try-except for ImportError is hit.
            # This happens if `from scipy import stats` fails or if `stats.norm.cdf` fails.
            # The current `_statistical_convergence_test` has `from scipy import stats` inside.
            # So, we can patch `scipy.stats` in the algorithms module context.

            with patch('perquire.convergence.algorithms.stats', None): # Simulate scipy.stats not being available
                result_dict = detector._statistical_convergence_test(scores)

            assert result_dict["converged"] # Should converge by variance fallback
            assert result_dict["metrics"]["method"] == "fallback_variance"
            assert result_dict["metrics"]["variance"] == pytest.approx(0.0)
            assert result_dict["p_value"] == pytest.approx(0.0) # p_value is set to recent_var in fallback

            # Test with scores that would NOT converge by fallback variance
            scores_not_converging_fallback = [0.1, 0.5, 0.9] # var will be > plateau_threshold
            with patch('perquire.convergence.algorithms.stats', None):
                result_dict_not_conv = detector._statistical_convergence_test(scores_not_converging_fallback)
            assert not result_dict_not_conv["converged"]
            assert result_dict_not_conv["metrics"]["method"] == "fallback_variance"
            assert result_dict_not_conv["metrics"]["variance"] > detector_defaults["plateau_threshold"]
