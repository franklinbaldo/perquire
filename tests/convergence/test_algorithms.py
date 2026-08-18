import numpy as np
import pytest

from perquire.convergence.algorithms import ConvergenceDetector, ConvergenceReason
from perquire.exceptions import ConvergenceError


@pytest.fixture
def detector() -> ConvergenceDetector:
    return ConvergenceDetector(
        similarity_threshold=0.90,
        min_improvement=0.001,
        convergence_window=3,
        plateau_threshold=0.0005,
        max_iterations=10,
        statistical_confidence=0.95,
    )


def test_empty_and_single_score_continue(detector) -> None:
    empty = detector.should_continue([], current_iteration=0)
    one = detector.should_continue([0.5], current_iteration=1)
    assert not empty.converged
    assert not one.converged


def test_max_iterations_is_hard_stop(detector) -> None:
    result = detector.should_continue([0.1, 0.2, 0.3], current_iteration=10)
    assert result.converged
    assert result.reason == ConvergenceReason.MAX_ITERATIONS


def test_high_similarity_with_plateau_stops(detector) -> None:
    result = detector.should_continue([0.899, 0.901, 0.902], current_iteration=3)
    assert result.converged
    assert result.reason == ConvergenceReason.SIMILARITY_THRESHOLD


def test_low_similarity_plateau_stops(detector) -> None:
    result = detector.should_continue([0.5, 0.5001, 0.5002], current_iteration=3)
    assert result.converged
    assert result.reason == ConvergenceReason.PLATEAU_DETECTED
    assert result.plateau_length >= 3


def test_clear_improvement_continues(detector) -> None:
    result = detector.should_continue([0.1, 0.3, 0.5, 0.7], current_iteration=4)
    assert not result.converged


def test_diminishing_returns_stop_when_not_plateau(detector) -> None:
    detector.plateau_threshold = 1e-12
    result = detector.should_continue([0.4, 0.4002, 0.4004], current_iteration=3)
    assert result.converged
    assert result.reason == ConvergenceReason.DIMINISHING_RETURNS


def test_plateau_length_counts_observations(detector) -> None:
    result = detector._detect_plateau([0.1, 0.5, 0.5001, 0.5000, 0.5002])
    assert result["is_plateau"]
    assert result["plateau_length"] >= detector.convergence_window


def test_recent_improvement_ignores_negative_steps(detector) -> None:
    assert detector._calculate_recent_improvement([0.5, 0.4, 0.3]) == 0.0
    assert detector._calculate_recent_improvement([0.1, 0.2]) == float("inf")


def test_statistical_check_is_scipy_free_and_finite(detector) -> None:
    result = detector._statistical_convergence_test([0.8, 0.8, 0.8, 0.8, 0.8])
    assert 0.0 <= result["p_value"] <= 1.0
    assert np.isfinite(result["p_value"])
    assert result["metrics"]["trend_direction"] == "no_trend"


def test_internal_numeric_failure_is_wrapped(detector, monkeypatch) -> None:
    monkeypatch.setattr(np, "var", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ConvergenceError, match="boom"):
        detector.should_continue([0.1, 0.2, 0.3], current_iteration=3)
