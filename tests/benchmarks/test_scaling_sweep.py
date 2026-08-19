"""Scaling sweep output must preserve the preregistered curve, not just winners."""

import pytest

from benchmarks.run_scaling_sweep import best_so_far, parse_budgets, summarize


def test_parse_budgets_sorts_and_rejects_duplicates():
    assert parse_budgets("32,2,8,4,16") == (2, 4, 8, 16, 32)
    with pytest.raises(Exception):
        parse_budgets("2,2,4")


def test_best_so_far_is_monotone_even_when_raw_scores_fall():
    observations = [
        {"step": 1, "target_similarity": 0.10},
        {"step": 2, "target_similarity": 0.05},
        {"step": 3, "target_similarity": 0.20},
    ]
    assert [point["best_target_similarity"] for point in best_so_far(observations)] == [
        0.10,
        0.10,
        0.20,
    ]


def test_summary_reports_marginal_gain_uncertainty_and_auc():
    records = [
        {"method": "adaptive_perquire", "budget": 2, "best_target_similarity": 0.10},
        {"method": "adaptive_perquire", "budget": 2, "best_target_similarity": 0.14},
        {"method": "adaptive_perquire", "budget": 4, "best_target_similarity": 0.20},
        {"method": "adaptive_perquire", "budget": 4, "best_target_similarity": 0.24},
    ]
    result = summarize(records, (2, 4))
    points = result["methods"]["adaptive_perquire"]["points"]
    assert points[0]["mean_best_target_similarity"] == pytest.approx(0.12)
    assert points[1]["mean_best_target_similarity"] == pytest.approx(0.22)
    assert points[1]["delta_mean_from_previous_budget"] == pytest.approx(0.10)
    assert points[0]["stdev_best_target_similarity"] > 0
    assert result["methods"]["adaptive_perquire"]["auc_mean_best_vs_log2_budget"] > 0
