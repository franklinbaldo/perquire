"""Scaling sweep output must preserve the preregistered curve and its failures."""

import pytest

import benchmarks.run_scaling_sweep as sweep
from benchmarks.run_semantic_inversion import MethodExecutionError, ResourceMeter
from benchmarks.semantic_inversion import BenchmarkCase, CandidateObservation, SearchTrace


def test_parse_budgets_sorts_and_rejects_duplicates():
    assert sweep.parse_budgets("32,2,8,4,16") == (2, 4, 8, 16, 32)
    with pytest.raises(Exception):
        sweep.parse_budgets("2,2,4")


def test_best_so_far_is_monotone_even_when_raw_scores_fall():
    observations = [
        {"step": 1, "target_similarity": 0.10},
        {"step": 2, "target_similarity": 0.05},
        {"step": 3, "target_similarity": 0.20},
    ]
    assert [point["best_target_similarity"] for point in sweep.best_so_far(observations)] == [
        0.10,
        0.10,
        0.20,
    ]


def test_summary_reports_marginal_gain_uncertainty_and_auc():
    records = [
        {"method": "adaptive_perquire", "budget": 2, "best_target_similarity": 0.10, "status": "valid"},
        {"method": "adaptive_perquire", "budget": 2, "best_target_similarity": 0.14, "status": "valid"},
        {"method": "adaptive_perquire", "budget": 4, "best_target_similarity": 0.20, "status": "valid"},
        {"method": "adaptive_perquire", "budget": 4, "best_target_similarity": 0.24, "status": "valid"},
    ]
    result = sweep.summarize(records, (2, 4))
    points = result["methods"]["adaptive_perquire"]["points"]
    assert points[0]["mean_best_target_similarity"] == pytest.approx(0.12)
    assert points[1]["mean_best_target_similarity"] == pytest.approx(0.22)
    assert points[1]["delta_mean_from_previous_budget"] == pytest.approx(0.10)
    assert points[0]["stdev_best_target_similarity"] > 0
    assert result["methods"]["adaptive_perquire"]["auc_mean_best_vs_log2_budget"] > 0


def test_invalid_cells_are_not_silently_folded_into_curve_summary():
    records = [
        {"method": "adaptive_perquire", "budget": 2, "best_target_similarity": 0.4, "status": "valid"},
        {"method": "adaptive_perquire", "budget": 2, "best_target_similarity": None, "status": "invalid"},
    ]
    result = sweep.summarize(records, (2,))
    point = result["methods"]["adaptive_perquire"]["points"][0]
    assert point["n_valid"] == 1
    assert point["mean_best_target_similarity"] == pytest.approx(0.4)


def test_validity_gate_blocks_clean_claim_below_95_percent():
    records = [{"status": "valid"}] * 18 + [{"status": "invalid"}] * 2
    validity = sweep.validity_summary(records)
    assert validity == {
        "expected_cells": 20,
        "valid_cells": 18,
        "invalid_cells": 2,
        "valid_fraction": 0.9,
        "minimum_valid_fraction": 0.95,
        "clean_v1_claim_eligible": False,
    }


def test_one_failing_method_cell_does_not_erase_other_methods(monkeypatch):
    case = BenchmarkCase(case_id="c1", domain="test", source_text="hidden")

    class Embedder:
        transport_attempts = 0

    def fake_prepare_target(_case, _embedder):
        return [1.0, 0.0], 1

    def fake_run_method(_case, *, method, **_kwargs):
        meter = ResourceMeter()
        meter.by_method[method] = {
            "llm_calls": 1,
            "llm_transport_attempts": 1,
            "embedding_transport_attempts": 1,
            "generated_candidates": 1,
            "empty_generations": 0,
        }
        if method == "mutation_hill_climber":
            raise MethodExecutionError(method, RuntimeError("synthetic failure"), meter.by_method[method])
        trace = SearchTrace(
            method=method,
            observations=[CandidateObservation(method, 1, f"{method} candidate", 0.25)],
        )
        return trace, meter

    monkeypatch.setattr(sweep, "prepare_target", fake_prepare_target)
    monkeypatch.setattr(sweep, "run_method", fake_run_method)

    records, overheads = sweep.execute_sweep(
        cases=[case], budgets=(1,), replicates=1, llm=object(), embedder=Embedder()
    )

    assert len(records) == 3
    assert [record["status"] for record in records] == ["valid", "invalid", "valid"]
    invalid = records[1]
    assert invalid["method"] == "mutation_hill_climber"
    assert invalid["error_type"] == "RuntimeError"
    assert invalid["llm_calls"] == 1
    assert "synthetic failure" not in str(invalid)
    assert overheads[0]["status"] == "valid"


def test_target_failure_marks_all_method_cells_invalid_and_continues(monkeypatch):
    cases = [
        BenchmarkCase(case_id="bad", domain="test", source_text="bad"),
        BenchmarkCase(case_id="good", domain="test", source_text="good"),
    ]

    class Embedder:
        transport_attempts = 0

    def fake_prepare_target(case, _embedder):
        if case.case_id.startswith("bad"):
            raise RuntimeError("target unavailable")
        return [1.0, 0.0], 0

    def fake_run_method(_case, *, method, **_kwargs):
        meter = ResourceMeter()
        meter.by_method[method] = {
            "llm_calls": 0,
            "llm_transport_attempts": 0,
            "embedding_transport_attempts": 1,
            "generated_candidates": 0,
            "empty_generations": 0,
        }
        return SearchTrace(
            method=method,
            observations=[CandidateObservation(method, 1, "candidate", 0.1)],
        ), meter

    monkeypatch.setattr(sweep, "prepare_target", fake_prepare_target)
    monkeypatch.setattr(sweep, "run_method", fake_run_method)

    records, _ = sweep.execute_sweep(
        cases=cases, budgets=(1,), replicates=1, llm=object(), embedder=Embedder()
    )
    assert len(records) == 6
    assert sum(record["status"] == "invalid" for record in records) == 3
    assert sum(record["status"] == "valid" for record in records) == 3
    assert {record["error_stage"] for record in records if record["status"] == "invalid"} == {
        "target_embedding"
    }
