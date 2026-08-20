from benchmarks.merge_scaling_shards import merge_shards


def _shard(case_id: str, budget: int, *, status: str = "valid"):
    record = {
        "case_id": case_id,
        "budget": budget,
        "replicate": 0,
        "method": "adaptive_perquire",
        "status": status,
        "best_target_similarity": 0.2 if status == "valid" else None,
    }
    return {
        "benchmark": "semantic-inversion-scaling-v1",
        "budgets": [budget],
        "replicates": 1,
        "llm_provider": "openrouter",
        "llm_model": "openrouter/google/gemini-2.5-flash-lite",
        "embedding_provider": "openrouter",
        "embedding_model": "openrouter/embed",
        "generation_config": {"requests_per_minute": 20, "max_retries": 2},
        "budget_unit": "target_similarity_evaluation",
        "records": [record],
        "case_overheads": [],
        "cache": {},
    }


def test_merge_recomputes_validity_and_detects_missing_method_cells():
    payload = merge_shards([_shard("c1", 2), _shard("c1", 4)])
    assert payload["budgets"] == [2, 4]
    assert payload["case_count"] == 1
    assert payload["validity"]["observed_cells"] == 2
    assert payload["validity"]["expected_cells"] == 6
    assert payload["validity"]["missing_cells"] == 4
    assert payload["validity"]["clean_v1_claim_eligible"] is False


def test_merge_rejects_duplicate_cells():
    shard = _shard("c1", 2)
    try:
        merge_shards([shard, shard])
    except ValueError as error:
        assert "duplicate scaling cell" in str(error)
    else:
        raise AssertionError("duplicate cells must fail aggregation")


def test_merge_rejects_model_drift_between_shards():
    first = _shard("c1", 2)
    second = _shard("c2", 2)
    second["llm_model"] = "openrouter/other-model"
    try:
        merge_shards([first, second])
    except ValueError as error:
        assert "llm_model" in str(error)
    else:
        raise AssertionError("model drift must fail aggregation")
