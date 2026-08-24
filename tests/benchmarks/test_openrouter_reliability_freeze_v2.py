from __future__ import annotations

from datetime import UTC, datetime, timedelta

from benchmarks.aggregate_openrouter_reliability_v2 import EVIDENCE_AFTER, REQUIRED_MODEL, aggregate


def window(*, model: str, when: datetime, successes: int = 10, quality: float = 50.0):
    calls = [
        {"success": index < successes, "transport_attempts": 1}
        for index in range(10)
    ]
    return {
        "observed_at_utc": when.astimezone(UTC).isoformat(),
        "selected_model": model,
        "observation_calls": calls,
        "qualification": [
            {
                "model": model,
                "quality": {
                    "intelligence_index": quality,
                    "agentic_index": quality,
                    "coding_index": quality,
                },
                "calls": [
                    {"success": True, "transport_attempts": 1},
                    {"success": True, "transport_attempts": 1},
                ],
            }
        ],
    }


def test_pre_rule_windows_never_count_toward_eligibility():
    old = [
        window(model=REQUIRED_MODEL, when=EVIDENCE_AFTER - timedelta(minutes=30 * index + 1))
        for index in range(60)
    ]
    result = aggregate(old)
    assert result["prospective_windows"] == 0
    assert result["excluded_pre_rule_windows"] == 60
    assert result["status"] == "insufficient_coverage"


def test_non_ox_alpha_windows_never_count_after_reset():
    rows = [
        window(model="some/other-model:free", when=EVIDENCE_AFTER + timedelta(minutes=30 * index))
        for index in range(60)
    ]
    result = aggregate(rows)
    assert result["required_model"] == REQUIRED_MODEL
    assert result["prospective_windows"] == 0
    assert result["excluded_wrong_model_windows"] == 60
    assert result["candidates"] == []
    assert result["status"] == "insufficient_coverage"


def test_48_clean_ox_alpha_windows_spanning_24h_make_candidate_reliability_eligible():
    rows = [
        window(
            model=REQUIRED_MODEL,
            when=EVIDENCE_AFTER + timedelta(minutes=(24 * 60 / 47) * index),
        )
        for index in range(48)
    ]
    result = aggregate(rows)
    candidate = result["candidates"][0]
    assert candidate["model"] == REQUIRED_MODEL
    assert candidate["selected_windows"] == 48
    assert candidate["observation_calls"] == 480
    assert candidate["span_hours"] >= 24.0
    assert candidate["observation_success_rate"] == 1.0
    assert candidate["eligible_for_freeze_review"] is True
    assert result["status"] == "eligible"
    assert result["selected_model_for_freeze_review"] == REQUIRED_MODEL
    assert result["freeze_authorized"] is False


def test_enough_coverage_but_low_call_success_is_reliability_failure():
    rows = [
        window(
            model=REQUIRED_MODEL,
            when=EVIDENCE_AFTER + timedelta(minutes=(24 * 60 / 47) * index),
            successes=9 if index in {10, 30, 45} else 10,
        )
        for index in range(48)
    ]
    result = aggregate(rows)
    candidate = result["candidates"][0]
    assert candidate["coverage_ok"] is True
    assert candidate["observation_success_rate"] < 0.995
    assert candidate["eligible_for_freeze_review"] is False
    assert result["status"] == "reliability_failure"


def test_consecutive_failure_windows_fail_even_if_aggregate_rate_could_pass():
    rows = []
    for index in range(100):
        successes = 9 if index in {50, 51} else 10
        rows.append(
            window(
                model=REQUIRED_MODEL,
                when=EVIDENCE_AFTER + timedelta(minutes=30 * index),
                successes=successes,
            )
        )
    result = aggregate(rows)
    candidate = result["candidates"][0]
    assert candidate["observation_success_rate"] >= 0.995
    assert candidate["consecutive_failed_windows"] is True
    assert candidate["reliability_ok"] is False
