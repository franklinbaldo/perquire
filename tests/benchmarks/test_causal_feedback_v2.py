from __future__ import annotations

import re

import pytest

from benchmarks.causal_feedback_v2 import (
    ARMS,
    CausalCase,
    CausalExecutionError,
    CausalObservation,
    arm_order,
    checkpoint_metrics,
    pair_decoys,
    render_prompt,
    run_causal_trace,
)


def test_decoy_pairing_is_prospective_ring_without_self_pairs():
    cases = [
        CausalCase("a", "x", "secret a"),
        CausalCase("b", "x", "secret b"),
        CausalCase("c", "x", "secret c"),
    ]
    assert pair_decoys(cases) == {"a": "b", "b": "c", "c": "a"}


def test_arm_order_is_deterministic_and_not_one_fixed_global_order():
    first = arm_order("concrete-001", 0)
    assert first == arm_order("concrete-001", 0)
    assert set(first) == set(ARMS)
    observed = {arm_order(f"case-{index}", 0) for index in range(20)}
    assert len(observed) > 1


def test_prompt_never_discloses_arm_or_hidden_source():
    hidden = "A red bicycle is leaning against a brick wall after the rain."
    history = [CausalObservation(1, "a bicycle outdoors", 0.8, 0.2)]
    prompt = render_prompt(history)
    assert hidden not in prompt
    assert "true_feedback" not in prompt
    assert "decoy_feedback" not in prompt
    assert "null_feedback" not in prompt
    assert "feedback=0.20000000" in prompt


def test_given_same_candidate_history_only_feedback_numbers_change_prompt():
    left = [CausalObservation(1, "same candidate", 0.7, 0.7)]
    right = [CausalObservation(1, "same candidate", 0.7, 0.1)]
    normalized_left = re.sub(r"feedback=-?\d+\.\d+", "feedback=<N>", render_prompt(left))
    normalized_right = re.sub(r"feedback=-?\d+\.\d+", "feedback=<N>", render_prompt(right))
    assert normalized_left == normalized_right


def test_true_decoy_and_null_arms_change_only_feedback_channel():
    outputs = iter(["candidate one", "candidate two"])

    def generate(_prompt: str, _step: int) -> str:
        return next(outputs)

    true_scores = {"candidate one": 0.2, "candidate two": 0.9}
    decoy_scores = {"candidate one": 0.8, "candidate two": 0.1}
    trace = run_causal_trace(
        case_id="case",
        arm="decoy_feedback",
        replicate=0,
        max_steps=2,
        generate=generate,
        score_true=true_scores.__getitem__,
        score_decoy=decoy_scores.__getitem__,
    )
    assert [item.true_target_score for item in trace.observations] == [0.2, 0.9]
    assert [item.feedback_score for item in trace.observations] == [0.8, 0.1]

    for arm, expected in (("true_feedback", 0.2), ("null_feedback", 0.0)):
        one = run_causal_trace(
            case_id="case",
            arm=arm,
            replicate=0,
            max_steps=1,
            generate=lambda _prompt, _step: "candidate one",
            score_true=true_scores.__getitem__,
            score_decoy=decoy_scores.__getitem__,
        )
        assert one.observations[0].feedback_score == expected
        assert one.observations[0].true_target_score == 0.2


def test_checkpoints_are_prefixes_of_one_trajectory_not_new_runs():
    calls: list[int] = []

    def generate(_prompt: str, step: int) -> str:
        calls.append(step)
        return f"candidate {step}"

    trace = run_causal_trace(
        case_id="case",
        arm="true_feedback",
        replicate=0,
        max_steps=16,
        generate=generate,
        score_true=lambda candidate: int(candidate.split()[-1]) / 100,
        score_decoy=lambda candidate: 1 - int(candidate.split()[-1]) / 100,
    )
    rows = checkpoint_metrics(trace)
    assert calls == list(range(1, 17))
    assert [row["checkpoint"] for row in rows] == [2, 4, 8, 16]
    assert [row["n_observed"] for row in rows] == [2, 4, 8, 16]


def test_exactly_one_generation_call_per_completed_step():
    calls = 0

    def generate(_prompt: str, step: int) -> str:
        nonlocal calls
        calls += 1
        return f"candidate {step}"

    trace = run_causal_trace(
        case_id="case",
        arm="null_feedback",
        replicate=0,
        max_steps=8,
        generate=generate,
        score_true=lambda _candidate: 0.5,
        score_decoy=lambda _candidate: 0.6,
    )
    assert len(trace.observations) == 8
    assert calls == 8


def test_failure_preserves_completed_observations_and_failure_step():
    def generate(_prompt: str, step: int) -> str:
        if step == 3:
            raise RuntimeError("provider unavailable")
        return f"candidate {step}"

    with pytest.raises(CausalExecutionError) as caught:
        run_causal_trace(
            case_id="case",
            arm="true_feedback",
            replicate=0,
            max_steps=5,
            generate=generate,
            score_true=lambda _candidate: 0.5,
            score_decoy=lambda _candidate: 0.4,
        )

    trace = caught.value.trace
    assert trace.status == "invalid"
    assert trace.failure_step == 3
    assert trace.error_type == "RuntimeError"
    assert [item.step for item in trace.observations] == [1, 2]


def test_source_text_has_no_path_into_core_proposer_api():
    # The causal runner accepts case_id only; source_text exists only on the case
    # object used by target construction outside the proposer.
    parameters = run_causal_trace.__annotations__
    assert "source_text" not in parameters
