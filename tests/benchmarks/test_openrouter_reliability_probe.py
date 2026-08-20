from benchmarks.probe_openrouter_reliability import (
    ELIGIBLE_MODELS,
    PROMPTS,
    REPETITIONS,
    select_model,
)


def result(model: str, failures: int = 0) -> dict:
    calls = len(PROMPTS) * REPETITIONS
    return {
        "model": model,
        "logical_successes": calls - failures,
        "logical_failures": failures,
    }


def test_cheaper_passing_candidate_wins_when_both_are_reliable():
    assert select_model([result(model) for model in ELIGIBLE_MODELS]) == (
        "google/gemini-2.5-flash-lite"
    )


def test_only_fully_reliable_candidate_is_eligible():
    assert select_model(
        [
            result("google/gemini-2.5-flash-lite", failures=1),
            result("google/gemini-3.1-flash-lite"),
        ]
    ) == "google/gemini-3.1-flash-lite"


def test_no_model_is_selected_when_both_have_a_failure():
    assert (
        select_model(
            [
                result("google/gemini-2.5-flash-lite", failures=1),
                result("google/gemini-3.1-flash-lite", failures=1),
            ]
        )
        is None
    )


def test_control_cannot_win_even_if_it_passes():
    assert select_model([result("openai/gpt-oss-20b:free")]) is None
