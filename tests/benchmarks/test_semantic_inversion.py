from benchmarks.semantic_inversion import (
    adaptive_feedback_search,
    best_similarity_curve,
    independent_best_of_n,
    mutation_hill_climber,
    paired_advantage,
)


def test_independent_best_of_n_uses_exact_budget() -> None:
    candidates = ["a", "bb", "ccc", "dddd"]
    trace = independent_best_of_n(
        generate=lambda _prompt, count: candidates[:count],
        score=lambda text: len(text) / 10,
        budget=3,
    )
    assert len(trace.observations) == 3
    assert trace.best is not None
    assert trace.best.candidate == "ccc"


def test_mutation_hill_climber_retains_best_context() -> None:
    mutations = iter([["aa"], ["a"], ["aaaa"]])
    trace = mutation_hill_climber(
        initial="x",
        mutate=lambda _candidate, _count: next(mutations),
        score=lambda text: len(text) / 10,
        budget=4,
    )
    assert len(trace.observations) == 4
    assert best_similarity_curve(trace) == [0.1, 0.2, 0.2, 0.4]


def test_adaptive_feedback_only_receives_best_candidate_and_score() -> None:
    seen: list[tuple[str | None, float | None, int]] = []
    candidates = iter(["a", "aaaa", "bb"])

    def propose(best: str | None, score: float | None, step: int) -> str:
        seen.append((best, score, step))
        return next(candidates)

    trace = adaptive_feedback_search(
        propose=propose,
        score=lambda text: len(text) / 10,
        budget=3,
    )
    assert seen == [(None, None, 1), ("a", 0.1, 2), ("aaaa", 0.4, 3)]
    assert trace.best is not None
    assert trace.best.candidate == "aaaa"


def test_paired_advantage_is_case_aligned() -> None:
    adaptive = [
        adaptive_feedback_search(
            propose=lambda *_: "aaaa",
            score=lambda text: len(text) / 10,
            budget=1,
        )
    ]
    baseline = [
        independent_best_of_n(
            generate=lambda *_: ["aa"],
            score=lambda text: len(text) / 10,
            budget=1,
        )
    ]
    assert paired_advantage(adaptive, baseline) == [0.2]
