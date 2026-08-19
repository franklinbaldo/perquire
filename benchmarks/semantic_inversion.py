"""Reproducible semantic-inversion benchmark primitives.

The benchmark deliberately separates target-vector scoring from candidate generation.
Search methods receive a scoring callback and never receive hidden source text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol


ScoreFn = Callable[[str], float]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    domain: str
    source_text: str


@dataclass(frozen=True)
class CandidateObservation:
    method: str
    step: int
    candidate: str
    target_similarity: float


@dataclass
class SearchTrace:
    method: str
    observations: list[CandidateObservation] = field(default_factory=list)

    @property
    def best(self) -> CandidateObservation | None:
        return max(self.observations, key=lambda item: item.target_similarity, default=None)


class CandidateGenerator(Protocol):
    def __call__(self, prompt: str, count: int) -> list[str]: ...


class CandidateMutator(Protocol):
    def __call__(self, candidate: str, count: int) -> list[str]: ...


def independent_best_of_n(
    *,
    generate: CandidateGenerator,
    score: ScoreFn,
    budget: int,
    prompt: str = "Generate diverse short descriptions of possible semantic concepts.",
) -> SearchTrace:
    """Generate candidates without target-score feedback, then select by score."""
    trace = SearchTrace(method="independent_best_of_n")
    candidates = generate(prompt, budget)
    for step, candidate in enumerate(candidates[:budget], start=1):
        trace.observations.append(
            CandidateObservation(trace.method, step, candidate, score(candidate))
        )
    return trace


def mutation_hill_climber(
    *,
    initial: str,
    mutate: CandidateMutator,
    score: ScoreFn,
    budget: int,
) -> SearchTrace:
    """Simple retained-best local baseline using one scored mutation per step."""
    trace = SearchTrace(method="mutation_hill_climber")
    current = initial
    current_score = score(current)
    trace.observations.append(CandidateObservation(trace.method, 1, current, current_score))

    for step in range(2, budget + 1):
        proposals = mutate(current, 1)
        if not proposals:
            break
        proposal = proposals[0]
        proposal_score = score(proposal)
        trace.observations.append(
            CandidateObservation(trace.method, step, proposal, proposal_score)
        )
        if proposal_score > current_score:
            current, current_score = proposal, proposal_score
    return trace


def adaptive_feedback_search(
    *,
    propose: Callable[[str | None, float | None, int], str],
    score: ScoreFn,
    budget: int,
) -> SearchTrace:
    """Minimal adaptive search contract mirroring Perquire's feedback loop.

    The proposer sees only the current best candidate, its score, and the step.
    It never sees hidden source text.
    """
    trace = SearchTrace(method="adaptive_perquire")
    best_text: str | None = None
    best_score: float | None = None

    for step in range(1, budget + 1):
        candidate = propose(best_text, best_score, step)
        candidate_score = score(candidate)
        trace.observations.append(
            CandidateObservation(trace.method, step, candidate, candidate_score)
        )
        if best_score is None or candidate_score > best_score:
            best_text, best_score = candidate, candidate_score
    return trace


def best_similarity_curve(trace: SearchTrace) -> list[float]:
    """Return best-so-far target similarity after each evaluation."""
    curve: list[float] = []
    best = float("-inf")
    for observation in trace.observations:
        best = max(best, observation.target_similarity)
        curve.append(best)
    return curve


def paired_advantage(
    adaptive: Iterable[SearchTrace], baseline: Iterable[SearchTrace]
) -> list[float]:
    """Case-aligned final best-score differences: adaptive minus baseline."""
    differences: list[float] = []
    for adaptive_trace, baseline_trace in zip(adaptive, baseline, strict=True):
        if adaptive_trace.best is None or baseline_trace.best is None:
            raise ValueError("Cannot compare an empty search trace")
        differences.append(
            adaptive_trace.best.target_similarity - baseline_trace.best.target_similarity
        )
    return differences
