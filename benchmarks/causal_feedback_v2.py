"""Pure primitives for the preregistered causal-feedback v2 experiment.

This module deliberately has no provider imports. It defines the causal arms,
prospective decoy pairing, deterministic arm counterbalancing, one shared prompt
surface, nested trajectories, and failure-preserving trace semantics.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable


EXPERIMENT_VERSION = "causal-feedback-v2"
ARMS = ("true_feedback", "decoy_feedback", "null_feedback")
CHECKPOINTS = (2, 4, 8, 16, 32)
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


@dataclass(frozen=True)
class CausalCase:
    case_id: str
    domain: str
    source_text: str


@dataclass(frozen=True)
class CausalObservation:
    step: int
    candidate: str
    true_target_score: float
    feedback_score: float


@dataclass
class CausalTrace:
    case_id: str
    arm: str
    replicate: int
    observations: list[CausalObservation] = field(default_factory=list)
    status: str = "valid"
    error_type: str | None = None
    failure_step: int | None = None

    @property
    def best_true_score(self) -> float | None:
        if not self.observations:
            return None
        return max(item.true_target_score for item in self.observations)


class CausalExecutionError(RuntimeError):
    """Failure carrying all observations completed before the failed step."""

    def __init__(self, trace: CausalTrace, error: Exception, failure_step: int) -> None:
        super().__init__(f"{trace.arm} failed at step {failure_step}: {error}")
        self.trace = trace
        self.error = error
        self.failure_step = failure_step


def parse_one_candidate(content: str) -> str:
    """Select the first non-empty candidate using one frozen parser rule."""
    for raw in content.splitlines():
        text = _LIST_MARKER.sub("", raw.strip()).strip()
        if text:
            return text
    raise ValueError("provider returned no parseable candidate")


def pair_decoys(cases: Iterable[CausalCase]) -> dict[str, str]:
    """Pair each case with the next corpus case, wrapping prospectively."""
    ordered = list(cases)
    if len(ordered) < 2:
        raise ValueError("causal feedback v2 requires at least two cases for decoy pairing")
    ids = [case.case_id for case in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("case_id values must be unique")
    return {
        case.case_id: ordered[(index + 1) % len(ordered)].case_id
        for index, case in enumerate(ordered)
    }


def arm_order(case_id: str, replicate: int, version: str = EXPERIMENT_VERSION) -> tuple[str, ...]:
    """Deterministically vary arm order without consulting target outcomes."""
    ranked = sorted(
        ARMS,
        key=lambda arm: hashlib.sha256(
            f"{version}|{case_id}|{replicate}|{arm}".encode("utf-8")
        ).digest(),
    )
    return tuple(ranked)


def render_prompt(history: list[CausalObservation]) -> str:
    """One shared proposer prompt. Arm identity never appears in the prompt."""
    if history:
        rendered = "\n".join(
            f"{item.step}. candidate={item.candidate!r} | feedback={item.feedback_score:.8f}"
            for item in history
        )
    else:
        rendered = "(none yet)"
    return (
        "You are searching for an unknown semantic target.\n"
        "Generate exactly one concise semantic description.\n"
        "Previous attempts are shown only as candidate text plus scalar feedback; higher feedback is better.\n"
        "Use the history to choose the next candidate. Do not explain your reasoning.\n"
        f"Previous attempts:\n{rendered}\n"
        "Return exactly one description and no commentary."
    )


def feedback_for_arm(
    arm: str,
    *,
    true_score: float,
    decoy_score: float,
) -> float:
    if arm == "true_feedback":
        return float(true_score)
    if arm == "decoy_feedback":
        return float(decoy_score)
    if arm == "null_feedback":
        return 0.0
    raise ValueError(f"unknown causal arm: {arm}")


def run_causal_trace(
    *,
    case_id: str,
    arm: str,
    replicate: int,
    max_steps: int,
    generate: Callable[[str, int], str],
    score_true: Callable[[str], float],
    score_decoy: Callable[[str], float],
) -> CausalTrace:
    """Run one arm; only feedback numbers differ across otherwise identical arms."""
    if arm not in ARMS:
        raise ValueError(f"unknown causal arm: {arm}")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    trace = CausalTrace(case_id=case_id, arm=arm, replicate=replicate)
    for step in range(1, max_steps + 1):
        try:
            prompt = render_prompt(trace.observations)
            candidate = parse_one_candidate(generate(prompt, step))
            true_score = float(score_true(candidate))
            decoy_score = float(score_decoy(candidate))
            if not (math.isfinite(true_score) and math.isfinite(decoy_score)):
                raise ValueError("non-finite feedback score")
            trace.observations.append(
                CausalObservation(
                    step=step,
                    candidate=candidate,
                    true_target_score=true_score,
                    feedback_score=feedback_for_arm(
                        arm,
                        true_score=true_score,
                        decoy_score=decoy_score,
                    ),
                )
            )
        except Exception as error:
            trace.status = "invalid"
            trace.error_type = type(error).__name__
            trace.failure_step = step
            raise CausalExecutionError(trace, error, step) from error
    return trace


def best_so_far(trace: CausalTrace) -> list[float]:
    curve: list[float] = []
    best = float("-inf")
    for observation in trace.observations:
        best = max(best, observation.true_target_score)
        curve.append(best)
    return curve


def checkpoint_metrics(
    trace: CausalTrace, checkpoints: Iterable[int] = CHECKPOINTS
) -> list[dict[str, float | int]]:
    """Summarize reached prefixes only; checkpoints never trigger generation."""
    rows: list[dict[str, float | int]] = []
    for checkpoint in checkpoints:
        if len(trace.observations) < checkpoint:
            continue
        prefix = trace.observations[:checkpoint]
        scores = [item.true_target_score for item in prefix]
        curve: list[float] = []
        best = float("-inf")
        improvements = 0
        previous_best: float | None = None
        for value in scores:
            best = max(best, value)
            curve.append(best)
            if previous_best is None or best > previous_best:
                improvements += 1
            previous_best = best
        if len(curve) == 1:
            auc = curve[0]
        else:
            auc = sum(
                (curve[index - 1] + curve[index]) / 2.0
                for index in range(1, len(curve))
            )
        ordered = sorted(scores)
        midpoint = len(ordered) // 2
        median = (
            ordered[midpoint]
            if len(ordered) % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
        )
        rows.append(
            {
                "checkpoint": checkpoint,
                "n_observed": len(prefix),
                "best_true_target_score": max(scores),
                "mean_true_target_score": sum(scores) / len(scores),
                "median_true_target_score": median,
                "auc_best_so_far_step": auc,
                "improvement_fraction": improvements / len(scores),
            }
        )
    return rows


def trace_to_dict(trace: CausalTrace) -> dict:
    return {
        "case_id": trace.case_id,
        "arm": trace.arm,
        "replicate": trace.replicate,
        "status": trace.status,
        "error_type": trace.error_type,
        "failure_step": trace.failure_step,
        "best_true_target_score": trace.best_true_score,
        "observations": [
            {
                "step": item.step,
                "candidate": item.candidate,
                "true_target_score": item.true_target_score,
                "feedback_score": item.feedback_score,
            }
            for item in trace.observations
        ],
        "best_so_far": best_so_far(trace),
        "checkpoint_metrics": checkpoint_metrics(trace),
    }
