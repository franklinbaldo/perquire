#!/usr/bin/env python3
"""Run semantic-inversion benchmark v1 with Perquire providers.

Hidden source text is used only to construct the target embedding. Candidate
proposers never receive it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from perquire.embeddings import embedding_registry
from perquire.embeddings.utils import cosine_similarity
from perquire.llm import provider_registry as llm_registry

from benchmarks.semantic_inversion import (
    BenchmarkCase,
    SearchTrace,
    adaptive_feedback_search,
    independent_best_of_n,
    mutation_hill_climber,
)

METHODS = ("independent_best_of_n", "mutation_hill_climber", "adaptive_perquire")


def load_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(BenchmarkCase(**json.loads(line)))
    return cases


_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def parse_candidates(content: str, count: int) -> list[str]:
    candidates: list[str] = []
    for raw in content.splitlines():
        text = _LIST_MARKER.sub("", raw.strip()).strip()
        if text and text not in candidates:
            candidates.append(text)
        if len(candidates) >= count:
            break
    if not candidates and content.strip():
        candidates.append(content.strip())
    return candidates[:count]


def trace_to_dict(
    case: BenchmarkCase, trace: SearchTrace, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    usage = usage or {}
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "method": trace.method,
        "best_target_similarity": trace.best.target_similarity if trace.best else None,
        "evaluations": len(trace.observations),
        "llm_calls": usage.get("llm_calls", 0),
        "llm_transport_attempts": usage.get("llm_transport_attempts", 0),
        "embedding_transport_attempts": usage.get("embedding_transport_attempts", 0),
        "generated_candidates": usage.get("generated_candidates", 0),
        "empty_generations": usage.get("empty_generations", 0),
        "observations": [
            {
                "step": obs.step,
                "candidate": obs.candidate,
                "target_similarity": obs.target_similarity,
            }
            for obs in trace.observations
        ],
    }


class ResourceMeter:
    """Track logical generation and real transport attempts per method."""

    def __init__(self) -> None:
        self.by_method: dict[str, dict[str, int]] = {}
        self.method = "unattributed"
        self._llm_start = 0
        self._embedding_start = 0

    @staticmethod
    def _transport_attempts(provider: Any) -> int:
        return int(getattr(provider, "transport_attempts", 0))

    def _bucket(self, method: str) -> dict[str, int]:
        return self.by_method.setdefault(
            method,
            {
                "llm_calls": 0,
                "llm_transport_attempts": 0,
                "embedding_transport_attempts": 0,
                "generated_candidates": 0,
                "empty_generations": 0,
            },
        )

    def start(self, method: str, *, llm: Any, embedder: Any) -> None:
        self.method = method
        self._bucket(method)
        self._llm_start = self._transport_attempts(llm)
        self._embedding_start = self._transport_attempts(embedder)

    def finish(self, *, llm: Any, embedder: Any) -> None:
        bucket = self._bucket(self.method)
        bucket["llm_transport_attempts"] += self._transport_attempts(llm) - self._llm_start
        bucket["embedding_transport_attempts"] += (
            self._transport_attempts(embedder) - self._embedding_start
        )

    def record_call(self) -> None:
        self._bucket(self.method)["llm_calls"] += 1

    def record_candidates(self, candidates: list[str]) -> None:
        bucket = self._bucket(self.method)
        bucket["generated_candidates"] += len(candidates)
        if not candidates:
            bucket["empty_generations"] += 1

    def record(self, candidates: list[str]) -> None:
        """Backward-compatible helper for tests/callers that record completed calls."""
        self.record_call()
        self.record_candidates(candidates)


class MethodExecutionError(RuntimeError):
    """A method cell failed after recording resource use observed so far."""

    def __init__(self, method: str, error: Exception, usage: dict[str, int]) -> None:
        super().__init__(f"{method} failed: {error}")
        self.method = method
        self.error = error
        self.usage = dict(usage)


def prepare_target(case: BenchmarkCase, embedder: Any) -> tuple[Any, int]:
    before = int(getattr(embedder, "transport_attempts", 0))
    target = embedder.embed_text(case.source_text).embedding
    attempts = int(getattr(embedder, "transport_attempts", 0)) - before
    return target, attempts


def run_method(
    case: BenchmarkCase,
    *,
    llm: Any,
    embedder: Any,
    budget: int,
    method: str,
    target: Any,
) -> tuple[SearchTrace, ResourceMeter]:
    """Run one frozen comparison method so failure is isolated to one method cell."""
    if method not in METHODS:
        raise ValueError(f"unknown semantic inversion method: {method}")

    meter = ResourceMeter()

    def score(candidate: str) -> float:
        candidate_embedding = embedder.embed_text(candidate).embedding
        return float(cosine_similarity(target, candidate_embedding))

    independent_call = 0

    def independent_generate(_prompt: str, count: int) -> list[str]:
        nonlocal independent_call
        candidates: list[str] = []
        max_calls = max(2, count)
        for _ in range(max_calls):
            remaining = count - len(candidates)
            if remaining <= 0:
                break
            prompt = (
                f"Generate {remaining} diverse, concise semantic descriptions. "
                "They must be independent guesses: do not assume any feedback about a hidden target. "
                "Return one description per line and no commentary."
            )
            request_id = f"{case.case_id}:independent_best_of_n:{independent_call}"
            independent_call += 1
            meter.record_call()
            response = llm.generate_response(prompt, cache_request_id=request_id)
            batch = parse_candidates(response.content, remaining)
            meter.record_candidates(batch)
            for candidate in batch:
                if candidate not in candidates:
                    candidates.append(candidate)
        if len(candidates) != count:
            raise RuntimeError(
                f"independent_best_of_n produced {len(candidates)} candidates for budget {count}"
            )
        return candidates

    mutation_call = 0

    def generate_hill_seed() -> str:
        prompt = (
            "Generate 1 concise semantic description as an independent initial guess. "
            "Do not assume any feedback about a hidden target. Return one description and no commentary."
        )
        meter.record_call()
        response = llm.generate_response(
            prompt,
            cache_request_id=f"{case.case_id}:mutation_hill_climber:seed",
        )
        candidates = parse_candidates(response.content, 1)
        meter.record_candidates(candidates)
        if len(candidates) != 1:
            raise RuntimeError("mutation_hill_climber failed to generate its initial seed")
        return candidates[0]

    def mutate(candidate: str, count: int) -> list[str]:
        nonlocal mutation_call
        prompt = (
            f"Produce {count} concise semantic variation of this candidate: {candidate!r}. "
            "Change its meaning enough to explore a nearby alternative. "
            "Return one candidate per line and no commentary."
        )
        request_id = f"{case.case_id}:mutation_hill_climber:{mutation_call}"
        mutation_call += 1
        meter.record_call()
        response = llm.generate_response(prompt, cache_request_id=request_id)
        candidates = parse_candidates(response.content, count)
        meter.record_candidates(candidates)
        if len(candidates) != count:
            raise RuntimeError(
                f"mutation_hill_climber produced {len(candidates)} candidates; expected {count}"
            )
        return candidates

    previous: list[str] = []

    def adaptive_propose(best: str | None, best_score: float | None, step: int) -> str:
        phase = "exploration" if step <= max(2, budget // 3) else "refinement"
        meter.record_call()
        questions = llm.generate_questions(
            current_description=best or "",
            target_similarity=best_score or 0.0,
            phase=phase,
            previous_questions=previous,
            cache_request_id=f"{case.case_id}:adaptive_perquire:{step}",
        )
        candidates = list(questions)
        meter.record_candidates(candidates)
        if not candidates:
            raise RuntimeError("adaptive_perquire received no candidate from provider")
        candidate = candidates[0]
        previous.append(candidate)
        return candidate

    meter.start(method, llm=llm, embedder=embedder)
    try:
        if method == "independent_best_of_n":
            trace = independent_best_of_n(generate=independent_generate, score=score, budget=budget)
        elif method == "mutation_hill_climber":
            trace = mutation_hill_climber(
                initial=generate_hill_seed(), mutate=mutate, score=score, budget=budget
            )
        else:
            trace = adaptive_feedback_search(propose=adaptive_propose, score=score, budget=budget)
    except Exception as error:
        meter.finish(llm=llm, embedder=embedder)
        raise MethodExecutionError(method, error, meter.by_method[method]) from error
    meter.finish(llm=llm, embedder=embedder)

    if len(trace.observations) != budget:
        raise MethodExecutionError(
            method,
            RuntimeError(f"{method} spent {len(trace.observations)} evaluations; expected {budget}"),
            meter.by_method[method],
        )
    return trace, meter


def run_case(
    case: BenchmarkCase, *, llm: Any, embedder: Any, budget: int
) -> tuple[list[SearchTrace], ResourceMeter, int]:
    """Compatibility runner: execute all frozen methods against one shared target."""
    target, target_attempts = prepare_target(case, embedder)
    traces: list[SearchTrace] = []
    combined = ResourceMeter()
    for method in METHODS:
        trace, meter = run_method(
            case, llm=llm, embedder=embedder, budget=budget, method=method, target=target
        )
        traces.append(trace)
        combined.by_method[method] = dict(meter.by_method[method])
    return traces, combined, target_attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/cases_v1.jsonl"))
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--embedding-provider", default="gemini")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/semantic_inversion_v1.json")
    )
    args = parser.parse_args()
    if args.budget < 1:
        raise SystemExit("--budget must be >= 1")

    llm = llm_registry.get_provider(args.llm_provider)
    embedder = embedding_registry.get_provider(args.embedding_provider)
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    records: list[dict[str, Any]] = []
    case_overheads: list[dict[str, Any]] = []
    for case in cases:
        traces, meter, target_attempts = run_case(case, llm=llm, embedder=embedder, budget=args.budget)
        records.extend(trace_to_dict(case, trace, meter.by_method.get(trace.method)) for trace in traces)
        case_overheads.append(
            {"case_id": case.case_id, "target_embedding_transport_attempts": target_attempts}
        )

    llm_info = llm.get_model_info() if hasattr(llm, "get_model_info") else {}
    embedding_info = embedder.get_model_info() if hasattr(embedder, "get_model_info") else {}
    payload = {
        "benchmark": "semantic-inversion-benchmark-v1",
        "budget": args.budget,
        "llm_provider": args.llm_provider,
        "embedding_provider": args.embedding_provider,
        "case_count": len(cases),
        "budget_unit": "target_similarity_evaluation",
        "case_overheads": case_overheads,
        "cache": {
            "llm_mode": llm_info.get("cache_mode"),
            "llm_hits": llm_info.get("cache_hits", 0),
            "llm_misses": llm_info.get("cache_misses", 0),
            "llm_writes": llm_info.get("cache_writes", 0),
            "embedding_hits": embedding_info.get("cache_hits", 0),
            "embedding_misses": embedding_info.get("cache_misses", 0),
            "embedding_writes": embedding_info.get("cache_writes", 0),
        },
        "notes": {
            "hill_climber_seed": (
                "mutation_hill_climber generates and pays for its own independent initial seed; "
                "the seed's generation and target-similarity evaluation are both attributed to that method."
            ),
            "transport_attempts": (
                "llm_transport_attempts and embedding_transport_attempts include bounded retries; "
                "llm_calls counts logical generation operations attempted by each method."
            ),
            "llm_replay": (
                "LLM replay cache keys include case/method/step logical identity plus model, prompt, "
                "temperature, and max_tokens. A hit replays an observed sample and is not a fresh draw."
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
