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


def load_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cases.append(BenchmarkCase(**item))
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
        bucket["llm_transport_attempts"] += (
            self._transport_attempts(llm) - self._llm_start
        )
        bucket["embedding_transport_attempts"] += (
            self._transport_attempts(embedder) - self._embedding_start
        )

    def record(self, candidates: list[str]) -> None:
        bucket = self._bucket(self.method)
        bucket["llm_calls"] += 1
        bucket["generated_candidates"] += len(candidates)
        if not candidates:
            bucket["empty_generations"] += 1


def run_case(
    case: BenchmarkCase, *, llm: Any, embedder: Any, budget: int
) -> tuple[list[SearchTrace], ResourceMeter, int]:
    target_before = int(getattr(embedder, "transport_attempts", 0))
    target = embedder.embed_text(case.source_text).embedding
    target_embedding_transport_attempts = (
        int(getattr(embedder, "transport_attempts", 0)) - target_before
    )
    meter = ResourceMeter()

    def score(candidate: str) -> float:
        candidate_embedding = embedder.embed_text(candidate).embedding
        return float(cosine_similarity(target, candidate_embedding))

    def independent_generate(_prompt: str, count: int) -> list[str]:
        candidates: list[str] = []
        # A single model response is usually enough, but partial formatting must
        # not silently shrink the target-similarity budget. Extra logical calls
        # are allowed and explicitly metered.
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
            response = llm.generate_response(prompt)
            batch = parse_candidates(response.content, remaining)
            meter.record(batch)
            for candidate in batch:
                if candidate not in candidates:
                    candidates.append(candidate)
        if len(candidates) != count:
            raise RuntimeError(
                f"independent_best_of_n produced {len(candidates)} candidates for budget {count}"
            )
        return candidates

    def mutate(candidate: str, count: int) -> list[str]:
        prompt = (
            f"Produce {count} concise semantic variation of this candidate: {candidate!r}. "
            "Change its meaning enough to explore a nearby alternative. "
            "Return one candidate per line and no commentary."
        )
        response = llm.generate_response(prompt)
        candidates = parse_candidates(response.content, count)
        meter.record(candidates)
        if len(candidates) != count:
            raise RuntimeError(
                f"mutation_hill_climber produced {len(candidates)} candidates; expected {count}"
            )
        return candidates

    previous: list[str] = []

    def adaptive_propose(best: str | None, best_score: float | None, step: int) -> str:
        phase = "exploration" if step <= max(2, budget // 3) else "refinement"
        questions = llm.generate_questions(
            current_description=best or "",
            target_similarity=best_score or 0.0,
            phase=phase,
            previous_questions=previous,
        )
        meter.record(list(questions))
        if not questions:
            raise RuntimeError("adaptive_perquire received no candidate from provider")
        candidate = questions[0]
        previous.append(candidate)
        return candidate

    meter.start("independent_best_of_n", llm=llm, embedder=embedder)
    independent = independent_best_of_n(
        generate=independent_generate,
        score=score,
        budget=budget,
    )
    meter.finish(llm=llm, embedder=embedder)

    initial = independent.observations[0].candidate
    meter.start("mutation_hill_climber", llm=llm, embedder=embedder)
    hill = mutation_hill_climber(
        initial=initial,
        mutate=mutate,
        score=score,
        budget=budget,
    )
    meter.finish(llm=llm, embedder=embedder)

    meter.start("adaptive_perquire", llm=llm, embedder=embedder)
    adaptive = adaptive_feedback_search(
        propose=adaptive_propose,
        score=score,
        budget=budget,
    )
    meter.finish(llm=llm, embedder=embedder)

    traces = [independent, hill, adaptive]
    for trace in traces:
        if len(trace.observations) != budget:
            raise RuntimeError(
                f"{trace.method} spent {len(trace.observations)} evaluations; expected {budget}"
            )
    return traces, meter, target_embedding_transport_attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/cases_v1.jsonl"))
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--embedding-provider", default="gemini")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/semantic_inversion_v1.json"))
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
        traces, meter, target_attempts = run_case(
            case, llm=llm, embedder=embedder, budget=args.budget
        )
        records.extend(
            trace_to_dict(case, trace, meter.by_method.get(trace.method)) for trace in traces
        )
        case_overheads.append(
            {
                "case_id": case.case_id,
                "target_embedding_transport_attempts": target_attempts,
            }
        )

    payload = {
        "benchmark": "semantic-inversion-benchmark-v1",
        "budget": args.budget,
        "llm_provider": args.llm_provider,
        "embedding_provider": args.embedding_provider,
        "case_count": len(cases),
        "budget_unit": "target_similarity_evaluation",
        "case_overheads": case_overheads,
        "notes": {
            "hill_climber_seed": (
                "mutation_hill_climber starts from the first independent_best_of_n candidate; "
                "that candidate's generation cost is attributed to independent_best_of_n, "
                "while its evaluation is charged to mutation_hill_climber."
            ),
            "transport_attempts": (
                "llm_transport_attempts and embedding_transport_attempts include bounded retries; "
                "llm_calls counts logical generation operations requested by each method."
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
