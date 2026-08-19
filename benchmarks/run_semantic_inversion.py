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


# Only an actual list marker is removed. A character-class strip would also eat
# the leading digits of a real candidate such as "1980s synthpop revival",
# silently changing the text that gets embedded and scored.
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
    """Track non-evaluation resources so equal-budget claims stay auditable.

    The contract fixes the budget in target-similarity evaluations only. LLM
    calls and generated candidates are counted per method so a win bought with
    extra generation compute cannot be reported as an adaptivity win.
    """

    def __init__(self) -> None:
        self.by_method: dict[str, dict[str, int]] = {}
        self.method = "unattributed"

    def start(self, method: str) -> None:
        self.method = method
        self.by_method.setdefault(
            method, {"llm_calls": 0, "generated_candidates": 0, "empty_generations": 0}
        )

    def record(self, candidates: list[str]) -> None:
        bucket = self.by_method.setdefault(
            self.method, {"llm_calls": 0, "generated_candidates": 0, "empty_generations": 0}
        )
        bucket["llm_calls"] += 1
        bucket["generated_candidates"] += len(candidates)
        if not candidates:
            bucket["empty_generations"] += 1


def run_case(
    case: BenchmarkCase, *, llm: Any, embedder: Any, budget: int
) -> tuple[list[SearchTrace], ResourceMeter]:
    target = embedder.embed_text(case.source_text).embedding
    meter = ResourceMeter()

    def score(candidate: str) -> float:
        candidate_embedding = embedder.embed_text(candidate).embedding
        return float(cosine_similarity(target, candidate_embedding))

    def independent_generate(_prompt: str, count: int) -> list[str]:
        prompt = (
            f"Generate {count} diverse, concise semantic descriptions. "
            "They must be independent guesses: do not assume any feedback about a hidden target. "
            "Return one description per line and no commentary."
        )
        response = llm.generate_response(prompt)
        candidates = parse_candidates(response.content, count)
        meter.record(candidates)
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
        candidate = questions[0] if questions else "A general semantic concept"
        meter.record(list(questions))
        previous.append(candidate)
        return candidate

    meter.start("independent_best_of_n")
    independent = independent_best_of_n(
        generate=independent_generate,
        score=score,
        budget=budget,
    )
    initial = independent.observations[0].candidate if independent.observations else "A general concept"
    meter.start("mutation_hill_climber")
    hill = mutation_hill_climber(
        initial=initial,
        mutate=mutate,
        score=score,
        budget=budget,
    )
    meter.start("adaptive_perquire")
    adaptive = adaptive_feedback_search(
        propose=adaptive_propose,
        score=score,
        budget=budget,
    )
    return [independent, hill, adaptive], meter


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
    for case in cases:
        traces, meter = run_case(case, llm=llm, embedder=embedder, budget=args.budget)
        records.extend(
            trace_to_dict(case, trace, meter.by_method.get(trace.method)) for trace in traces
        )

    payload = {
        "benchmark": "semantic-inversion-benchmark-v1",
        "budget": args.budget,
        "llm_provider": args.llm_provider,
        "embedding_provider": args.embedding_provider,
        "case_count": len(cases),
        "budget_unit": "target_similarity_evaluation",
        "notes": {
            "hill_climber_seed": (
                "mutation_hill_climber starts from the first independent_best_of_n candidate; "
                "that candidate's generation cost is attributed to independent_best_of_n, "
                "while its evaluation is charged to mutation_hill_climber."
            )
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
