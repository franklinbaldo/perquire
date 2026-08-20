#!/usr/bin/env python3
"""Run the preregistered semantic-inversion scaling sweep.

Each (case, budget, replicate) receives a distinct replay identity so cached
provider observations cannot masquerade as fresh stochastic replicates. The
hidden source text remains available only to target-embedding construction.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from perquire.embeddings import embedding_registry
from perquire.llm import provider_registry as llm_registry

from benchmarks.run_semantic_inversion import load_cases, run_case, trace_to_dict
from benchmarks.semantic_inversion import BenchmarkCase

PREREGISTERED_BUDGETS = (2, 4, 8, 16, 32)


def parse_budgets(raw: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from error
    if not budgets or any(budget < 1 for budget in budgets):
        raise argparse.ArgumentTypeError("budgets must contain positive integers")
    if len(set(budgets)) != len(budgets):
        raise argparse.ArgumentTypeError("budgets must be unique")
    return tuple(sorted(budgets))


def best_so_far(observations: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    best = -math.inf
    curve: list[dict[str, float | int]] = []
    for observation in observations:
        best = max(best, float(observation["target_similarity"]))
        curve.append({"step": int(observation["step"]), "best_target_similarity": best})
    return curve


def summarize(records: list[dict[str, Any]], budgets: tuple[int, ...]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], int(record["budget"]))].append(
            float(record["best_target_similarity"])
        )

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (method, budget), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        mean = statistics.fmean(values)
        median = statistics.median(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        stderr = stdev / math.sqrt(len(values)) if values else 0.0
        by_method[method].append(
            {
                "budget": budget,
                "n": len(values),
                "mean_best_target_similarity": mean,
                "median_best_target_similarity": median,
                "stdev_best_target_similarity": stdev,
                "stderr_best_target_similarity": stderr,
            }
        )

    method_summaries: dict[str, Any] = {}
    for method, points in by_method.items():
        previous_mean: float | None = None
        previous_budget: int | None = None
        auc = 0.0
        enriched: list[dict[str, Any]] = []
        for point in points:
            current = dict(point)
            if previous_mean is None:
                current["delta_mean_from_previous_budget"] = None
                current["previous_budget"] = None
            else:
                current["delta_mean_from_previous_budget"] = (
                    current["mean_best_target_similarity"] - previous_mean
                )
                current["previous_budget"] = previous_budget
                x0 = math.log2(previous_budget)
                x1 = math.log2(current["budget"])
                auc += (x1 - x0) * (
                    previous_mean + current["mean_best_target_similarity"]
                ) / 2.0
            previous_mean = current["mean_best_target_similarity"]
            previous_budget = current["budget"]
            enriched.append(current)
        method_summaries[method] = {
            "points": enriched,
            "auc_mean_best_vs_log2_budget": auc,
        }

    return {
        "budgets": list(budgets),
        "methods": method_summaries,
        "uncertainty_note": (
            "stdev/stderr summarize replicate-and-case dispersion; raw paired records remain canonical"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/cases_v1.jsonl"))
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--embedding-provider", default="gemini")
    parser.add_argument(
        "--budgets",
        type=parse_budgets,
        default=PREREGISTERED_BUDGETS,
        help="comma-separated evaluation budgets",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/semantic_inversion_scaling_v1.json"),
    )
    args = parser.parse_args()

    if args.replicates < 1:
        raise SystemExit("--replicates must be >= 1")

    budgets = tuple(args.budgets)
    llm = llm_registry.get_provider(args.llm_provider)
    embedder = embedding_registry.get_provider(args.embedding_provider)
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    records: list[dict[str, Any]] = []
    case_overheads: list[dict[str, Any]] = []
    for budget in budgets:
        for replicate in range(args.replicates):
            for case in cases:
                replay_case = BenchmarkCase(
                    case_id=f"{case.case_id}:budget:{budget}:replicate:{replicate}",
                    domain=case.domain,
                    source_text=case.source_text,
                )
                traces, meter, target_attempts = run_case(
                    replay_case,
                    llm=llm,
                    embedder=embedder,
                    budget=budget,
                )
                case_overheads.append(
                    {
                        "case_id": case.case_id,
                        "budget": budget,
                        "replicate": replicate,
                        "target_embedding_transport_attempts": target_attempts,
                    }
                )
                for trace in traces:
                    record = trace_to_dict(
                        replay_case,
                        trace,
                        meter.by_method.get(trace.method),
                    )
                    record["replay_case_id"] = record["case_id"]
                    record["case_id"] = case.case_id
                    record["budget"] = budget
                    record["replicate"] = replicate
                    record["best_so_far"] = best_so_far(record["observations"])
                    records.append(record)

    llm_info = llm.get_model_info() if hasattr(llm, "get_model_info") else {}
    embedding_info = embedder.get_model_info() if hasattr(embedder, "get_model_info") else {}
    payload = {
        "benchmark": "semantic-inversion-scaling-v1",
        "preregistered_budgets": list(PREREGISTERED_BUDGETS),
        "budgets": list(budgets),
        "replicates": args.replicates,
        "case_count": len(cases),
        "llm_provider": args.llm_provider,
        "llm_model": llm_info.get("model"),
        "embedding_provider": args.embedding_provider,
        "embedding_model": embedding_info.get("model"),
        "generation_config": {
            "requests_per_minute": llm_info.get("requests_per_minute"),
            "max_retries": llm_info.get("max_retries"),
        },
        "budget_unit": "target_similarity_evaluation",
        "replicate_semantics": (
            "each case/budget/replicate has a distinct LLM replay identity; cache hits replay only the exact same experimental cell"
        ),
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
        "summary": summarize(records, budgets),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
