#!/usr/bin/env python3
"""Run the preregistered semantic-inversion scaling sweep.

Each (case, budget, replicate, method) is an experimental cell. Provider failure
invalidates only that cell; it never disappears and never aborts the remaining
sweep. Hidden source text remains available only to target construction.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import platform
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from perquire.embeddings import embedding_registry
from perquire.llm import provider_registry as llm_registry

from benchmarks.run_semantic_inversion import (
    METHODS,
    MethodExecutionError,
    load_cases,
    prepare_target,
    run_method,
    trace_to_dict,
)
from benchmarks.semantic_inversion import BenchmarkCase

PREREGISTERED_BUDGETS = (2, 4, 8, 16, 32)
MIN_VALID_FRACTION = 0.95
ENVIRONMENT_PACKAGES = (
    "perquire",
    "litellm",
    "numpy",
    "openai",
    "httpx",
    "httpcore",
    "pydantic",
    "pydantic-core",
    "pydantic-settings",
    "aiohttp",
    "tiktoken",
    "tokenizers",
)


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


def environment_manifest(embedding_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record software provenance plus observable remote embedding routing metadata."""
    git_sha = os.getenv("GITHUB_SHA")
    if not git_sha:
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_sha = None

    packages: dict[str, str | None] = {}
    for package in ENVIRONMENT_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    manifest: dict[str, Any] = {
        "git_sha": git_sha,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }
    if embedding_info:
        manifest["embedding_transport"] = {
            "router": embedding_info.get("provider"),
            "model": embedding_info.get("model"),
            "served_upstream_providers": embedding_info.get("served_upstream_providers", []),
            "served_upstream_provider_counts": embedding_info.get(
                "served_upstream_provider_counts", {}
            ),
            "successful_responses_without_upstream_provider": embedding_info.get(
                "successful_responses_without_upstream_provider", 0
            ),
            "upstream_provider_note": (
                "best-effort response metadata only; null/empty means the OpenRouter/LiteLLM "
                "embedding response did not expose the served upstream, not that routing was fixed"
            ),
        }
    return manifest


def best_so_far(observations: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    best = -math.inf
    curve: list[dict[str, float | int]] = []
    for observation in observations:
        best = max(best, float(observation["target_similarity"]))
        curve.append({"step": int(observation["step"]), "best_target_similarity": best})
    return curve


def summarize(records: list[dict[str, Any]], budgets: tuple[int, ...]) -> dict[str, Any]:
    """Summarize valid observations only; invalid cells remain explicit outside curves."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        if record.get("status", "valid") != "valid":
            continue
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
                "n_valid": len(values),
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
            "summaries exclude invalid cells; validity counts are mandatory context and raw paired records remain canonical"
        ),
    }


def _usage_fields(usage: dict[str, int] | None = None) -> dict[str, int]:
    usage = usage or {}
    return {
        "llm_calls": usage.get("llm_calls", 0),
        "llm_transport_attempts": usage.get("llm_transport_attempts", 0),
        "embedding_transport_attempts": usage.get("embedding_transport_attempts", 0),
        "generated_candidates": usage.get("generated_candidates", 0),
        "empty_generations": usage.get("empty_generations", 0),
    }


def invalid_record(
    *,
    case: BenchmarkCase,
    replay_case_id: str,
    budget: int,
    replicate: int,
    method: str,
    error: Exception,
    stage: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Serialize only stable error class/stage, never provider payload or secret-bearing text."""
    return {
        "case_id": case.case_id,
        "replay_case_id": replay_case_id,
        "domain": case.domain,
        "budget": budget,
        "replicate": replicate,
        "method": method,
        "status": "invalid",
        "error_stage": stage,
        "error_type": type(error).__name__,
        "best_target_similarity": None,
        "evaluations": 0,
        **_usage_fields(usage),
        "observations": [],
        "best_so_far": [],
    }


def execute_sweep(
    *,
    cases: list[BenchmarkCase],
    budgets: tuple[int, ...],
    replicates: int,
    llm: Any,
    embedder: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute every cell independently and preserve failures as first-class records."""
    records: list[dict[str, Any]] = []
    case_overheads: list[dict[str, Any]] = []

    for budget in budgets:
        for replicate in range(replicates):
            for case in cases:
                replay_case = BenchmarkCase(
                    case_id=f"{case.case_id}:budget:{budget}:replicate:{replicate}",
                    domain=case.domain,
                    source_text=case.source_text,
                )
                target_before = int(getattr(embedder, "transport_attempts", 0))
                try:
                    target, target_attempts = prepare_target(replay_case, embedder)
                except Exception as error:
                    target_attempts = int(getattr(embedder, "transport_attempts", 0)) - target_before
                    case_overheads.append(
                        {
                            "case_id": case.case_id,
                            "budget": budget,
                            "replicate": replicate,
                            "target_embedding_transport_attempts": target_attempts,
                            "status": "invalid",
                            "error_type": type(error).__name__,
                        }
                    )
                    for method in METHODS:
                        records.append(
                            invalid_record(
                                case=case,
                                replay_case_id=replay_case.case_id,
                                budget=budget,
                                replicate=replicate,
                                method=method,
                                error=error,
                                stage="target_embedding",
                            )
                        )
                    continue

                case_overheads.append(
                    {
                        "case_id": case.case_id,
                        "budget": budget,
                        "replicate": replicate,
                        "target_embedding_transport_attempts": target_attempts,
                        "status": "valid",
                    }
                )
                for method in METHODS:
                    try:
                        trace, meter = run_method(
                            replay_case,
                            llm=llm,
                            embedder=embedder,
                            budget=budget,
                            method=method,
                            target=target,
                        )
                    except MethodExecutionError as failure:
                        records.append(
                            invalid_record(
                                case=case,
                                replay_case_id=replay_case.case_id,
                                budget=budget,
                                replicate=replicate,
                                method=method,
                                error=failure.error,
                                stage="method",
                                usage=failure.usage,
                            )
                        )
                        continue

                    record = trace_to_dict(
                        replay_case,
                        trace,
                        meter.by_method.get(trace.method),
                    )
                    record["replay_case_id"] = record["case_id"]
                    record["case_id"] = case.case_id
                    record["budget"] = budget
                    record["replicate"] = replicate
                    record["status"] = "valid"
                    record["best_so_far"] = best_so_far(record["observations"])
                    records.append(record)

    return records, case_overheads


def validity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(record.get("status") == "valid" for record in records)
    total = len(records)
    invalid = total - valid
    fraction = valid / total if total else 0.0
    return {
        "expected_cells": total,
        "valid_cells": valid,
        "invalid_cells": invalid,
        "valid_fraction": fraction,
        "minimum_valid_fraction": MIN_VALID_FRACTION,
        "clean_v1_claim_eligible": fraction >= MIN_VALID_FRACTION,
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

    records, case_overheads = execute_sweep(
        cases=cases,
        budgets=budgets,
        replicates=args.replicates,
        llm=llm,
        embedder=embedder,
    )
    validity = validity_summary(records)

    llm_info = llm.get_model_info() if hasattr(llm, "get_model_info") else {}
    embedding_info = embedder.get_model_info() if hasattr(embedder, "get_model_info") else {}
    payload = {
        "benchmark": "semantic-inversion-scaling-v1",
        "environment": environment_manifest(embedding_info),
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
            "each case/budget/replicate/method is a frozen cell; replay of the same identity is not a fresh cell"
        ),
        "failure_policy": (
            "an exhausted method cell is invalid and remains in the artifact; it does not abort or get silently rerun"
        ),
        "validity": validity,
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
