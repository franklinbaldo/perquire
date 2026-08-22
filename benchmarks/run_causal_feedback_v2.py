#!/usr/bin/env python3
"""Execute the preregistered causal-feedback v2 mechanism experiment.

No live run should begin until a generation substrate is frozen prospectively by
the target-free reliability program. The runner therefore requires explicit
model/failure-policy arguments instead of silently inheriting a moving default.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.causal_feedback_v2 import (
    ARMS,
    CHECKPOINTS,
    EXPERIMENT_VERSION,
    CausalCase,
    CausalExecutionError,
    arm_order,
    pair_decoys,
    run_causal_trace,
    trace_to_dict,
)
from perquire.embeddings.openrouter_embeddings import OpenRouterEmbeddingProvider
from perquire.embeddings.utils import cosine_similarity
from perquire.llm.openrouter_provider import OpenRouterProvider


DEFAULT_CASES = Path("benchmarks/cases_v1.jsonl")
DEFAULT_OUTPUT = Path("benchmark_results/causal_feedback_v2.json")
ENVIRONMENT_PACKAGES = ("perquire", "litellm", "numpy", "openai", "httpx", "pydantic")


def load_cases(path: Path) -> list[CausalCase]:
    cases: list[CausalCase] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            cases.append(CausalCase(**json.loads(raw)))
    return cases


def environment_manifest() -> dict[str, Any]:
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
    return {
        "git_sha": git_sha,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def checkpoint_row(record: dict[str, Any], checkpoint: int) -> dict[str, Any] | None:
    for row in record["checkpoint_metrics"]:
        if int(row["checkpoint"]) == checkpoint and int(row["n_observed"]) >= checkpoint:
            return row
    return None


def target_level_effects(records: list[dict[str, Any]], checkpoints: tuple[int, ...]) -> dict[str, Any]:
    """Aggregate replicates within target before computing paired arm effects."""
    result: dict[str, Any] = {}
    for checkpoint in checkpoints:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["status"] != "valid":
                continue
            row = checkpoint_row(record, checkpoint)
            if row is not None:
                grouped[(record["case_id"], record["arm"])].append(row)

        target_rows: list[dict[str, Any]] = []
        case_ids = sorted({case_id for case_id, _ in grouped})
        for case_id in case_ids:
            arm_values: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                rows = grouped.get((case_id, arm), [])
                if not rows:
                    break
                arm_values[arm] = {
                    "best": statistics.fmean(float(row["best_true_target_score"]) for row in rows),
                    "auc": statistics.fmean(float(row["auc_best_so_far_step"]) for row in rows),
                    "mean_candidate": statistics.fmean(float(row["mean_true_target_score"]) for row in rows),
                    "improvement_fraction": statistics.fmean(
                        float(row["improvement_fraction"]) for row in rows
                    ),
                }
            if len(arm_values) != len(ARMS):
                continue
            target_rows.append(
                {
                    "case_id": case_id,
                    "true_minus_decoy_best": arm_values["true_feedback"]["best"]
                    - arm_values["decoy_feedback"]["best"],
                    "true_minus_null_best": arm_values["true_feedback"]["best"]
                    - arm_values["null_feedback"]["best"],
                    "true_minus_decoy_auc": arm_values["true_feedback"]["auc"]
                    - arm_values["decoy_feedback"]["auc"],
                    "true_minus_null_auc": arm_values["true_feedback"]["auc"]
                    - arm_values["null_feedback"]["auc"],
                    "arm_values": arm_values,
                }
            )

        def summarize(field: str) -> dict[str, Any]:
            values = [float(row[field]) for row in target_rows]
            return {
                "n_targets": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "positive_fraction": (
                    sum(value > 0 for value in values) / len(values) if values else None
                ),
                "target_effects": values,
            }

        result[str(checkpoint)] = {
            "targets": target_rows,
            "true_minus_decoy_best": summarize("true_minus_decoy_best"),
            "true_minus_null_best": summarize("true_minus_null_best"),
            "true_minus_decoy_auc": summarize("true_minus_decoy_auc"),
            "true_minus_null_auc": summarize("true_minus_null_auc"),
            "inference_note": (
                "targets are the inferential units; provider replicates were averaged within target"
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--max-steps", type=int, choices=(16, 32), default=16)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--requests-per-minute", type=int, required=True)
    parser.add_argument("--max-retries", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.replicates < 1:
        raise SystemExit("--replicates must be >= 1")

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if len(cases) < 2:
        raise SystemExit("at least two cases are required for prospective decoy pairing")

    decoy_ids = pair_decoys(cases)
    by_id = {case.case_id: case for case in cases}

    llm = OpenRouterProvider(
        config={
            "model": args.llm_model,
            "requests_per_minute": args.requests_per_minute,
            "max_retries": args.max_retries,
            "cache_mode": "fresh",
        }
    )
    embedder = OpenRouterEmbeddingProvider(
        config={
            "model": args.embedding_model,
            "requests_per_minute": args.requests_per_minute,
            "max_retries": args.max_retries,
        }
    )

    target_vectors: dict[str, Any] = {}
    target_embedding_attempts: dict[str, int] = {}
    for case in cases:
        before = embedder.transport_attempts
        target_vectors[case.case_id] = embedder.embed_text(case.source_text).embedding
        target_embedding_attempts[case.case_id] = embedder.transport_attempts - before

    records: list[dict[str, Any]] = []
    execution_order: list[dict[str, Any]] = []

    for replicate in range(args.replicates):
        for case in cases:
            order = arm_order(case.case_id, replicate)
            execution_order.append(
                {"case_id": case.case_id, "replicate": replicate, "arms": list(order)}
            )
            true_target = target_vectors[case.case_id]
            decoy_case_id = decoy_ids[case.case_id]
            decoy_target = target_vectors[decoy_case_id]

            for arm in order:
                logical_calls = 0
                llm_transport_before = llm.transport_attempts
                embedding_transport_before = embedder.transport_attempts
                vector_cache: dict[str, Any] = {}

                def vector_for(candidate: str):
                    if candidate not in vector_cache:
                        vector_cache[candidate] = embedder.embed_text(candidate).embedding
                    return vector_cache[candidate]

                def score_true(candidate: str) -> float:
                    return float(cosine_similarity(true_target, vector_for(candidate)))

                def score_decoy(candidate: str) -> float:
                    return float(cosine_similarity(decoy_target, vector_for(candidate)))

                def generate(prompt: str, step: int) -> str:
                    nonlocal logical_calls
                    logical_calls += 1
                    request_id = (
                        f"{EXPERIMENT_VERSION}:{case.case_id}:replicate:{replicate}:"
                        f"arm:{arm}:step:{step}"
                    )
                    return llm.generate_response(prompt, cache_request_id=request_id).content

                try:
                    trace = run_causal_trace(
                        case_id=case.case_id,
                        arm=arm,
                        replicate=replicate,
                        max_steps=args.max_steps,
                        generate=generate,
                        score_true=score_true,
                        score_decoy=score_decoy,
                    )
                except CausalExecutionError as failure:
                    trace = failure.trace

                record = trace_to_dict(trace)
                record.update(
                    {
                        "domain": case.domain,
                        "decoy_case_id": decoy_case_id,
                        "logical_llm_calls": logical_calls,
                        "llm_transport_attempts": llm.transport_attempts - llm_transport_before,
                        "embedding_transport_attempts": (
                            embedder.transport_attempts - embedding_transport_before
                        ),
                    }
                )
                records.append(record)

    checkpoints = tuple(value for value in CHECKPOINTS if value <= args.max_steps)
    valid = sum(record["status"] == "valid" for record in records)
    payload = {
        "experiment": EXPERIMENT_VERSION,
        "preregistration": "docs/experiments/causal_feedback_v2_preregistration.md",
        "environment": environment_manifest(),
        "cases": [case.case_id for case in cases],
        "case_count": len(cases),
        "decoy_pairing": decoy_ids,
        "arms": list(ARMS),
        "execution_order": execution_order,
        "max_steps": args.max_steps,
        "checkpoints": list(checkpoints),
        "replicates": args.replicates,
        "inferential_unit": "target_case",
        "llm": llm.get_model_info(),
        "embedding": embedder.get_model_info(),
        "target_embedding_transport_attempts": target_embedding_attempts,
        "validity": {
            "expected_trajectories": len(cases) * args.replicates * len(ARMS),
            "valid_trajectories": valid,
            "invalid_trajectories": len(records) - valid,
            "minimum_validity_threshold": None,
            "note": "Gate-B threshold must be frozen with the generation substrate before live target scoring",
        },
        "target_level_effects": target_level_effects(records, checkpoints),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
