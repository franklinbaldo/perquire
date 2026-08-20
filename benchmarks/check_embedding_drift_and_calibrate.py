#!/usr/bin/env python3
"""Check cross-era embedding drift before calibrating semantic similarity.

The drift control compares one vector already present in the persistent cache with
a fresh embedding of the exact same text. Only if that control passes do we
compare cached v1 targets with fresh, fixed paraphrases. Fresh diagnostic vectors
bypass cache reads and writes so this script cannot overwrite the frozen targets
it is testing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.run_semantic_inversion import load_cases
from perquire.embeddings.openrouter_embeddings import OpenRouterEmbeddingProvider
from perquire.embeddings.utils import cosine_similarity

DEFAULT_PARAPHRASES = Path("benchmarks/calibration_paraphrases_v1.jsonl")
DEFAULT_CASES = Path("benchmarks/cases_v1.jsonl")
DEFAULT_OUTPUT = Path("benchmark_results/embedding_drift_calibration_v1.json")
DEFAULT_MIN_CONTROL_COSINE = 0.999999


def load_paraphrases(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row["case_id"])
            paraphrase = str(row["paraphrase"])
            if case_id in rows:
                raise ValueError(f"duplicate paraphrase for {case_id} on line {line_number}")
            rows[case_id] = paraphrase
    return rows


def provider_identity(result: Any) -> str | None:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("upstream_provider")
        return str(value) if value is not None else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--paraphrases", type=Path, default=DEFAULT_PARAPHRASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-control-cosine",
        type=float,
        default=DEFAULT_MIN_CONTROL_COSINE,
        help="minimum same-text cached-vs-fresh cosine required before calibration",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    paraphrases = load_paraphrases(args.paraphrases)
    case_ids = {case.case_id for case in cases}
    if set(paraphrases) != case_ids:
        missing = sorted(case_ids - set(paraphrases))
        extra = sorted(set(paraphrases) - case_ids)
        raise SystemExit(f"paraphrase coverage mismatch: missing={missing}, extra={extra}")

    embedder = OpenRouterEmbeddingProvider(config={})
    cached_targets: dict[str, np.ndarray] = {}
    missing_cached_targets: list[str] = []
    for case in cases:
        cached = embedder.cached_embedding(case.source_text)
        if cached is None:
            missing_cached_targets.append(case.case_id)
        else:
            cached_targets[case.case_id] = cached

    if not cached_targets:
        raise SystemExit("no v1 source target is present in the restored embedding cache")

    control_case = next(case for case in cases if case.case_id in cached_targets)
    cached_control = cached_targets[control_case.case_id]
    fresh_control_result = embedder.embed_text_uncached(control_case.source_text)
    fresh_control = fresh_control_result.embedding
    control_cosine = cosine_similarity(cached_control, fresh_control)
    max_abs_delta = float(np.max(np.abs(cached_control - fresh_control)))
    exact_equal = bool(np.array_equal(cached_control, fresh_control))
    control_passed = control_cosine >= args.min_control_cosine

    payload: dict[str, Any] = {
        "diagnostic": "embedding-drift-and-paraphrase-calibration-v1",
        "embedding_model": embedder.model,
        "cache_path": str(embedder._cache.path),
        "cached_target_count": len(cached_targets),
        "missing_cached_target_count": len(missing_cached_targets),
        "missing_cached_target_case_ids": missing_cached_targets,
        "control": {
            "case_id": control_case.case_id,
            "cached_vs_fresh_cosine": control_cosine,
            "min_required_cosine": args.min_control_cosine,
            "passed": control_passed,
            "exact_vector_equal": exact_equal,
            "max_abs_component_delta": max_abs_delta,
            "fresh_upstream_provider": provider_identity(fresh_control_result),
        },
        "calibration": [],
        "calibration_summary": None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not control_passed:
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(args.output)
        raise SystemExit(
            f"embedding drift control failed: cosine={control_cosine:.12f} "
            f"< {args.min_control_cosine:.12f}; calibration intentionally not run"
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        target = cached_targets.get(case.case_id)
        if target is None:
            continue
        fresh = embedder.embed_text_uncached(paraphrases[case.case_id])
        similarity = cosine_similarity(target, fresh.embedding)
        rows.append(
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "target_origin": "restored_cache",
                "paraphrase_origin": "versioned_fixed_paraphrase",
                "target_paraphrase_cosine": similarity,
                "fresh_upstream_provider": provider_identity(fresh),
            }
        )

    values = [float(row["target_paraphrase_cosine"]) for row in rows]
    payload["calibration"] = rows
    if values:
        payload["calibration_summary"] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    payload["embedding_transport"] = embedder.get_model_info()
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
