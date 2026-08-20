#!/usr/bin/env python3
"""Merge independently executed scaling-v1 shard artifacts.

The merger performs no scoring or model calls. It checks that shard metadata
matches the frozen experiment, combines canonical cell records, and recomputes
validity/curve summaries from those records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.run_scaling_sweep import PREREGISTERED_BUDGETS, summarize, validity_summary


def load_shards(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("*.json"))
    if not paths:
        raise ValueError(f"no scaling shard JSON files found under {root}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def merge_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one shard is required")

    first = shards[0]
    invariant_keys = (
        "benchmark",
        "llm_provider",
        "llm_model",
        "embedding_provider",
        "embedding_model",
        "generation_config",
        "replicates",
        "budget_unit",
    )
    for shard in shards[1:]:
        for key in invariant_keys:
            if shard.get(key) != first.get(key):
                raise ValueError(f"shard metadata mismatch for {key}")

    records: list[dict[str, Any]] = []
    overheads: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, int, int, str]] = set()
    cache_totals = {
        "llm_hits": 0,
        "llm_misses": 0,
        "llm_writes": 0,
        "embedding_hits": 0,
        "embedding_misses": 0,
        "embedding_writes": 0,
    }

    for shard in shards:
        records.extend(shard.get("records", []))
        overheads.extend(shard.get("case_overheads", []))
        for key in cache_totals:
            cache_totals[key] += int(shard.get("cache", {}).get(key, 0) or 0)

    for record in records:
        identity = (
            str(record["case_id"]),
            int(record["budget"]),
            int(record["replicate"]),
            str(record["method"]),
        )
        if identity in seen_cells:
            raise ValueError(f"duplicate scaling cell: {identity}")
        seen_cells.add(identity)

    budgets = tuple(sorted({int(record["budget"]) for record in records}))
    case_ids = sorted({str(record["case_id"]) for record in records})
    validity = validity_summary(records)
    expected = len(case_ids) * len(budgets) * int(first["replicates"]) * 3
    validity["expected_cells"] = expected
    validity["observed_cells"] = len(records)
    validity["missing_cells"] = max(0, expected - len(records))
    if validity["missing_cells"]:
        validity["clean_v1_claim_eligible"] = False

    return {
        "benchmark": first["benchmark"],
        "preregistered_budgets": list(PREREGISTERED_BUDGETS),
        "budgets": list(budgets),
        "replicates": first["replicates"],
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "llm_provider": first["llm_provider"],
        "llm_model": first["llm_model"],
        "embedding_provider": first["embedding_provider"],
        "embedding_model": first["embedding_model"],
        "generation_config": first["generation_config"],
        "budget_unit": first["budget_unit"],
        "replicate_semantics": first.get("replicate_semantics"),
        "failure_policy": first.get("failure_policy"),
        "validity": validity,
        "case_overheads": overheads,
        "cache": {"aggregation": "sum_of_shard_counters", **cache_totals},
        "summary": summarize(records, budgets),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = merge_shards(load_shards(args.root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
