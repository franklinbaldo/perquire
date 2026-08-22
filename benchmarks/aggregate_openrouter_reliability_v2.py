#!/usr/bin/env python3
"""Aggregate target-free OpenRouter observatory windows under the frozen v2 rule."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RULE_VERSION = "openrouter-reliability-freeze-v2"
EVIDENCE_AFTER = datetime(2026, 8, 22, 12, 30, tzinfo=UTC)
MIN_WINDOWS = 48
MIN_SPAN_HOURS = 24.0
MIN_OBSERVATION_CALLS = 480
MIN_CLEAN_WINDOW_FRACTION = 0.95
MIN_CALL_SUCCESS_RATE = 0.995
MAX_TRANSPORT_LOGICAL_RATIO = 1.01
MAX_FAILURES_PER_WINDOW = 1


def _parse_time(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("observed_at_utc must be timezone-aware")
    return value.astimezone(UTC)


def load_windows(paths: list[Path]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_source_path"] = str(path)
        windows.append(data)
    return windows


def _quality_for_selected(window: dict[str, Any], model: str) -> dict[str, float | None]:
    for row in window.get("qualification", []):
        if row.get("model") == model:
            quality = row.get("quality") or {}
            return {
                "intelligence_index": quality.get("intelligence_index"),
                "agentic_index": quality.get("agentic_index"),
                "coding_index": quality.get("coding_index"),
            }
    return {"intelligence_index": None, "agentic_index": None, "coding_index": None}


def summarize_candidate(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: _parse_time(item["observed_at_utc"]))
    first = _parse_time(ordered[0]["observed_at_utc"])
    last = _parse_time(ordered[-1]["observed_at_utc"])
    span_hours = (last - first).total_seconds() / 3600.0

    observation_calls = [call for row in ordered for call in row.get("observation_calls", [])]
    successes = sum(bool(call.get("success")) for call in observation_calls)
    failures = len(observation_calls) - successes
    transport_attempts = sum(int(call.get("transport_attempts", 0)) for call in observation_calls)
    logical_calls = len(observation_calls)

    window_failures = [
        sum(not bool(call.get("success")) for call in row.get("observation_calls", []))
        for row in ordered
    ]
    clean_windows = sum(count == 0 for count in window_failures)
    consecutive_failed_windows = any(
        left > 0 and right > 0
        for left, right in zip(window_failures, window_failures[1:], strict=False)
    )

    qualification_calls = [
        call
        for row in ordered
        for candidate in row.get("qualification", [])
        if candidate.get("model") == model
        for call in candidate.get("calls", [])
    ]
    qualification_successes = sum(bool(call.get("success")) for call in qualification_calls)

    success_rate = successes / logical_calls if logical_calls else 0.0
    clean_fraction = clean_windows / len(ordered) if ordered else 0.0
    transport_ratio = transport_attempts / logical_calls if logical_calls else float("inf")
    max_window_failures = max(window_failures, default=0)

    coverage_ok = (
        len(ordered) >= MIN_WINDOWS
        and span_hours >= MIN_SPAN_HOURS
        and logical_calls >= MIN_OBSERVATION_CALLS
    )
    reliability_ok = (
        clean_fraction >= MIN_CLEAN_WINDOW_FRACTION
        and success_rate >= MIN_CALL_SUCCESS_RATE
        and transport_ratio <= MAX_TRANSPORT_LOGICAL_RATIO
        and max_window_failures <= MAX_FAILURES_PER_WINDOW
        and not consecutive_failed_windows
    )

    latest_quality = _quality_for_selected(ordered[-1], model)
    return {
        "model": model,
        "selected_windows": len(ordered),
        "first_observed_at_utc": first.isoformat(),
        "last_observed_at_utc": last.isoformat(),
        "span_hours": span_hours,
        "observation_calls": logical_calls,
        "observation_successes": successes,
        "observation_failures": failures,
        "observation_success_rate": success_rate,
        "clean_windows": clean_windows,
        "clean_window_fraction": clean_fraction,
        "transport_attempts": transport_attempts,
        "transport_logical_ratio": transport_ratio,
        "max_failures_in_window": max_window_failures,
        "consecutive_failed_windows": consecutive_failed_windows,
        "qualification_calls": len(qualification_calls),
        "qualification_successes": qualification_successes,
        "latest_quality": latest_quality,
        "coverage_ok": coverage_ok,
        "reliability_ok": reliability_ok,
        "eligible_for_freeze_review": coverage_ok and reliability_ok,
        "identity_note": (
            "reliability eligibility is necessary but not sufficient; provider routing must still be explicitly constrained in the freeze record"
        ),
    }


def _score(value: Any) -> float:
    try:
        return float(value) if value is not None else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    quality = row["latest_quality"]
    return (
        -_score(quality.get("intelligence_index")),
        -_score(quality.get("agentic_index")),
        -_score(quality.get("coding_index")),
        -float(row["observation_success_rate"]),
        -float(row["clean_window_fraction"]),
        float(row["transport_logical_ratio"]),
        str(row["model"]),
    )


def aggregate(windows: list[dict[str, Any]]) -> dict[str, Any]:
    prospective: list[dict[str, Any]] = []
    excluded_pre_rule = 0
    for window in windows:
        timestamp = _parse_time(window["observed_at_utc"])
        if timestamp < EVIDENCE_AFTER:
            excluded_pre_rule += 1
            continue
        prospective.append(window)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in prospective:
        model = window.get("selected_model")
        if model:
            by_model[str(model)].append(window)

    candidates = [summarize_candidate(model, rows) for model, rows in sorted(by_model.items())]
    eligible = sorted(
        [row for row in candidates if row["eligible_for_freeze_review"]],
        key=selection_key,
    )

    if eligible:
        status = "eligible"
        selected_model = eligible[0]["model"]
    elif any(row["coverage_ok"] for row in candidates):
        status = "reliability_failure"
        selected_model = None
    else:
        status = "insufficient_coverage"
        selected_model = None

    return {
        "rule_version": RULE_VERSION,
        "evidence_after_utc": EVIDENCE_AFTER.isoformat(),
        "status": status,
        "selected_model_for_freeze_review": selected_model,
        "freeze_authorized": False,
        "freeze_authorization_note": (
            "Even an eligible model needs a separate versioned routing/configuration freeze record before Gate B"
        ),
        "input_windows": len(windows),
        "excluded_pre_rule_windows": excluded_pre_rule,
        "prospective_windows": len(prospective),
        "thresholds": {
            "min_selected_windows": MIN_WINDOWS,
            "min_span_hours": MIN_SPAN_HOURS,
            "min_observation_calls": MIN_OBSERVATION_CALLS,
            "min_clean_window_fraction": MIN_CLEAN_WINDOW_FRACTION,
            "min_call_success_rate": MIN_CALL_SUCCESS_RATE,
            "max_transport_logical_ratio": MAX_TRANSPORT_LOGICAL_RATIO,
            "max_failures_per_window": MAX_FAILURES_PER_WINDOW,
            "consecutive_failed_windows_allowed": False,
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(load_windows(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
