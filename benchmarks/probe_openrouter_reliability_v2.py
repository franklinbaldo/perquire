#!/usr/bin/env python3
"""Target-free longitudinal OpenRouter reliability observation for scaling v2."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

CATALOG_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_BASE_URL = "https://openrouter.ai"
DISCOVERY_CANDIDATES = 5
QUALIFICATION_CALLS_PER_MODEL = 2
OBSERVATION_CALLS = 10
MIN_MEAN_UPTIME_PERCENT = 99.5
MIN_ENDPOINT_UPTIME_PERCENT = 95.0
PROMPTS = (
    "Generate one concise semantic description of an unspecified topic. Return only the description.",
    "Rewrite this candidate into one nearby but meaningfully different semantic alternative: 'a public institution processes records'. Return only the alternative.",
    "A previous generic description received similarity 0.2400 to an unspecified hidden target. Propose one concise refinement without assuming the target text. Return only the refinement.",
    "Synthesize one concise description from these generic hints: public records; procedural state; time-sensitive work. Return only the description.",
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _zero_price(value: Any) -> bool:
    number = _number(value)
    return number == 0.0


def _is_free_text_model(model: dict[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    pricing = model.get("pricing") or {}
    architecture = model.get("architecture") or {}
    output_modalities = architecture.get("output_modalities") or []
    return (
        model_id.endswith(":free")
        and _zero_price(pricing.get("prompt"))
        and _zero_price(pricing.get("completion"))
        and "text" in output_modalities
    )


def _quality_metrics(model: dict[str, Any]) -> dict[str, float | None]:
    artificial_analysis = ((model.get("benchmarks") or {}).get("artificial_analysis") or {})
    return {
        "intelligence_index": _number(artificial_analysis.get("intelligence_index")),
        "agentic_index": _number(artificial_analysis.get("agentic_index")),
        "coding_index": _number(artificial_analysis.get("coding_index")),
    }


def _uptime_for_endpoint(endpoint: dict[str, Any]) -> tuple[float | None, str | None]:
    for field in ("uptime_last_5m", "uptime_last_30m", "uptime_last_1d"):
        if (value := _number(endpoint.get(field))) is not None:
            return value, field
    return None, None


def _health_summary(endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    operational = [endpoint for endpoint in endpoints if int(endpoint.get("status", 1)) == 0]
    uptime_rows = [
        (value, source)
        for endpoint in operational
        if (value_source := _uptime_for_endpoint(endpoint))[0] is not None
        for value, source in [value_source]
    ]
    uptime_values = [float(value) for value, _source in uptime_rows]
    throughput_p50 = [
        value
        for endpoint in operational
        if (value := _number((endpoint.get("throughput_last_30m") or {}).get("p50"))) is not None
    ]
    latency_p50 = [
        value
        for endpoint in operational
        if (value := _number((endpoint.get("latency_last_30m") or {}).get("p50"))) is not None
    ]
    mean_uptime = statistics.fmean(uptime_values) if uptime_values else None
    min_uptime = min(uptime_values) if uptime_values else None
    return {
        "endpoint_count": len(endpoints),
        "operational_endpoint_count": len(operational),
        "reported_uptime_endpoint_count": len(uptime_values),
        "uptime_sources": sorted({source for _value, source in uptime_rows if source}),
        "mean_uptime_percent": mean_uptime,
        "minimum_uptime_percent": min_uptime,
        "best_throughput_p50": max(throughput_p50) if throughput_p50 else None,
        "best_latency_p50": min(latency_p50) if latency_p50 else None,
        "health_eligible": (
            bool(operational)
            and len(uptime_values) == len(operational)
            and mean_uptime is not None
            and min_uptime is not None
            and mean_uptime >= MIN_MEAN_UPTIME_PERCENT
            and min_uptime >= MIN_ENDPOINT_UPTIME_PERCENT
        ),
    }


def _descending(value: float | None) -> float:
    return -(value if value is not None else -math.inf)


def _ascending(value: float | None) -> float:
    return value if value is not None else math.inf


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    quality = candidate["quality"]
    health = candidate["health"]
    return (
        _descending(quality["intelligence_index"]),
        _descending(quality["agentic_index"]),
        _descending(quality["coding_index"]),
        _descending(health["mean_uptime_percent"]),
        _descending(health["minimum_uptime_percent"]),
        _descending(health["best_throughput_p50"]),
        _ascending(health["best_latency_p50"]),
        str(candidate["model"]),
    )


def discover_free_models(api_key: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        CATALOG_URL,
        headers=headers,
        params={"output_modalities": "text"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    free_models = [model for model in data if _is_free_text_model(model)]

    checked: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for model in free_models:
        model_id = str(model["id"])
        details_path = str((model.get("links") or {}).get("details") or "")
        row: dict[str, Any] = {
            "model": model_id,
            "quality": _quality_metrics(model),
            "health": None,
            "endpoint_error": None,
        }
        try:
            if not details_path.startswith("/api/v1/models/"):
                raise RuntimeError(f"missing canonical endpoints link for {model_id}")
            endpoint_response = requests.get(
                f"{OPENROUTER_BASE_URL}{details_path}",
                headers=headers,
                timeout=10,
            )
            endpoint_response.raise_for_status()
            endpoints = endpoint_response.json().get("data", {}).get("endpoints", [])
            row["health"] = _health_summary(endpoints)
            if row["health"]["health_eligible"]:
                eligible.append(row)
        except Exception as exc:
            row["endpoint_error"] = f"{type(exc).__name__}: {exc}"
        checked.append(row)

    eligible.sort(key=_candidate_sort_key)
    candidates = eligible[:DISCOVERY_CANDIDATES]
    return {
        "catalog_url": CATALOG_URL,
        "catalog_model_count": len(data),
        "catalog_free_text_model_count": len(free_models),
        "minimum_mean_uptime_percent": MIN_MEAN_UPTIME_PERCENT,
        "minimum_endpoint_uptime_percent": MIN_ENDPOINT_UPTIME_PERCENT,
        "selection_rule": (
            "query current OpenRouter catalog at runtime; filter :free + zero prompt/completion price + text output; "
            "fetch each model's canonical Endpoints API record; for every operational endpoint use 5m uptime, "
            "falling back to 30m then 1d only when unavailable; require complete uptime coverage, route mean >=99.5% "
            "and every endpoint >=95%; rank healthy models by OpenRouter Artificial Analysis intelligence, then "
            "agentic, then coding index; qualify the top five with two target-free calls each; choose the "
            "highest-capability candidate that passes 2/2 on our account"
        ),
        "checked_models": checked,
        "candidates": candidates,
    }


def probe_call(model: str, prompt: str) -> dict[str, Any]:
    from perquire.llm.openrouter_provider import OpenRouterProvider

    provider = OpenRouterProvider(
        config={
            "model": model,
            "temperature": 0.7,
            "max_tokens": 64,
            "max_retries": 0,
            "cache_mode": "off",
        }
    )
    started = time.perf_counter()
    error: str | None = None
    content = ""
    try:
        response = provider.generate_response(prompt)
        content = response.content.strip()
        if not content:
            raise RuntimeError("empty content after provider success")
    except Exception as exc:  # evidence artifact must preserve failures
        error = f"{type(exc).__name__}: {exc}"
    return {
        "model": model,
        "success": error is None,
        "transport_attempts": provider.transport_attempts,
        "elapsed_seconds": time.perf_counter() - started,
        "output_chars": len(content),
        "error": error,
    }


def qualify_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    qualification: list[dict[str, Any]] = []
    selected_model: str | None = None
    for candidate_index, candidate in enumerate(candidates):
        model = str(candidate["model"])
        rows: list[dict[str, Any]] = []
        for probe_index in range(QUALIFICATION_CALLS_PER_MODEL):
            prompt_index = (candidate_index + probe_index) % len(PROMPTS)
            row = probe_call(model, PROMPTS[prompt_index])
            row.update({"candidate_index": candidate_index, "probe_index": probe_index, "prompt_index": prompt_index})
            rows.append(row)
        successes = sum(bool(row["success"]) for row in rows)
        qualification.append(
            {
                "model": model,
                "quality": candidate["quality"],
                "health": candidate["health"],
                "successes": successes,
                "attempts": len(rows),
                "calls": rows,
            }
        )
        if selected_model is None and successes == QUALIFICATION_CALLS_PER_MODEL:
            selected_model = model

    return qualification, selected_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    max_calls_per_window = DISCOVERY_CANDIDATES * QUALIFICATION_CALLS_PER_MODEL + OBSERVATION_CALLS
    payload: dict[str, Any] = {
        "probe": "openrouter-generation-reliability-v2",
        "target_scoring": False,
        "window_id": args.window_id,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "temperature": 0.7,
        "max_tokens": 64,
        "max_retries": 0,
        "cache_mode": "off",
        "daily_budget_contract": {
            "schedule_windows_per_day": 48,
            "max_qualification_calls_per_window": DISCOVERY_CANDIDATES * QUALIFICATION_CALLS_PER_MODEL,
            "observation_calls_per_window": OBSERVATION_CALLS,
            "max_inference_calls_per_window": max_calls_per_window,
            "max_inference_calls_per_day": 48 * max_calls_per_window,
        },
        "discovery": None,
        "qualification": [],
        "selected_model": None,
        "observation_calls": [],
    }

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")
        discovery = discover_free_models(api_key)
        payload["discovery"] = discovery
        candidates = discovery["candidates"]
        if not candidates:
            raise RuntimeError("no free model satisfies the OpenRouter health and capability discovery gate")

        qualification, selected_model = qualify_candidates(candidates)
        payload["qualification"] = qualification
        payload["selected_model"] = selected_model
        if selected_model is None:
            raise RuntimeError("no quality-ranked free candidate passed 2/2 target-free account probes")

        observation_calls: list[dict[str, Any]] = []
        for observation_index in range(OBSERVATION_CALLS):
            prompt_index = observation_index % len(PROMPTS)
            row = probe_call(selected_model, PROMPTS[prompt_index])
            row.update({"observation_index": observation_index, "prompt_index": prompt_index})
            observation_calls.append(row)
        payload["observation_calls"] = observation_calls
    except Exception as exc:
        payload["window_error"] = f"{type(exc).__name__}: {exc}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    observation_calls = payload["observation_calls"]
    qualification_calls = [call for candidate in payload["qualification"] for call in candidate["calls"]]
    print(
        json.dumps(
            {
                "window_id": args.window_id,
                "qualification_calls": len(qualification_calls),
                "qualification_successes": sum(bool(call["success"]) for call in qualification_calls),
                "selected_model": payload["selected_model"],
                "observation_calls": len(observation_calls),
                "observation_successes": sum(bool(call["success"]) for call in observation_calls),
            }
        )
    )


if __name__ == "__main__":
    main()
