#!/usr/bin/env python3
"""Target-free longitudinal OpenRouter reliability observation for scaling v2."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

CATALOG_URL = "https://openrouter.ai/api/v1/models"
DISCOVERY_CANDIDATES = 10
QUALIFICATION_CALLS_PER_MODEL = 2
OBSERVATION_CALLS = 20
PROMPTS = (
    "Generate one concise semantic description of an unspecified topic. Return only the description.",
    "Rewrite this candidate into one nearby but meaningfully different semantic alternative: 'a public institution processes records'. Return only the alternative.",
    "A previous generic description received similarity 0.2400 to an unspecified hidden target. Propose one concise refinement without assuming the target text. Return only the refinement.",
    "Synthesize one concise description from these generic hints: public records; procedural state; time-sensitive work. Return only the description.",
)


def _zero_price(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


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


def discover_free_models(api_key: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        CATALOG_URL,
        headers=headers,
        params={"output_modalities": "text", "sort": "throughput-high-to-low"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    free_ids = [str(model["id"]) for model in data if _is_free_text_model(model)]

    endpoint_checks: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for model_id in free_ids:
        author, slug = model_id.split("/", 1)
        endpoint_url = (
            "https://openrouter.ai/api/v1/models/"
            f"{quote(author, safe='')}/{quote(slug, safe=':')}/endpoints"
        )
        check: dict[str, Any] = {"model": model_id, "endpoint_count": None, "error": None}
        try:
            endpoint_response = requests.get(endpoint_url, headers=headers, timeout=10)
            endpoint_response.raise_for_status()
            endpoints = endpoint_response.json().get("data", {}).get("endpoints", [])
            check["endpoint_count"] = len(endpoints)
            if endpoints:
                candidates.append({"model": model_id, "endpoint_count": len(endpoints)})
        except Exception as exc:
            check["error"] = f"{type(exc).__name__}: {exc}"
        endpoint_checks.append(check)
        if len(candidates) == DISCOVERY_CANDIDATES:
            break

    return {
        "catalog_url": CATALOG_URL,
        "catalog_sort": "throughput-high-to-low",
        "catalog_model_count": len(data),
        "catalog_free_text_models": free_ids,
        "selection_rule": (
            "query current OpenRouter catalog at runtime; filter :free + zero prompt/completion price + text output; "
            "preserve OpenRouter throughput-high-to-low order; require at least one currently listed endpoint; "
            "qualify first 10 active candidates with two target-free calls each; choose among 2/2 candidates by "
            "endpoint_count descending, qualification median latency ascending, then model id"
        ),
        "endpoint_checks": endpoint_checks,
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
    eligible: list[tuple[int, float, str]] = []
    for candidate_index, candidate in enumerate(candidates):
        model = str(candidate["model"])
        rows: list[dict[str, Any]] = []
        for probe_index in range(QUALIFICATION_CALLS_PER_MODEL):
            prompt_index = (candidate_index + probe_index) % len(PROMPTS)
            row = probe_call(model, PROMPTS[prompt_index])
            row.update({"candidate_index": candidate_index, "probe_index": probe_index, "prompt_index": prompt_index})
            rows.append(row)
        successes = sum(bool(row["success"]) for row in rows)
        median_latency = statistics.median(row["elapsed_seconds"] for row in rows)
        qualification.append(
            {
                "model": model,
                "endpoint_count": int(candidate["endpoint_count"]),
                "successes": successes,
                "attempts": len(rows),
                "median_latency_seconds": median_latency,
                "calls": rows,
            }
        )
        if successes == QUALIFICATION_CALLS_PER_MODEL:
            eligible.append((-int(candidate["endpoint_count"]), median_latency, model))

    selected_model = min(eligible)[2] if eligible else None
    return qualification, selected_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
            "schedule_windows_per_day": 24,
            "max_qualification_calls_per_window": DISCOVERY_CANDIDATES * QUALIFICATION_CALLS_PER_MODEL,
            "observation_calls_per_window": OBSERVATION_CALLS,
            "max_inference_calls_per_window": DISCOVERY_CANDIDATES * QUALIFICATION_CALLS_PER_MODEL + OBSERVATION_CALLS,
            "max_inference_calls_per_day": 24 * (DISCOVERY_CANDIDATES * QUALIFICATION_CALLS_PER_MODEL + OBSERVATION_CALLS),
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
        if len(candidates) != DISCOVERY_CANDIDATES:
            raise RuntimeError(
                f"only {len(candidates)} endpoint-active free text models discovered; need {DISCOVERY_CANDIDATES}"
            )

        qualification, selected_model = qualify_candidates(candidates)
        payload["qualification"] = qualification
        payload["selected_model"] = selected_model
        if selected_model is None:
            raise RuntimeError("no free candidate achieved 2/2 target-free qualification")

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
