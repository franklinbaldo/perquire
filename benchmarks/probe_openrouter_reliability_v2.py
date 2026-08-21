#!/usr/bin/env python3
"""Target-free longitudinal OpenRouter reliability observation for scaling v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

CATALOG_URL = "https://openrouter.ai/api/v1/models"
SELECTED_MODELS = 5
MAX_ENDPOINT_CHECKS = 25
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


def _stable_rank(model_id: str) -> str:
    """Target-independent ordering stable across catalog response ordering."""
    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()


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
    free_ids = sorted(
        (str(model["id"]) for model in data if _is_free_text_model(model)),
        key=lambda model_id: (_stable_rank(model_id), model_id),
    )

    endpoint_checks: list[dict[str, Any]] = []
    selected: list[str] = []
    for model_id in free_ids[:MAX_ENDPOINT_CHECKS]:
        author, slug = model_id.split("/", 1)
        endpoint_url = (
            "https://openrouter.ai/api/v1/models/"
            f"{quote(author, safe='')}/{quote(slug, safe=':')}/endpoints"
        )
        check: dict[str, Any] = {"model": model_id, "endpoint_count": None, "error": None}
        try:
            endpoint_response = requests.get(endpoint_url, headers=headers, timeout=30)
            endpoint_response.raise_for_status()
            endpoints = endpoint_response.json().get("data", {}).get("endpoints", [])
            check["endpoint_count"] = len(endpoints)
            if endpoints:
                selected.append(model_id)
        except Exception as exc:
            check["error"] = f"{type(exc).__name__}: {exc}"
        endpoint_checks.append(check)
        if len(selected) == SELECTED_MODELS:
            break

    return {
        "catalog_url": CATALOG_URL,
        "catalog_model_count": len(data),
        "catalog_free_text_models": free_ids,
        "selection_rule": (
            "filter id suffix :free + zero prompt/completion price + text output; "
            "rank by sha256(model id); require at least one currently listed endpoint; take first 5"
        ),
        "endpoint_checks": endpoint_checks,
        "selected_models": selected,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calls <= 0 or args.calls % SELECTED_MODELS:
        raise SystemExit(f"--calls must be a positive multiple of {SELECTED_MODELS}")

    observed_at = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "probe": "openrouter-generation-reliability-v2",
        "target_scoring": False,
        "window_id": args.window_id,
        "observed_at_utc": observed_at,
        "logical_calls": 0,
        "temperature": 0.7,
        "max_tokens": 64,
        "max_retries": 0,
        "cache_mode": "off",
        "discovery": None,
        "calls": [],
    }

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        payload["discovery"] = {"error": "OPENROUTER_API_KEY missing"}
    else:
        try:
            discovery = discover_free_models(api_key)
            payload["discovery"] = discovery
            models = discovery["selected_models"]
            if len(models) != SELECTED_MODELS:
                raise RuntimeError(
                    f"only {len(models)} endpoint-active free text models discovered; "
                    f"need {SELECTED_MODELS}"
                )

            calls: list[dict[str, Any]] = []
            rounds = args.calls // len(models)
            for round_index in range(rounds):
                for model_index, model in enumerate(models):
                    prompt_index = (round_index + model_index) % len(PROMPTS)
                    row = probe_call(model, PROMPTS[prompt_index])
                    row.update({"round": round_index, "prompt_index": prompt_index})
                    calls.append(row)
            payload["calls"] = calls
            payload["logical_calls"] = len(calls)
        except Exception as exc:
            discovery = payload.get("discovery")
            if not isinstance(discovery, dict):
                discovery = {}
                payload["discovery"] = discovery
            discovery["error"] = f"{type(exc).__name__}: {exc}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    calls = payload["calls"]
    print(
        json.dumps(
            {
                "window_id": args.window_id,
                "calls": len(calls),
                "successes": sum(bool(call["success"]) for call in calls),
                "discovered": len((payload.get("discovery") or {}).get("selected_models", [])),
            }
        )
    )


if __name__ == "__main__":
    main()
