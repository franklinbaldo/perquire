#!/usr/bin/env python3
"""Target-free OpenRouter generation reliability probe for scaling v1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from perquire.llm.openrouter_provider import OpenRouterProvider

CONTROL_MODEL = "openai/gpt-oss-20b:free"
ELIGIBLE_MODELS = (
    "google/gemini-2.5-flash-lite",
    "google/gemini-3.1-flash-lite",
)
MODELS = (CONTROL_MODEL, *ELIGIBLE_MODELS)
REPETITIONS = 3
PROMPTS = (
    "Generate one concise semantic description of an unspecified topic. Return only the description.",
    "Rewrite this candidate into one nearby but meaningfully different semantic alternative: 'a public institution processes records'. Return only the alternative.",
    "A previous generic description received similarity 0.2400 to an unspecified hidden target. Propose one concise refinement without assuming the target text. Return only the refinement.",
    "Synthesize one concise description from these generic hints: public records; procedural state; time-sensitive work. Return only the description.",
)
LIST_PRICE_PER_M = {
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "google/gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
}


def select_model(results: list[dict[str, Any]]) -> str | None:
    by_model = {result["model"]: result for result in results}
    passing = [
        model
        for model in ELIGIBLE_MODELS
        if model in by_model
        and by_model[model]["logical_failures"] == 0
        and by_model[model]["logical_successes"] == len(PROMPTS) * REPETITIONS
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda model: (
            LIST_PRICE_PER_M[model]["input"] + LIST_PRICE_PER_M[model]["output"],
            model,
        ),
    )


def probe_model(model: str) -> dict[str, Any]:
    provider = OpenRouterProvider(
        config={
            "model": model,
            "temperature": 0.7,
            "max_tokens": 64,
            "max_retries": 2,
            "cache_mode": "off",
        }
    )
    successes = 0
    failures: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    started = time.perf_counter()

    for repetition in range(REPETITIONS):
        for prompt_index, prompt in enumerate(PROMPTS):
            before = provider.transport_attempts
            call_started = time.perf_counter()
            error: str | None = None
            content = ""
            try:
                response = provider.generate_response(prompt)
                content = response.content.strip()
                if not content:
                    raise RuntimeError("empty content after provider success")
                successes += 1
            except Exception as exc:  # probe must preserve failures in its artifact
                error = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "repetition": repetition,
                        "prompt_index": prompt_index,
                        "error": error,
                    }
                )
            calls.append(
                {
                    "repetition": repetition,
                    "prompt_index": prompt_index,
                    "success": error is None,
                    "transport_attempts": provider.transport_attempts - before,
                    "elapsed_seconds": time.perf_counter() - call_started,
                    "output_chars": len(content),
                    "error": error,
                }
            )

    return {
        "model": model,
        "role": "control" if model == CONTROL_MODEL else "eligible_candidate",
        "logical_calls": len(PROMPTS) * REPETITIONS,
        "logical_successes": successes,
        "logical_failures": len(failures),
        "transport_attempts": provider.transport_attempts,
        "elapsed_seconds": time.perf_counter() - started,
        "list_price_per_m_tokens": LIST_PRICE_PER_M.get(model),
        "failures": failures,
        "calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/openrouter_reliability_probe_v1.json"),
    )
    args = parser.parse_args()

    results = [probe_model(model) for model in MODELS]
    selected = select_model(results)
    payload = {
        "probe": "openrouter-generation-reliability-v1",
        "target_scoring": False,
        "models": list(MODELS),
        "eligible_models": list(ELIGIBLE_MODELS),
        "calls_per_model": len(PROMPTS) * REPETITIONS,
        "temperature": 0.7,
        "max_tokens": 64,
        "max_retries": 2,
        "cache_mode": "off",
        "selection_rule": "all 12 calls succeed; tie -> lowest frozen list price",
        "selected_model": selected,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_model": selected, "results": results}, ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
