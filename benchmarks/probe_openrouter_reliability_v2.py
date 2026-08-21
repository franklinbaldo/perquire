#!/usr/bin/env python3
"""Target-free longitudinal OpenRouter reliability observation for scaling v2."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Frozen initial free-first candidate set. Changing this set requires a new
# preregistered contract before benchmark target scores are observed.
MODELS = (
    "openai/gpt-oss-20b:free",
    "google/gemini-2.5-flash-lite:free",
    "google/gemini-3.1-flash-lite:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-4b:free",
)
PROMPTS = (
    "Generate one concise semantic description of an unspecified topic. Return only the description.",
    "Rewrite this candidate into one nearby but meaningfully different semantic alternative: 'a public institution processes records'. Return only the alternative.",
    "A previous generic description received similarity 0.2400 to an unspecified hidden target. Propose one concise refinement without assuming the target text. Return only the refinement.",
    "Synthesize one concise description from these generic hints: public records; procedural state; time-sensitive work. Return only the description.",
)


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
    if args.calls <= 0 or args.calls % len(MODELS):
        raise SystemExit(f"--calls must be a positive multiple of {len(MODELS)}")

    calls: list[dict[str, Any]] = []
    rounds = args.calls // len(MODELS)
    for round_index in range(rounds):
        for model_index, model in enumerate(MODELS):
            prompt_index = (round_index + model_index) % len(PROMPTS)
            row = probe_call(model, PROMPTS[prompt_index])
            row.update({"round": round_index, "prompt_index": prompt_index})
            calls.append(row)

    payload = {
        "probe": "openrouter-generation-reliability-v2",
        "target_scoring": False,
        "window_id": args.window_id,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "models": list(MODELS),
        "logical_calls": len(calls),
        "temperature": 0.7,
        "max_tokens": 64,
        "max_retries": 0,
        "cache_mode": "off",
        "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"window_id": args.window_id, "calls": len(calls), "successes": sum(c["success"] for c in calls)}))


if __name__ == "__main__":
    main()
