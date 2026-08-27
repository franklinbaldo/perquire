#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm==1.97.0",
# ]
# ///
"""Target-free Ox Alpha apparatus diagnostic for #79.

This runner is intentionally separate from the reliability observatory: its calls
must never count toward #65 qualification evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    for name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                return _plain(method())
            except TypeError:
                pass
    return str(value)


def _usage(response: Any) -> dict[str, Any]:
    raw = _plain(getattr(response, "usage", None))
    return raw if isinstance(raw, dict) else {}


def run_call(
    cell: dict[str, Any],
    prompt: str,
    temperature: float,
    *,
    completion_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if completion_fn is None:
        from litellm import completion as completion_fn

    started = time.perf_counter()
    request: dict[str, Any] = {
        "model": f"openrouter/{cell['model']}",
        "messages": [{"role": "user", "content": prompt}],
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "temperature": temperature,
        "max_tokens": int(cell["max_tokens"]),
        "max_retries": 0,
    }
    if cell.get("reasoning") is not None:
        request["extra_body"] = {"reasoning": cell["reasoning"]}

    row: dict[str, Any] = {
        "transport_success": False,
        "visible_text_success": False,
        "elapsed_seconds": None,
        "response_model": None,
        "finish_reason": None,
        "visible_content_length": 0,
        "reasoning_length": 0,
        "usage": {},
        "error_class": None,
        "error": None,
    }
    try:
        response = completion_fn(**request)
        row["transport_success"] = True
        row["response_model"] = getattr(response, "model", None)
        choice = response.choices[0]
        message = choice.message
        content = (getattr(message, "content", None) or "").strip()
        reasoning = getattr(message, "reasoning", None) or ""
        row["finish_reason"] = getattr(choice, "finish_reason", None)
        row["visible_content_length"] = len(content)
        row["reasoning_length"] = len(str(reasoning))
        row["usage"] = _usage(response)
        row["visible_text_success"] = bool(content)
        if not content:
            row["error_class"] = "empty_visible_completion"
            row["error"] = "transport succeeded but visible content was empty"
    except Exception as exc:
        row["error_class"] = type(exc).__name__
        row["error"] = str(exc)
    finally:
        row["elapsed_seconds"] = time.perf_counter() - started
    return row


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("target_scoring") is not False:
        raise RuntimeError("diagnostic config must explicitly set target_scoring=false")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY missing")

    rows: list[dict[str, Any]] = []
    for matrix_index, frozen in enumerate(config["matrix"]):
        cell = {**frozen, "model": config["model"]}
        calls = [
            run_call(cell, config["prompt"], float(config["temperature"]))
            for _ in range(int(config["calls_per_cell"]))
        ]
        operational = all(
            call["transport_success"] and call["visible_text_success"] for call in calls
        )
        rows.append(
            {
                "matrix_index": matrix_index,
                "id": cell["id"],
                "max_tokens": cell["max_tokens"],
                "reasoning": cell.get("reasoning"),
                "operational": operational,
                "calls": calls,
            }
        )

    operational = [row for row in rows if row["operational"]]
    selected = (
        min(operational, key=lambda row: (row["max_tokens"], row["matrix_index"]))
        if operational
        else None
    )
    return {
        "diagnostic": "ox-alpha-target-free-v1",
        "target_scoring": False,
        "counts_toward_reliability_freeze": False,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "cells": rows,
        "decision": {
            "status": (
                "operational_configuration_found" if selected else "no_eligible_ox_alpha_substrate"
            ),
            "selected_cell": selected["id"] if selected else None,
            "requires_new_evidence_boundary": bool(selected),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("benchmarks/ox_alpha_diagnostic_v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["decision"]))


if __name__ == "__main__":
    main()
