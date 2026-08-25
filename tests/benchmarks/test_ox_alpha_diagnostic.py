from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import benchmarks.diagnose_ox_alpha as diagnostic


def _response(content: str, *, reasoning: str = "", finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, reasoning=reasoning)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = {"completion_tokens": 12, "completion_tokens_details": {"reasoning_tokens": 8}}
    return SimpleNamespace(model="stealth/ox-alpha", choices=[choice], usage=usage)


def test_run_call_distinguishes_transport_from_visible_text():
    row = diagnostic.run_call(
        {"id": "x", "model": "stealth/ox-alpha", "max_tokens": 64, "reasoning": None},
        "prompt",
        0.7,
        completion_fn=lambda **kwargs: _response("", reasoning="thinking"),
    )
    assert row["transport_success"] is True
    assert row["visible_text_success"] is False
    assert row["reasoning_length"] > 0
    assert row["error_class"] == "empty_visible_completion"
    assert row["usage"]["completion_tokens_details"]["reasoning_tokens"] == 8


def test_run_selects_smallest_operational_frozen_cell(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    outcomes = iter(["", "", "ok", "ok", "ok", "ok", "ok", "ok"])
    monkeypatch.setattr(
        diagnostic,
        "run_call",
        lambda cell, prompt, temperature: {
            "transport_success": True,
            "visible_text_success": bool(text := next(outcomes)),
            "visible_content_length": len(text),
        },
    )
    config = {
        "schema_version": 1,
        "target_scoring": False,
        "model": "stealth/ox-alpha",
        "prompt": "target-free",
        "calls_per_cell": 2,
        "temperature": 0.7,
        "matrix": [
            {"id": "64", "max_tokens": 64, "reasoning": None},
            {"id": "256", "max_tokens": 256, "reasoning": None},
            {"id": "1024", "max_tokens": 1024, "reasoning": {"effort": "low"}},
            {"id": "2048", "max_tokens": 2048, "reasoning": {"effort": "low"}},
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = diagnostic.run(path)
    assert result["target_scoring"] is False
    assert result["counts_toward_reliability_freeze"] is False
    assert result["decision"]["selected_cell"] == "256"
    assert result["decision"]["requires_new_evidence_boundary"] is True


def test_config_refuses_target_scoring(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"target_scoring": True}), encoding="utf-8")
    try:
        diagnostic.run(path)
    except RuntimeError as exc:
        assert "target_scoring=false" in str(exc)
    else:
        raise AssertionError("target-scored diagnostic must fail closed")
