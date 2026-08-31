"""A window that collected no evidence must not report success."""

import json

import pytest

pytest.importorskip("requests")

from benchmarks import probe_openrouter_reliability_v2 as probe


def run(tmp_path, monkeypatch, argv_id="w1"):
    output = tmp_path / "window.json"
    monkeypatch.setattr("sys.argv", ["probe", "--window-id", argv_id, "--output", str(output)])
    return probe.main(), output


def test_window_without_credential_fails_and_still_writes_the_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    status, output = run(tmp_path, monkeypatch)

    assert status == 1
    payload = json.loads(output.read_text())
    assert payload["window_error"].startswith("MissingCredentialError:")
    assert payload["window_error_kind"] == "credential"
    assert payload["observation_calls"] == []


def test_window_that_collected_nothing_fails_even_though_discovery_succeeded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.setattr(
        probe,
        "discover_free_models",
        lambda _key: {"catalog_model_count": 396, "candidates": [], "checked_models": []},
    )

    status, output = run(tmp_path, monkeypatch)

    assert status == 1
    payload = json.loads(output.read_text())
    assert probe.TARGET_MODEL in payload["window_error"]
    assert payload["window_error_kind"] == "health_gate"
    assert payload["selected_model"] is None
    assert payload["discovery"]["candidates"] == []


def test_complete_window_reports_success(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.setattr(
        probe,
        "discover_free_models",
        lambda _key: {"candidates": [{"model": probe.TARGET_MODEL}], "checked_models": []},
    )
    monkeypatch.setattr(
        probe,
        "qualify_candidates",
        lambda _candidates: ([{"model": probe.TARGET_MODEL, "calls": []}], probe.TARGET_MODEL),
    )
    monkeypatch.setattr(probe, "probe_call", lambda _model, _prompt: {"success": True})

    status, output = run(tmp_path, monkeypatch)

    assert status == 0
    payload = json.loads(output.read_text())
    assert payload.get("window_error") is None
    assert payload.get("window_error_kind") is None
    assert len(payload["observation_calls"]) == probe.OBSERVATION_CALLS


def test_a_qualification_failure_is_not_labelled_a_health_gate_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.setattr(
        probe,
        "discover_free_models",
        lambda _key: {"candidates": [{"model": probe.TARGET_MODEL}], "checked_models": []},
    )
    monkeypatch.setattr(
        probe,
        "qualify_candidates",
        lambda _candidates: ([{"model": probe.TARGET_MODEL, "calls": []}], None),
    )

    status, output = run(tmp_path, monkeypatch)

    assert status == 1
    payload = json.loads(output.read_text())
    assert payload["window_error_kind"] == "qualification"
