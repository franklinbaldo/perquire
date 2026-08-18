import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from perquire.cli.main import app
from perquire.core.result import InvestigationResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def investigation_result() -> InvestigationResult:
    start = datetime(2026, 1, 1, 10, 0, 0)
    return InvestigationResult(
        investigation_id="cli_mock_id_123",
        description="CLI mocked description",
        final_similarity=0.99,
        iterations=5,
        start_time=start,
        end_time=start + timedelta(seconds=2),
        strategy_name="default",
    )


def test_cli_entry_point(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "perquire" in result.output.lower()
    assert "investigate" in result.output
    assert "batch" in result.output
    assert "status" in result.output


def test_cli_providers_command(runner: CliRunner) -> None:
    provider_data = {
        "embedding": {"mock_emb": {"installed": True, "extra": "mock_extra"}},
        "llm": {"mock_llm": {"installed": False, "extra": "mock_llm_extra"}},
    }
    with patch("perquire.cli.main.list_available_providers", return_value=provider_data):
        result = runner.invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "Embedding Providers" in result.output
    assert "mock_emb" in result.output
    assert "LLM Providers" in result.output
    assert "mock_llm" in result.output


def test_cli_investigate_command(
    runner: CliRunner,
    tmp_path: Path,
    investigation_result: InvestigationResult,
) -> None:
    embedding_file = tmp_path / "target.npy"
    np.save(embedding_file, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    investigator = MagicMock()
    investigator.investigate.return_value = investigation_result

    with patch(
        "perquire.cli.main.create_investigator_from_cli_options",
        return_value=investigator,
    ):
        result = runner.invoke(app, ["investigate", str(embedding_file)])

    assert result.exit_code == 0, result.output
    investigator.investigate.assert_called_once()
    assert "CLI mocked description" in result.output
    assert "0.9900" in result.output


def test_cli_investigate_missing_file(runner: CliRunner) -> None:
    result = runner.invoke(app, ["investigate", "missing.npy"])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_cli_batch_command(
    runner: CliRunner,
    tmp_path: Path,
    investigation_result: InvestigationResult,
) -> None:
    for index in range(3):
        np.save(
            tmp_path / f"target_{index}.npy",
            np.array([index + 0.1, 0.2, 0.3], dtype=np.float32),
        )

    investigator = MagicMock()
    investigator.investigate.return_value = investigation_result

    with patch(
        "perquire.cli.main.create_investigator_from_cli_options",
        return_value=investigator,
    ), patch("perquire.cli.main.Confirm.ask", return_value=True):
        result = runner.invoke(app, ["batch", str(tmp_path), "--format", "npy"])

    assert result.exit_code == 0, result.output
    assert investigator.investigate.call_count == 3
    assert "Batch Investigation Summary (3 files)" in result.output
    assert "target_0.npy" in result.output


def test_cli_status_command(runner: CliRunner) -> None:
    database = MagicMock()
    database.get_investigation_stats.return_value = {
        "total_investigations": 10,
        "total_questions": 100,
        "average_similarity": 0.85,
        "average_iterations": 7.5,
    }
    database.list_investigations.return_value = [
        {
            "investigation_id": "id_123456789",
            "description": "A test description",
            "final_similarity": 0.9,
            "strategy_name": "default",
        }
    ]

    with patch("perquire.cli.main._open_database", return_value=database):
        result = runner.invoke(app, ["status", "--database", "dummy.db"])

    assert result.exit_code == 0, result.output
    database.get_investigation_stats.assert_called_once()
    database.list_investigations.assert_called_once_with(limit=5)
    database.disconnect.assert_called_once()
    assert "Total Investigations" in result.output
    assert "A test description" in result.output


def test_cli_configure_round_trip(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = runner.invoke(
        app,
        [
            "configure",
            "--provider",
            "test_llm",
            "--api-key",
            "secret",
            "--database",
            str(tmp_path / "test.db"),
        ],
    )
    assert result.exit_code == 0, result.output

    config_file = tmp_path / ".perquire" / "config.json"
    config = json.loads(config_file.read_text())
    assert config["default_provider"] == "test_llm"
    assert config["default_llm_provider"] == "test_llm"
    assert config["default_embedding_provider"] == "test_llm"
    assert config["test_llm_api_key"] == "secret"

    shown = runner.invoke(app, ["configure", "--show"])
    assert shown.exit_code == 0
    assert "test_llm" in shown.output
    assert "secret" not in shown.output
    assert "***" in shown.output
