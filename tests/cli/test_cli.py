import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import numpy as np
import json
from pathlib import Path

# Import the CLI entry point from main (for full command set)
from perquire.cli.main import cli as perquire_cli_app
from perquire.core.result import InvestigationResult # For typing and creating mock results

# Mock PerquireInvestigator for testing CLI commands that use it
@pytest.fixture
def mock_investigator_instance():
    mock = MagicMock(name="MockPerquireInvestigatorInstance")

    # Create a mock InvestigationResult object
    # Use actual InvestigationResult to ensure interface compatibility
    mock_inv_result_data = {
        "investigation_id": "cli_mock_id_123",
        "description": "CLI Mocked Description from Test",
        "final_similarity": 0.99,
        "iterations": 5,
        "start_time": "2024-01-01T10:00:00", # Use ISO format string
        "end_time": "2024-01-01T10:01:00",
        "strategy_name": "cli_mock_strategy",
        "model_config": {"llm_provider": {"provider":"mock"}, "embedding_provider":{"provider":"mock"}},
        "questions_history": [], # Empty for simple mock
        "convergence_reason": "max_iterations",
        "phase_reached": "convergence"
    }
    mock_result_obj = InvestigationResult.from_dict(mock_inv_result_data) # Create from dict

    mock.investigate.return_value = mock_result_obj
    return mock

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_entry_point(runner):
    """Test that the perquire command runs and shows help."""
    result = runner.invoke(perquire_cli_app, ['--help']) # MODIFIED
    assert result.exit_code == 0
    # Default group name for cli in main.py is 'cli', so usage shows 'cli'
    assert "Usage: cli [OPTIONS] COMMAND [ARGS]..." in result.output # main.py also uses 'cli'
    assert "Perquire: Reverse Embedding Search Through Systematic Questioning" in result.output

def test_cli_providers_command(runner):
    """Test the 'providers' command."""
    # providers command is in main.py, so patch its list_available_providers
    with patch('perquire.cli.main.list_available_providers') as mock_list_providers: # MODIFIED patch target
        mock_list_providers.return_value = {
            "embedding": {"mock_emb": {"installed": True, "extra": "mock_extra"}},
            "llm": {"mock_llm": {"installed": False, "extra": "mock_llm_extra"}}
        }
        result = runner.invoke(perquire_cli_app, ['providers']) # MODIFIED
        assert result.exit_code == 0
        assert "Embedding Providers" in result.output
        assert "mock_emb" in result.output
        assert "Installed" in result.output
        assert "LLM Providers" in result.output
        assert "mock_llm" in result.output
        assert "Not installed" in result.output
        mock_list_providers.assert_called_once()

def test_cli_investigate_command(runner, mock_investigator_instance, tmp_path):
    """Test the 'investigate' command with mocks."""
    dummy_emb_file = tmp_path / "dummy.npy"
    np.save(dummy_emb_file, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    # Patch the create_investigator_from_cli_options where it's defined (in perquire.cli.main)
    with patch('perquire.cli.main.create_investigator_from_cli_options', return_value=mock_investigator_instance) as mock_create_inv:
        result = runner.invoke(perquire_cli_app, [ # MODIFIED
            'investigate', str(dummy_emb_file),
            '--llm-provider', 'mock_llm', # These args are passed to create_investigator
            '--embedding-provider', 'mock_emb'
        ])

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    mock_create_inv.assert_called_once()
    mock_investigator_instance.investigate.assert_called_once()
    assert "CLI Mocked Description from Test" in result.output
    assert "0.9900" in result.output # Similarity formatted

def test_cli_investigate_file_not_found(runner):
    result = runner.invoke(perquire_cli_app, ['investigate', 'nonexistent.npy']) # MODIFIED
    assert result.exit_code != 0
    # Click's message for Path(exists=True)
    assert "Error: Invalid value for 'EMBEDDING_FILE': Path 'nonexistent.npy' does not exist." in result.output


def test_cli_status_command(runner):
    """Test the 'status' command with a mocked database provider."""
    mock_db_provider = MagicMock(name="MockDuckDBProvider")
    mock_db_provider.get_investigation_stats.return_value = {
        'total_investigations': 10, 'total_questions': 100,
        'average_similarity': 0.85, 'average_iterations': 7.5
    }
    mock_db_provider.list_investigations.return_value = [
        {'investigation_id': 'id_123', 'description': 'A test desc',
         'final_similarity': 0.9, 'strategy_name': 'default', 'start_time': '2023-01-01T10:00:00'}
    ]
    mock_db_provider.check_tables_exist.return_value = True

    # Patch DuckDBProvider instantiation within perquire.cli.main (where status cmd is)
    with patch('perquire.cli.main.DuckDBProvider', return_value=mock_db_provider) as mock_duckdb_init:
        result = runner.invoke(perquire_cli_app, ['status', '--database', 'dummy.db']) # MODIFIED

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    mock_duckdb_init.assert_called_once()
    mock_db_provider.get_investigation_stats.assert_called_once()
    mock_db_provider.list_investigations.assert_called_once()
    assert "Total Investigations" in result.output
    assert "10" in result.output
    assert "A test desc" in result.output

def test_cli_configure_command(runner, tmp_path, monkeypatch): # Added monkeypatch
    """Test the 'configure' command writing to a temporary config file."""
    config_dir = tmp_path / ".perquire" # This is where lean_main.configure will write
    # No need to create config_dir, the command should do it.
    config_file = config_dir / "config.json"

    # Redirect Path.home() to tmp_path so config is written within tmp_path
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)

    result = runner.invoke(perquire_cli_app, [ # MODIFIED
        'configure',
        '--provider', 'test_llm',
        '--api-key', 'test_key_123',
        '--database', str(tmp_path / 'test.db') # Ensure path is string
    ])
    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    assert config_file.exists(), "Config file was not created"

    with open(config_file, 'r') as f:
        config_data = json.load(f)
    assert config_data.get('default_provider') == 'test_llm'
    assert config_data.get('test_llm_api_key') == 'test_key_123' # configure command forms key this way
    assert config_data.get('default_database') == str(tmp_path / 'test.db')

    result_show = runner.invoke(perquire_cli_app, ['configure', '--show']) # MODIFIED
    assert result_show.exit_code == 0
    assert "default_provider" in result_show.output
    assert "test_llm" in result_show.output
    assert "test_llm_api_key" in result_show.output # Key name
    assert "***" in result_show.output # Masked API key value
    # monkeypatch is function-scoped, no need to undo explicitly here.


def test_cli_batch_command(runner, mock_investigator_instance, tmp_path):
    """Test the 'batch' command with mocks."""
    batch_dir = tmp_path / "embeddings_batch"
    batch_dir.mkdir()
    num_files = 3
    dummy_files = []
    for i in range(num_files):
        f = batch_dir / f"dummy_emb_{i}.npy"
        np.save(f, np.array([0.1 * i, 0.2 * i, 0.3 * i], dtype=np.float32))
        dummy_files.append(f)

    # Patch create_investigator_from_cli_options in perquire.cli.main
    with patch('perquire.cli.main.create_investigator_from_cli_options', return_value=mock_investigator_instance) as mock_create_inv:
        # Patch Confirm.ask to always return True to proceed with batch
        with patch('rich.prompt.Confirm.ask', return_value=True):
            result = runner.invoke(perquire_cli_app, [
                'batch', str(batch_dir),
                '--llm-provider', 'mock_llm_batch', # Passed to create_investigator
                '--format', 'npy' # Specify format for dummy files
            ])

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    mock_create_inv.assert_called_once() # Investigator created once for the batch
    assert mock_investigator_instance.investigate.call_count == num_files # Called for each file

    # Check that investigate was called with the loaded embeddings
    # For more robustness, you could check call_args_list if embeddings are distinct enough
    # For example, if mock_investigator_instance.investigate stored its args or had side effects based on them.
    # Here, we rely on the call_count and the fact that dummy files are created.

    assert f"Batch Investigation Summary ({num_files} files processed)" in result.output
    # Based on the mock_investigator_instance, each call to investigate returns the same mocked description
    # So, we expect to see this description in the output (likely multiple times if verbose, or summarized)
    assert "CLI Mocked Description from Test" in result.output
    assert "dummy_emb_0.npy" in result.output # File names in summary
    assert "dummy_emb_1.npy" in result.output
    assert "dummy_emb_2.npy" in result.output


# --- Test for Demo Command ---
# Mock the DemoEmbeddingProvider and PerquireInvestigator used in the demo command
@pytest.fixture
def mock_demo_embedding_provider():
    mock_provider = MagicMock(name="MockDemoEmbeddingProviderInstance")
    mock_provider.get_embedding.return_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    mock_provider.get_embedding_model_name.return_value = "mock-demo-model"
    return mock_provider

@pytest.fixture
def mock_demo_investigator_instance(): # Similar to mock_investigator_instance but for demo
    mock = MagicMock(name="MockDemoPerquireInvestigatorInstance")
    mock_inv_result_data = {
        "investigation_id": "demo_mock_id_456",
        "description": "Demo Command Mocked Description",
        "final_similarity": 0.95,
        "iterations": 3,
        "start_time": "2024-01-02T10:00:00",
        "end_time": "2024-01-02T10:01:00",
        "strategy_name": "demo_mock_strategy",
        "model_config": {"llm_provider": {"provider":"mock_demo_llm"}, "embedding_provider":{"provider":"mock-demo-model"}},
        "questions_history": [],
        "convergence_reason": "similarity_threshold",
        "phase_reached": "convergence"
    }
    mock_result_obj = InvestigationResult.from_dict(mock_inv_result_data)
    mock.investigate.return_value = mock_result_obj
    return mock

# Patch targets for the demo command are in 'perquire.cli.demo'
def test_cli_demo_text_command(runner, mock_demo_embedding_provider, mock_demo_investigator_instance):
    """Test the 'demo text' command with mocks."""
    test_input_text = "This is a test sentence for the demo."

    # Mock PerquireInvestigator instantiation within perquire.cli.demo
    with patch('perquire.cli.demo.PerquireInvestigator', return_value=mock_demo_investigator_instance) as mock_investigator_init:
        # Mock DemoEmbeddingProvider instantiation within perquire.cli.demo
        with patch('perquire.cli.demo.DemoEmbeddingProvider', return_value=mock_demo_embedding_provider) as mock_emb_provider_init:
            # Mock OpenAIProvider instantiation (or other default LLM provider)
            with patch('perquire.cli.demo.OpenAIProvider') as mock_llm_provider_init: # Assuming OpenAI is default
                mock_llm_service_instance = MagicMock(name="MockOpenAIService")
                mock_llm_provider_init.return_value = mock_llm_service_instance

                result = runner.invoke(perquire_cli_app, [
                    'demo', 'text',
                    '--text', test_input_text,
                    '--llm-provider', 'openai' # Specify to match mock
                ])

    assert result.exit_code == 0, f"CLI demo command exited with error: {result.output}"

    # Check that our mocks were used
    mock_emb_provider_init.assert_called_once()
    mock_demo_embedding_provider.get_embedding.assert_called_once_with(test_input_text)

    mock_llm_provider_init.assert_called_once() # Check LLM provider was init

    mock_investigator_init.assert_called_once()
    # Example: Check some args passed to PerquireInvestigator constructor
    args, kwargs = mock_investigator_init.call_args
    assert kwargs.get('embedding_provider') == mock_demo_embedding_provider
    assert kwargs.get('llm_provider') == mock_llm_service_instance
    assert kwargs.get('db_path') == ":memory:"

    mock_demo_investigator_instance.investigate.assert_called_once()
    # Check that the embedding from DemoEmbeddingProvider was passed to investigate
    np.testing.assert_array_equal(
        mock_demo_investigator_instance.investigate.call_args[0][0], # First positional arg (target_embedding)
        np.array([0.5, 0.5, 0.5], dtype=np.float32)
    )

    assert "Investigating text:" in result.output
    assert test_input_text in result.output
    assert "Demo Command Mocked Description" in result.output
    assert "0.9500" in result.output # Similarity
    assert "Iterations: 3" in result.output

def test_cli_demo_text_command_missing_text_option(runner):
    """Test 'demo text' command fails if --text option is missing."""
    result = runner.invoke(perquire_cli_app, ['demo', 'text'])
    assert result.exit_code != 0
    assert "Error: Missing option '--text'." in result.output

def test_cli_demo_text_command_error_handling(runner):
    """Test 'demo text' command error handling during investigation."""
    test_input_text = "Error inducing text."
    with patch('perquire.cli.demo.PerquireInvestigator') as mock_investigator_init:
        mock_investigator_instance = MagicMock()
        mock_investigator_instance.investigate.side_effect = Exception("Test investigation error")
        mock_investigator_init.return_value = mock_investigator_instance

        with patch('perquire.cli.demo.DemoEmbeddingProvider'): # Mock this too
            with patch('perquire.cli.demo.OpenAIProvider'):
                result = runner.invoke(perquire_cli_app, [
                    'demo', 'text',
                    '--text', test_input_text
                ])

    assert result.exit_code == 0 # Command itself handles exception and prints error, doesn't exit with error code
    assert "An error occurred: Test investigation error" in result.output
