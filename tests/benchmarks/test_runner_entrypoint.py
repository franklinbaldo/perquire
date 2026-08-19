"""The live benchmark is invoked from CI, so its entrypoint needs a guard.

Running the runner as a file path puts `benchmarks/` on `sys.path` instead of the
repository root, which breaks its own `benchmarks.*` imports. Only the module form
works, and a broken entrypoint is otherwise invisible until it burns a live run.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/live-semantic-inversion.yml"


def test_runner_is_importable_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.run_semantic_inversion", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--budget" in result.stdout


def test_live_workflow_uses_the_module_entrypoint():
    content = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m benchmarks.run_semantic_inversion" in content
    assert "python benchmarks/run_semantic_inversion.py" not in content


def test_live_workflow_fails_fast_without_a_credential():
    content = WORKFLOW.read_text(encoding="utf-8")
    assert 'if [ -z "${OPENROUTER_API_KEY}" ]' in content
