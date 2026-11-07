import os
import sys

from agent_optimize.process import run_worker, worker_environment
from agent_optimize.store import SESSION_ENV


def test_provider_credentials_and_recording_marker_are_removed():
    source = {"PATH": "keep", "OPENAI_API_KEY": "sentinel", "azure_openai_api_key": "sentinel", "ANTHROPIC_API_KEY": "sentinel", "HF_TOKEN": "sentinel", "AWS_SECRET_ACCESS_KEY": "sentinel", "GOOGLE_APPLICATION_CREDENTIALS": "sentinel", SESSION_ENV: "session", "SystemRoot": "keep"}
    assert worker_environment(source) == {"PATH": "keep", "SystemRoot": "keep"}
    environment = dict(os.environ, OPENAI_API_KEY="credential-sentinel", AGENT_OPTIMIZE_SESSION="session-sentinel")
    result = run_worker([sys.executable, "-c", "import os; assert 'OPENAI_API_KEY' not in os.environ; assert 'AGENT_OPTIMIZE_SESSION' not in os.environ; print('clean')"], environment=environment)
    assert result.status == "passed"
    assert result.stdout.strip() == "clean"
    assert environment["OPENAI_API_KEY"] == "credential-sentinel"


def test_worker_crash_is_an_explicit_failed_status():
    result = run_worker([sys.executable, "-c", "import os; os._exit(23)"])
    assert result.status == "failed"
    assert result.returncode == 23


def test_worker_timeout_terminates_the_process():
    result = run_worker([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.2)
    assert result.status == "timeout"
    assert result.returncode is not None


def test_worker_output_is_bounded_and_oversized_output_stops_the_worker():
    result = run_worker([sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000); sys.stdout.flush()"], max_output_bytes=4096)
    assert result.status == "output_limit"
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 4096


def test_missing_worker_executable_is_an_explicit_status(tmp_path):
    result = run_worker([str(tmp_path / "missing worker")])
    assert result.status == "startup_error"
    assert result.returncode is None
