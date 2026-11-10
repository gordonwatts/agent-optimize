import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent_optimize.cli import _finish, main
from agent_optimize.reporting import command_name, render_report, write_report
from agent_optimize.usage import summarize_usage
from test_dispatch import accepted_base


def test_report_shows_real_frozen_source_and_labels_controlled_evidence(accepted_base, tmp_path):
    report = json.loads((accepted_base["session"] / "report.json").read_text())
    report.pop("session_directory", None)
    result = _finish(report, "completed", 0)
    text = render_report(result, "analysis")
    assert "def ht(" in text and "def kernel(" in text
    assert "controlled test fixture" in text
    assert "No eligible replacement with measured timing" in text
    assert "2.00x" not in text
    assert "0 tokens" in text
    path = write_report(result, ["--", sys.executable, "analysis.py", "--api-key", "do-not-copy"], tmp_path)
    assert path.name.startswith("analysis-") and "do-not-copy" not in path.read_text()
    with pytest.raises(FileExistsError):
        write_report(result, ["--", sys.executable, "analysis.py"], tmp_path)


def test_markdown_fences_hash_validation_and_partial_usage(tmp_path):
    code = 'def reference(x):\n    """``` source text </details>"""\n    return x\n'
    (tmp_path / "reference.py").write_bytes(code.encode("utf-8"))
    (tmp_path / "reference-job.json").write_text(json.dumps({"source_sha256": hashlib.sha256(code.encode()).hexdigest()}))
    (tmp_path / "candidate.py").write_text("THIS FILE MUST NEVER EXECUTE")
    evaluation = {"status": "optimized", "workspace": str(tmp_path), "candidate_sha256": "incorrect", "case_count": 12,
                  "generator": {"kind": "openai", "model": "unknown"}, "provider_usage": {"total_tokens": 20}, "provider_request_count": 1}
    function = {"identity": {"qualified_name": "f"}, "status": "optimized", "reason": "saved", "evaluations": [evaluation],
                "pending_attempt": {"attempt_id": "pending", "generator": {"kind": "openai"}, "provider_request_count": None}}
    result = {"status": "interrupted", "functions": [function], "next_action": "Inspect checkpoints"}
    result["usage_summary"] = summarize_usage(result)
    text = render_report(result, "<script>|name")
    assert "````python" in text
    assert "THIS FILE" not in text and "Verified source is unavailable" in text
    assert "20 known (partial)" in text and "unavailable total" in text
    assert "<script>" not in text


@pytest.mark.parametrize("failure", [False, True])
def test_report_cli_runs_analysis_once_and_preserves_json_and_exit(tmp_path, failure):
    script = tmp_path / "analysis with spaces.py"
    script.write_text("from pathlib import Path\nPath('once').open('a').write('x')\nprint('noisy child')\n" + ("raise SystemExit(7)\n" if failure else ""))
    result = subprocess.run([sys.executable, "-m", "agent_optimize", "run", "--json", "--report", "--", sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert result.returncode == payload["exit_code"] == (7 if failure else 0)
    assert (tmp_path / "once").read_text() == "x"
    path = Path(payload["markdown_report_path"])
    assert path.name.startswith("analysis-with-spaces-") and path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "No decorated function" in text and "0 tokens" in text
    assert "Analysis failed" in text if failure else "Completed" in text
    assert payload["run_timing"]["analysis_seconds"] >= 0
    assert payload["usage_summary"]["cost"]["estimated_total_usd"] == "0"
    assert json.loads(Path(payload["report_path"]).read_text())["markdown_report_path"] == str(path)


def test_script_report_argument_is_forwarded_without_export(tmp_path):
    script = tmp_path / "analysis.py"
    script.write_text("import sys\nassert sys.argv[1:] == ['--report']\n")
    result = subprocess.run([sys.executable, "-m", "agent_optimize", "run", "--json", "--", sys.executable, str(script), "--report"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "markdown_report_path" not in json.loads(result.stdout)
    assert not list(tmp_path.glob("*.md"))


def test_report_write_failure_is_actionable_and_preserves_child_exit(tmp_path, monkeypatch, capfd):
    import agent_optimize.reporting as reporting
    monkeypatch.chdir(tmp_path)
    def fail(*args): raise OSError("disk full")
    monkeypatch.setattr(reporting, "write_report", fail)
    assert main(["run", "--json", "--report", "--", sys.executable, "-c", "raise SystemExit(7)"]) == 7
    payload = json.loads(capfd.readouterr().out)
    assert payload["error"]["phase"] == "markdown_report"
    assert payload["child_exit_code"] == 7


def test_report_command_names_do_not_contain_inline_program_or_arguments():
    assert command_name(["--", "python", "-c", "secret.py", "private"]) == "python-command"
    assert command_name(["--", "python", "-m", "analysis", "private"]) == "analysis"
