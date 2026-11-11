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
    assert "@optimize\ndef ht(" in text
    assert "from agent_optimize import optimize" in text
    assert "restored for display" not in text
    assert "controlled test fixture" in text
    assert "No Numba function with measured timing" in text
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


def test_report_explains_measured_inputs_effort_and_checks_without_internal_counters():
    recorded = [{"case_id": f"case-{i:03d}", "name": f"recording-{i}", "status": "passed"} for i in range(2)]
    generated = [{"case_id": "edge", "name": "empty-lists", "status": "passed"},
                 {"case_id": "public", "name": "public-synthetic", "status": "passed"},
                 {"case_id": "extra", "name": "provider-extra-0", "status": "passed"}]
    accepted = {"status": "optimized", "case_count": 5, "recorded_case_ids": [c["case_id"] for c in recorded],
                "checks": recorded + generated, "generator": {"kind": "mock"}, "provider_request_count": 0,
                "timing": {"mode": "measured", "status": "passed", "compilation_ns": 2_000_000,
                           "first_use_ns": 3_000_000, "recordings": [
                               {"case_id": c["case_id"], "events": 24, "batches": [
                                   {"reference_ns": [10_000_000, 12_000_000], "candidate_ns": [1_000_000, 1_200_000]},
                                   {"reference_ns": [20_000_000], "candidate_ns": [1_000_000]}]}
                               for c in recorded]}}
    rejected = {"status": "rejected", "checks": [{"status": "failed"}], "generator": {"kind": "mock"}, "provider_request_count": 0}
    result = {"status": "completed", "run_timing": {"analysis_seconds": 1, "optimization_seconds": 2, "controller_seconds": 3.1},
              "functions": [{"identity": {"qualified_name": "reconstruct_events"}, "status": "optimized", "captured_calls": 2,
                             "evaluations": [rejected, accepted], "skipped": {"call_limit": {"count": 233}}}]}
    result["usage_summary"] = summarize_usage(result)
    text = render_report(result, "q16")
    assert text.index("## Function speedup") < text.index("## Time and API cost") < text.index("## Checks and saved code")
    assert "1 of 1 functions optimized" in text and "The count includes" not in text
    assert "verification group 1 | 24" in text and "verification group 2 | 24" in text and "case-000" not in text
    assert "10.00x" in text and "15.00x" not in text  # Lowest round ratio, not the ratio of displayed medians.
    assert "Verification group 1 (24 events) | 1 | Passed" in text
    assert "Verification group 2 (24 events) | 1 | Passed" in text
    assert "Synthetic test groups for edge cases | 3 | Passed" in text
    assert "**Total** | **5**" in text
    assert "Verified the Numba function against 2 real-data verification groups" in text and "Saved example inputs from" not in text
    assert "#### Saved replacement" not in text
    assert "2 attempts to reach the final code" in text
    assert "Generated inputs comprise" not in text
    assert "independently validate" not in text
    assert "Numba function | Conservative speedup" in text
    assert "Replacement outcomes:" not in text
    assert "rejected" not in text
    assert "Verified the Numba function against 2 real-data verification groups (out of 235 total function calls)" in text
    assert "Another 233" not in text
    assert "regardless of how many events" not in text
    assert "The other 3 checks" not in text
    assert "compilation took 2.000 ms within the first Numba function call of 3.000 ms" in text
    assert "included in the optimization phase" in text
    assert "deterministic local code generator" in text and "Searches using a model have their own" in text
    assert all(word not in text for word in ("HTTP", "Provider attempts", "dispatch", "Controller", "## Next step"))


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


def test_report_introduction_uses_only_generic_run_settings():
    result = {"status": "completed", "functions": [],
              "limits": {"max_calls_per_function": 7}, "search_limits": {"proposals": 3},
              "run_command": "agent-optimize run --report --verification-groups 7 --max-attempts 3 -- python unrelated-analysis.py --events 12"}
    result["usage_summary"] = summarize_usage(result)
    text = render_report(result, "unrelated-analysis")
    assert result["run_command"] in text
    assert "<analysis command>" not in text
    assert all(word not in text for word in ("q16", "ServiceX", "future", "after review"))


def test_actual_command_preserves_script_arguments_and_redacts_credentials():
    from agent_optimize.cli import _recorded_command
    arguments = ["run", "--api-key", "private-key", "--report", "--", "python",
                 "analysis with spaces.py", "--input-file", "events.root", "--token=private-token"]
    command = _recorded_command(arguments, "private-key")
    assert "private-key" not in command and "private-token" not in command
    assert "analysis with spaces.py" in command and "--input-file events.root" in command
    assert "[redacted]" in command
    assert "private inline program" not in _recorded_command(["run", "--", "python", "-c", "private inline program"])
