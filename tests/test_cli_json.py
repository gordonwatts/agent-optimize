import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import textwrap

import pytest

from agent_optimize.store import read_json
from test_dispatch import SOURCE

CONSOLE = Path(sysconfig.get_path("scripts")) / ("agent-optimize.exe" if os.name == "nt" else "agent-optimize")


def invoke(tmp_path, source, *options, bootstrap=None):
    analysis = tmp_path / "analysis with spaces.py"
    analysis.write_text(textwrap.dedent(source), encoding="utf-8")
    prefix = [str(CONSOLE)]
    if bootstrap is not None:
        launcher = tmp_path / "launcher.py"
        launcher.write_text(textwrap.dedent(bootstrap), encoding="utf-8")
        prefix = [sys.executable, str(launcher)]
    result = subprocess.run([*prefix, "run", "--json", *options, "--", sys.executable, str(analysis)], cwd=tmp_path, capture_output=True, text=True, check=False)
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["exit_code"] == result.returncode
    return result, payload


def test_json_stream_is_one_result_with_arbitrary_noisy_child_output(tmp_path):
    result, payload = invoke(tmp_path, r"""
        import os, sys
        os.write(1, b'not-json\x00noise\\n')
        print('{invalid: json}')
        print('child stderr', file=sys.stderr)
    """)
    assert result.returncode == 0 and payload["status"] == "completed"
    assert payload["functions"] == []
    assert "not-json" in result.stderr and "{invalid: json}" in result.stderr and "child stderr" in result.stderr
    assert "nothing to optimize" in result.stderr
    assert read_json(Path(payload["report_path"])) == payload
    assert payload["search_limits"]["proposals"] == 3


@pytest.mark.parametrize("options,phase", [(["--max-attempts", "0"], "configuration"), (["--unknown-option"], "configuration")])
def test_json_configuration_errors_are_structured(tmp_path, options, phase):
    result, payload = invoke(tmp_path, "raise AssertionError('must not run')", *options)
    assert result.returncode == 2 and payload["status"] == "error"
    assert payload["error"]["phase"] == phase
    assert "--help" in payload["error"]["message"]
    assert not (tmp_path / ".agent-optimize").exists()


def test_json_missing_command_and_launch_failure(tmp_path):
    for arguments, phase in [(["run", "--json", "--"], "configuration"), (["run", "--json", "--", str(tmp_path / "missing")], "launch")]:
        result = subprocess.run([str(CONSOLE), *arguments], cwd=tmp_path, capture_output=True, text=True, check=False)
        payload = json.loads(result.stdout)
        assert result.returncode == payload["exit_code"] == 2
        assert payload["error"]["phase"] == phase


@pytest.mark.parametrize("option", ["--js", "--max-prop"])
def test_abbreviated_options_are_rejected_without_launching_child(tmp_path, option):
    value = ["1"] if option == "--max-prop" else []
    result = subprocess.run([str(CONSOLE), "run", option, *value, "--", sys.executable, "-c", "print('child must not launch')"], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and result.stdout == ""
    assert "--help" in result.stderr and "child must not launch" not in result.stderr
    assert not (tmp_path / ".agent-optimize").exists()


def test_json_unwritable_artifact_root_reports_session_error(tmp_path):
    root = tmp_path / "a file"
    root.write_text("not a directory")
    result, payload = invoke(tmp_path, "pass", "--artifact-root", str(root))
    assert result.returncode == 2 and payload["error"]["phase"] == "session"


MULTIPLE = """
import awkward as ak
import numpy as np
from agent_optimize import optimize
@optimize
def first(events):
    jets = events.Jet
    return ak.sum(jets[jets.pt > 30].pt, axis=1)
@optimize
def second(events):
    jets = events.Jet
    return ak.sum(jets[jets.pt > 40].pt, axis=1)
@optimize
def unsupported(value):
    return value
events = ak.zip({'Jet': ak.zip({'pt': ak.unflatten(np.array([20, 35, 50], dtype=np.int64), [2, 0, 1])})}, depth_limit=1)
print(ak.to_list(first(events)))
print(ak.to_list(second(events)))
print(unsupported('unsupported input'))
"""


def test_json_reports_several_functions_and_unsupported_calls(tmp_path):
    result, payload = invoke(tmp_path, MULTIPLE, "--max-request-bytes", "1")
    assert result.returncode == 0
    functions = {entry["identity"]["qualified_name"]: entry for entry in payload["functions"]}
    assert set(functions) == {"first", "second", "unsupported"}
    for name in ("first", "second"):
        assert functions[name]["captured_calls"] == 1
        assert functions[name]["replays"][0]["status"] == "passed"
        assert functions[name]["evaluations"][0]["status"] == "request_limit"
    assert functions["unsupported"]["captured_calls"] == 0 and functions["unsupported"]["skipped"]


def test_json_no_usable_captures_is_completed_with_actionable_diagnostics(tmp_path):
    result, payload = invoke(tmp_path, """
        from agent_optimize import optimize
        @optimize
        def reference(value):
            return value
        print(reference('opaque'))
    """)
    assert result.returncode == 0 and payload["status"] == "completed"
    assert payload["functions"][0]["captured_calls"] == 0
    assert "skipped-capture" in payload["next_action"]
    assert "no usable calls captured" in result.stderr


def test_json_failed_child_preserves_status_and_skips_replay_and_search(tmp_path):
    result, payload = invoke(tmp_path, MULTIPLE + "\nraise SystemExit(7)\n")
    assert result.returncode == 7 and payload["status"] == "child_failed"
    assert payload["child_exit_code"] == 7
    assert all(not entry["replays"] and not entry.get("evaluations") for entry in payload["functions"])
    assert not (Path(payload["session_directory"]) / "experiments").exists()


CONTROLLED_BOOTSTRAP = """
import sys
from agent_optimize import cli
from agent_optimize import evaluator
from agent_optimize.generator import propose
original = evaluator.evaluate_session
def timings(evaluation):
    return {'schema_version': 1, 'status': 'passed', 'mode': 'controlled-test-fixture',
            'recordings': [{'case_id': identifier, 'batches': [{'reference_ns': [1000]*5, 'candidate_ns': [500]*5} for _ in range(3)]}
                           for identifier in evaluation['recorded_case_ids']]}
def provider(problem, history):
    if history and INTERRUPT:
        raise KeyboardInterrupt()
    return propose(problem, [{'feedback': 'correct first proposal in this controlled acceptance test'}])
def evaluate(*args, **kwargs):
    return original(*args, **kwargs, timing_provider=timings, proposer=provider)
evaluator.evaluate_session = evaluate
raise SystemExit(cli.main())
"""


@pytest.mark.parametrize("interrupted", [False, True])
def test_json_successful_persistence_and_interruption_keep_real_artifact(tmp_path, interrupted):
    bootstrap = CONTROLLED_BOOTSTRAP.replace("INTERRUPT", repr(interrupted))
    source = SOURCE + "\nprint(ak.to_list(ht(make_events())))\n"
    result, payload = invoke(tmp_path, source, "--max-attempts", "2" if interrupted else "1", bootstrap=bootstrap)
    assert result.returncode == (130 if interrupted else 0)
    assert payload["status"] == ("interrupted" if interrupted else "completed")
    evaluation = payload["functions"][0]["evaluations"][0]
    assert evaluation["status"] == "optimized"
    manifest = read_json(Path(evaluation["artifact_path"]))
    assert manifest["timing"]["mode"] == "controlled-test-fixture"
    assert "not a measured performance claim" in result.stderr
    assert "saved manifest" in result.stderr
    assert read_json(Path(payload["report_path"])) == payload
    ordinary = subprocess.run([sys.executable, str(tmp_path / "analysis with spaces.py")], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert ordinary.returncode == 0 and ordinary.stdout == "[0, 0, 153]\n"


def test_json_report_failure_is_structured_without_losing_prior_checkpoint(tmp_path):
    bootstrap = """
        from agent_optimize import cli
        def fail(*args, **kwargs):
            raise OSError('read-only destination')
        cli.write_json = fail
        raise SystemExit(cli.main())
    """
    result, payload = invoke(tmp_path, "print('ordinary output')", bootstrap=bootstrap)
    assert result.returncode == 2 and payload["status"] == "error"
    assert "checkpoint" in payload["report_error"]
    assert read_json(Path(payload["report_path"]))["functions"] == []


def test_json_search_exception_uses_latest_checkpoint(tmp_path):
    bootstrap = """
        from agent_optimize import cli, evaluator
        def fail(*args, **kwargs):
            raise RuntimeError('controlled local failure')
        evaluator.evaluate_session = fail
        raise SystemExit(cli.main())
    """
    result, payload = invoke(tmp_path, "pass", bootstrap=bootstrap)
    assert result.returncode == 2 and payload["error"]["phase"] == "search"
    assert "controlled local failure" in payload["error"]["message"]


def test_human_timing_summary_reports_complete_path_and_separate_compilation(capsys):
    from agent_optimize.cli import _show_timings

    _show_timings({'mode': 'measured', 'status': 'passed', 'compilation_ns': 3000000, 'first_use_ns': 5000000,
                   'recordings': [{'case_id': 'case-000', 'events': 4, 'elements': 12, 'batches': [{'reference_ns': [1000000]*5, 'candidate_ns': [500000]*5}]}]})
    message = capsys.readouterr().err
    assert "compilation 3.000 ms" in message and "first replacement use 5.000 ms" in message
    assert "reference 1.000 ms/call, replacement 0.500 ms/call" in message
    assert "speedup 2.00x" in message and "4 events, 12 elements" in message
