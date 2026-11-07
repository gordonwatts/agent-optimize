import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from agent_optimize.replay import replay_session
from agent_optimize.store import create_session, read_json, SESSION_ENV
from agent_optimize.worker import replay_call

CONSOLE = Path(sys.executable).with_name("agent-optimize.exe" if os.name == "nt" else "agent-optimize")


def run_script(tmp_path, source, *options):
    script = tmp_path / "analysis with spaces.py"
    script.write_text(textwrap.dedent(source), encoding="utf-8")
    result = subprocess.run(
        [str(CONSOLE), "run", *options, "--", sys.executable, str(script)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    roots = list(tmp_path.rglob("session.json"))
    assert len(roots) == 1, result.stderr
    directory = roots[0].parent
    return result, directory, read_json(directory / "report.json")


def test_script_runs_once_two_functions_repeated_calls_and_frozen_replay(tmp_path):
    result, directory, report = run_script(tmp_path, """
        import os
        from pathlib import Path
        import awkward as ak
        import numpy as np
        from agent_optimize import optimize

        marker = Path('script-count.txt')
        marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')
        assert Path(os.environ['AGENT_OPTIMIZE_SESSION']).is_absolute()
        LIMIT = 2.4

        @optimize
        def ht(events, /, threshold=30, *, enabled=True):
            jets = events.Jet
            selected = jets[(jets.pt > threshold) & (abs(jets.eta) < LIMIT) & enabled]
            return ak.sum(selected.pt, axis=1)

        @optimize
        def raw(events):
            return np.asarray(ak.sum(events.Jet.pt, axis=1))

        jets = ak.zip({
            'pt': ak.unflatten(np.array([20, 31, 50], dtype=np.int32), [2, 0, 1]),
            'eta': ak.unflatten(np.array([0., 2., 3.], dtype=np.float32), [2, 0, 1]),
        })
        events = ak.zip({'Jet': jets}, depth_limit=1)
        print(ak.to_list(ht(events, threshold=np.int32(30), enabled=np.bool_(True))))
        print(ak.to_list(ht(events[:0])))
        print(raw(events).tolist())
    """, "--artifact-root", "artifacts with spaces")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["[31, 0, 0]", "[]", "[51, 0, 50]"]
    assert (tmp_path / "script-count.txt").read_text() == "1"
    assert directory.is_absolute()
    assert directory.parents[1].name == "artifacts with spaces"
    assert report["limits"] == {"max_capture_bytes": 64 * 1024 * 1024, "max_calls_per_function": 5}
    assert len(report["functions"]) == 2
    assert sorted(entry["captured_calls"] for entry in report["functions"]) == [1, 2]
    assert all(replay["status"] == "passed" for entry in report["functions"] for replay in entry["replays"])
    captures = [read_json(path) for path in (directory / "calls").glob("*/capture.json")]
    assert sorted(capture["reference_result"]["dtype"] for capture in captures) == ["<i8", "<i8", "<i8"]
    assert {capture["reference_result"]["kind"] for capture in captures} == {"awkward", "numpy"}
    # Deleting the original script proves replay depends only on the frozen module.
    (tmp_path / "analysis with spaces.py").unlink()
    for capture in captures:
        assert replay_call(directory, capture["call_id"])["status"] == "passed"
    assert (tmp_path / "script-count.txt").read_text() == "1"


def test_child_failure_preserves_status_and_never_starts_replay(tmp_path):
    result, directory, report = run_script(tmp_path, """
        import awkward as ak
        from agent_optimize import optimize
        @optimize
        def total(events):
            return ak.sum(events, axis=1)
        print(ak.to_list(total(ak.Array([[1, 2], []]))))
        raise SystemExit(9)
    """)
    assert result.returncode == report["child_exit_code"] == 9
    assert report["functions"][0]["captured_calls"] == 1
    assert report["functions"][0]["replays"] == []
    assert "no replay" in report["functions"][0]["reason"]
    assert not list(directory.rglob("replay.json"))


def test_unsupported_helper_and_arguments_still_return_reference_results(tmp_path):
    result, _, report = run_script(tmp_path, """
        import awkward as ak
        from agent_optimize import optimize
        def helper(events):
            return ak.sum(events, axis=1)
        @optimize
        def uses_helper(events):
            return helper(events)
        @optimize
        def list_argument(events):
            return events
        print(ak.to_list(uses_helper(ak.Array([[1, 2], []]))))
        print(list_argument([4, 5]))
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["[3, 0]", "[4, 5]"]
    assert all(entry["status"] == "unsupported" for entry in report["functions"])
    reasons = [entry["skipped"]["unsupported"]["diagnostic"] for entry in report["functions"]]
    assert any("helper" in reason for reason in reasons)
    assert any("list" in reason for reason in reasons)


def test_cli_limit_controls_are_reported_and_skip_full_inputs(tmp_path):
    result, _, report = run_script(tmp_path, """
        import awkward as ak
        import numpy as np
        from agent_optimize import optimize
        @optimize
        def identity(events):
            return events
        for index in range(3):
            print(len(identity(ak.Array(np.arange(10000)))))
    """, "--max-capture-bytes", "5000", "--max-calls", "1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["10000"] * 3
    assert report["limits"] == {"max_capture_bytes": 5000, "max_calls_per_function": 1}
    function = report["functions"][0]
    assert function["captured_calls"] == 0
    assert function["skipped"]["byte_limit"]["count"] == 1
    assert function["skipped"]["call_limit"]["count"] == 2


def test_caught_reference_error_is_reported_without_partial_capture(tmp_path):
    result, directory, report = run_script(tmp_path, """
        import awkward as ak
        from agent_optimize import optimize
        @optimize
        def missing(events):
            return events.missing
        try:
            missing(ak.Array([1, 2]))
        except AttributeError:
            print('original exception')
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "original exception"
    assert "reference_exception" in report["functions"][0]["skipped"]
    assert not list((directory / "calls").iterdir())


def test_replay_worker_does_not_inherit_recording_marker(tmp_path, monkeypatch):
    directory = create_session(tmp_path)
    from agent_optimize.recorder import Recorder
    from test_recorder import identity_reference
    import awkward as ak

    Recorder(directory).invoke(identity_reference, (ak.Array([1]),), {})
    original_popen = subprocess.Popen
    environments = []

    def observed(*args, **kwargs):
        environments.append(kwargs["env"])
        return original_popen(*args, **kwargs)

    monkeypatch.setenv(SESSION_ENV, str(directory))
    monkeypatch.setattr(subprocess, "Popen", observed)
    report = replay_session(directory, 0)
    assert report["functions"][0]["replays"][0]["status"] == "passed"
    assert environments
    assert all(SESSION_ENV not in env for env in environments)


def test_worker_rejects_changed_frozen_source_before_execution(tmp_path):
    result, directory, report = run_script(tmp_path, """
        import awkward as ak
        from agent_optimize import optimize
        @optimize
        def identity(events):
            return events
        identity(ak.Array([1, 2]))
    """)
    assert result.returncode == 0, result.stderr
    reference = next((directory / "references").glob("*/reference.py"))
    reference.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    call_id = report["functions"][0]["replays"][0]["call_id"]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        replay_call(directory, call_id)


def test_supported_inner_function_is_recorded_when_outer_function_is_unsupported(tmp_path):
    result, _, report = run_script(tmp_path, """
        import awkward as ak
        from agent_optimize import optimize
        @optimize
        def inner(events):
            return ak.sum(events, axis=1)
        @optimize
        def outer(events):
            return inner(events)
        print(ak.to_list(outer(ak.Array([[1, 2], []]))))
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[3, 0]"
    functions = {entry["identity"]["qualified_name"]: entry for entry in report["functions"]}
    assert functions["outer"]["status"] == "unsupported"
    assert functions["inner"]["captured_calls"] == 1
    assert functions["inner"]["replays"][0]["status"] == "passed"


def test_to_numpy_conversion_retains_python_with_clear_subset_diagnostic(tmp_path):
    result, _, report = run_script(tmp_path, """
        import awkward as ak
        from agent_optimize import optimize
        @optimize
        def ht(events):
            jets = events.Jet
            return ak.to_numpy(ak.sum(jets[jets.pt > 30].pt, axis=1))
        print(ht(ak.Array([{'Jet': [{'pt': 40}]}])).tolist())
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[40]"
    function = report["functions"][0]
    assert function["captured_calls"] == 0
    assert "convert the result outside" in function["skipped"]["unsupported"]["diagnostic"]
    assert function["evaluations"] == []
