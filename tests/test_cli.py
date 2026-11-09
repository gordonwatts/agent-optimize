import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest

from agent_optimize.cli import main

ROOT = Path(__file__).resolve().parents[1]
CONSOLE = Path(sysconfig.get_path("scripts")) / ("agent-optimize.exe" if os.name == "nt" else "agent-optimize")


def test_installed_console_forwards_arguments_spaces_and_flags(tmp_path):
    script = tmp_path / "script with spaces.py"
    script.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    forwarded = ["a value with spaces", "--json", "--max-events", "12", "--", "-x", "semi;colon", ""]
    result = subprocess.run(
        [str(CONSOLE), "run", "--", sys.executable, str(script), *forwarded],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == forwarded
    assert "recording at" in result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [".agent-optimize", script.name]


def test_child_failure_status_is_preserved(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "agent_optimize", "run", "--", sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7


def test_json_configuration_errors_remain_machine_readable(capsys):
    assert main(["run", "--json", "--max-calls", "not-an-integer"]) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "error"
    assert result["exit_code"] == 2
    assert result["error"]["phase"] == "configuration"


def test_help_and_unknown_option_exit_status(capsys):
    with pytest.raises(SystemExit) as help_exit:
        main(["run", "--help"])
    assert help_exit.value.code == 0
    assert "--max-calls" in capsys.readouterr().out
    with pytest.raises(SystemExit) as invalid_exit:
        main(["run", "--unknown-option"])
    assert invalid_exit.value.code == 2


@pytest.mark.parametrize("arguments", [[], ["run"], ["run", "--"], ["run", "python", "script.py"]])
def test_missing_command_or_separator_is_configuration_error(arguments):
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2


def test_launch_failure_has_actionable_message(tmp_path, capsys):
    assert main(["run", "--", str(tmp_path / "missing executable")]) == 2
    assert "could not launch command" in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("--max-proposals", "0"), ("--function-seconds", "nan"), ("--worker-seconds", "-1"), ("--search-seconds", "inf"), ("--max-request-bytes", "0"), ("--max-response-bytes", "0"), ("--max-extra-cases", "33")])
def test_invalid_search_limits_are_configuration_errors(option, value):
    with pytest.raises(SystemExit) as caught:
        main(["run", option, value, "--", sys.executable, "-c", "pass"])
    assert caught.value.code == 2


def test_import_and_normal_calls_do_not_record_or_load_optimization_modules(tmp_path):
    program = """
import sys
import agent_optimize
@agent_optimize.optimize
def reference(value):
    return value
assert reference('ordinary Python value') == 'ordinary Python value'
assert reference.__wrapped__('reference') == 'reference'
assert 'agent_optimize.arguments' not in sys.modules
for name in ('recorder', 'generator', 'evaluator', 'store'):
    assert 'agent_optimize.' + name not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_documented_example_ordinary_and_launched_results_match(tmp_path):
    command = [sys.executable, str(ROOT / "examples" / "filter_sum.py"), "--max-events", "12"]
    ordinary = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    launched = subprocess.run(
        [str(CONSOLE), "run", "--", *command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert ordinary.returncode == launched.returncode == 0, ordinary.stderr + launched.stderr
    assert ordinary.stdout == launched.stdout == "events=12 total_ht=700.0\n"
    assert [path.name for path in tmp_path.iterdir()] == [".agent-optimize"]
