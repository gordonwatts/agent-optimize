import json
from pathlib import Path
import subprocess
import sys

import awkward as ak
import pytest

import agent_optimize.dispatch as dispatch
from agent_optimize.dispatch import clear_dispatch_cache, observe_dispatch
from agent_optimize.performance import complete_evaluation, measure_evaluation
from agent_optimize.store import read_json
from test_dispatch import accepted, accepted_base, controlled, fast_kernel, invoke
from test_persistence import copy_evaluation


@pytest.fixture
def fast_accepted(accepted, tmp_path, monkeypatch):
    evaluation, metadata = copy_evaluation(accepted, tmp_path, "fast evaluation")
    result = complete_evaluation(evaluation, metadata, accepted["function_id"], accepted["root"], timing_provider=controlled, dispatch_mode="fast")
    assert result["status"] == "optimized", result
    monkeypatch.delenv("AGENT_OPTIMIZE_MODE", raising=False)
    return {**accepted, "artifact_id": result["artifact_id"], "evaluation": result}


def test_fast_is_default_and_hot_calls_execute_saved_generated_adapter_without_checks(fast_accepted, fast_kernel, monkeypatch):
    case = fast_accepted
    assert invoke(case)["used"] is True
    adapter = case["root"] / "artifacts" / case["artifact_id"] / "adapter.py"
    assert "def replacement(arguments, kernel)" in adapter.read_text()
    assert "layout = arguments['events'].layout" in adapter.read_text()

    def unexpected(*args, **kwargs):
        raise AssertionError("hot fast call re-entered compatibility or artifact loading")

    for name in ("freeze_reference", "environment_signature", "adapt_inputs", "read_json", "validate_manifest", "validate_reusable_source"):
        monkeypatch.setattr(dispatch, name, unexpected)
    stats = invoke(case, case["module"].make_events((0, 20, 0, 3)))
    assert stats["used"] is True and stats["dispatch_mode"] == "fast"


def test_fast_and_careful_artifacts_coexist_and_mode_selects_its_own_evidence(fast_accepted, fast_kernel, monkeypatch):
    case = fast_accepted
    assert invoke(case)["artifact_id"] == case["artifact_id"]
    monkeypatch.setenv("AGENT_OPTIMIZE_MODE", "careful")
    careful_id = invoke(case)["artifact_id"]
    assert careful_id != case["artifact_id"]
    manifest = read_json(case["root"] / "artifacts" / careful_id / "manifest.json")
    assert manifest["dispatch_mode"] == "careful"
    case["module"].CUT = 1.5
    assert not invoke(case)["used"]
    case["module"].CUT = 2.4


def test_fast_mode_documents_and_exercises_stable_code_assumption(fast_accepted, fast_kernel):
    case = fast_accepted
    array = case["module"].make_events((5,))
    array = ak.with_field(array, ak.with_field(array.Jet, array.Jet.eta + 2, "eta"), "Jet")
    first = case["module"].ht(array)
    case["module"].CUT = 1.0
    # Deliberately no dynamic dependency check, even after a cache reset: the
    # user must regenerate saved code when its reference/dependencies change.
    clear_dispatch_cache()
    stats = {}
    with observe_dispatch(stats):
        actual = case["module"].ht(array)
    assert stats["used"]
    assert ak.to_list(actual) == ak.to_list(first)
    assert ak.to_list(case["module"].ht.__wrapped__(array)) != ak.to_list(actual)
    case["module"].CUT = 2.4


def test_fast_saved_implementation_runs_in_fresh_process(fast_accepted):
    program = """import importlib.util, json, sys
import awkward as ak
from agent_optimize.dispatch import observe_dispatch
spec = importlib.util.spec_from_file_location('accepted_analysis', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
stats = {}
with observe_dispatch(stats):
    value = module.ht(module.make_events((0, 7, 1, 0)))
print(json.dumps({'stats': stats, 'value': ak.to_list(value)}))
"""
    process = subprocess.run([sys.executable, "-c", program, str(fast_accepted["source"])], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["stats"]["used"] and result["stats"]["dispatch_mode"] == "fast"
    assert result["value"] == [0, 238, 0, 0]


def test_fast_load_corruption_falls_back_and_quarantines(fast_accepted, fast_kernel):
    case = fast_accepted
    adapter = case["root"] / "artifacts" / case["artifact_id"] / "adapter.py"
    adapter.write_text("invalid generated code", encoding="utf-8")
    assert not invoke(case)["used"]
    assert (case["root"] / "quarantine" / f"{case['artifact_id']}.json").exists()


def test_fast_runtime_failure_uses_reference_and_quarantines(fast_accepted, monkeypatch):
    class Broken:
        def compile(self, signature):
            pass

        def __call__(self, *args):
            raise RuntimeError("candidate failed")

    monkeypatch.setattr(dispatch, "compile_kernel", lambda source: Broken())
    assert not invoke(fast_accepted)["used"]
    assert not invoke(fast_accepted)["used"]
    assert (fast_accepted["root"] / "quarantine" / f"{fast_accepted['artifact_id']}.json").exists()


def test_real_fast_timing_uses_ordinary_decorator_and_labels_its_path(fast_accepted, tmp_path):
    evaluation, metadata = copy_evaluation(fast_accepted, tmp_path, "measured fast")
    timing = measure_evaluation(evaluation, metadata, fast_accepted["function_id"], worker_timeout=30, dispatch_mode="fast")
    assert timing["status"] == "passed", timing
    assert timing["dispatch_mode"] == "fast"
    assert "cached ordinary fast dispatch" in timing["path"]
    assert timing["compilation_ns"] > 0
    assert len(timing["recordings"][0]["batches"]) == 3


def test_careful_only_store_is_not_silently_used_as_fast(accepted, fast_kernel, monkeypatch):
    monkeypatch.delenv("AGENT_OPTIMIZE_MODE", raising=False)
    assert not invoke(accepted)["used"]


def test_invalid_mode_retains_reference_for_ordinary_calls(accepted, monkeypatch):
    monkeypatch.setenv("AGENT_OPTIMIZE_MODE", "unknown")
    assert not invoke(accepted).get("used", False)


@pytest.mark.parametrize("environment,option,expected", [(None, None, "fast"), ("careful", None, "careful"), ("careful", "fast", "fast"), ("fast", "careful", "careful")])
def test_cli_mode_precedence_and_report(tmp_path, monkeypatch, environment, option, expected):
    if environment is None:
        monkeypatch.delenv("AGENT_OPTIMIZE_MODE", raising=False)
    else:
        monkeypatch.setenv("AGENT_OPTIMIZE_MODE", environment)
    command = [sys.executable, "-m", "agent_optimize", "run", "--json"]
    if option:
        command += ["--dispatch-mode", option]
    command += ["--", sys.executable, "-c", "import os; assert os.environ['AGENT_OPTIMIZE_MODE'] == '" + expected + "'"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["dispatch_mode"] == expected
