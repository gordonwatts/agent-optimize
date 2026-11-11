from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import agent_optimize.artifacts as artifacts
import agent_optimize.dispatch as dispatch
import agent_optimize.evaluator as evaluator
import agent_optimize.performance as performance
from agent_optimize.evaluator import evaluate_contract, evaluate_session
from agent_optimize.experiment import location_key
from agent_optimize.operations import graph_from_dict
from agent_optimize.candidate import contract_from_dict
from agent_optimize.cases import TestCase as EvaluationCase
from agent_optimize.purity import validate_reusable_source
from agent_optimize.serialization import load_value
from agent_optimize.store import create_session, read_json
from agent_optimize.recorder import Recorder
from agent_optimize.replay import replay_session
from test_dispatch import accepted, accepted_base, controlled, fast_kernel, invoke


def copy_evaluation(case, tmp_path, name="evaluation"):
    original = Path(case["evaluation"]["workspace"])
    workspace = tmp_path / name
    shutil.copytree(original, workspace, ignore=shutil.ignore_patterns("prospective-store", "timing-*", "completed-evaluation.json"))
    evaluation = read_json(workspace / "evaluation.json")
    evaluation["workspace"] = str(workspace)
    metadata = read_json(next((case["session"] / "references").glob("*/reference.json")))
    return evaluation, metadata


def test_completed_checkpoint_precedes_index_change_and_failure_preserves_old_artifact(accepted, fast_kernel, tmp_path, monkeypatch):
    evaluation, metadata = copy_evaluation(accepted, tmp_path)
    index = accepted["root"] / "accepted" / accepted["function_id"] / "index.json"
    original = index.read_bytes()
    write = performance.write_json

    def failing(path, value):
        if path.name == "completed-evaluation.json":
            raise OSError("checkpoint unavailable")
        return write(path, value)

    monkeypatch.setattr(performance, "write_json", failing)
    with pytest.raises(OSError, match="checkpoint"):
        performance.complete_evaluation(evaluation, metadata, accepted["function_id"], accepted["root"], timing_provider=controlled)
    assert index.read_bytes() == original
    assert invoke(accepted)["used"]


def test_atomic_index_failure_preserves_previous_accepted_artifact_and_completed_record(accepted, fast_kernel, tmp_path, monkeypatch):
    evaluation, metadata = copy_evaluation(accepted, tmp_path)
    index = accepted["root"] / "accepted" / accepted["function_id"] / "index.json"
    original = index.read_bytes()
    write = artifacts.write_json

    def failing(path, value):
        if path.name == "index.json":
            raise OSError("index unavailable")
        return write(path, value)

    monkeypatch.setattr(artifacts, "write_json", failing)
    result = performance.complete_evaluation(evaluation, metadata, accepted["function_id"], accepted["root"], timing_provider=controlled)
    assert result["status"] == "failed"
    assert index.read_bytes() == original
    assert read_json(Path(evaluation["workspace"]) / "completed-evaluation.json")["timing"]["eligibility"]["eligible"]
    assert invoke(accepted)["used"]


def test_later_interrupt_preserves_completed_evaluation_and_accepted_implementation(accepted, fast_kernel, tmp_path):
    first, metadata = copy_evaluation(accepted, tmp_path, "first")
    result = performance.complete_evaluation(first, metadata, accepted["function_id"], accepted["root"], timing_provider=controlled)
    assert result["status"] == "optimized"
    second, _ = copy_evaluation(accepted, tmp_path, "second")

    def interrupt(evaluation):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        performance.complete_evaluation(second, metadata, accepted["function_id"], accepted["root"], timing_provider=interrupt)
    assert read_json(Path(first["workspace"]) / "completed-evaluation.json")["status"] == "optimized"
    assert invoke(accepted)["used"]


def test_new_accepted_identity_replaces_quarantined_candidate(accepted, fast_kernel, tmp_path, monkeypatch):
    compile_source = dispatch.compile_kernel
    monkeypatch.setattr(dispatch, "compile_kernel", lambda source: (_ for _ in ()).throw(RuntimeError("failure")))
    assert not invoke(accepted)["used"]
    assert (accepted["root"] / "quarantine" / f"{accepted['artifact_id']}.json").exists()
    evaluation, metadata = copy_evaluation(accepted, tmp_path)
    result = performance.complete_evaluation(evaluation, metadata, accepted["function_id"], accepted["root"], timing_provider=controlled)
    assert result["status"] == "optimized"
    assert result["artifact_id"] != accepted["artifact_id"]
    monkeypatch.setattr(dispatch, "compile_kernel", compile_source)
    assert invoke(accepted)["used"]


@pytest.mark.parametrize("source", [
    "def kernel(offsets, columns, scalars):\n    columns[0][0] = 1\n    return columns[0]\n",
    "def kernel(offsets, columns, scalars):\n    alias = columns[0]\n    alias[0] = 1\n    return alias\n",
    "def kernel(offsets, columns, scalars):\n    output = np.empty(1, dtype=np.int64)\n    print('external effect')\n    return output\n",
    "def kernel(offsets, columns, scalars):\n    import os\n    return columns[0]\n",
    "def kernel(offsets, columns, scalars):\n    output = np.empty(1, dtype=np.int64)\n    output = columns[0]\n    output[0] = 1\n    return output\n",
])
def test_input_mutation_aliases_and_external_side_effects_are_not_reusable(source):
    with pytest.raises(ValueError):
        validate_reusable_source(source)


def test_changed_candidate_source_after_correctness_cannot_be_accepted(accepted, tmp_path):
    evaluation, metadata = copy_evaluation(accepted, tmp_path)
    (Path(evaluation["workspace"]) / "candidate.py").write_text("def kernel(offsets, columns, scalars):\n    return columns[0]\n", encoding="utf-8")
    result = performance.complete_evaluation(evaluation, metadata, accepted["function_id"], accepted["root"], timing_provider=controlled)
    assert result["status"] == "rejected"
    assert "source changed" in result["reason"]


def test_correctness_passing_candidate_with_print_is_rejected_before_timing_or_acceptance(accepted, tmp_path):
    evaluation, metadata = copy_evaluation(accepted, tmp_path, "copy")
    workspace = Path(evaluation["workspace"])
    job = read_json(workspace / "reference-job.json")
    first_case = job["cases"][0]
    directory = workspace / "cases" / first_case["id"]
    inputs = read_json(directory / "inputs.json")
    arguments = {name: load_value(value, directory) for name, value in inputs["arguments"].items()}
    source = (workspace / "candidate.py").read_text().replace("    output =", "    print('unwanted effect')\n    output =", 1)
    checked = evaluate_contract((workspace / "reference.py").read_text(), graph_from_dict(evaluation["graph"]), contract_from_dict(evaluation["contract"]), [EvaluationCase("recording", arguments)], tmp_path / "print evaluation", candidate_source=source)
    assert checked["status"] == "passed", checked

    def unexpected(evaluation):
        raise AssertionError("unsafe candidate reached timing")

    result = performance.complete_evaluation(checked, metadata, accepted["function_id"], accepted["root"], timing_provider=unexpected)
    assert result["status"] == "rejected"
    assert "safe for reuse" in result["reason"]
