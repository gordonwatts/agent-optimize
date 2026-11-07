import itertools
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

from agent_optimize.candidate import fused_candidate_source
from agent_optimize.cases import generate_cases, TestCase as EvaluationCase
import agent_optimize.evaluator as evaluator
from agent_optimize.evaluator import compare_results, evaluate_contract, evaluate_session
from agent_optimize.recorder import Recorder
from agent_optimize.replay import replay_session
from agent_optimize.serialization import plan_value, write_value
from agent_optimize.store import create_session, read_json
from test_operations import floating_ht, integer_ht, problem


def evaluate_problem(tmp_path, *, function=integer_ht, dtype=np.int64, transform=None):
    frozen, graph, contract, arguments = problem(function, dtype)
    source = fused_candidate_source(graph, contract)
    if transform:
        source = transform(source)
    cases = itertools.chain([EvaluationCase("recording", arguments)], generate_cases(graph, contract, arguments, observed_max_length=3))
    workspace = tmp_path / "experiment with spaces"
    result = evaluate_contract(frozen.source, graph, contract, cases, workspace, candidate_source=source)
    return result, workspace


@pytest.mark.parametrize("dtype, expected", [(np.int8, "<i8"), (np.int64, "<i8"), (np.uint64, "<u8")])
def test_real_numba_candidate_passes_integer_contract_and_all_required_cases(tmp_path, dtype, expected):
    result, workspace = evaluate_problem(tmp_path, dtype=dtype)
    assert result["status"] == "passed", result
    assert result["correctness_only"] is True
    assert result["contract"]["output_dtype"] == expected
    assert all(check["status"] == "passed" for check in result["checks"])
    assert result["case_count"] == len(result["checks"])
    assert result["candidate_worker"]["compiled_signatures"]
    assert result["candidate_worker"]["fastmath"] is False
    candidate_job = read_json(workspace / "candidate-job.json")
    assert "source_sha256" not in candidate_job
    assert "positional_names" not in candidate_job
    assert "reference_result" not in str(candidate_job)
    assert not list(workspace.rglob("*accepted*"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_reference_and_numba_are_checked_bitwise_on_difficult_cases(tmp_path, dtype):
    result, _ = evaluate_problem(tmp_path, function=floating_ht, dtype=dtype)
    assert result["status"] == "passed", result
    names = {check["name"] for check in result["checks"]}
    assert {"cancellation-order", "large-small-difficult-sums", "negative-zero-sums"} <= names


def test_boolean_sum_uses_observed_awkward_integer_promotion(tmp_path):
    result, _ = evaluate_problem(tmp_path, function=floating_ht, dtype=np.bool_)
    assert result["status"] == "passed", result
    assert result["contract"]["output_dtype"] == "<i8"


@pytest.mark.parametrize("transform, diagnostic", [
    (lambda source: source.replace(" > ", " >= ", 1), "boundaries-pt"),
    (lambda source: source.replace("return output", "return output[output != 0]"), "events"),
    (lambda source: source.replace("np.int64", "np.float64"), "dtype"),
    (lambda source: source.replace("return output", "return list(output)"), "flat NumPy"),
])
def test_wrong_boundaries_dropped_events_and_dtype_are_rejected(tmp_path, transform, diagnostic):
    result, _ = evaluate_problem(tmp_path, transform=transform)
    assert result["status"] == "rejected", result
    if "checks" in result:
        assert any(check["name"] == diagnostic and check["status"] == "failed" for check in result["checks"])
    else:
        assert diagnostic in result["candidate_worker"]["reason"]


def test_candidate_specializing_on_zero_sample_outputs_is_rejected_by_generated_cases(tmp_path):
    result, _ = evaluate_problem(tmp_path, transform=lambda source: "def kernel(offsets, columns, scalars):\n    return np.zeros(len(offsets) - 1, dtype=np.int64)\n")
    assert result["status"] == "rejected", result
    assert next(check for check in result["checks"] if check["name"] == "recording")["status"] == "passed"
    assert any(check["name"] == "all-pass" and check["status"] == "failed" for check in result["checks"])


def test_candidate_cannot_hardcode_a_single_observed_scalar_value(tmp_path):
    result, _ = evaluate_problem(tmp_path, transform=lambda source: source.replace("scalars[0]", "30"))
    assert result["status"] == "rejected", result
    assert all(check["status"] == "passed" for check in result["checks"] if not check["name"].startswith("scalar-"))
    assert any(check["name"].startswith("scalar-threshold-") and check["status"] == "failed" for check in result["checks"])


def test_candidate_cannot_modify_readonly_input_buffers(tmp_path):
    result, _ = evaluate_problem(tmp_path, transform=lambda source: source.replace("    output =", "    columns[0][0] = 0\n    output =", 1))
    assert result["status"] == "rejected", result
    assert result["candidate_worker"]["phase"] == "compilation"
    assert "readonly" in result["candidate_worker"]["reason"].lower()


def test_changed_signed_zero_from_compiled_candidate_is_rejected(tmp_path):
    result, _ = evaluate_problem(tmp_path, function=floating_ht, dtype=np.float64, transform=lambda source: source.replace("output[event] = total", "output[event] = -0.0 if total == 0 else total"))
    assert result["status"] == "rejected", result
    assert any(check["name"] == "empty-lists" and check["status"] == "failed" for check in result["checks"])


def test_changed_floating_sum_order_is_rejected(tmp_path):
    result, _ = evaluate_problem(tmp_path, function=floating_ht, dtype=np.float64, transform=lambda source: source.replace("range(offsets[event], offsets[event + 1])", "range(offsets[event + 1] - 1, offsets[event] - 1, -1)"))
    assert result["status"] == "rejected", result
    assert any(check["name"] == "cancellation-order" and check["status"] == "failed" for check in result["checks"])


@pytest.mark.parametrize("candidate, reason", [
    (np.array([1, 2], dtype=np.int32), "dtype"),
    (np.array([1], dtype=np.int64), "length"),
    (ak.Array([1, 2]), "container"),
    (np.array([[1, 2]], dtype=np.int64), "structure"),
])
def test_comparison_rejects_dtype_container_length_and_structure(candidate, reason):
    passed, diagnostic = compare_results(np.array([1, 2], dtype=np.int64), candidate)
    assert not passed
    assert reason in diagnostic


def test_comparison_is_exact_for_integers_booleans_signed_zero_and_nan_payloads():
    assert compare_results(np.array([2**63 - 1], dtype=np.int64), np.array([2**63 - 2], dtype=np.int64))[0] is False
    assert compare_results(np.array([True]), np.array([False]))[0] is False
    assert compare_results(np.array([0.0]), np.array([-0.0]))[0] is False
    first = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    second = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)
    assert compare_results(first, first.copy())[0] is True
    assert compare_results(first, second)[0] is False


@pytest.mark.parametrize("failure", ["crash", "timeout"])
def test_candidate_worker_crashes_and_timeouts_reject_safely(tmp_path, monkeypatch, failure):
    original_run = evaluator.run_worker

    def injected(command, **kwargs):
        if any("candidate-job.json" in argument for argument in command):
            if failure == "crash":
                return original_run([sys.executable, "-c", "import os; os._exit(19)"], **kwargs)
            return original_run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.1)
        return original_run(command, **kwargs)

    monkeypatch.setattr(evaluator, "run_worker", injected)
    result, _ = evaluate_problem(tmp_path)
    assert result["status"] == "rejected", result
    assert result["candidate_worker"]["phase"] == "worker"
    assert "19" in result["candidate_worker"]["reason"] if failure == "crash" else "timeout" in result["candidate_worker"]["reason"]


def test_frozen_reference_must_match_recording_before_candidate_runs(tmp_path):
    frozen, graph, contract, arguments = problem()
    recording = tmp_path / "retained recording"
    recording.mkdir()
    wrong = plan_value(ak.Array(np.array([999, 999, 999], dtype=np.int64)), "result", output=True)
    write_value(wrong, recording)
    case = EvaluationCase("recording", arguments, wrong.metadata, str(recording))
    workspace = tmp_path / "experiment"
    result = evaluate_contract(frozen.source, graph, contract, [case], workspace)
    assert result["status"] == "reference_failed"
    assert not (workspace / "candidate-job-result.json").exists()


def test_all_recordings_and_each_scalar_configuration_get_boundary_cases(tmp_path, monkeypatch):
    directory = create_session(tmp_path / "artifacts")
    _, _, _, arguments = problem()
    recorder = Recorder(directory)
    recorder.invoke(integer_ht, (), arguments)
    recorder.invoke(integer_ht, (), {**arguments, "threshold": 50})
    report = replay_session(directory, 0)
    from agent_optimize.generator import propose
    from agent_optimize.search import SearchLimits

    def specialized(problem, history):
        response = propose(problem, [{"feedback": "test"}])
        response["source"] = response["source"].replace("scalars[0]", "30")
        return response

    report = evaluate_session(directory, report, proposer=specialized, search_limits=SearchLimits(proposals=1))
    result = report["functions"][0]["evaluations"][0]
    assert result["status"] == "rejected", result
    recordings = [check for check in result["checks"] if check["name"].startswith("recording-")]
    assert len(recordings) == 2 and all(check["status"] == "passed" for check in recordings)
    assert sum(check["name"] == "boundaries-pt" for check in result["checks"]) == 2
    assert any(check["name"] == "boundaries-pt" and check["status"] == "failed" for check in result["checks"])


@pytest.mark.parametrize("phase", ["retained", "candidate"])
@pytest.mark.parametrize("expiration", ["load", "comparison"])
def test_output_verification_stops_host_work_at_deadline(tmp_path, monkeypatch, phase, expiration):
    frozen, graph, contract, arguments = problem()
    clock = [100.0]
    jobs, loads, comparisons = [], [], []
    def run_job(path, **kwargs):
        jobs.append(path.name)
        return {"status": "passed", "environment": {"test": True}, "outputs": [
            {"case_id": f"case-{index:03d}", "result": {"test": index}} for index in range(2)
        ]}
    def load(metadata, path):
        loads.append(path)
        if expiration == "load":
            clock[0] += 0.3
        return np.array([1], dtype=np.int64)
    def compare(left, right, **kwargs):
        comparisons.append(1)
        clock[0] += 0.5
        return True, "exact match"
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(evaluator, "_run_job", run_job)
    monkeypatch.setattr(evaluator, "load_value", load)
    monkeypatch.setattr(evaluator, "compare_results", compare)
    cases = [EvaluationCase(f"case-{index}", arguments, {"test": index} if phase == "retained" else None, str(tmp_path)) for index in range(2)]
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        evaluate_contract(frozen.source, graph, contract, cases, tmp_path / "experiment", deadline=100.5)
    assert len(jobs) == (1 if phase == "retained" else 2)
    assert len(loads) == 2
    assert len(comparisons) == (0 if expiration == "load" else 1)


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_output_comparison_checks_deadline_between_bounded_chunks(monkeypatch, dtype):
    clock = [100.0]
    array = np.zeros(evaluator.CHUNK_BYTES // np.dtype(dtype).itemsize + 1, dtype=dtype)
    work = []
    if dtype == np.int64:
        original = np.array_equal
        def equal(left, right):
            work.append(len(left))
            clock[0] += 0.5
            return original(left, right)
        monkeypatch.setattr(evaluator.np, "array_equal", equal)
    else:
        original = evaluator.buffer_chunks
        def chunks(value):
            for chunk in original(value):
                work.append(len(chunk))
                clock[0] += 0.25
                yield chunk
        monkeypatch.setattr(evaluator, "buffer_chunks", chunks)
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: clock[0])
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        compare_results(array, array.copy(), deadline=100.5)
    assert len(work) == (1 if dtype == np.int64 else 2)
