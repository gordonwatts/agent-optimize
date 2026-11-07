import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pytest

import agent_optimize.evaluator as evaluator
import agent_optimize.performance as performance
import agent_optimize.search as search
from agent_optimize.cases import TestCase as Case, generate_cases
from agent_optimize.generator import IDENTITY, propose
from agent_optimize.process import ProcessResult
from agent_optimize.recorder import Recorder
from agent_optimize.replay import replay_session
from agent_optimize.store import create_session, read_json, write_json
from test_operations import integer_ht, problem
from test_dispatch import controlled


def group():
    frozen, graph, contract, arguments = problem()
    return {"source": frozen.source, "metadata": frozen.metadata, "graph": graph, "contract": contract, "cases": [Case("recording-private", arguments)], "mandatory": list(generate_cases(graph, contract, arguments))}


def harness(tmp_path, monkeypatch, *, status="rejected"):
    directory = create_session(tmp_path / "artifacts")
    outcome = {"function_id": "function", "status": "replayed", "evaluations": []}
    report = {"schema_version": 1, "functions": [outcome]}
    checked = []

    def evaluate(source, graph, contract, cases, workspace, **kwargs):
        checked.append(cases)
        workspace.mkdir(parents=True)
        return {"schema_version": 1, "status": status, "reason": "private diagnostic 8877665544332211", "workspace": str(workspace), "checks": [{"name": "private recording 8877665544332211", "reason": "8877665544332211"}]}

    def complete(evaluation, *args, **kwargs):
        write_json(Path(evaluation["workspace"]) / "completed-evaluation.json", evaluation)
        return evaluation

    monkeypatch.setattr(evaluator, "evaluate_contract", evaluate)
    monkeypatch.setattr(performance, "complete_evaluation", complete)
    return directory, outcome, report, checked


def run(directory, outcome, report, *, groups=None, limits=None, deadline=None, proposer=propose):
    search.search_groups(groups or {"contract": group()}, outcome, directory, report, limits or search.SearchLimits(), deadline if deadline is not None else time.monotonic() + 30, proposer=proposer)


def test_mock_is_deterministic_and_feedback_changes_source():
    public = search.provider_problem(group())
    first = propose(public, [])
    assert first == propose(public, [])
    second = propose(public, [{"feedback": {"status": "rejected"}}])
    assert "total + 1" in first["source"] and "total + 1" not in second["source"]
    assert first["extra_cases"]


def test_real_mock_wrong_then_correct_checkpoints_and_records_identity(tmp_path):
    directory = create_session(tmp_path / "artifacts")
    arguments = problem()[3]
    Recorder(directory).invoke(integer_ht, (), arguments)
    report = evaluator.evaluate_session(directory, replay_session(directory, 0), timing_provider=controlled, search_limits=search.SearchLimits(proposals=2))
    outcome = report["functions"][0]
    assert [item["status"] for item in outcome["evaluations"]] == ["rejected", "optimized"]
    paths = sorted((directory / "experiments").rglob("completed-proposal.json"))
    assert len(paths) == 2
    requests = sorted((directory / "experiments").rglob("request.json"))
    assert read_json(requests[1])["history"][0]["feedback"]["status"] == "rejected"
    assert "synthetic_counterexample" in read_json(requests[1])["history"][0]["feedback"]
    accepted = read_json(Path(outcome["evaluations"][1]["artifact_path"]))
    assert accepted["generator"] == IDENTITY
    assert outcome["search"]["stop"] == "proposal_limit"


@pytest.mark.parametrize("response,status", [
    (None, "malformed"), ({}, "malformed"),
    ({"status": "refused", "reason": "private"}, "refused"),
    ({"status": "incomplete"}, "incomplete"), ({"status": "error"}, "provider_error"),
    ({"status": "proposal", "source": "bad", "description": "invalid"}, "invalid_source"),
    ({"status": "proposal", "source": "", "description": "empty"}, "malformed"),
])
def test_response_failures_are_distinct_checkpointed_and_bounded(tmp_path, monkeypatch, response, status):
    directory, outcome, report, checked = harness(tmp_path, monkeypatch)
    calls = []
    def provider(problem, history):
        calls.append(history)
        if len(calls) > 1:
            assert len(list(directory.rglob("completed-proposal.json"))) == len(calls) - 1
        return response
    run(directory, outcome, report, proposer=provider)
    assert len(calls) == 3 and not checked
    assert {item["status"] for item in outcome["evaluations"]} == {status}
    assert len(list(directory.rglob("completed-proposal.json"))) == 3


def test_provider_exception_does_not_expose_message(tmp_path, monkeypatch):
    directory, outcome, report, _ = harness(tmp_path, monkeypatch)
    def provider(problem, history):
        raise RuntimeError("secret-8877665544332211")
    run(directory, outcome, report, proposer=provider)
    assert all(item["status"] == "provider_error" for item in outcome["evaluations"])
    assert "secret" not in json.dumps(outcome)


def test_provider_views_exclude_captured_scalars_arrays_and_raw_feedback(tmp_path, monkeypatch):
    directory, outcome, report, _ = harness(tmp_path, monkeypatch)
    private = group()
    private["cases"][0].arguments["threshold"] = 8877665544332211
    private["cases"][0].arguments["events"] = ak.Array({"Jet": [[{"pt": 8877665544332211, "eta": 0.0}]]})
    run(directory, outcome, report, groups={"one": private})
    serialized = "".join(path.read_text() for path in directory.rglob("request.json"))
    assert "8877665544332211" not in serialized
    assert "recording-private" not in serialized
    assert "synthetic_examples" in serialized and "signature" in serialized
    assert all("8877665544332211" not in json.dumps(item["provider_feedback"]) for item in outcome["evaluations"])


@pytest.mark.parametrize("proposal", ["__import__('os')", {"kind": "code", "source": "raise Exception()"}, {"kind": "lengths", "lengths": [-1]}, {"kind": "lengths", "lengths": [True]}, {"kind": "lengths", "lengths": [4096, 1]}, {"kind": "boundaries", "field": "unknown"}, {"kind": "zeros", "source": "anything"}])
def test_extra_case_vocabulary_rejects_executable_or_out_of_bounds_data(proposal):
    cases, diagnostics = search.extra_cases([proposal], group(), 8)
    assert cases == [] and diagnostics


def test_extra_cases_preserve_schema_and_never_replace_mandatory_cases(tmp_path, monkeypatch):
    directory, outcome, report, checked = harness(tmp_path, monkeypatch)
    current = group()
    def provider(problem, history):
        response = propose(problem, history)
        response["extra_cases"] = [{"kind": "lengths", "lengths": [1, 0, 4]}, {"kind": "boundaries", "field": "pt"}, {"kind": "zeros"}]
        return response
    run(directory, outcome, report, groups={"one": current}, limits=search.SearchLimits(proposals=1, extra_cases=2), proposer=provider)
    assert len(checked[0]) == len(current["cases"]) + len(current["mandatory"]) + 3
    assert checked[0][:len(current["cases"])] == current["cases"]
    assert checked[0][-2].arguments["events"].layout.form.to_dict() == current["contract"].schema
    assert "count limit" in outcome["evaluations"][0]["provider_feedback"]["diagnostics"][0]


@pytest.mark.parametrize("kind", ["request", "response"])
def test_size_limits_before_use(tmp_path, monkeypatch, kind):
    directory, outcome, report, checked = harness(tmp_path, monkeypatch)
    calls = []
    def provider(problem, history):
        calls.append(1)
        return {"status": "proposal", "source": "x" * 1000, "description": "large"}
    limits = search.SearchLimits(request_bytes=1) if kind == "request" else search.SearchLimits(response_bytes=64)
    run(directory, outcome, report, limits=limits, proposer=provider)
    assert outcome["evaluations"][0]["status"] == f"{kind}_limit"
    assert not checked and (not calls if kind == "request" else len(calls) == 3)


def test_function_deadline_does_not_start_provider(tmp_path, monkeypatch):
    directory, outcome, report, checked = harness(tmp_path, monkeypatch)
    def provider(*args):
        pytest.fail("deadline must prevent provider call")
    run(directory, outcome, report, deadline=time.monotonic() - 1, proposer=provider)
    assert outcome["search"] == {"proposals": 0, "stop": "deadline", "provider": IDENTITY}


def test_provider_worker_timeout_is_capped_to_remaining_deadline(tmp_path, monkeypatch):
    directory, outcome, report, _ = harness(tmp_path, monkeypatch)
    observed = []
    def worker(command, **kwargs):
        observed.append(kwargs["timeout"])
        return ProcessResult("timeout", None, "", "")
    monkeypatch.setattr(search, "run_worker", worker)
    run(directory, outcome, report, proposer=None, deadline=time.monotonic() + 2)
    assert all(0 < value <= 2 for value in observed)
    assert all(item["status"] == "provider_timeout" for item in outcome["evaluations"])


@pytest.mark.parametrize("contents,status", [("bad json", "malformed"), ("x" * 500, "response_limit")])
def test_provider_transport_checks_response_file_before_use(tmp_path, monkeypatch, contents, status):
    directory, outcome, report, checked = harness(tmp_path, monkeypatch)
    def worker(command, **kwargs):
        Path(command[-2]).write_text(contents)
        return ProcessResult("passed", 0, "", "")
    monkeypatch.setattr(search, "run_worker", worker)
    run(directory, outcome, report, proposer=None, limits=search.SearchLimits(proposals=1, response_bytes=64))
    assert outcome["evaluations"][0]["status"] == status and not checked


def test_completed_evaluation_checkpoint_failure_prevents_next_provider(tmp_path, monkeypatch):
    directory, outcome, report, _ = harness(tmp_path, monkeypatch)
    calls = []
    def fail(*args, **kwargs):
        raise OSError("evaluation checkpoint failed")
    monkeypatch.setattr(performance, "complete_evaluation", fail)
    def provider(problem, history):
        calls.append(1)
        return propose(problem, history)
    with pytest.raises(OSError, match="evaluation checkpoint failed"):
        run(directory, outcome, report, proposer=provider)
    assert calls == [1]


def test_interrupt_preserves_completed_evaluation_before_next_proposal(tmp_path, monkeypatch):
    directory, outcome, report, _ = harness(tmp_path, monkeypatch, status="optimized")
    def provider(problem, history):
        if history:
            assert read_json(directory / "report.json")["functions"][0]["status"] == "optimized"
            raise KeyboardInterrupt()
        return propose(problem, history)
    with pytest.raises(KeyboardInterrupt):
        run(directory, outcome, report, proposer=provider)
    checkpoints = list(directory.rglob("completed-proposal.json"))
    assert len(checkpoints) == 1 and read_json(checkpoints[0])["status"] == "optimized"


@pytest.mark.parametrize("checkpoint", ["completed-proposal.json", "report.json"])
def test_checkpoint_failure_prevents_next_request(tmp_path, monkeypatch, checkpoint):
    directory, outcome, report, _ = harness(tmp_path, monkeypatch)
    calls = []
    original = search.write_json
    def write(path, value):
        if path.name == checkpoint:
            raise OSError("full disk")
        original(path, value)
    monkeypatch.setattr(search, "write_json", write)
    def provider(problem, history):
        calls.append(1)
        return propose(problem, history)
    with pytest.raises(OSError, match="full disk"):
        run(directory, outcome, report, proposer=provider)
    assert calls == [1]


def test_count_limit_is_per_function_across_contracts(tmp_path, monkeypatch):
    directory, outcome, report, checked = harness(tmp_path, monkeypatch)
    run(directory, outcome, report, groups={"one": group(), "two": group()})
    assert len(checked) == 3
    assert len(list(directory.rglob("request.json"))) == 3


@pytest.mark.parametrize("kwargs", [{"proposals": 0}, {"worker_seconds": float("nan")}, {"function_seconds": -1}, {"session_seconds": float("inf")}, {"response_bytes": True}, {"extra_cases": 33}, {"proposals": 1.5}])
def test_search_limits_validate_finite_positive_bounds(kwargs):
    with pytest.raises(ValueError):
        search.SearchLimits(**kwargs)


def test_empty_capture_never_requests_proposal(tmp_path):
    directory = create_session(tmp_path / "artifacts")
    def provider(*args):
        pytest.fail("no capture must mean no proposal")
    result = evaluator.evaluate_session(directory, {"schema_version": 1, "child_exit_code": 0, "functions": []}, proposer=provider)
    assert result["functions"] == [] and result["search_limits"]["proposals"] == 3


def test_total_search_deadline_applies_across_functions(tmp_path, monkeypatch):
    directory = create_session(tmp_path / "artifacts")
    Recorder(directory).invoke(integer_ht, (), problem()[3])
    report = replay_session(directory, 0)
    report["functions"].append(dict(report["functions"][0]))
    clock = [100.0]
    calls = []
    monkeypatch.setattr(search.time, "monotonic", lambda: clock[0])
    def provider(problem, history):
        calls.append(1)
        clock[0] += 6
        return propose(problem, history)
    report = evaluator.evaluate_session(directory, report, proposer=provider, search_limits=search.SearchLimits(session_seconds=5))
    assert calls == [1]
    assert report["functions"][0]["evaluations"][0]["status"] == "deadline"
    assert report["functions"][1]["search"]["proposals"] == 0


def test_interrupt_after_real_acceptance_preserves_artifact_and_report(tmp_path):
    directory = create_session(tmp_path / "artifacts")
    Recorder(directory).invoke(integer_ht, (), problem()[3])
    def provider(problem, history):
        if history:
            raise KeyboardInterrupt()
        return propose(problem, [{"feedback": "correct first attempt for this test"}])
    with pytest.raises(KeyboardInterrupt):
        evaluator.evaluate_session(directory, replay_session(directory, 0), proposer=provider, timing_provider=controlled)
    report = read_json(directory / "report.json")
    evaluation = report["functions"][0]["evaluations"][0]
    assert evaluation["status"] == "optimized"
    assert read_json(Path(evaluation["artifact_path"]))["generator"] == IDENTITY
    assert len(list(directory.rglob("completed-proposal.json"))) == 1


@pytest.mark.parametrize("limit", ["function_seconds", "session_seconds"])
def test_case_generation_checks_deadline_before_advancing_next_case_or_configuration(tmp_path, monkeypatch, limit):
    directory = create_session(tmp_path / "artifacts")
    recorder = Recorder(directory)
    arguments = problem()[3]
    recorder.invoke(integer_ht, (), arguments)
    recorder.invoke(integer_ht, (), {**arguments, "threshold": 50})
    identifier = next(iter(read_json(directory / "session.json")["functions"]))
    outcome = {"function_id": identifier, "replays": [{"status": "passed"}], "status": "replayed"}
    report = {"schema_version": 1, "child_exit_code": 0, "functions": [outcome]}
    clock = [100.0]
    constructed, configurations = [], []
    original = evaluator.generate_cases
    def generate(*args, **kwargs):
        configurations.append(1)
        for case in original(*args, **kwargs):
            constructed.append(case.name)
            clock[0] += 0.25
            yield case
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(evaluator, "generate_cases", generate)
    def provider(*args):
        pytest.fail("expired case preparation must not request a proposal")
    evaluator.evaluate_session(directory, report, proposer=provider, search_limits=search.SearchLimits(**{limit: 0.5}))
    assert constructed == ["zero-events", "empty-lists"]
    assert configurations == [1]
    assert outcome["search"]["stop"] == "deadline" and outcome["search"]["proposals"] == 0
    assert read_json(directory / "report.json")["functions"][0]["evaluations"][0]["status"] == "ineligible"
