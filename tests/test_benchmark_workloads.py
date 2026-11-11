"""Offline checks: synthetic fixtures match advertised support and semantics."""

import inspect
import json
from pathlib import Path
import subprocess

import awkward as ak
import numpy as np
import pytest

from agent_optimize.candidate import make_contract
from agent_optimize.cases import generate_cases
from agent_optimize.experiment import freeze_reference, UnsupportedReference
from agent_optimize.operations import parse_filter_sum, UnsupportedGraph
from agent_optimize.serialization import plan_value
from benchmarks.compare_models import summarize
from benchmarks import compare_models
from benchmarks.workloads import WORKLOADS, array_digest, make_events, nearest_jet_matching


@pytest.mark.parametrize("name", [name for name, item in WORKLOADS.items() if item.supported])
def test_supported_workloads_reach_real_graph_contract_and_mandatory_cases(name):
    reference = WORKLOADS[name].function.__wrapped__
    arguments = inspect.signature(reference).bind(make_events(8))
    arguments.apply_defaults()
    result = reference(**arguments.arguments)
    graph = parse_filter_sum(freeze_reference(reference).source)
    contract = make_contract(graph, arguments.arguments, plan_value(result, "result", output=True).metadata)
    cases = list(generate_cases(graph, contract, arguments.arguments))
    assert {"zero-events", "empty-lists", "all-pass", "none-pass", "larger-than-recording"} <= {case.name for case in cases}
    assert len(result) == 8
    assert all(case.arguments["events"].layout.form.to_dict() == contract.schema for case in cases)


@pytest.mark.parametrize("name", [name for name, item in WORKLOADS.items() if not item.supported])
def test_unsupported_controls_execute_python_but_cannot_enter_candidate_graph(name):
    reference = WORKLOADS[name].function.__wrapped__
    output = reference(make_events(8))
    assert len(output) == 8
    with pytest.raises((UnsupportedReference, UnsupportedGraph)):
        parse_filter_sum(freeze_reference(reference).source)


def test_seed_and_empty_inputs_preserve_schema_and_digests():
    events = make_events(16, seed=123)
    assert array_digest(events) == array_digest(make_events(16, seed=123))
    assert array_digest(events) != array_digest(make_events(16, seed=124))
    assert events.layout.form == make_events(0).layout.form
    lengths = ak.to_list(ak.num(events.Jet))
    assert lengths[0] == 0 and lengths[1] == 32 and len(set(lengths)) > 2
    assert ak.to_numpy(ak.flatten(events.Jet.score)).dtype == np.dtype("int32")
    assert ak.to_numpy(ak.flatten(events.Jet.bad)).dtype == np.dtype("bool")


def test_matching_wraps_phi_and_handles_missing_reference_jets():
    events = ak.Array([
        {"Jet": [{"eta": 0.0, "phi": 3.13}], "RefJet": [{"eta": 0.0, "phi": -3.13}]},
        {"Jet": [{"eta": 0.0, "phi": 0.0}], "RefJet": []},
        {"Jet": [], "RefJet": []},
    ])
    assert ak.to_list(nearest_jet_matching(events)) == [1, 0, 0]


def test_usage_absence_and_unsupported_fallback_are_reported_truthfully():
    report = {"status": "completed", "child_exit_code": 0, "functions": [{"status": "unsupported"}]}
    result = summarize(report, expected_supported=False)
    assert result["unsupported_fallback_verified"] is True
    assert result["provider_attempts"] == 0
    assert result["http_request_count"] == 0
    assert result["reported_token_totals"]["total_tokens"] == 0
    report["functions"] = [{"status": "reference", "search": {"proposals": 2}, "evaluations": [{"status": "provider_error", "provider_usage": {}, "provider_request_count": None}, {"status": "rejected", "provider_request_count": 1, "provider_usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}]}]
    result = summarize(report, expected_supported=True)
    assert result["provider_attempts"] == 2
    assert result["usage_reported_attempts"] == 1
    assert result["http_request_count"] is None
    assert result["request_count_reported_attempts"] == 1
    assert result["reported_token_totals"]["total_tokens"] == 30
    assert result["reported_token_totals"]["cached_input_tokens"] is None


def test_mock_runner_explicitly_overrides_inherited_live_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_OPTIMIZE_PROVIDER", "openai")
    invocations = []

    def offline_process(command, **kwargs):
        invocations.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps({"status": "completed", "child_exit_code": 0, "functions": [{"status": "reference", "evaluations": [{"status": "ineligible", "correctness_status": "passed"}]}]}))

    monkeypatch.setattr(compare_models, "run_benchmark_process", offline_process)
    output = tmp_path / "comparison"
    assert compare_models.main(["--provider", "mock", "--cases", "integer_threshold", "--output", str(output)]) == 0
    command, kwargs = invocations[0]
    assert command[command.index("--provider") + 1] == "mock"
    assert "--model" not in command
    assert kwargs["timeout"] is None


def test_runner_checkpoints_timeout_and_continues_next_case(monkeypatch, tmp_path):
    invocations = []

    def offline_process(command, **kwargs):
        invocations.append(command)
        assert kwargs["timeout"] is None
        if len(invocations) == 1:
            raise subprocess.TimeoutExpired(command, 12, output=b"partial output")
        return subprocess.CompletedProcess(command, 0, json.dumps({"status": "completed", "child_exit_code": 0, "functions": [{"status": "unsupported"}]}))

    monkeypatch.setattr(compare_models, "run_benchmark_process", offline_process)
    output = tmp_path / "comparison"
    assert compare_models.main(["--provider", "mock", "--cases", "integer_threshold", "dijet_combinatorics", "--output", str(output)]) == 1
    report = json.loads((output / "comparison.json").read_text())
    assert len(invocations) == 2
    first, second = report["results"]
    assert first["status"] == "timed_out"
    assert first["timeout_seconds"] == 12
    assert first["http_request_count"] is None
    assert Path(first["stdout_path"]).read_text() == "partial output"
    assert second["status"] == "completed"


def test_unavailable_model_makes_comparison_incomplete_without_fabricated_measurements(monkeypatch, tmp_path):
    def offline_process(command, **kwargs):
        assert command[command.index("--provider") + 1] == "openai"
        model = command[command.index("--model") + 1]
        evaluation = {"status": "provider_error", "provider_error": "model_unavailable"} if model == "missing-model" else {"status": "rejected", "correctness_status": "rejected"}
        return subprocess.CompletedProcess(command, 0, json.dumps({"status": "completed", "child_exit_code": 0, "functions": [{"status": "reference", "evaluations": [evaluation]}]}))

    monkeypatch.setattr(compare_models, "run_benchmark_process", offline_process)
    output = tmp_path / "comparison"
    assert compare_models.main(["--provider", "openai", "--model", "missing-model", "--model", "working-model", "--cases", "integer_threshold", "--output", str(output)]) == 1
    report = json.loads((output / "comparison.json").read_text())
    assert report["comparison_status"] == "failed"
    assert report["model_outcomes"]["missing-model"]["status"] == "no_evaluated_candidate"
    assert report["model_outcomes"]["working-model"]["status"] == "completed"
    assert report["model_outcomes"]["missing-model"]["supported_runs_with_evaluated_candidates"] == 0
