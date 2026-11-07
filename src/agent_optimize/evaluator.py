"""Independent correctness checks for the first compiled filter/sum candidate."""

import ast
from dataclasses import asdict
import hashlib
import itertools
from pathlib import Path
import sys
import time

import awkward as ak
import numpy as np

from .candidate import collection_layout, fused_candidate_source, make_contract
from .cases import generate_cases, TestCase
from .experiment import UnsupportedReference, validate_frozen_source
from .operations import parse_filter_sum, UnsupportedGraph
from .process import run_worker
from .serialization import CHUNK_BYTES, buffer_chunks, load_value, plan_value, UnsupportedValue, write_value
from .store import json_bytes, read_json, write_json


def _check_deadline(deadline, phase):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"experiment deadline exhausted while {phase}")


def _with_deadline(values, deadline, phase):
    """Check before advancing a lazy producer as well as after its bounded work."""
    iterator = iter(values)
    while True:
        _check_deadline(deadline, phase)
        try:
            value = next(iterator)
        except StopIteration:
            return
        _check_deadline(deadline, phase)
        yield value


def compare_results(reference, candidate, *, deadline=None):
    """Match containers/structure/dtype first; floats then match every bit."""
    _check_deadline(deadline, "checking output structure")
    try:
        expected = plan_value(reference, "result", output=True)
        actual = plan_value(candidate, "result", output=True)
    except UnsupportedValue as error:
        return False, f"unsupported output structure: {error}"
    _check_deadline(deadline, "checking output structure")
    if type(reference) is not type(candidate) or expected.metadata["kind"] != actual.metadata["kind"]:
        return False, "output container kind differs from the reference"
    for field in ("length", "dtype", "form"):
        if expected.metadata.get(field) != actual.metadata.get(field):
            return False, f"output {field} differs from the reference"
    left = reference.layout.data if type(reference) is ak.Array else reference
    right = candidate.layout.data if type(candidate) is ak.Array else candidate
    equal = True
    if left.dtype.kind in "biu":
        count = max(1, CHUNK_BYTES // left.dtype.itemsize)
        for start in _with_deadline(range(0, len(left), count), deadline, "comparing output values"):
            equal = np.array_equal(left[start:start + count], right[start:start + count])
            if not equal:
                break
    else:
        chunks = itertools.zip_longest(buffer_chunks(left), buffer_chunks(right))
        for first, second in _with_deadline(chunks, deadline, "comparing output bits"):
            if first != second:
                equal = False
                break
    _check_deadline(deadline, "comparing output values")
    return (True, "exact output match") if equal else (False, "output values/float bit patterns differ from the reference")


def _run_job(path, *, timeout):
    result_path = path.with_name(path.stem + "-result.json")
    process = run_worker([sys.executable, "-m", "agent_optimize.evaluation_worker", str(path.resolve()), str(result_path.resolve())], timeout=timeout)
    if process.status not in ("passed", "failed"):
        return {"schema_version": 1, "status": "failed", "phase": "worker", "reason": f"worker {process.status}", "returncode": process.returncode}
    if not result_path.exists():
        return {"schema_version": 1, "status": "failed", "phase": "worker", "reason": f"worker exited {process.returncode} without a result", "returncode": process.returncode}
    result = read_json(result_path)
    if process.returncode and result.get("status") == "passed":
        return {"schema_version": 1, "status": "failed", "phase": "worker", "reason": "worker exit status disagrees with its result"}
    return result


def evaluate_contract(reference_source, graph, contract, cases, workspace: Path, *, candidate_source=None, worker_timeout=30, deadline=None):
    """Keep the frozen reference and mandatory cases outside candidate control.

    This records correctness only. It makes no timing, eligibility, persistence,
    or automatic-dispatch decision.
    """
    workspace.mkdir(parents=True)
    tree, function_name = validate_frozen_source(reference_source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    positional_names = [argument.arg for argument in function.args.posonlyargs]
    source = candidate_source if candidate_source is not None else fused_candidate_source(graph, contract)
    (workspace / "reference.py").write_bytes(reference_source.encode("utf-8"))
    (workspace / "candidate.py").write_bytes(source.encode("utf-8"))
    case_root = workspace / "cases"
    case_root.mkdir()
    descriptions, recordings = [], {}
    for index, case in enumerate(_with_deadline(cases, deadline, "preparing cases")):
        if index >= 512:
            raise UnsupportedGraph("correctness evaluation is limited to 512 cases per contract")
        identifier = f"case-{index:03d}"
        directory = case_root / identifier
        directory.mkdir()
        plans = {name: plan_value(value, f"arg{number}") for number, (name, value) in enumerate(case.arguments.items())}
        for plan in plans.values():
            _check_deadline(deadline, "writing case inputs")
            write_value(plan, directory)
        write_json(directory / "inputs.json", {"schema_version": 1, "arguments": {name: plan.metadata for name, plan in plans.items()}})
        descriptions.append({"id": identifier, "name": case.name})
        if case.recorded_output is not None:
            recordings[identifier] = (case.recorded_output, Path(case.recorded_directory))
    if not descriptions:
        raise UnsupportedGraph("no captured or generated cases are available")
    base = {"schema_version": 1, "contract": asdict(contract), "cases": descriptions}
    # Candidate jobs intentionally contain no reference source/hash/output metadata.
    write_json(workspace / "reference-job.json", {
        **base, "kind": "reference", "output_directory": "reference-outputs",
        "source_sha256": hashlib.sha256(reference_source.encode("utf-8")).hexdigest(),
        "positional_names": positional_names,
    })
    candidate_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    write_json(workspace / "candidate-job.json", {**base, "kind": "candidate", "candidate_sha256": candidate_sha256, "output_directory": "candidate-outputs"})
    summary = {"schema_version": 1, "status": "failed", "correctness_only": True, "graph": asdict(graph), "contract": asdict(contract), "case_count": len(descriptions), "recorded_case_ids": list(recordings), "workspace": str(workspace.resolve()), "candidate_sha256": candidate_sha256}
    def remaining_timeout():
        if deadline is None:
            return worker_timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("experiment deadline exhausted before worker launch")
        return min(worker_timeout, remaining)
    reference = _run_job(workspace / "reference-job.json", timeout=remaining_timeout())
    if reference["status"] != "passed":
        summary.update(status="reference_failed", reason="frozen reference worker failed", reference_worker=reference)
        write_json(workspace / "evaluation.json", summary)
        return summary
    expected = {output["case_id"]: output for output in reference["outputs"]}
    for identifier, (metadata, directory) in _with_deadline(recordings.items(), deadline, "verifying retained outputs"):
        replayed = load_value(expected[identifier]["result"], workspace / "reference-outputs" / identifier)
        _check_deadline(deadline, "loading retained outputs")
        retained = load_value(metadata, directory)
        _check_deadline(deadline, "loading retained outputs")
        matched, reason = compare_results(retained, replayed, deadline=deadline)
        _check_deadline(deadline, "verifying retained outputs")
        if not matched:
            summary.update(status="reference_failed", reason=f"frozen reference disagrees with retained recording: {reason}")
            write_json(workspace / "evaluation.json", summary)
            return summary
    candidate = _run_job(workspace / "candidate-job.json", timeout=remaining_timeout())
    summary["candidate_worker"] = {key: value for key, value in candidate.items() if key != "outputs"}
    if candidate["status"] != "passed":
        summary.update(status="rejected", reason="candidate worker failed")
        write_json(workspace / "evaluation.json", summary)
        return summary
    if candidate.get("environment") != reference.get("environment") or not candidate.get("environment"):
        summary.update(status="rejected", reason="reference and candidate worker environments differ")
        write_json(workspace / "evaluation.json", summary)
        return summary
    summary["environment"] = candidate["environment"]
    actual = {output["case_id"]: output for output in candidate["outputs"]}
    checks = []
    for case in _with_deadline(descriptions, deadline, "checking candidate outputs"):
        identifier = case["id"]
        if identifier not in expected or identifier not in actual:
            matched, reason = False, "worker omitted a required case"
        else:
            left = load_value(expected[identifier]["result"], workspace / "reference-outputs" / identifier)
            _check_deadline(deadline, "loading candidate outputs")
            right = load_value(actual[identifier]["result"], workspace / "candidate-outputs" / identifier)
            _check_deadline(deadline, "loading candidate outputs")
            matched, reason = compare_results(left, right, deadline=deadline)
        _check_deadline(deadline, "checking candidate outputs")
        checks.append({"case_id": identifier, "name": case["name"], "status": "passed" if matched else "failed", "reason": reason})
    summary.update(
        status="passed" if all(check["status"] == "passed" for check in checks) else "rejected",
        reason="candidate passed strict correctness checks; performance and reuse have not been evaluated" if all(check["status"] == "passed" for check in checks) else "candidate disagreed with the frozen reference; Python remains in use",
        checks=checks,
    )
    write_json(workspace / "evaluation.json", summary)
    return summary


def evaluate_session(directory: Path, report: dict, *, worker_timeout=30, timing_provider=None, experiment_timeout=120, search_limits=None, proposer=None, provider=None, api_key=None, dispatch_mode=None):
    from .generator import IDENTITY
    from .search import SearchLimits, search_groups
    identity = provider.identity if provider else IDENTITY
    from .modes import dispatch_mode as select_mode
    dispatch_mode = select_mode(dispatch_mode)
    report["dispatch_mode"] = dispatch_mode
    report["provider"] = identity

    limits = search_limits or SearchLimits(worker_seconds=worker_timeout, function_seconds=experiment_timeout)
    report["search_limits"] = asdict(limits)
    session_deadline = time.monotonic() + limits.session_seconds
    state = read_json(directory / "session.json")
    for outcome in report["functions"]:
        deadline = min(time.monotonic() + limits.function_seconds, session_deadline)
        outcome["evaluations"] = []
        if report["child_exit_code"] or not outcome["replays"] or any(replay["status"] != "passed" for replay in outcome["replays"]):
            continue
        if time.monotonic() >= deadline:
            outcome["search"] = {"proposals": 0, "stop": "deadline", "provider": identity}
            outcome["reason"] = "search deadline exhausted; Python remains in use"
            continue
        entry = state["functions"][outcome["function_id"]]
        groups = {}
        try:
            for call_id in _with_deadline(entry["captures"], deadline, "loading captures"):
                call_directory = directory / "calls" / call_id
                capture = read_json(call_directory / "capture.json")
                reference_directory = directory / "references" / capture["reference_fingerprint"]
                source = (reference_directory / "reference.py").read_text(encoding="utf-8")
                metadata = read_json(reference_directory / "reference.json")
                if hashlib.sha256(source.encode("utf-8")).hexdigest() != metadata["source_sha256"]:
                    raise UnsupportedGraph("frozen reference source fingerprint mismatch")
                graph = parse_filter_sum(source)
                arguments = {name: load_value(value, call_directory) for name, value in capture["arguments"].items()}
                contract = make_contract(graph, arguments, capture["reference_result"])
                group_id = hashlib.sha256(json_bytes({"fingerprint": capture["reference_fingerprint"], "contract": asdict(contract)})).hexdigest()[:24]
                group = groups.setdefault(group_id, {"source": source, "metadata": metadata, "graph": graph, "contract": contract, "cases": [], "maximum": 0})
                group["cases"].append(TestCase(f"recording-{call_id}", arguments, capture["reference_result"], str(call_directory)))
                offsets = collection_layout(arguments[graph.array_argument], graph.collection_path).offsets.data
                group["maximum"] = max(group["maximum"], int(np.max(np.diff(offsets))) if len(offsets) > 1 else 0)
            for group_id, group in _with_deadline(groups.items(), deadline, "preparing contracts"):
                configurations = {}
                for case in _with_deadline(group["cases"], deadline, "preparing scalar configurations"):
                    scalar_key = json_bytes({name: plan_value(case.arguments[name], "scalar").metadata for name in group["contract"].scalar_names})
                    configurations.setdefault(scalar_key, case.arguments)
                synthetic = itertools.chain.from_iterable(
                    generate_cases(group["graph"], group["contract"], arguments, observed_max_length=group["maximum"])
                    for arguments in _with_deadline(configurations.values(), deadline, "generating scalar configurations")
                )
                group["mandatory"] = list(itertools.islice(_with_deadline(synthetic, deadline, "generating mandatory cases"), 513))
                if len(group["cases"]) + len(group["mandatory"]) >= 512:
                    raise UnsupportedGraph("correctness evaluation is limited to 512 cases per contract")
            if groups:
                search_groups(groups, outcome, directory, report, limits, deadline, proposer=proposer, timing_provider=timing_provider, provider=provider, api_key=api_key, dispatch_mode=dispatch_mode)
        except (UnsupportedGraph, UnsupportedReference, UnsupportedValue, ValueError) as error:
            outcome["evaluations"].append({"schema_version": 1, "status": "unsupported", "correctness_only": True, "reason": str(error)[:500]})
            outcome["reason"] = "compiled candidate unavailable; Python remains in use"
        except TimeoutError:
            outcome["search"] = {"proposals": len(outcome["evaluations"]), "stop": "deadline", "provider": identity}
            outcome["evaluations"].append({"schema_version": 1, "status": "ineligible", "correctness_only": False, "reason": "experiment deadline exhausted"})
            outcome["reason"] = "experiment deadline exhausted; Python remains in use"
    write_json(directory / "report.json", report)
    return report
