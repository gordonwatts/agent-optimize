"""Bounded sequential proposal search and the provider's deliberately small view."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import awkward as ak
import numpy as np

from .cases import TestCase, build_schema_array, generate_cases
from .generator import IDENTITY
from .diagnostics import code_diagnostic
from .process import run_worker
from .provider import contains_credential
from .store import json_bytes, read_json, write_json
from .usage import normalize_usage, estimate_cost, update_usage


@dataclass(frozen=True)
class SearchLimits:
    proposals: int = 3
    request_bytes: int = 128 * 1024
    response_bytes: int = 64 * 1024
    extra_cases: int = 8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"search limit {name} must be finite and positive")
            if name in ("proposals", "request_bytes", "response_bytes", "extra_cases") and type(value) is not int:
                raise ValueError(f"search limit {name} must be an integer")
        if self.extra_cases > 32:
            raise ValueError("at most 32 extra cases per proposal are supported")


def public_arguments(contract, lengths=(0, 1, 3)):
    """Independent of recordings, including scalar values and list lengths."""
    values = {}
    for name, descriptor in contract.scalar_types.items():
        kind = descriptor["type"]
        values[name] = np.dtype(descriptor["dtype"]).type(1) if "dtype" in descriptor else {"bool": bool, "int": int, "float": float}[kind](1)
    values[contract.array_argument] = build_schema_array(contract.schema, list(lengths), contract.collection_path, {})
    return values


def provider_problem(group):
    contract = group["contract"]
    arguments = public_arguments(contract)
    example = {name: ak.to_list(value) if type(value) is ak.Array else value.item() if isinstance(value, np.generic) else value for name, value in arguments.items()}
    return {
        "source": group["source"], "graph": asdict(group["graph"]), "contract": asdict(contract),
        "signature": "def kernel(offsets, columns, scalars)",
        "constraints": ["numeric read-only inputs", "separately allocated flat output", "exact dtype and float bits", "no imports, I/O, file changes, installations, or test changes", "evaluator alone decides acceptance"],
        "synthetic_examples": [{"arguments": example, "origin": "schema-only zeros, scalar one, fixed lengths"}],
        "extra_case_vocabulary": {"zeros": {}, "lengths": {"lengths": "0..64 lengths, each 0..4096, total <=4096"}, "boundaries": {"field": list(contract.fields)}},
    }


def extra_cases(proposals, group, limit, *, deadline=None):
    """Interpret data, never code. Mandatory cases are outside this interface."""
    from .evaluator import _with_deadline

    diagnostics, cases = [], []
    if not isinstance(proposals, list):
        return [], ["extra cases must be a list"]
    if len(proposals) > limit:
        diagnostics.append("extra case count limit exceeded")
    for index, proposal in enumerate(_with_deadline(proposals[:limit], deadline, "building proposed cases")):
        try:
            if type(proposal) is not dict:
                raise ValueError()
            kind = proposal.get("kind")
            if kind == "zeros" and set(proposal) == {"kind"}:
                arguments = public_arguments(group["contract"])
            elif kind == "lengths" and set(proposal) == {"kind", "lengths"}:
                lengths = proposal["lengths"]
                if type(lengths) is not list or len(lengths) > 64 or any(type(value) is not int or value < 0 or value > 4096 for value in lengths) or sum(lengths) > 4096:
                    raise ValueError()
                arguments = public_arguments(group["contract"], lengths)
            elif kind == "boundaries" and set(proposal) == {"kind", "field"} and proposal["field"] in group["contract"].fields:
                base = public_arguments(group["contract"])
                generated = _with_deadline(generate_cases(group["graph"], group["contract"], base), deadline, "generating proposed boundaries")
                selected = next(case for case in generated if case.name == f"boundaries-{proposal['field']}")
                arguments = selected.arguments
            else:
                raise ValueError()
            cases.append(TestCase(f"provider-extra-{index}", arguments))
        except (ValueError, TypeError, KeyError, StopIteration):
            diagnostics.append(f"extra case {index} is unsupported or outside bounds")
    return cases, diagnostics


def _billing_evidence(response, status):
    """Retain only bounded public accounting fields from an invalid proposal."""
    import re
    if type(response) is not dict:
        return None
    count = response.get("provider_request_count")
    usage = normalize_usage(response.get("provider_usage"))
    if not usage and not (type(count) is int and count >= 0):
        return None
    model = response.get("provider_response_model")
    tier = response.get("provider_service_tier", "default")
    result = {"status": status, "provider_usage": usage,
              "provider_request_count": count if type(count) is int and count >= 0 else None,
              "provider_service_tier": tier if tier in ("default", "flex", "priority", "fast", "ultrafast") else "unknown"}
    if model is not None:
        result["provider_response_model"] = model if isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,100}", model) else "unknown"
    return result


def request_proposal(request, path, limits, deadline, proposer=None, *, provider=None, api_key=None):
    """Default transport is a bounded local subprocess. Injection is for tests."""
    encoded = json_bytes(request)
    if len(encoded) > limits.request_bytes:
        return None, "request_limit"
    write_json(path / "request.json", request)
    remaining = provider.timeout if provider and provider.kind == "openai" else None
    if deadline is not None:
        remaining = deadline - time.monotonic() if remaining is None else min(remaining, deadline - time.monotonic())
    if remaining is not None and remaining <= 0:
        return None, "deadline"
    try:
        if proposer is None:
            response_path = path / "response.json"
            module = "agent_optimize.generator"
            private = {}
            if provider and provider.kind == "openai":
                from dataclasses import replace
                module = "agent_optimize.openai_provider"
                private["input_data"] = json_bytes({"key": api_key, "settings": asdict(replace(provider, timeout=remaining))})
            process = run_worker([sys.executable, "-m", module, str((path / "request.json").resolve()), str(response_path.resolve()), str(limits.response_bytes)], timeout=remaining, **private)
            if process.status != "passed":
                return None, "provider_timeout" if process.status == "timeout" else "provider_error"
            if not response_path.exists():
                return None, "malformed"
            # The envelope is bounded too, before reading its contents.
            if response_path.stat().st_size > limits.response_bytes + 128:
                return None, "response_limit"
            try:
                response = read_json(response_path).get("response")
            except (ValueError, UnicodeError):
                return None, "malformed"
        else:
            # Copy only the serialized provider view, never evaluator objects.
            detached = json.loads(encoded)
            response = proposer(detached["problem"], detached["history"])
        if deadline is not None and time.monotonic() >= deadline:
            return _billing_evidence(response, "deadline"), "deadline"
        encoded_response = json_bytes(response)
    except Exception:
        return None, "provider_error"
    if len(encoded_response) > limits.response_bytes:
        return _billing_evidence(response, "response_limit"), "response_limit"
    if contains_credential(response, api_key):
        return None, "provider_error"
    write_json(path / "response.json", {"schema_version": 1, "response": response})
    if type(response) is not dict:
        return None, "malformed"
    status = response.get("status")
    if status in ("refused", "incomplete", "error", "response_limit", "provider_timeout", "malformed"):
        return response, "provider_error" if status == "error" else status
    if status != "proposal" or type(response.get("source")) is not str or not response["source"].strip() or type(response.get("description")) is not str:
        return _billing_evidence(response, "malformed"), "malformed"
    return response, None


def feedback(evaluation, diagnostics, public_example=None):
    """Send code diagnostics, not runtime errors or recorded input evidence."""
    status = evaluation["status"]
    result = {"status": status, "diagnostics": diagnostics}
    worker = evaluation.get("candidate_worker", {})
    compiler = worker.get("compiler_diagnostic")
    validation = evaluation.get("validation_diagnostic")
    if worker.get("phase") == "compilation" and isinstance(compiler, str) and compiler:
        result.update(phase="compilation", diagnostic="Numba could not compile the proposed code. Correct it using the compiler diagnostic.",
                      compiler_diagnostic=code_diagnostic(compiler))
    elif status == "invalid_source" and isinstance(validation, str) and validation:
        result.update(phase="source_validation", diagnostic="The proposed code failed source validation. Correct it using the validation diagnostic.",
                      validation_diagnostic=code_diagnostic(validation))
    elif evaluation.get("correctness_status", status) == "rejected":
        result["diagnostic"] = "candidate failed independent correctness checks; preserve exact reduction semantics and all events"
    elif status in ("ineligible", "optimized"):
        result["diagnostic"] = "correctness passed; eligible implementation saved" if status == "optimized" else "candidate did not satisfy timing eligibility"
    else:
        result["diagnostic"] = "proposal could not complete evaluation; reference remains available"
    if public_example is not None and any(check.get("name") == "public-synthetic" and check.get("status") == "failed" for check in evaluation.get("checks", [])):
        result["synthetic_counterexample"] = public_example
    return result


def search_groups(groups, outcome, directory, report, limits, deadline, *, proposer=None, timing_provider=None, provider=None, api_key=None, dispatch_mode="careful"):
    from .evaluator import evaluate_contract
    from .performance import complete_evaluation
    from .purity import validate_reusable_source
    identity = provider.identity if provider else IDENTITY

    histories = {key: [] for key in groups}
    group_items = list(groups.items())
    stop = "proposal_limit"
    for number in range(limits.proposals):
        if deadline is not None and time.monotonic() >= deadline:
            stop = "deadline"
            break
        group_id, group = group_items[number % len(group_items)]
        path = directory / "experiments" / outcome["function_id"] / f"proposal-{number:03d}-{group_id}"
        path.mkdir(parents=True)
        request = {"schema_version": 1, "provider": identity, "problem": provider_problem(group), "history": histories[group_id]}
        attempt_id = path.relative_to(directory).as_posix()
        outcome["pending_attempt"] = {"attempt_id": attempt_id, "generator": identity, "status": "requesting", "provider_request_count": 0 if identity.get("kind") == "mock" else None}
        update_usage(report)
        write_json(directory / "report.json", report)
        response, failure = request_proposal(request, path, limits, deadline, proposer, provider=provider, api_key=api_key)
        diagnostics = []
        raw_usage = response.get("provider_usage", {}) if response else {}
        usage = normalize_usage(raw_usage)
        request_count = response.get("provider_request_count") if response else None
        if type(request_count) is not int or request_count < 0:
            request_count = None
        diagnostic = response.get("provider_error") if response else None
        if diagnostic not in {"authentication_error", "permission_denied", "rate_limit", "connection_error", "invalid_request", "model_unavailable"}:
            diagnostic = failure
        if not provider or provider.kind == "mock" or failure == "request_limit":
            request_count = 0
        tier = response.get("provider_service_tier", "default") if response else "unknown"
        response_model = response.get("provider_response_model", identity.get("model")) if response else identity.get("model")
        cost_identity = identity
        if identity.get("kind") == "openai" and response_model != identity.get("model"):
            # A dated snapshot of the requested model follows the same model-page rates.
            import re
            if not isinstance(response_model, str) or not re.fullmatch(re.escape(identity["model"]) + r"-\d{4}-\d{2}-\d{2}", response_model):
                cost_identity = {**identity, "model": "unknown"}
        provider_metadata = {"attempt_id": attempt_id, "provider_cost": estimate_cost(usage, cost_identity, request_count, service_tier=tier), "provider_usage": usage, "provider_request_count": request_count, **({"provider_error": diagnostic} if failure else {})}
        outcome["pending_attempt"].update(provider_metadata, status="evaluating")
        update_usage(report)
        write_json(directory / "report.json", report)
        if failure:
            response = None
        evaluation = {"schema_version": 1, "status": failure, "reason": f"proposal {failure}; Python remains in use", "generator": identity, **provider_metadata}
        if response is not None:
            try:
                proposed, diagnostics = extra_cases(response.get("extra_cases", []), group, limits.extra_cases, deadline=deadline)
            except TimeoutError:
                evaluation.update(status="deadline", reason="search deadline exhausted; Python remains in use")
                response = None
        if response is not None:
            room = max(0, 511 - len(group["cases"]) - len(group["mandatory"]))
            if len(proposed) > room:
                diagnostics.append("extra cases exceed remaining correctness case capacity")
                proposed = proposed[:room]
            try:
                validate_reusable_source(response["source"])
            except (ValueError, SyntaxError, RecursionError) as error:
                evaluation.update(status="invalid_source", reason="candidate source does not satisfy the fixed computation interface",
                                  validation_diagnostic=code_diagnostic(f"{type(error).__name__}: {error}"))
            else:
                try:
                    public_case = TestCase("public-synthetic", public_arguments(group["contract"]))
                    evaluation = evaluate_contract(group["source"], group["graph"], group["contract"], [*group["cases"], *group["mandatory"], public_case, *proposed], path / "evaluation", candidate_source=response["source"], deadline=deadline)
                    evaluation["generator"] = identity
                    evaluation.update(provider_metadata)
                    evaluation = complete_evaluation(evaluation, group["metadata"], outcome["function_id"], directory.parent.parent, timing_provider=timing_provider, deadline=deadline, dispatch_mode=dispatch_mode)
                except TimeoutError:
                    evaluation = {"schema_version": 1, "status": "deadline", "reason": "search deadline exhausted; Python remains in use", "generator": identity, **provider_metadata}
        safe_feedback = feedback(evaluation, diagnostics, request["problem"]["synthetic_examples"][0])
        evaluation["provider_feedback"] = safe_feedback
        # No next provider request occurs until BOTH local checkpoints succeed.
        write_json(path / "completed-proposal.json", evaluation)
        outcome["evaluations"].append(evaluation)
        outcome.pop("pending_attempt", None)
        if evaluation["status"] == "optimized":
            suffix = "run with AGENT_OPTIMIZE_MODE=careful to reuse it" if dispatch_mode == "careful" else "run the script normally to reuse it"
            outcome.update(status="optimized", reason=f"eligible {dispatch_mode} implementation saved; {suffix}")
        histories[group_id].append({"attempt": number + 1, "feedback": safe_feedback, **({"source": response["source"]} if response else {})})
        outcome["search"] = {"proposals": number + 1, "stop": "running", "provider": identity}
        update_usage(report)
        write_json(directory / "report.json", report)
        if evaluation["status"] in ("deadline", "request_limit"):
            stop = evaluation["status"]
            break
    outcome["search"] = {"proposals": len(outcome["evaluations"]), "stop": stop, "provider": identity}
    if outcome["status"] != "optimized":
        outcome["reason"] = "no eligible implementation saved; Python remains in use"
