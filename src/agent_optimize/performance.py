"""Complete, checkpoint, and optionally retain a correctness-checked proposal."""

import hashlib
from pathlib import Path
import sys
import time
import uuid

from .artifacts import install_artifact
from .compatibility import environment_signature
from .process import run_worker
from .purity import validate_reusable_source
from .store import read_json, write_json
from .timing import assess_eligibility


def _manifest(evaluation, metadata, function_id, timing, dispatch_mode="careful"):
    correctness = {"status": "passed", "case_count": evaluation["case_count"], "checks": evaluation["checks"]}
    eligibility = timing.get("eligibility", {})
    return {
        "schema_version": 1, "artifact_id": uuid.uuid4().hex,
        "dispatch_mode": dispatch_mode,
        "function_id": function_id, "identity": metadata["identity"],
        "reference_fingerprint": metadata["fingerprint"],
        "dependency_fingerprint": metadata["dependency_fingerprint"],
        "reference_code_fingerprint": metadata["code_fingerprint"],
        "contract": evaluation["contract"], "environment": evaluation["environment"],
        "correctness": {"status": "passed", "case_count": evaluation["case_count"]},
        "timing": {"mode": timing.get("mode"), "dispatch_mode": dispatch_mode, "eligibility": {key: eligibility[key] for key in ("eligible", "reason", "minimum_batch_speedup") if key in eligibility}},
        "evaluation_evidence": {"schema_version": 1, "correctness": correctness, "timing": timing},
        "generator": evaluation.get("generator", {"kind": "handwritten", "identity": "fused-filter-sum-v1"}),
        **{key: evaluation[key] for key in ("attempt_id", "provider_cost") if key in evaluation},
        "provider_usage": evaluation.get("provider_usage", {}),
        "provider_request_count": evaluation.get("provider_request_count", 0),
    }


def measure_evaluation(evaluation, metadata, function_id, *, worker_timeout=None, deadline=None, dispatch_mode="careful"):
    if deadline is not None and deadline - time.monotonic() <= 0.2:
        return {"schema_version": 1, "mode": "measured", "status": "failed", "recordings": [], "reason": "experiment deadline exhausted before timing"}
    workspace = Path(evaluation["workspace"])
    reference_job = read_json(workspace / "reference-job.json")
    retained = set(evaluation["recorded_case_ids"])
    cases = [case for case in reference_job["cases"] if case["id"] in retained]
    source = (workspace / "candidate.py").read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != evaluation["candidate_sha256"]:
        raise ValueError("candidate source changed since correctness checks")
    prospective = _manifest(evaluation, metadata, function_id, {"mode": "prospective", "eligibility": {"eligible": False}}, dispatch_mode)
    install_artifact(workspace / "prospective-store", prospective, source, prospective=True)
    remaining = worker_timeout
    if deadline is not None:
        remaining = deadline - time.monotonic() if remaining is None else min(remaining, deadline - time.monotonic())
    if remaining is not None and remaining <= 0.2:
        return {"schema_version": 1, "mode": "measured", "status": "failed", "recordings": [], "reason": "experiment deadline exhausted before timing"}
    job = {
        "schema_version": 1, "identity": metadata["identity"], "contract": evaluation["contract"],
        "source_sha256": reference_job["source_sha256"], "positional_names": reference_job["positional_names"],
        "cases": cases, "budget_seconds": remaining - 0.1 if remaining is not None else None,
        "dispatch_mode": dispatch_mode,
    }
    job_path, result_path = workspace / "timing-job.json", workspace / "timing-result.json"
    write_json(job_path, job)
    result = run_worker([sys.executable, "-m", "agent_optimize.timing_worker", str(job_path.resolve()), str(result_path.resolve())], timeout=remaining)
    if result.status not in ("passed", "failed") or not result_path.exists():
        return {"schema_version": 1, "mode": "measured", "status": "failed", "recordings": [], "reason": f"timing worker {result.status}"}
    timing = read_json(result_path)
    if result.returncode and timing.get("status") == "passed":
        return {"schema_version": 1, "mode": "measured", "status": "failed", "recordings": [], "reason": "timing worker exit status disagrees with its result"}
    return timing


def complete_evaluation(evaluation, metadata, function_id, artifact_root, *, timing_provider=None, worker_timeout=None, deadline=None, dispatch_mode="careful"):
    workspace = Path(evaluation["workspace"])
    completed = {**evaluation, "correctness_only": False, "correctness_status": evaluation["status"], "dispatch_mode": dispatch_mode}
    checkpoint = workspace / "completed-evaluation.json"
    if evaluation["status"] != "passed":
        write_json(checkpoint, completed)
        return completed
    source = (workspace / "candidate.py").read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != evaluation.get("candidate_sha256"):
        completed.update(status="rejected", reason="candidate source changed since correctness checks")
        write_json(checkpoint, completed)
        return completed
    if evaluation.get("environment") != environment_signature():
        completed.update(status="ineligible", reason="environment changed since correctness checks")
        write_json(checkpoint, completed)
        return completed
    try:
        validate_reusable_source(source)
    except ValueError as error:
        completed.update(status="rejected", reason=f"candidate is not safe for reuse: {error}")
        write_json(checkpoint, completed)
        return completed
    if timing_provider is None:
        timing = measure_evaluation(evaluation, metadata, function_id, worker_timeout=worker_timeout, deadline=deadline, dispatch_mode=dispatch_mode)
    else:
        timing = timing_provider(evaluation)
        if timing.get("mode") != "controlled-test-fixture":
            raise ValueError("injected timing fixtures must identify themselves as controlled-test-fixture")
    eligibility = assess_eligibility(timing, evaluation["recorded_case_ids"])
    timing = {**timing, "eligibility": eligibility, "dispatch_mode": dispatch_mode}
    completed.update(status="eligible" if eligibility["eligible"] else "ineligible", reason=eligibility["reason"], timing=timing)
    manifest = _manifest(evaluation, metadata, function_id, timing, dispatch_mode) if eligibility["eligible"] else None
    if manifest is not None:
        completed["artifact_id"] = manifest["artifact_id"]
    # A completed evaluation is durable before any accepted index is changed.
    write_json(checkpoint, completed)
    if manifest is not None:
        try:
            path = install_artifact(artifact_root, manifest, source)
            completed.update(status="optimized", reason="eligible implementation saved", artifact_path=str(path.resolve()))
        except (OSError, ValueError) as error:
            completed.update(status="failed", reason=f"artifact could not be saved: {type(error).__name__}")
        write_json(checkpoint, completed)
    return completed
