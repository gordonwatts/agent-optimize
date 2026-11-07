"""Separate frozen-reference and Numba-candidate workers for correctness."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .candidate import adapt_inputs, compile_kernel, contract_from_dict, wrap_output
from .compatibility import environment_signature
from .experiment import validate_frozen_source
from .serialization import load_value, plan_value, write_value
from .store import read_json, write_json


def evaluate_job(job_path: Path):
    job = read_json(job_path)
    root = job_path.parent
    contract = contract_from_dict(job["contract"])
    output_directory = root / job["output_directory"]
    output_directory.mkdir()
    if job["kind"] == "reference":
        source = (root / "reference.py").read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != job["source_sha256"]:
            raise ValueError("frozen reference source fingerprint mismatch")
        tree, name = validate_frozen_source(source)
        namespace = {"__name__": "agent_optimize_frozen_reference"}
        exec(compile(tree, str(root / "reference.py"), "exec"), namespace)
        reference = namespace[name]
    elif job["kind"] == "candidate":
        candidate_source = (root / "candidate.py").read_text(encoding="utf-8")
        if hashlib.sha256(candidate_source.encode("utf-8")).hexdigest() != job["candidate_sha256"]:
            raise ValueError("candidate source changed before correctness evaluation")
        kernel = compile_kernel(candidate_source)
    else:
        raise ValueError("unknown evaluation worker kind")
    outputs = []
    for case in job["cases"]:
        metadata = read_json(root / "cases" / case["id"] / "inputs.json")
        arguments = {name: load_value(value, root / "cases" / case["id"]) for name, value in metadata["arguments"].items()}
        try:
            if job["kind"] == "reference":
                positional = [arguments.pop(name) for name in job["positional_names"]]
                result = reference(*positional, **arguments)
            else:
                offsets, columns, scalars = adapt_inputs(contract, arguments)
                raw = kernel(offsets, columns, scalars)
                if type(raw) is np.ndarray and any(np.shares_memory(raw, array) for array in [offsets, *columns]):
                    raise ValueError("candidate output must be separately allocated")
                result = wrap_output(contract, raw, len(offsets) - 1)
            plan = plan_value(result, "result", output=True)
            write_value(plan, _case_directory(output_directory, case["id"]))
            outputs.append({"case_id": case["id"], "name": case["name"], "result": plan.metadata})
        except Exception as error:
            return {
                "schema_version": 1, "status": "failed", "kind": job["kind"],
                "phase": "compilation" if type(error).__module__.startswith("numba") else "execution",
                "case_id": case["id"], "case_name": case["name"],
                "reason": f"{type(error).__name__}: {str(error)[:500]}", "outputs": outputs,
            }
    result = {"schema_version": 1, "status": "passed", "kind": job["kind"], "outputs": outputs, "environment": environment_signature()}
    if job["kind"] == "candidate":
        result["compiled_signatures"] = [str(signature) for signature in kernel.signatures]
        result["fastmath"] = kernel.targetoptions["fastmath"]
    return result


def _case_directory(root, name):
    directory = root / name
    directory.mkdir()
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_job(args.job)
    except Exception as error:
        result = {"schema_version": 1, "status": "failed", "phase": "setup", "reason": f"{type(error).__name__}: {str(error)[:500]}"}
    write_json(args.result, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
