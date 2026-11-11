"""Measure warmed ordinary dispatch against frozen direct-reference calls."""

import argparse
import ast
import hashlib
import os
from pathlib import Path
import time

from .candidate import collection_layout, contract_from_dict
from .decorator import optimize
from .dispatch import clear_dispatch_cache, observe_dispatch
from .evaluator import compare_results
from .experiment import validate_frozen_source
from .serialization import load_value
from .store import read_json, write_json
from .timing import MIN_BATCHES, MIN_SAMPLES


def _measure(function, args, kwargs, iterations, deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("timing deadline exhausted")
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function(*args, **kwargs)
    return (time.perf_counter_ns() - started) / iterations


def _calibrate(function, args, kwargs, deadline):
    iterations = 1
    while True:
        duration = _measure(function, args, kwargs, iterations, deadline)
        if duration * iterations >= 1_000_000 or iterations >= 4096:
            return iterations
        iterations *= 2


def measure_job(path):
    job = read_json(path)
    workspace = path.parent
    source = (workspace / "reference.py").read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != job["source_sha256"]:
        raise ValueError("frozen reference source fingerprint mismatch")
    tree, name = validate_frozen_source(source)
    namespace = {"__name__": job["identity"]["module"]}
    exec(compile(tree, str(workspace / "reference.py"), "exec"), namespace)
    reference = namespace[name]
    # Preserve the original location without importing or executing its script.
    reference.__code__ = reference.__code__.replace(co_filename=job["identity"]["source_file"], co_firstlineno=job["identity"]["first_line"])
    reference.__module__ = job["identity"]["module"]
    reference.__qualname__ = job["identity"]["qualified_name"]
    os.environ["AGENT_OPTIMIZE_ARTIFACT_ROOT"] = str((workspace / "prospective-store").resolve())
    mode = job.get("dispatch_mode", "careful")
    os.environ["AGENT_OPTIMIZE_MODE"] = mode
    replacement = optimize(reference)
    clear_dispatch_cache()
    contract = contract_from_dict(job["contract"])
    deadline = time.monotonic() + job["budget_seconds"] if job.get("budget_seconds") is not None else None
    stats, recordings = {}, []
    compilation_ns = first_use_ns = reference_first_use_ns = 0
    with observe_dispatch(stats, allow_prospective=True):
        for index, case in enumerate(job["cases"]):
            directory = workspace / "cases" / case["id"]
            metadata = read_json(directory / "inputs.json")
            arguments = {name: load_value(value, directory) for name, value in metadata["arguments"].items()}
            array = arguments[contract.array_argument]
            offsets = collection_layout(array, contract.collection_path).offsets.data
            args = tuple(arguments.pop(name) for name in job["positional_names"])
            kwargs = arguments
            if index == 0:
                started = time.perf_counter_ns()
                expected = reference(*args, **kwargs)
                reference_first_use_ns = time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                actual = replacement(*args, **kwargs)
                first_use_ns = time.perf_counter_ns() - started
                compilation_ns = stats.get("compilation_ns", 0)
                if not stats.get("used") or not compare_results(expected, actual)[0]:
                    raise ValueError("prospective ordinary dispatch was inapplicable or disagreed with its reference")
            for _ in range(3):
                reference(*args, **kwargs)
                replacement(*args, **kwargs)
            counts = {"reference": _calibrate(reference, args, kwargs, deadline), "candidate": _calibrate(replacement, args, kwargs, deadline)}
            functions = {"reference": reference, "candidate": replacement}
            batches = []
            for batch_index in range(MIN_BATCHES):
                batch = {"reference_ns": [], "candidate_ns": [], "iterations": counts, "order": []}
                for sample in range(MIN_SAMPLES):
                    order = ["reference", "candidate"] if (batch_index + sample) % 2 == 0 else ["candidate", "reference"]
                    batch["order"].append(order)
                    for kind in order:
                        batch[f"{kind}_ns"].append(_measure(functions[kind], args, kwargs, counts[kind], deadline))
                    if not stats.get("used"):
                        raise ValueError("prospective dispatch fell back during timing")
                batches.append(batch)
            recordings.append({"case_id": case["id"], "events": len(array), "elements": int(offsets[-1] - offsets[0]), "batches": batches})
    description = "cached ordinary fast dispatch including argument binding, generated adapter, kernel, and output" if mode == "fast" else "ordinary decorator dispatch including compatibility, conversion, kernel, and output"
    return {"schema_version": 1, "status": "passed", "mode": "measured", "dispatch_mode": mode, "path": description, "reference_path": "direct frozen reference call", "compilation_ns": compilation_ns, "first_use_ns": first_use_ns, "reference_first_use_ns": reference_first_use_ns, "recordings": recordings}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        result = measure_job(args.job)
    except Exception as error:
        result = {"schema_version": 1, "status": "failed", "mode": "measured", "recordings": [], "reason": f"{type(error).__name__}: {str(error)[:400]}"}
    write_json(args.result, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
