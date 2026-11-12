"""Compare assignment-loop scheduling against a pinned historical generator."""

import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import awkward as ak
import numba
import numpy as np
import typer

from benchmarks.q16_reconstruction import load, ROOT
from agent_optimize.candidate import make_contract, fused_candidate_source, compile_kernel, adapt_inputs, wrap_output
from agent_optimize.experiment import freeze_reference
from agent_optimize.operations import parse_filter_sum
from agent_optimize.serialization import plan_value
from agent_optimize.evaluator import compare_results
from agent_optimize.fast_adapter import adapter_source

app = typer.Typer()


def same_values(left, right):
    left, right = np.asarray(ak.to_numpy(left)), np.asarray(ak.to_numpy(right))
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


@app.command()
def main(
    input_file: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("docs/evidence/q16-codegen.json")),
    baseline_revision: str = typer.Option("9c9004a"),
    repeats: int = typer.Option(5, min=3),
    coalesce_groups: int = typer.Option(1, min=1),
) -> None:
    baseline_commit = subprocess.run(["git", "rev-parse", baseline_revision], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    baseline_module_source = subprocess.run(["git", "show", f"{baseline_commit}:src/agent_optimize/assignments.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    baseline_module = types.ModuleType("agent_optimize._q16_baseline_assignments")
    baseline_module.__package__ = "agent_optimize"
    sys.modules[baseline_module.__name__] = baseline_module
    exec(compile(baseline_module_source, "<historical assignment generator>", "exec"), baseline_module.__dict__)
    module = load("q16_codegen_benchmark", ROOT / "examples/q16_ttbar.py")
    original = load("q16_codegen_supplied", ROOT / "tests/fixtures/q16_reference.py")
    batches = list(module.read_events([input_file]))
    if not batches:
        raise typer.BadParameter("No selected events in the input file")
    original_groups = len(batches)
    if coalesce_groups > 1:
        batches = [tuple(ak.concatenate([batch[field] for batch in batches[start:start + coalesce_groups]], axis=0)
                         for field in range(4)) for start in range(0, len(batches), coalesce_groups)]

    def packed(batch):
        return ak.to_packed(ak.zip(dict(zip(("jets", "lepton", "metx", "mety"), batch)), depth_limit=1))

    arrays = [packed(batch) for batch in batches]
    reference = module.reconstruct_events.__wrapped__
    frozen = freeze_reference(reference)
    graph = parse_filter_sum(frozen.source)
    contract = make_contract(graph, {"events": arrays[0]}, plan_value(reference(arrays[0]), "result", output=True).metadata)
    sources = {"before": baseline_module.candidate_source(graph, contract), "scheduled": fused_candidate_source(graph, contract)}
    kernels = {name: compile_kernel(source) for name, source in sources.items()}
    adapter_namespace = {}
    exec(adapter_source(asdict(contract)), adapter_namespace)

    def compiled(name):
        results = []
        for batch in batches:
            value = packed(batch)
            results.append(wrap_output(contract, kernels[name](*adapt_inputs(contract, {"events": value})), len(value)))
        return results

    def fast_adapter(name):
        return [adapter_namespace["replacement"]({"events": packed(batch)}, kernels[name]) for batch in batches]

    expected = [original.reconstruct(*batch) for batch in batches]
    revised = [reference(value) for value in arrays]
    first = {}
    calls = {"original": lambda: [original.reconstruct(*batch) for batch in batches],
             "before": lambda: compiled("before"), "scheduled": lambda: compiled("scheduled"),
             "before_fast_adapter": lambda: fast_adapter("before"),
             "scheduled_fast_adapter": lambda: fast_adapter("scheduled")}
    for name in list(calls)[1:]:
        start = time.perf_counter()
        actual = calls[name]()
        first[name] = time.perf_counter() - start
        assert all(same_values(left, right) for left, right in zip(expected, actual, strict=True)), name
        assert all(compare_results(left, right)[0] for left, right in zip(revised, actual, strict=True)), name
    timings = {name: [] for name in calls}
    for repeat in range(repeats):
        for name in (list(calls) if repeat % 2 == 0 else list(reversed(calls))):
            start = time.perf_counter()
            result = calls[name]()
            timings[name].append(time.perf_counter() - start)
            assert all(same_values(left, right) for left, right in zip(expected, result, strict=True)), name
    inputs = [adapt_inputs(contract, {"events": value}) for value in arrays]
    component = {name: [] for name in sources}
    for repeat in range(repeats):
        for name in (list(sources) if repeat % 2 == 0 else list(reversed(sources))):
            start = time.perf_counter()
            for arguments in inputs:
                kernels[name](*arguments)
            component[name].append(time.perf_counter() - start)
    result = {
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "awkward": ak.__version__, "numpy": np.__version__, "numba": numba.__version__},
        "events": sum(len(a) for a in arrays), "groups": len(arrays), "original_groups": original_groups,
        "coalesce_groups": coalesce_groups, "repeats": repeats,
        "baseline_commit": baseline_commit,
        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(frozen.source.encode()).hexdigest(),
        "generator_before_sha256": hashlib.sha256(baseline_module_source.encode()).hexdigest(),
        "generator_scheduled_sha256": hashlib.sha256((ROOT / "src/agent_optimize/assignments.py").read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        "first_full_pass_seconds": first, "full_reconstruction_seconds": timings,
        "full_median_seconds": {name: statistics.median(values) for name, values in timings.items()},
        "kernel_only_seconds": component,
        "kernel_only_median_seconds": {name: statistics.median(values) for name, values in component.items()},
        "all_real_outputs_bitwise_exact": True,
        "api_requests": 0,
        "scope": "full reconstruction includes packing, adaptation, native loop and output wrapping; before/scheduled use checked adapt_inputs; *_fast_adapter use generated unchecked fast adapter, without dispatch lookup; excludes data loading/imports and first compilation; component kernel-only uses preadapted packed arrays; original option outputs compared as NumPy values, compiled outputs also strictly compared with revised reference",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps({key: result[key] for key in ("events", "full_median_seconds", "kernel_only_median_seconds", "first_full_pass_seconds")}, indent=2))


if __name__ == "__main__":
    app()
