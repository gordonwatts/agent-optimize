"""Compare original, revised-reference, and saved-dispatch q16 reconstruction.

Run in the checkout's q16 environment with an existing transformed ROOT file:
python benchmarks/q16_reconstruction.py --input-file delivered.root \
    --artifact-root .agent-optimize/q16 --repeats 5

Data loading, first compilation, and plotting are excluded from warm timings.
Every selected event is checked against the unmodified supplied reconstruction.
"""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import awkward as ak
import numpy as np
import numba
import typer

from agent_optimize.dispatch import observe_dispatch
from agent_optimize.evaluator import compare_results

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer(pretty_exceptions_show_locals=False)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.command()
def main(
    input_file: list[Path] | None = typer.Option(None, exists=True, dir_okay=False),
    synthetic_events: int = typer.Option(0, min=0, help="Offline software/performance evidence only."),
    artifact_root: Path = typer.Option(...),
    repeats: int = typer.Option(5, min=1),
    events_per_read: int = typer.Option(128, min=1, help="ROOT entries per read before event selection."),
):
    os.environ["AGENT_OPTIMIZE_ARTIFACT_ROOT"] = str(artifact_root.resolve())
    os.environ["AGENT_OPTIMIZE_MODE"] = "fast"
    if os.environ.get("AGENT_OPTIMIZE_SESSION"):
        raise typer.BadParameter("run this benchmark outside a recording session")
    module = load("q16_benchmark", ROOT / "examples/q16_ttbar.py")
    # Match the identity of the same function when the example runs as a script.
    module.reconstruct_events.__wrapped__.__module__ = "__main__"
    original = load("q16_supplied_reference", ROOT / "tests/fixtures/q16_reference.py")
    batches = list(module.synthetic_batches(synthetic_events) if synthetic_events else module.read_events(input_file or [], step_size=events_per_read))
    if not batches:
        raise typer.BadParameter("no selected events")
    control_groups = None
    if not synthetic_events and events_per_read != 128:
        control = list(module.read_events(input_file or [], step_size=128))
        control_groups = len(control)
        for field in range(4):
            left = ak.to_packed(ak.concatenate([batch[field] for batch in control], axis=0))
            right = ak.to_packed(ak.concatenate([batch[field] for batch in batches], axis=0))
            if left.fields != right.fields:
                raise RuntimeError("ROOT read grouping changed input fields")
            if field == 0:
                columns = [(ak.num(left, axis=1), ak.num(right, axis=1))]
                columns += [(ak.flatten(left[name]), ak.flatten(right[name])) for name in left.fields]
            elif left.fields:
                columns = [(left[name], right[name]) for name in left.fields]
            else:
                columns = [(left, right)]
            for expected_column, actual_column in columns:
                equal, reason = compare_results(expected_column, actual_column)
                if not equal:
                    raise RuntimeError(f"ROOT read grouping changed selected inputs: {reason}")
    decorated = module.reconstruct_events

    def run(kind):
        module.reconstruct_events = decorated.__wrapped__ if kind == "revised-reference" else decorated
        function = original.reconstruct if kind == "original" else module.reconstruct
        return [np.asarray(ak.to_numpy(function(*batch))) for batch in batches]

    expected = run("original")
    first = time.perf_counter()
    stats = {}
    with observe_dispatch(stats):
        dispatched = run("saved-dispatch")
    first_seconds = time.perf_counter() - first
    if not stats.get("used"):
        raise RuntimeError("saved replacement was not used; inspect artifact identity and environment")
    for actual in (run("revised-reference"), dispatched):
        for left, right in zip(expected, actual, strict=True):
            if left.dtype != right.dtype or left.tobytes() != right.tobytes():
                raise RuntimeError("reconstructed masses differ from the supplied reference")
    timings = {kind: [] for kind in ("original", "revised-reference", "saved-dispatch")}
    # Alternate order to reduce systematic warmup and scheduling bias.
    for repeat in range(repeats):
        order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
        for kind in order:
            start = time.perf_counter()
            run(kind)
            timings[kind].append(time.perf_counter() - start)
    medians = {kind: statistics.median(values) for kind, values in timings.items()}
    print(json.dumps({
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "awkward": ak.__version__, "numpy": np.__version__, "numba": numba.__version__},
        "input_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in input_file or []],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), ROOT / "examples/q16_ttbar.py", ROOT / "tests/fixtures/q16_reference.py")},
        "events": sum(len(chunk) for chunk in expected), "batches": len(batches),
        "exact_original_match": True, "data": "synthetic" if synthetic_events else "transformed ROOT",
        "events_per_read": events_per_read,
        "reader_inputs_exactly_match_128": True if control_groups is not None else None,
        "control_groups_at_128": control_groups,
        "mean_mass_GeV": float(np.mean(np.concatenate(expected))), "raw_reconstruction_seconds": timings, "saved_dispatch_observed": True,
        "repeats": repeats, "first_full_pass_seconds": first_seconds,
        "median_reconstruction_seconds": medians,
        "speedup_over_original": medians["original"] / medians["saved-dispatch"],
        "scope": "warm reconstruction only; excludes ROOT reading, ServiceX and plotting",
    }, indent=2))


if __name__ == "__main__":
    app()
