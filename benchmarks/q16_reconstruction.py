"""Compare original, revised-reference, and saved-dispatch q16 reconstruction.

Run in the checkout's q16 environment with an existing transformed ROOT file:
python benchmarks/q16_reconstruction.py --input-file delivered.root \
    --artifact-root .agent-optimize/q16 --repeats 5

Data loading, first compilation, and plotting are excluded from warm timings.
Every selected event is checked against the unmodified supplied reconstruction.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import awkward as ak
import numpy as np

from agent_optimize.dispatch import observe_dispatch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, action="append")
    parser.add_argument("--synthetic-events", type=int, default=0, help="offline software/performance evidence only")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--events-per-read", type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.events_per_read < 1:
        parser.error("--events-per-read must be positive")
    os.environ["AGENT_OPTIMIZE_ARTIFACT_ROOT"] = str(args.artifact_root.resolve())
    os.environ["AGENT_OPTIMIZE_MODE"] = "fast"
    if os.environ.get("AGENT_OPTIMIZE_SESSION"):
        parser.error("run this benchmark outside a recording session")
    module = load("q16_benchmark", ROOT / "examples/q16_ttbar.py")
    # Match the identity of the same function when the example runs as a script.
    module.reconstruct_events.__wrapped__.__module__ = "__main__"
    original = load("q16_supplied_reference", ROOT / "tests/fixtures/q16_reference.py")
    batches = list(module.synthetic_batches(args.synthetic_events) if args.synthetic_events else module.read_events(args.input_file or [], step_size=args.events_per_read))
    if not batches:
        parser.error("no selected events")
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
    for repeat in range(args.repeats):
        order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
        for kind in order:
            start = time.perf_counter()
            run(kind)
            timings[kind].append(time.perf_counter() - start)
    medians = {kind: statistics.median(values) for kind, values in timings.items()}
    print(json.dumps({
        "events": sum(len(chunk) for chunk in expected), "batches": len(batches),
        "exact_original_match": True, "data": "synthetic" if args.synthetic_events else "transformed ROOT",
        "mean_mass_GeV": float(np.mean(np.concatenate(expected))), "raw_reconstruction_seconds": timings, "saved_dispatch_observed": True,
        "repeats": args.repeats, "first_full_pass_seconds": first_seconds,
        "median_reconstruction_seconds": medians,
        "speedup_over_original": medians["original"] / medians["saved-dispatch"],
        "scope": "warm reconstruction only; excludes ROOT reading, ServiceX and plotting",
    }, indent=2))


if __name__ == "__main__":
    main()
