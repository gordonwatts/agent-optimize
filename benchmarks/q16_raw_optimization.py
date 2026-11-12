"""Profile and compare q16 raw-source changes with strict output checking."""

import ast
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import time

import awkward as ak
import numpy as np
import typer

if __package__:
    from .q16_raw_candidates import assignment_indices, grouped_numpy, hadronic_indices
else:
    from q16_raw_candidates import assignment_indices, grouped_numpy, hadronic_indices

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(left, right):
    left, right = np.ma.asarray(ak.to_numpy(left)), np.ma.asarray(ak.to_numpy(right))
    return (left.dtype == right.dtype and left.shape == right.shape
            and np.array_equal(np.ma.getmaskarray(left), np.ma.getmaskarray(right))
            and left.compressed().tobytes() == right.compressed().tobytes())


@app.command()
def main(input_file: Path = typer.Option(..., exists=True),
         output: Path = typer.Option(Path(".agent-optimize/q16-raw-results.json")),
         repeats: int = typer.Option(5, min=1),
         artifact_root: Path | None = typer.Option(None),
         coalesce: int = typer.Option(1, min=1),
         profile: bool = typer.Option(True)):
    if artifact_root is not None:
        os.environ["AGENT_OPTIMIZE_ARTIFACT_ROOT"] = str(artifact_root.resolve())
        os.environ["AGENT_OPTIMIZE_MODE"] = "fast"
    example = load("q16_raw_example", ROOT / "examples/q16_ttbar.py")
    reference = load("q16_raw_reference", ROOT / "tests/fixtures/q16_reference.py")
    # Keep the original expression ordering, but move the final mass calculation
    # after winner selection. This is a standalone source candidate.
    source = inspect.getsource(reference)
    source = source.replace("return ak.firsts(mass(add(thad, tlep))[best])",
                            "return ak.firsts(mass(add(thad[best], tlep[best])))")
    namespace = {}
    exec(compile(source, "<winner-only-q16>", "exec"), namespace)
    original_batches = list(example.read_events([input_file]))
    expected = [reference.reconstruct(*batch) for batch in original_batches]
    batches = original_batches
    coalesce_seconds = 0.0
    if coalesce > 1:
        coalesce_started = time.perf_counter()
        batches = [tuple(ak.concatenate([batch[column] for batch in original_batches[start:start + coalesce]], axis=0)
                         for column in range(4))
                   for start in range(0, len(original_batches), coalesce)]
        coalesce_seconds = time.perf_counter() - coalesce_started
        expected = [ak.concatenate(expected[start:start + coalesce], axis=0)
                    for start in range(0, len(expected), coalesce)]
    functions = {
        "supplied-python": reference.reconstruct,
        "winner-mass-only": namespace["reconstruct"],
        "grouped-numpy-separate-flatten": lambda *args: grouped_numpy(*args, fallback=reference.reconstruct, flatten_once=False),
        "grouped-numpy": lambda *args: grouped_numpy(*args, fallback=reference.reconstruct),
        "grouped-numpy-reused": lambda *args: grouped_numpy(*args, fallback=reference.reconstruct, reuse_subexpressions=True),
    }
    dispatch_stats = {}
    if artifact_root is not None:
        from agent_optimize.dispatch import observe_dispatch
        example.reconstruct_events.__wrapped__.__module__ = "__main__"
        functions["saved-dispatch"] = example.reconstruct
    checks = {}
    first = {}
    for name, function in functions.items():
        started = time.perf_counter()
        if name == "saved-dispatch":
            with observe_dispatch(dispatch_stats):
                actual = [function(*batch) for batch in batches]
            if not dispatch_stats.get("used"):
                raise RuntimeError(f"saved replacement was not used: {dispatch_stats}")
        else:
            assignment_indices.cache_clear()
            hadronic_indices.cache_clear()
            actual = [function(*batch) for batch in batches]
        first[name] = time.perf_counter() - started
        checks[name] = all(exact(left, right) for left, right in zip(expected, actual, strict=True))
        if not checks[name]:
            raise RuntimeError(f"{name}: complete-file bitwise mismatch")
    synthetic_checks = 0
    branch_observations = {}
    reused_branch_observations = {}
    for batch in batches:
        grouped_numpy(*batch, fallback=reference.reconstruct, observe=branch_observations)
        grouped_numpy(*batch, fallback=reference.reconstruct, reuse_subexpressions=True, observe=reused_branch_observations)
    edge_observations = {}
    for batch in example.synthetic_batches(256, batch_size=32):
        for name, function in functions.items():
            if not exact(reference.reconstruct(*batch), function(*batch)):
                raise RuntimeError(f"{name}: synthetic bitwise mismatch")
            synthetic_checks += 1
        jets, lepton, mx, my = batch
        for value in (np.nan, np.inf, -np.inf, -0.0, 0.0, np.finfo(np.float64).max,
                      np.nextafter(0.0, 1.0)):
            changed = ak.with_field(jets, ak.full_like(jets.e, value), "e")
            edge_expected = reference.reconstruct(changed, lepton, mx, my)
            for reuse in (False, True):
                if not exact(edge_expected, grouped_numpy(changed, lepton, mx, my, fallback=reference.reconstruct,
                                                         reuse_subexpressions=reuse, observe=edge_observations)):
                    raise RuntimeError("nonfinite/zero edge mismatch")
                synthetic_checks += 1
    jets, lepton, mx, my = next(example.synthetic_batches(32))
    special = [
        (jets[ak.local_index(jets, axis=1) < ak.Array(np.arange(len(jets)) % 5)[:, None]], lepton, mx, my),
        tuple(value[:0] for value in (jets, lepton, mx, my)),
        (ak.zip({key: ak.full_like(jets[key], 10.0 if key == "e" else 1.0) for key in ("px", "py", "pz", "e")}), lepton, mx, my),
        tuple(ak.values_astype(value, np.float32) for value in (jets, lepton, mx, my)),
        (ak.mask(jets, ak.local_index(jets, axis=1) != 0), lepton, mx, my),
        (jets, ak.mask(lepton, ak.Array(np.arange(len(lepton)) % 3 != 0)), mx, my),
        (jets, ak.with_field(lepton, ak.mask(lepton.e, ak.Array(np.arange(len(lepton)) % 3 != 0)), "e"), mx, my),
    ]
    for batch in special:
        for reuse in (False, True):
            if not exact(reference.reconstruct(*batch), grouped_numpy(*batch, fallback=reference.reconstruct,
                                                                    reuse_subexpressions=reuse, observe=edge_observations)):
                raise RuntimeError("empty / low multiplicity / tie edge mismatch")
            synthetic_checks += 1
    timings = {name: [] for name in functions}
    for repeat in range(repeats):
        order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
        for name in order:
            started = time.perf_counter()
            for batch in batches:
                functions[name](*batch)
            timings[name].append(time.perf_counter() - started)
    profiles = {}
    for name, function in (functions.items() if profile else []):
        profiler = cProfile.Profile()
        profiler.enable()
        for batch in batches:
            function(*batch)
        profiler.disable()
        stats = pstats.Stats(profiler)
        profiles[name] = [{"function": f"{key[0]}:{key[1]}:{key[2]}",
                           "calls": value[1], "self_seconds": value[2],
                           "cumulative_seconds": value[3]}
                          for key, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:15]]
    from agent_optimize.assignments import parse_assignments
    from agent_optimize.experiment import freeze_reference
    try:
        freeze_reference(grouped_numpy)
        freeze_status = "frozen"
    except Exception as error:
        freeze_status = f"{type(error).__name__}: {error}"
    try:
        parse_assignments(ast.parse(inspect.getsource(grouped_numpy)))
        translation = "recognized"
    except Exception as error:
        translation = f"{type(error).__name__}: {error}"
    multiplicities, frequencies = np.unique(np.concatenate([
        ak.to_numpy(ak.num(batch[0], axis=1)) for batch in original_batches
    ]), return_counts=True)
    evidence = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(),
        "awkward": ak.__version__, "numpy": np.__version__,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (ROOT / "benchmarks/q16_raw_candidates.py", Path(__file__), ROOT / "tests/fixtures/q16_reference.py")},
        "events": sum(len(x) for x in expected), "groups": len(batches),
        "original_groups": len(original_batches), "coalesce": coalesce,
        "coalesce_preparation_seconds": coalesce_seconds,
        "jet_multiplicity": {str(int(count)): int(frequency)
                             for count, frequency in zip(multiplicities, frequencies, strict=True)},
        "all_real_outputs_bitwise_equal": checks, "synthetic_checks": synthetic_checks,
        "grouped_real_branch_observations": branch_observations,
        "grouped_reused_real_branch_observations": reused_branch_observations,
        "grouped_edge_branch_observations": edge_observations,
        "saved_dispatch_observations": dispatch_stats,
        "assignment_cache": "cleared before each first raw pass; populated during warm passes",
        "real_assignment_cache_array_bytes": sum(assignment_indices(int(count)).nbytes
                                                  + hadronic_indices(int(count)).nbytes
                                                  for count in multiplicities if count >= 4),
        "first_full_pass_seconds": first, "warm_seconds": timings,
        "median_seconds": {name: statistics.median(values) for name, values in timings.items()},
        "median_including_coalesce_seconds": {name: statistics.median(values) + coalesce_seconds
                                               for name, values in timings.items()},
        "profile": profiles, "grouped_numpy_assignment_translation": translation,
        "grouped_numpy_freeze_attempt": freeze_status,
        "scope": "reconstruction only; excludes data loading, batch coalescing, ServiceX and plotting; raw candidates have no JIT; saved-dispatch first pass includes compilation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in evidence.items() if key != "profile"}, indent=2))


if __name__ == "__main__":
    app()
