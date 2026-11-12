"""Shared synthetic workloads for an adapted auto-opt workflow comparison.

The explicit loops are source edits chosen by an agent, not auto-opt software.
Run recording through agent-optimize first, then run this module ordinarily.
"""

import cProfile
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import statistics
import sys
import time

import awkward as ak
import numba
import numpy as np
import typer

from agent_optimize.dispatch import observe_dispatch
from benchmarks.workloads import WORKLOADS, array_digest, make_events

app = typer.Typer()
CASES = ("central_ht", "dijet_combinatorics", "nearest_jet_matching")


@numba.njit(boundscheck=True)
def loop_ht(events, threshold=30.0, eta_limit=2.4):
    output = np.zeros(len(events), dtype=np.float64)
    for i, event in enumerate(events):
        total = 0.0
        for jet in event.Jet:
            if jet.pt > threshold and abs(jet.eta) < eta_limit:
                total += jet.pt
        output[i] = total
    return output


@numba.njit(boundscheck=True)
def loop_dijet(events, minimum_pair_pt=80.0):
    output = np.zeros(len(events), dtype=np.float64)
    for i, event in enumerate(events):
        total = 0.0
        for left in range(len(event.Jet)):
            for right in range(left + 1, len(event.Jet)):
                pair_pt = event.Jet[left].pt + event.Jet[right].pt
                if pair_pt > minimum_pair_pt:
                    total += pair_pt
        output[i] = total
    return output


@numba.njit(boundscheck=True)
def loop_matching(events, radius=0.4):
    output = np.zeros(len(events), dtype=np.int64)
    for i, event in enumerate(events):
        count = 0
        for jet in event.Jet:
            nearest = np.inf
            for reference in event.RefJet:
                deta = jet.eta - reference.eta
                dphi = (jet.phi - reference.phi + np.pi) % (2 * np.pi) - np.pi
                distance = deta * deta + dphi * dphi
                if distance < nearest:
                    nearest = distance
            if nearest < radius * radius:
                count += 1
        output[i] = count
    return output


LOOPS = dict(zip(CASES, (loop_ht, loop_dijet, loop_matching), strict=True))


@numba.njit(boundscheck=True)
def loop_matching_threshold(events, radius=0.4):
    """Count existence directly; prune eta before calculating wrapped phi."""
    output = np.zeros(len(events), dtype=np.int64)
    radius_squared = radius * radius
    for i, event in enumerate(events):
        for jet in event.Jet:
            for reference in event.RefJet:
                deta = jet.eta - reference.eta
                if deta * deta >= radius_squared:
                    continue
                dphi = (jet.phi - reference.phi + np.pi) % (2 * np.pi) - np.pi
                if deta * deta + dphi * dphi < radius_squared:
                    output[i] += 1
                    break
    return output


def peak_process_rss():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak", "working", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi", use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak


def checked(reference, actual):
    left, right = np.asarray(ak.to_numpy(reference)), np.asarray(ak.to_numpy(actual))
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def adapted(function, events):
    # Include output wrapping in the prospective replacement timing.
    return ak.Array(function(events))


@app.command()
def main(
    artifact_root: Path = typer.Option(Path(".agent-optimize/workflow-comparison")),
    output: Path = typer.Option(Path("docs/evidence/workflow-comparison.json")),
    events: int = typer.Option(8192, min=1),
    repeats: int = typer.Option(11, min=3),
    record_only: bool = typer.Option(False),
    memory_case: str = typer.Option("", hidden=True),
    memory_variant: str = typer.Option("python", hidden=True),
) -> None:
    data = make_events(events)
    if memory_case:
        if memory_case not in CASES or memory_variant not in ("python", "loop", "threshold", "dispatch"):
            raise typer.BadParameter("invalid isolated memory case/variant")
        os.environ["AGENT_OPTIMIZE_ARTIFACT_ROOT"] = str(artifact_root.resolve())
        os.environ["AGENT_OPTIMIZE_MODE"] = "fast"
        started = time.perf_counter()
        if memory_variant == "python":
            WORKLOADS[memory_case].function.__wrapped__(data)
        elif memory_variant == "dispatch":
            WORKLOADS[memory_case].function(data)
        else:
            adapted(loop_matching_threshold if memory_variant == "threshold" else LOOPS[memory_case], data)
        typer.echo(json.dumps({"first_function_call_seconds": time.perf_counter() - started,
                               "peak_process_rss_bytes": peak_process_rss()}))
        return
    if record_only:
        for case in CASES:
            for count in (128, events):
                WORKLOADS[case].function(make_events(count))
        return
    os.environ["AGENT_OPTIMIZE_ARTIFACT_ROOT"] = str(artifact_root.resolve())
    os.environ["AGENT_OPTIMIZE_MODE"] = "fast"
    profile_dir = artifact_root / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in CASES:
        decorated = WORKLOADS[case].function
        original = decorated.__wrapped__
        baseline = original(data)
        profiler = cProfile.Profile()
        profiler.runcall(original, data)
        profiler.dump_stats(str(profile_dir / f"{case}-baseline.prof"))
        loop = LOOPS[case]
        start = time.perf_counter()
        candidate = adapted(loop, data)
        first = time.perf_counter() - start
        verification = []
        for count in (0, 1, 2, 17, 128, events):
            for seed in (16, 20261106, 20261112):
                sample = make_events(count, seed)
                verification.append(checked(original(sample), adapted(loop, sample)))
        # Exercise nonfinite coordinates, empty collections, and signed zeros.
        edge = make_events(128, 42)
        for collection in ("Jet", "RefJet"):
            for field in ("pt", "eta", "phi"):
                values = np.asarray(ak.to_numpy(ak.flatten(edge[collection][field]))).copy()
                values[:8] = [np.nan, np.inf, -np.inf, 0.0, -0.0, 0.4, -0.4, np.pi]
                edge = ak.with_field(edge, ak.with_field(edge[collection], ak.unflatten(values, ak.num(edge[collection])), field), collection)
        with np.errstate(invalid="ignore"):
            verification.append(checked(original(edge), adapted(loop, edge)))
        scalar_checks = []
        for value in (-1.0, 0.0, 0.4, 1000.0):
            scalar_checks.append(checked(original(data, value), ak.Array(loop(data, value))))
        boundary = ak.Array([
            {"Jet": [{"eta": eta, "phi": phi}], "RefJet": [{"eta": 0.0, "phi": 0.0}]}
            for eta in (0.0, -0.0, 0.4, np.nextafter(0.4, 0.0), np.nan, np.inf)
            for phi in (0.0, 0.4, np.nextafter(0.4, 0.0), np.pi, -np.pi, 2 * np.pi, 1e20)
        ])
        stats = {}
        start = time.perf_counter()
        with observe_dispatch(stats):
            dispatched = decorated(data)
        dispatch_first = time.perf_counter() - start
        calls = {
            "python": lambda: original(data),
            "adapted_auto_opt_loop": lambda: adapted(loop, data),
            "agent_optimize_dispatch": lambda: decorated(data),
        }
        threshold_verification = None
        threshold_first = None
        if case == "nearest_jet_matching":
            start = time.perf_counter()
            adapted(loop_matching_threshold, data)
            threshold_first = time.perf_counter() - start
            with np.errstate(invalid="ignore"):
                threshold_verification = all(
                    checked(original(sample), adapted(loop_matching_threshold, sample))
                    for sample in [edge, boundary] + [make_events(count, seed) for count in (0, 1, 2, 17, 128, events) for seed in (16, 20261106, 20261112)]
                ) and all(checked(original(data, radius), ak.Array(loop_matching_threshold(data, radius)))
                          for radius in (-1.0, 0.0, 0.4, 1000.0))
            calls["adapted_auto_opt_threshold"] = lambda: adapted(loop_matching_threshold, data)
        for call in calls.values():
            call()
        timings = {name: [] for name in calls}
        for repeat in range(repeats):
            order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
            for name in order:
                start = time.perf_counter()
                calls[name]()
                timings[name].append(time.perf_counter() - start)
        profiler = cProfile.Profile()
        profiler.runcall(adapted, loop, data)
        profiler.dump_stats(str(profile_dir / f"{case}-candidate.prof"))
        results[case] = {
            "loop_exact_main": checked(baseline, candidate),
            "loop_verification_passed": sum(verification),
            "loop_verification_total": len(verification),
            "scalar_verification_passed": sum(scalar_checks),
            "scalar_verification_total": len(scalar_checks),
            "threshold_verification_passed": threshold_verification,
            "threshold_first_call_seconds": threshold_first,
            "dispatch_exact": checked(baseline, dispatched),
            "dispatch_observed": stats,
            "loop_first_call_seconds": first,
            "dispatch_first_call_seconds": dispatch_first,
            "warm_seconds": timings,
            "warm_median_seconds": {key: statistics.median(value) for key, value in timings.items()},
            "output_sha256": array_digest(baseline),
        }
        memory = {}
        variants = ("python", "loop", "threshold", "dispatch") if case == "nearest_jet_matching" else ("python", "loop", "dispatch")
        for variant in variants:
            measured = subprocess.run([sys.executable, "-m", "benchmarks.workflow_comparison", "--events", str(events),
                                       "--memory-case", case, "--memory-variant", variant,
                                       "--artifact-root", str(artifact_root)],
                                      check=True, capture_output=True, text=True, timeout=120)
            memory[variant] = json.loads(measured.stdout)
        results[case]["isolated_process_memory_including_imports_and_jit"] = memory
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "auto_opt_source_commit": "56cbad7d561097037eefaecc5440268d83c6b9a3",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "awkward": ak.__version__, "numpy": np.__version__, "numba": numba.__version__},
        "events": events, "input_sha256": array_digest(data), "repeats": repeats,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "in-memory full function including ordinary dispatch/output wrapping; excludes generation and imports",
        "method": "adapted auto-opt workflow with agent-written Numba loops versus deterministic mock agent-optimize generation; not equal model/budget comparison",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    typer.echo(str(output))


if __name__ == "__main__":
    app()
