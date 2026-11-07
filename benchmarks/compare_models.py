"""Opt-in, sequential end-to-end comparison using the actual library CLI.

Offline smoke run::

    python -m benchmarks.compare_models --provider mock --cases integer_threshold
        dijet_combinatorics nearest_jet_matching --events 16 --max-proposals 2

Live run (requires configured credentials; bills the selected provider)::

    python -m benchmarks.compare_models --provider openai --model gpt-6.1-sol
        --model gpt-6.1-terra --model gpt-6-luna

Every model/case/repeat gets a fresh artifact root. Synthetic inputs are seeded;
raw CLI reports, workload digests, logs, settings and platform are retained.
Unsupported controls should make no model requests. No speedup is inferred from
failed correctness, unavailable models, or absent timing evidence. Token totals
cover only attempts whose provider reports usage; billing is not estimated.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from .workloads import WORKLOADS
from .process_tree import run as run_benchmark_process


ROOT = Path(__file__).resolve().parents[1]
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_output_tokens")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _platform():
    versions = {}
    for package in ("agent-optimize", "awkward", "numpy", "numba", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(), "executable": sys.executable, "logical_cpu_count": os.cpu_count(), "packages": versions}


def summarize(report, *, expected_supported):
    """Keep absence distinct from zero and retain per-proposal evidence."""
    functions = report.get("functions", [])
    evaluations = [evaluation for function in functions for evaluation in function.get("evaluations", [])]
    attempts = sum(function.get("search", {}).get("proposals", 0) for function in functions)
    request_counts = [evaluation["provider_request_count"] for evaluation in evaluations if type(evaluation.get("provider_request_count")) is int]
    usage = [evaluation["provider_usage"] for evaluation in evaluations if isinstance(evaluation.get("provider_usage"), dict) and any(isinstance(evaluation["provider_usage"].get(field), int) for field in TOKEN_FIELDS)]
    totals = {field: sum(item[field] for item in usage if isinstance(item.get(field), int)) if any(isinstance(item.get(field), int) for item in usage) else None for field in TOKEN_FIELDS}
    fallback_verified = None
    if not expected_supported:
        fallback_verified = bool(functions) and report.get("child_exit_code") == 0 and attempts == 0 and all(function.get("status") == "unsupported" for function in functions)
    return {
        "status": report.get("status"), "child_exit_code": report.get("child_exit_code"),
        "dispatch_mode": report.get("dispatch_mode"),
        "provider_attempts": attempts, "http_request_count": sum(request_counts) if len(request_counts) == attempts else None,
        "request_count_reported_attempts": len(request_counts), "usage_reported_attempts": len(usage), "reported_token_totals": totals,
        "unsupported_fallback_verified": fallback_verified,
        "functions": [{"name": function.get("identity", {}).get("qualified_name"), "status": function.get("status"), "reason": function.get("reason"), "captured_calls": function.get("captured_calls"), "skipped": function.get("skipped"), "search": function.get("search"), "evaluations": [{key: evaluation[key] for key in ("status", "reason", "correctness_status", "case_count", "timing", "provider_usage", "provider_request_count", "provider_error", "generator", "candidate_sha256", "artifact_path") if key in evaluation} for evaluation in function.get("evaluations", [])]} for function in functions],
        **({"error": report["error"]} if "error" in report else {}),
    }


def _command(options, model, artifact_root, name):
    command = [sys.executable, "-m", "agent_optimize", "run", "--json", "--provider", options.provider, "--artifact-root", str(artifact_root), "--max-proposals", str(options.max_proposals), "--function-seconds", str(options.function_seconds), "--search-seconds", str(options.search_seconds), "--worker-seconds", str(options.worker_seconds)]
    if getattr(options, "dispatch_mode", None):
        command += ["--dispatch-mode", options.dispatch_mode]
    # Explicit provider selection overrides inherited live-provider configuration.
    if options.provider == "openai":
        command += ["--model", model, "--provider-seconds", str(options.provider_seconds), "--max-output-tokens", str(options.max_output_tokens)]
    command += ["--", sys.executable, "-m", "benchmarks.workloads", "--case", name, "--seed", str(options.seed), "--events", *map(str, options.events)]
    return command


def _workload_evidence(log_path):
    evidence = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get("kind") == "synthetic-workload":
                evidence.append(value)
    return evidence


def _comparison_outcome(results, models):
    """Transport completion alone is not a completed model comparison."""
    outcomes = {}
    for model in models:
        rows = [row for row in results if row["model"] == model and row["expected_supported_graph"]]
        evaluated = sum(any(evaluation.get("correctness_status") in ("passed", "rejected") or evaluation.get("status") in ("passed", "rejected", "ineligible", "eligible", "optimized") for function in row.get("functions", []) for evaluation in function.get("evaluations", [])) for row in rows)
        status = "fallback_controls_only" if not rows else "no_evaluated_candidate" if not evaluated else "partial" if evaluated < len(rows) else "completed"
        outcomes[model] = {"status": status, "supported_runs": len(rows), "supported_runs_with_evaluated_candidates": evaluated}
    failed_runs = any(row.get("status") != "completed" or row.get("unsupported_fallback_verified") is False for row in results)
    status = "failed" if failed_runs or any(item["status"] == "no_evaluated_candidate" for item in outcomes.values()) else "partial" if any(item["status"] == "partial" for item in outcomes.values()) else "completed"
    return status, outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=("mock", "openai"), default="mock")
    parser.add_argument("--dispatch-mode", choices=("fast", "careful"), help="select the ordinary-call path to measure and save (default: library/environment policy)")
    parser.add_argument("--model", action="append", help="repeat to compare models; required for openai")
    parser.add_argument("--cases", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--events", nargs="+", type=int, default=[128, 2048])
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--function-seconds", type=float, default=180)
    parser.add_argument("--search-seconds", type=float, default=240)
    parser.add_argument("--worker-seconds", type=float, default=45)
    parser.add_argument("--provider-seconds", type=float, default=60)
    parser.add_argument("--run-seconds", type=float, default=900, help="wall-time limit for each model/case/repeat, including recording and replay")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, help="new directory; defaults under .agent-optimize/benchmarks")
    parser.add_argument("--list", action="store_true", help="describe cases without running")
    options = parser.parse_args(argv)
    if options.list:
        print(json.dumps([{"name": item.name, "description": item.description, "supported_graph": item.supported} for item in WORKLOADS.values()], indent=2))
        return 0
    if options.provider == "openai" and not options.model:
        parser.error("openai requires at least one explicit --model")
    if options.provider == "mock" and options.model:
        parser.error("--model requires --provider openai")
    if options.model and len(set(options.model)) != len(options.model):
        parser.error("select each model once; use --repeats for repeated measurements")
    if options.repeats < 1 or any(count < 0 for count in options.events):
        parser.error("repeats must be positive and event counts nonnegative")
    for key in ("max_proposals", "function_seconds", "search_seconds", "worker_seconds", "provider_seconds", "run_seconds", "max_output_tokens"):
        value = getattr(options, key)
        if value <= 0 or value == float("inf") or value != value:
            parser.error(f"{key.replace('_', '-')} must be finite and positive")
    models = options.model if options.provider == "openai" else ["mock"]
    started = datetime.now(timezone.utc)
    output = (options.output or ROOT / ".agent-optimize" / "benchmarks" / started.strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = {key: value for key, value in vars(options).items() if key != "output"}
    sources = [*sorted((ROOT / "benchmarks").glob("*.py")), *sorted((ROOT / "src" / "agent_optimize").glob("*.py"))]
    summary = {"schema_version": 1, "kind": "synthetic-end-to-end-model-comparison", "comparison_status": "running", "started_utc": started.isoformat(), "platform": _platform(), "settings": settings, "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, "results": []}
    summary_path = output / "comparison.json"
    _write_json(summary_path, summary)
    for repeat in range(options.repeats):
        # Rotate order across repeats to reduce a fixed model-order bias.
        ordered_models = models[repeat % len(models):] + models[:repeat % len(models)]
        for model in ordered_models:
            for name in options.cases:
                tag = re.sub(r"[^A-Za-z0-9._-]", "_", model)
                directory = output / f"r{repeat:02d}-m{models.index(model):02d}-{tag}" / name
                directory.mkdir(parents=True)
                command = _command(options, model, directory / "artifacts", name)
                environment = os.environ.copy()
                environment.pop("AGENT_OPTIMIZE_SESSION", None)
                environment["AGENT_OPTIMIZE_ARTIFACT_ROOT"] = str(directory / "artifacts")
                row = {"model": model, "provider": options.provider, "case": name, "repeat": repeat, "expected_supported_graph": WORKLOADS[name].supported, "command": command, "log_path": str(directory / "run.log")}
                start = time.monotonic()
                print(f"Running {model}: {name} (repeat {repeat + 1})", flush=True)
                try:
                    with (directory / "run.log").open("w", encoding="utf-8") as log:
                        process = run_benchmark_process(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", check=False, timeout=options.run_seconds)
                    (directory / "cli-stdout.json").write_text(process.stdout, encoding="utf-8")
                    row.update(exit_code=process.returncode, wall_seconds=time.monotonic() - start, workload_evidence=_workload_evidence(directory / "run.log"))
                    try:
                        report = json.loads(process.stdout)
                        _write_json(directory / "report.json", report)
                        row.update(summarize(report, expected_supported=WORKLOADS[name].supported), report_path=str(directory / "report.json"))
                    except (ValueError, TypeError, AttributeError) as error:
                        row.update(status="invalid_cli_report", error={"kind": type(error).__name__}, stdout_path=str(directory / "cli-stdout.json"))
                except subprocess.TimeoutExpired as error:
                    partial = error.stdout or ""
                    if isinstance(partial, bytes):
                        partial = partial.decode("utf-8", errors="replace")
                    partial_path = directory / "cli-stdout.partial.txt"
                    partial_path.write_text(partial, encoding="utf-8")
                    row.update(status="timed_out", exit_code=None, wall_seconds=time.monotonic() - start, timeout_seconds=options.run_seconds, http_request_count=None, stdout_path=str(partial_path), workload_evidence=_workload_evidence(directory / "run.log"))
                except OSError as error:
                    row.update(status="launch_error", wall_seconds=time.monotonic() - start, error={"kind": type(error).__name__, "message": str(error)})
                except KeyboardInterrupt:
                    row.update(status="interrupted", wall_seconds=time.monotonic() - start)
                    summary["results"].append(row)
                    summary["comparison_status"] = "interrupted"
                    _write_json(summary_path, summary)
                    print(f"Interrupted; completed results: {summary_path}", file=sys.stderr)
                    return 130
                summary["results"].append(row)
                _write_json(summary_path, summary)
                statuses = ", ".join(function.get("status", "unknown") for function in row.get("functions", []))
                print(f"  {row.get('status')}: {statuses}; {row['wall_seconds']:.1f}s", flush=True)
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    summary["comparison_status"], summary["model_outcomes"] = _comparison_outcome(summary["results"], models)
    _write_json(summary_path, summary)
    print(f"Comparison saved: {summary_path}")
    return 0 if summary["comparison_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
