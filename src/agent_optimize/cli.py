"""Run an analysis once and report bounded local optimization outcomes."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .store import create_session, DEFAULT_CALL_LIMIT, DEFAULT_CAPTURE_BYTES, SESSION_ENV, json_bytes, read_json, write_json


class ConfigurationError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ConfigurationError(message)


def _parser():
    parser = _Parser(prog="agent-optimize", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", allow_abbrev=False, help="run an analysis once, then check and save eligible implementations")
    run.add_argument("--json", action="store_true", help="write one versioned JSON result to stdout; send child output and progress to stderr")
    run.add_argument("--artifact-root", type=Path, default=Path(".agent-optimize"))
    run.add_argument("--dispatch-mode", choices=("fast", "careful"), default=None, help="measure/save fast ordinary dispatch (default) or opt into repeated careful compatibility checks")
    run.add_argument("--max-capture-bytes", type=int, default=DEFAULT_CAPTURE_BYTES)
    run.add_argument("--max-calls", type=int, default=DEFAULT_CALL_LIMIT)
    run.add_argument("--max-proposals", type=int, default=3)
    run.add_argument("--function-seconds", type=float, default=120)
    run.add_argument("--worker-seconds", type=float, default=30)
    run.add_argument("--search-seconds", type=float, default=300)
    run.add_argument("--max-request-bytes", type=int, default=128 * 1024)
    run.add_argument("--max-response-bytes", type=int, default=64 * 1024)
    run.add_argument("--max-extra-cases", type=int, default=8)
    run.add_argument("--provider", choices=("mock", "openai"), default="mock")
    from .provider import MODELS
    run.add_argument("--model", choices=MODELS, default="gpt-6.1-sol")
    run.add_argument("--api-key", help="OpenAI credential; prefer AGENT_OPTIMIZE_OAI_API_KEY or .env to avoid shell history")
    run.add_argument("--provider-seconds", type=float, default=30)
    run.add_argument("--max-output-tokens", type=int, default=4096)
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _checkpoint(session, fallback):
    try:
        return read_json(session / "report.json")
    except (OSError, ValueError):
        return fallback


def _finish(report, status, exit_code, *, error=None):
    result = {**report, "status": status, "exit_code": exit_code}
    if error:
        result["error"] = error
    if status == "interrupted":
        result["next_action"] = "Inspect completed evaluations and saved artifacts in the report; rerun the command to start a new search."
    elif status in ("error", "child_failed"):
        result["next_action"] = "Resolve the command or report diagnostic and run again; the Python reference remains available."
    elif any(function["status"] == "optimized" for function in result["functions"]):
        result["next_action"] = "Run the script with AGENT_OPTIMIZE_MODE=careful to reuse the saved careful-mode implementations." if result.get("dispatch_mode") == "careful" else "Run the script normally to reuse the saved fast-mode implementations."
    elif not result["functions"]:
        result["next_action"] = "Add @optimize to a supported function and call it in the script; Python remains in use."
    elif not any(function["captured_calls"] for function in result["functions"]):
        result["next_action"] = "Inspect skipped-capture diagnostics and use supported inputs; Python remains in use."
    else:
        result["next_action"] = "Continue using Python; inspect the report for failed, unsupported, rejected, or ineligible evaluations."
    if result.get("session_directory"):
        path = Path(result["session_directory"]) / "report.json"
        result["report_path"] = str(path)
        try:
            write_json(path, result)
        except (OSError, ValueError) as failure:
            result["report_error"] = f"could not checkpoint final report: {type(failure).__name__}"
            if status == "completed":
                result.update(status="error", exit_code=2, next_action="Check the artifact directory permissions and free space; completed evaluations remain in their workspaces.")
    return result


def _run(arguments):
    parser = _parser()
    options = parser.parse_args(arguments)
    command = options.command
    if not command or command[0] != "--" or len(command) == 1:
        parser.error("run requires a command after --, for example: run -- python analysis.py")
    if options.max_capture_bytes <= 0 or options.max_calls <= 0:
        parser.error("--max-capture-bytes and --max-calls must be positive")
    from .search import SearchLimits
    from .provider import ProviderSettings, resolve_api_key

    try:
        from .modes import dispatch_mode
        selected_mode = dispatch_mode(options.dispatch_mode)
        limits = SearchLimits(options.max_proposals, options.function_seconds, options.worker_seconds, options.search_seconds, options.max_request_bytes, options.max_response_bytes, options.max_extra_cases)
        provider = ProviderSettings(options.provider, options.model, options.provider_seconds, options.max_output_tokens)
        api_key = resolve_api_key(options.api_key) if provider.kind == "openai" else None
        if provider.kind == "openai":
            import importlib.util
            if any(importlib.util.find_spec(name) is None for name in ("openai", "pydantic")):
                raise ValueError("OpenAI provider requires the optional package: pip install 'agent-optimize[openai]'")
    except ValueError as error:
        parser.error(str(error))
    base = {"schema_version": 1, "child_exit_code": None, "functions": [], "limits": {"max_capture_bytes": options.max_capture_bytes, "max_calls_per_function": options.max_calls}, "search_limits": asdict(limits)}
    base["provider"] = provider.identity
    base["dispatch_mode"] = selected_mode
    try:
        session = create_session(options.artifact_root, max_bytes=options.max_capture_bytes, max_calls=options.max_calls, provider=provider.identity)
    except (OSError, ValueError) as error:
        return _finish(base, "error", 2, error={"phase": "session", "kind": type(error).__name__, "message": f"could not create recording session: {error}"[:500]})
    base.update(session_id=session.name, session_directory=str(session), artifact_root=str(session.parent.parent), captured_bytes=0)
    print(f"agent-optimize: recording at {session}; limits: {options.max_capture_bytes} bytes, {options.max_calls} attempts/function", file=sys.stderr)
    print(f"agent-optimize: dispatch mode: {selected_mode}; correctness checks remain enabled during evaluation", file=sys.stderr)
    print(f"agent-optimize: {provider.kind} search limits: {limits.proposals} proposals/function, {limits.function_seconds:g}s/function, {limits.worker_seconds:g}s/worker, {limits.session_seconds:g}s/session, {limits.request_bytes} request bytes, {limits.response_bytes} response bytes, {limits.extra_cases} extra cases/proposal", file=sys.stderr)
    from .process import worker_environment
    environment = worker_environment()
    environment[SESSION_ENV] = str(session)
    environment["AGENT_OPTIMIZE_MODE"] = selected_mode
    try:
        # JSON mode directly inherits stderr for BOTH child streams, keeping
        # arbitrary analysis output out of stdout without buffering it in memory.
        streams = {"stdout": sys.stderr, "stderr": sys.stderr} if options.json else {}
        child = subprocess.run(command[1:], env=environment, check=False, **streams)
    except OSError as error:
        return _finish(base, "error", 2, error={"phase": "launch", "kind": type(error).__name__, "message": f"could not launch command: {error}"[:500]})
    except KeyboardInterrupt:
        return _finish(_checkpoint(session, base), "interrupted", 130)
    base["child_exit_code"] = child.returncode
    phase = "replay"
    try:
        from .replay import replay_session

        report = {**base, **replay_session(session, child.returncode, timeout=limits.worker_seconds)}
        if not child.returncode:
            phase = "search"
            from .evaluator import evaluate_session

            report = evaluate_session(session, report, search_limits=limits, dispatch_mode=selected_mode, **({"provider": provider, "api_key": api_key} if provider.kind == "openai" else {}))
    except KeyboardInterrupt:
        return _finish({**base, **_checkpoint(session, base)}, "interrupted", 130)
    except Exception as error:
        message = f"local {phase} failed: {type(error).__name__}" if provider.kind == "openai" else f"local {phase} failed: {type(error).__name__}: {error}"[:500]
        return _finish({**base, **_checkpoint(session, base)}, "error", child.returncode or 2, error={"phase": phase, "kind": type(error).__name__, "message": message})
    return _finish(report, "child_failed" if child.returncode else "completed", child.returncode)


def _show_timings(timing):
    if timing.get("mode") != "measured":
        if timing:
            print("    controlled test timings; not a measured performance claim", file=sys.stderr)
        return
    if timing.get("status") != "passed":
        print(f"    timing unavailable: {timing.get('reason', 'measurement did not complete')}", file=sys.stderr)
        return
    print(f"    compilation {timing.get('compilation_ns', 0) / 1e6:.3f} ms; first replacement use {timing.get('first_use_ns', 0) / 1e6:.3f} ms", file=sys.stderr)
    for recording in timing.get("recordings", []):
        pairs = [(statistics.median(batch["reference_ns"]), statistics.median(batch["candidate_ns"])) for batch in recording.get("batches", []) if batch.get("reference_ns") and batch.get("candidate_ns")]
        if pairs and all(left > 0 and right > 0 for left, right in pairs):
            reference = statistics.median(left for left, _ in pairs) / 1e6
            candidate = statistics.median(right for _, right in pairs) / 1e6
            ratio = min(left / right for left, right in pairs)
            print(f"    {recording['case_id']}: {recording.get('events', '?')} events, {recording.get('elements', '?')} elements; reference {reference:.3f} ms/call, replacement {candidate:.3f} ms/call; minimum measured batch speedup {ratio:.2f}x", file=sys.stderr)


def _show_result(result):
    if "error" in result:
        print(f"agent-optimize: {result['error']['message']}", file=sys.stderr)
    if result["status"] == "interrupted":
        print("agent-optimize: interrupted; completed evaluations and saved implementations remain available", file=sys.stderr)
    if result["status"] == "child_failed":
        print(f"agent-optimize: analysis command exited {result['child_exit_code']}; no replay or optimization was started", file=sys.stderr)
    if not result["functions"] and result["status"] == "completed":
        print("agent-optimize: no decorated function calls encountered; nothing to optimize", file=sys.stderr)
    for outcome in result["functions"]:
        name = outcome["identity"]["qualified_name"]
        passed = sum(replay["status"] == "passed" for replay in outcome["replays"])
        print(f"{name}: {outcome['captured_calls']} captured, {passed} replayed; {outcome['reason']}", file=sys.stderr)
        for skip in outcome["skipped"].values():
            print(f"  skipped {skip['count']}: {skip['diagnostic']}", file=sys.stderr)
        for replay in outcome["replays"]:
            if replay["status"] != "passed":
                print(f"  replay failed: {replay.get('reason', 'inspect the local report')}", file=sys.stderr)
        for evaluation in outcome.get("evaluations", []):
            print(f"  compiled candidate: {evaluation['status']}; {evaluation['reason']}", file=sys.stderr)
            _show_timings(evaluation.get("timing", {}))
            if evaluation.get("artifact_path"):
                print(f"    saved manifest: {evaluation['artifact_path']}", file=sys.stderr)
        if outcome.get("search"):
            print(f"  search stopped: {outcome['search']['stop']}; {outcome['search']['proposals']} proposals", file=sys.stderr)
    if result.get("report_path"):
        print(f"agent-optimize: report: {result['report_path']}", file=sys.stderr)
    if result.get("report_error"):
        print(f"agent-optimize: {result['report_error']}", file=sys.stderr)
    print(f"Next: {result['next_action']}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # A script's own --json after the separator is forwarded unchanged.
    prefix = arguments[:arguments.index("--")] if "--" in arguments else arguments
    json_mode = "--json" in prefix
    try:
        result = _run(arguments)
    except ConfigurationError as error:
        message = str(error)
        secrets = [os.environ.get("AGENT_OPTIMIZE_OAI_API_KEY")]
        for index, argument in enumerate(prefix):
            if argument == "--api-key" and index + 1 < len(prefix):
                secrets.append(prefix[index + 1])
            elif argument.startswith("--api-key="):
                secrets.append(argument.partition("=")[2])
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[redacted]")
        result = _finish({"schema_version": 1, "child_exit_code": None, "functions": []}, "error", 2, error={"phase": "configuration", "kind": "ConfigurationError", "message": f"{message}; use agent-optimize run --help"})
        _show_result(result)
        if not json_mode:
            raise SystemExit(2) from None
    else:
        _show_result(result)
    if json_mode:
        sys.stdout.write(json_bytes(result).decode("utf-8"))
    return result["exit_code"]
