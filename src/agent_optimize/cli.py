"""Run an analysis once and report bounded local optimization outcomes."""

from collections.abc import Sequence
from dataclasses import asdict
from enum import Enum
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import click
import typer
from typer.main import get_command

from .store import create_session, DEFAULT_CALL_LIMIT, DEFAULT_CAPTURE_BYTES, SESSION_ENV, json_bytes, read_json, write_json


class ConfigurationError(ValueError):
    pass


class DispatchMode(str, Enum):
    fast = "fast"
    careful = "careful"


class Provider(str, Enum):
    mock = "mock"
    openai = "openai"


app = typer.Typer(no_args_is_help=False, add_completion=False, pretty_exceptions_enable=False)


@app.command(context_settings={"allow_interspersed_args": False})
def run(
    json_output: bool = typer.Option(False, "--json", help="Write one versioned JSON result to stdout; send child output and progress to stderr."),
    report: bool = typer.Option(False, "--report", help="Write a Markdown summary with timings and Python/Numba source in the current directory."),
    artifact_root: Path = typer.Option(Path(".agent-optimize")),
    dispatch_mode: DispatchMode = typer.Option(None),
    max_capture_bytes: int = typer.Option(DEFAULT_CAPTURE_BYTES),
    max_calls: int = typer.Option(DEFAULT_CALL_LIMIT, "--verification-groups", "--max-calls", help="Save up to this many real-data verification groups per function for correctness checks and timing. One group is one function call."),
    max_proposals: int = typer.Option(3, "--max-attempts", "--max-proposals", help="Maximum replacement-code attempts per function. If none passes the correctness and speed checks, keep the original Python function."),
    function_seconds: float = typer.Option(120),
    worker_seconds: float = typer.Option(30),
    search_seconds: float = typer.Option(300),
    max_request_bytes: int = typer.Option(128 * 1024),
    max_response_bytes: int = typer.Option(64 * 1024),
    max_extra_cases: int = typer.Option(8),
    provider: Provider = typer.Option(Provider.mock),
    model: str = typer.Option("gpt-6.1-sol"),
    api_key: str = typer.Option(None, help="OpenAI credential; prefer AGENT_OPTIMIZE_OAI_API_KEY or .env to avoid shell history."),
    provider_seconds: float = typer.Option(30),
    max_output_tokens: int = typer.Option(4096),
    command: list[str] = typer.Argument(None, help="script.py [arguments], or -- <command> [arguments]; optimizer options precede the script."),
):
    """Run an analysis once, then check and save eligible implementations."""
    from .provider import MODELS
    if model not in MODELS:
        raise ConfigurationError(f"invalid model: {model!r} (choose from {', '.join(MODELS)})")
    return SimpleNamespace(json=json_output, report=report, artifact_root=artifact_root,
        dispatch_mode=dispatch_mode.value if dispatch_mode else None,
        max_capture_bytes=max_capture_bytes, max_calls=max_calls, max_proposals=max_proposals,
        function_seconds=function_seconds, worker_seconds=worker_seconds, search_seconds=search_seconds,
        max_request_bytes=max_request_bytes, max_response_bytes=max_response_bytes,
        max_extra_cases=max_extra_cases, provider=provider.value, model=model, api_key=api_key,
        provider_seconds=provider_seconds, max_output_tokens=max_output_tokens, command=command or [])


# Force a group even though there is currently one subcommand.
@app.callback()
def _root():
    """Record and optimize an analysis."""


def _parse(arguments):
    cli = get_command(app)
    try:
        with cli.make_context("agent-optimize", arguments.copy()) as ctx:
            options = cli.invoke(ctx)
    except click.exceptions.Exit as exit_request:
        raise SystemExit(exit_request.exit_code) from None
    except click.ClickException as error:
        raise ConfigurationError(error.format_message()) from error
    # Click consumes the separator. Restore it only when it introduced the
    # command, leaving separators in the child's arguments untouched.
    command_start = len(arguments) - len(options.command)
    if command_start and arguments[command_start - 1] == "--":
        options.command.insert(0, "--")
    return options


def _checkpoint(session, fallback):
    try:
        return read_json(session / "report.json")
    except (OSError, ValueError):
        return fallback


def _finish(report, status, exit_code, *, error=None):
    from .usage import update_usage
    result = update_usage({**report, "status": status, "exit_code": exit_code})
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
    options = _parse(arguments)
    command = options.command
    started = time.perf_counter()
    analysis_seconds = None
    optimization_started = None

    def finish(value, status, exit_code, *, error=None):
        from .reporting import write_report
        now = time.perf_counter()
        value = {**value, "run_timing": {"analysis_seconds": analysis_seconds,
            "optimization_seconds": now - optimization_started if optimization_started is not None else None,
            "controller_seconds": now - started}}
        result = _finish(value, status, exit_code, error=error)
        if options.report:
            try:
                result["markdown_report_path"] = str(write_report(result, command))
            except (OSError, ValueError) as failure:
                return _finish(result, "error", exit_code or 2, error={"phase": "markdown_report", "kind": type(failure).__name__, "message": "Could not write Markdown report; inspect the current directory permissions and existing report files."})
            result = _finish(result, result["status"], result["exit_code"])
        return result

    if not command or command[0] != "--" or len(command) == 1:
        raise ConfigurationError("run requires script.py [arguments] or a command after --, for example: run -- python analysis.py")
    if options.max_capture_bytes <= 0 or options.max_calls <= 0:
        raise ConfigurationError("--max-capture-bytes and --verification-groups must be positive")
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
        raise ConfigurationError(str(error))
    base = {"schema_version": 1, "child_exit_code": None, "functions": [], "limits": {"max_capture_bytes": options.max_capture_bytes, "max_calls_per_function": options.max_calls}, "search_limits": asdict(limits)}
    base["provider"] = provider.identity
    base["dispatch_mode"] = selected_mode
    base["python_executable"] = sys.executable
    try:
        session = create_session(options.artifact_root, max_bytes=options.max_capture_bytes, max_calls=options.max_calls, provider=provider.identity)
    except (OSError, ValueError) as error:
        return finish(base, "error", 2, error={"phase": "session", "kind": type(error).__name__, "message": f"could not create recording session: {error}"[:500]})
    base.update(session_id=session.name, session_directory=str(session), artifact_root=str(session.parent.parent), captured_bytes=0)
    print(f"agent-optimize: recording at {session}; limits: {options.max_capture_bytes} bytes, {options.max_calls} verification groups/function", file=sys.stderr)
    print(f"agent-optimize: dispatch mode: {selected_mode}; correctness checks remain enabled during evaluation", file=sys.stderr)
    print(f"agent-optimize: {provider.kind} search limits: {limits.proposals} attempts/function, {limits.function_seconds:g}s/function, {limits.worker_seconds:g}s/worker, {limits.session_seconds:g}s/session, {limits.request_bytes} request bytes, {limits.response_bytes} response bytes, {limits.extra_cases} extra cases/proposal", file=sys.stderr)
    from .process import worker_environment
    environment = worker_environment()
    environment[SESSION_ENV] = str(session)
    environment["AGENT_OPTIMIZE_MODE"] = selected_mode
    child_started = time.perf_counter()
    try:
        # JSON mode directly inherits stderr for BOTH child streams, keeping
        # arbitrary analysis output out of stdout without buffering it in memory.
        streams = {"stdout": sys.stderr, "stderr": sys.stderr} if options.json else {}
        child = subprocess.run(command[1:], env=environment, check=False, **streams)
    except OSError as error:
        return finish(base, "error", 2, error={"phase": "launch", "kind": type(error).__name__, "message": f"could not launch command: {error}"[:500]})
    except KeyboardInterrupt:
        analysis_seconds = time.perf_counter() - child_started
        return finish(_checkpoint(session, base), "interrupted", 130)
    analysis_seconds = time.perf_counter() - child_started
    optimization_started = time.perf_counter()
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
        return finish({**base, **_checkpoint(session, base)}, "interrupted", 130)
    except Exception as error:
        message = f"local {phase} failed: {type(error).__name__}" if provider.kind == "openai" else f"local {phase} failed: {type(error).__name__}: {error}"[:500]
        return finish({**base, **_checkpoint(session, base)}, "error", child.returncode or 2, error={"phase": phase, "kind": type(error).__name__, "message": message})
    return finish(report, "child_failed" if child.returncode else "completed", child.returncode)


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
    from .usage import usage_line
    if result.get("usage_summary"):
        print(usage_line(result["usage_summary"]), file=sys.stderr)
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
            print(f"  search stopped: {outcome['search']['stop']}; {outcome['search']['proposals']} attempts", file=sys.stderr)
    if result.get("markdown_report_path"):
        print(f"agent-optimize: Markdown report: {result['markdown_report_path']}", file=sys.stderr)
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
        options = _parse(arguments)
        command = options.command
        if command and command[0] != "--" and Path(command[0]).suffix.lower() == ".py":
            prefix = arguments[:len(arguments) - len(command)]
            json_mode = options.json
            if options.max_capture_bytes <= 0 or options.max_calls <= 0:
                raise ConfigurationError("--max-capture-bytes and --verification-groups must be positive")
            from .search import SearchLimits
            from .provider import ProviderSettings
            from .modes import dispatch_mode
            from .script import launch_script
            try:
                dispatch_mode(options.dispatch_mode)
                SearchLimits(options.max_proposals, options.function_seconds, options.worker_seconds, options.search_seconds, options.max_request_bytes, options.max_response_bytes, options.max_extra_cases)
                ProviderSettings(options.provider, options.model, options.provider_seconds, options.max_output_tokens)
                return launch_script(arguments, command, json_mode=json_mode, provider=options.provider)
            except ValueError as error:
                raise ConfigurationError(str(error)) from error
            except KeyboardInterrupt:
                return 130
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
