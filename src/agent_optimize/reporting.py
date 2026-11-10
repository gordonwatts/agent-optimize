"""A shareable Markdown account of one run, built from evaluator-owned evidence."""

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import uuid

from .usage import usage_line


def command_name(command):
    """Name the script/command without copying arguments or inline code."""
    args = list(command[1:] if command and command[0] == "--" else command)
    if not args:
        return "analysis"
    first = Path(args[0]).name
    if first.lower().startswith("python"):
        if "-c" in args[1:]:
            return "python-command"
        if len(args) > 2 and args[1] == "-m":
            return args[2]
        for arg in args[1:]:
            if not arg.startswith("-"):
                return Path(arg).stem
    return Path(first).stem


def _text(value):
    value = html.escape(str(value)).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([`*\[\]|_])", r"\\\1", value)


def _fence(code):
    fence = "`" * max(3, max((len(x) + 1 for x in re.findall(r"`+", code)), default=3))
    return f"{fence}python\n{code.rstrip()}\n{fence}"


def _source(evaluation, kind):
    """Read frozen source without importing the analysis or executing code."""
    try:
        directory = Path(evaluation["workspace"])
        if kind == "reference":
            expected = json.loads((directory / "reference-job.json").read_text(encoding="utf-8"))["source_sha256"]
        else:
            expected = evaluation["candidate_sha256"]
        path = directory / f"{kind}.py"
        if path.stat().st_size > 256 * 1024:
            return None
        data = path.read_bytes()
        return data.decode("utf-8") if hashlib.sha256(data).hexdigest() == expected else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _accepted(function):
    chosen = {}
    for evaluation in function.get("evaluations", []):
        if evaluation.get("status") == "optimized":
            chosen[json.dumps(evaluation.get("contract", {}), sort_keys=True)] = evaluation
    return list(chosen.values())


def _timing_rows(evaluation):
    timing = evaluation.get("timing", {})
    if timing.get("mode") != "measured" or timing.get("status") != "passed":
        return []
    rows = []
    for recording in timing.get("recordings", []):
        pairs = [(statistics.median(batch["reference_ns"]), statistics.median(batch["candidate_ns"]))
                 for batch in recording.get("batches", []) if batch.get("reference_ns") and batch.get("candidate_ns")]
        if pairs and all(a > 0 and b > 0 for a, b in pairs):
            rows.append((recording, statistics.median(a for a, _ in pairs) / 1e6,
                         statistics.median(b for _, b in pairs) / 1e6, min(a / b for a, b in pairs)))
    return rows


def render_report(result, name):
    functions = result.get("functions", [])
    optimized = sum(bool(_accepted(f)) for f in functions)
    status_label = "Analysis failed" if result["status"] == "child_failed" else result["status"].replace("_", " ").capitalize()
    lines = [f"# Optimization report: {_text(name)}", "", f"**Result:** {_text(status_label)}. **{optimized} of {len(functions)} functions optimized.**", ""]
    timings = result.get("run_timing", {})
    lines += ["## Run at a glance", "", "| Phase | Elapsed time |", "| --- | ---: |"]
    for key, label in (("analysis_seconds", "Analysis (recording run)"), ("optimization_seconds", "Replay, correctness and optimization"), ("controller_seconds", "Controller total")):
        value = timings.get(key)
        lines.append(f"| {label} | {value:.3f} s |" if isinstance(value, (int, float)) else f"| {label} | Unavailable |")
    lines += ["", "Controller timings exclude uv environment preparation. The analysis executes once during recording.", "", usage_line(result["usage_summary"]), ""]
    summary = result["usage_summary"]
    if summary["attempts"]:
        lines += [f"Provider attempts: {summary['attempts']}. HTTP requests: {summary['requests']['total'] if summary['requests']['total'] is not None else 'total unavailable'}.", ""]
    if summary.get("models") and all(m["provider"] == "mock" for m in summary["models"]):
        lines += ["Generator: deterministic local mock. No model calls were made.", ""]
    if summary.get("models"):
        lines += ["| Provider / model | Reported tokens | Estimated API cost |", "| --- | ---: | ---: |"]
        for model in summary["models"]:
            tokens = model["tokens"]["total_tokens"]
            known = " (partial)" if not model["coverage"]["total_tokens"]["complete"] else ""
            cost = model["cost"]["estimated_total_usd"]
            lines.append(f"| {_text(model['provider'])} / {_text(model['model'])} | {str(tokens) + known if tokens is not None else 'Unavailable'} | {'USD ' + cost if cost is not None else 'Unavailable'} |")
        lines.append("")
    if result.get("error"):
        lines += ["**Run diagnostic:** " + _text(result["error"]["message"]), ""]
    lines += ["## Timing comparison", "", "Measurements below compare function calls on recorded batches, including ordinary dispatch. They are not whole-analysis speedups. Correctness means agreement with the Python reference, not independent validation of the underlying analysis.", "",
              "| Function / batch | Events | Python reference | Replacement | Minimum batch speedup |", "| --- | ---: | ---: | ---: | ---: |"]
    rows = 0
    for function in functions:
        fname = function["identity"]["qualified_name"]
        for evaluation in _accepted(function):
            for recording, reference, candidate, ratio in _timing_rows(evaluation):
                rows += 1
                lines.append(f"| {_text(fname)} / {_text(recording['case_id'])} | {recording.get('events', '?')} | {reference:.3f} ms | {candidate:.3f} ms | {ratio:.2f}x |")
    if not rows:
        lines += ["", "No eligible replacement with measured timing evidence was saved."]
    lines += ["", "## Functions", ""]
    if not functions:
        lines += ["No decorated function calls were recorded.", ""]
    prices = {}
    for function in functions:
        lines += [f"### {_text(function['identity']['qualified_name'])}", "", f"**Status:** {_text(function['status'])}. {_text(function['reason'])}", "", f"Recorded calls: {function.get('captured_calls', 0)}. Proposals attempted: {len(function.get('evaluations', []))}.", ""]
        outcomes = Counter(e.get("status", "unknown").replace("_", " ") for e in function.get("evaluations", []))
        if outcomes:
            lines += ["Attempt outcomes: " + ", ".join(f"{count} {_text(status)}" for status, count in outcomes.items()) + ".", ""]
        for skip in function.get("skipped", {}).values():
            lines += [f"Not recorded: {skip['count']} calls. {_text(skip['diagnostic'])}. These calls still used the Python reference.", ""]
        for evaluation in function.get("evaluations", []):
            pricing = evaluation.get("provider_cost", {}).get("pricing")
            if pricing:
                prices[(pricing["version"], pricing["model"])] = pricing
        accepted = _accepted(function)
        if not accepted:
            lines += ["No replacement saved; the Python reference remains in use.", ""]
            continue
        for number, evaluation in enumerate(accepted, 1):
            lines += [f"#### Saved replacement {number}", "", f"Correctness: **{evaluation.get('case_count', '?')} cases passed**. Timing and compatibility gates passed.", ""]
            timing = evaluation.get("timing", {})
            if timing.get("mode") == "measured":
                lines += [f"Compilation: {timing.get('compilation_ns', 0) / 1e6:.3f} ms. First replacement use: {timing.get('first_use_ns', 0) / 1e6:.3f} ms. These are separate from warm measurements.", ""]
            else:
                lines += ["Timing evidence is a controlled test fixture; it is not a measured performance claim.", ""]
            for kind, title in (("reference", "Python reference"), ("candidate", "Numba replacement")):
                source = _source(evaluation, kind)
                lines += [f"**{title}**", "", _fence(source) if source is not None else "Verified source is unavailable in the retained workspace.", ""]
    if prices:
        lines += ["## Cost estimate basis", "", "Costs are estimates of agent-optimize's API requests, not a reconciled invoice. Reasoning tokens are included in output; cache reads/writes are subsets of input. Missing usage/rates leave the total unavailable.", ""]
        for pricing in prices.values():
            lines += [f"- {_text(pricing['model'])}: {_text(pricing['version'])}, verified {_text(pricing['verified_on'])}. {_text(pricing['scope'])}. [Pricing source]({pricing['source']})."]
        lines.append("")
    lines += ["## Next step", "", _text(result["next_action"]), ""]
    return "\n".join(lines)


def write_report(result, command, directory=None):
    name = command_name(command)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")[:80] or "analysis"
    identifier = result.get("session_id", uuid.uuid4().hex)[:12]
    path = (Path.cwd() if directory is None else Path(directory)) / f"{safe}-{identifier}.agent-optimize.md"
    content = render_report(result, name)
    # Exclusive creation preserves existing logs, including reports from a prior run.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return path.resolve()
