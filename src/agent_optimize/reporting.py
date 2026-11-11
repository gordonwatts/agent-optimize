"""A shareable Markdown account of one run, built from evaluator-owned evidence."""

import ast
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


def _fence(code, language="python"):
    fence = "`" * max(3, max((len(x) + 1 for x in re.findall(r"`+", code)), default=3))
    return f"{fence}{language}\n{code.rstrip()}\n{fence}"


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


def _reference_listing(source):
    """Restore the opt-in marker for readers, without changing frozen evidence."""
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    lines = source.splitlines()
    lines.insert(function.lineno - 1, "@optimize")
    return "from agent_optimize import optimize\n" + "\n".join(lines)


def _correctness(evaluation):
    count = evaluation.get("case_count", "?")
    checks = evaluation.get("checks", [])
    lines = [f"**The Numba function passed all {count} correctness checks.** It also passed the speed and compatibility requirements.", ""]
    if checks and len(checks) == count and all(c.get("status") == "passed" for c in checks):
        recorded_ids = set(evaluation.get("recorded_case_ids", []))
        recorded = [c for c in checks if c["case_id"] in recorded_ids]
        generated = [c for c in checks if c["case_id"] not in recorded_ids]
        measured = {row[0]["case_id"]: (number, row[0].get("events"))
                    for number, row in enumerate(_timing_rows(evaluation), 1)}
        lines += ["Each correctness check compares the original function and the Numba function on one complete verification group, including every event in that group.", "",
                  "| Verification groups checked | Number of checks | Result |", "| --- | ---: | --- |"]
        for check in recorded:
            group = measured.get(check["case_id"])
            label = f"Verification group {group[0]}" if group else "Saved verification group"
            if group and isinstance(group[1], int):
                label += f" ({group[1]} events)"
            lines.append(f"| {label} | 1 | Passed |")
        lines += [f"| Synthetic test groups for edge cases | {len(generated)} | Passed |",
                  f"| **Total** | **{count}** | **Passed** |", ""]
    else:
        lines += ["Each correctness check compares the original function and the Numba function on one complete verification group, including every event in that group. A detailed breakdown of saved and generated verification groups is unavailable in this result.", ""]
    return lines


def render_report(result, name):
    functions = result.get("functions", [])
    optimized = sum(bool(_accepted(f)) for f in functions)
    status_label = "Analysis failed" if result["status"] == "child_failed" else result["status"].replace("_", " ").capitalize()
    lines = [f"# Optimization report: {_text(name)}", "", f"**Result:** {_text(status_label)}. **{optimized} of {len(functions)} functions optimized.**", ""]
    if result.get("run_command"):
        lines += ["Command used for this run:", "", _fence(result["run_command"], "text"), ""]
    lines += ["## Function speedup", "", "Each row is a different saved verification group. The Events column gives the number of events in that group. The original Python function and Numba function are timed using the same events, and repeated measurements reuse those events.", "",
              "| Function / verification group | Events | Original Python function | Numba function | Conservative speedup |", "| --- | ---: | ---: | ---: | ---: |"]
    rows = 0
    for function in functions:
        fname = function["identity"]["qualified_name"]
        for evaluation in _accepted(function):
            for number, (recording, reference, candidate, ratio) in enumerate(_timing_rows(evaluation), 1):
                rows += 1
                lines.append(f"| {_text(fname)} / verification group {number} | {recording.get('events', '?')} | {reference:.3f} ms | {candidate:.3f} ms | {ratio:.2f}x |")
    if rows:
        lines += ["", "Times are medians of repeated measurements after warm-up, including the overhead of calling the Numba function through `@optimize`. The conservative speedup is the lowest ratio across timing rounds for each verification group, so it can differ from the ratio of the displayed medians. These are function timings; reading data, plotting and other analysis work are outside this comparison."]
    else:
        lines += ["", "No Numba function with measured timing evidence was saved."]
    lines += ["", "## Time and API cost to obtain the result", "", "| Phase | Elapsed time |", "| --- | ---: |"]
    timings = result.get("run_timing", {})
    for key, label in (("analysis_seconds", "Run analysis once and save verification groups"), ("optimization_seconds", "Generate, check, compile and time Numba functions"), ("controller_seconds", "Total elapsed time")):
        value = timings.get(key)
        lines.append(f"| {label} | {value:.3f} s |" if isinstance(value, (int, float)) else f"| {label} | Unavailable |")
    lines += ["", "The total covers the analysis and optimization work, plus coordination overhead. Preparing the Python environment is excluded. The analysis executes once; subsequent checks rerun only the saved function inputs.", ""]
    for function in functions:
        for evaluation in _accepted(function):
            timing = evaluation.get("timing", {})
            if timing.get("mode") == "measured" and timing.get("status") == "passed":
                lines += [f"For {_text(function['identity']['qualified_name'])}, compilation took {timing.get('compilation_ns', 0) / 1e6:.3f} ms within the first Numba function call of {timing.get('first_use_ns', 0) / 1e6:.3f} ms. That first call is included in the optimization phase above; do not add it or compilation again to the total. The speedup table uses later, warm calls.", ""]
    lines += [usage_line(result["usage_summary"]), ""]
    summary = result["usage_summary"]
    local_only = summary.get("models") and all(m["provider"] == "mock" for m in summary["models"])
    if local_only:
        lines += ["This run used a deterministic local code generator (the mock backend). It made no model API calls, which explains the zero tokens and zero API cost for this run. Searches using a model have their own token usage and API costs.", ""]
    elif summary.get("models"):
        lines += ["| Code generator / model | Reported tokens | Estimated API cost |", "| --- | ---: | ---: |"]
        for model in summary["models"]:
            tokens = model["tokens"]["total_tokens"]
            known = " (partial)" if not model["coverage"]["total_tokens"]["complete"] else ""
            cost = model["cost"]["estimated_total_usd"]
            lines.append(f"| {_text(model['provider'])} / {_text(model['model'])} | {str(tokens) + known if tokens is not None else 'Unavailable'} | {'USD ' + cost if cost is not None else 'Unavailable'} |")
        lines.append("")
    if result.get("error"):
        lines += ["**Run diagnostic:** " + _text(result["error"]["message"]), ""]
    lines += ["## Checks and saved code", "", "Correctness here means exact agreement with the original Python function, including floating-point bit patterns.", ""]
    if not functions:
        lines += ["No decorated function calls were recorded.", ""]
    prices = {}
    for function in functions:
        accepted = _accepted(function)
        iterations = len(function.get("evaluations", []))
        status = f"The Numba function passed all required checks ({iterations} attempts to reach the final code)." if accepted else "No Numba function was saved; the original Python function remains in use."
        saved = function.get("captured_calls", 0)
        total_calls = saved + sum(skip.get("count", 0) for skip in function.get("skipped", {}).values())
        lines += [f"### Function {_text(function['identity']['qualified_name'])}", "", status, "",
                  f"Verified the Numba function against {saved} real-data verification groups (out of {total_calls} total function calls)." if accepted else f"Collected {saved} verification groups (out of {total_calls} total function calls).", ""]
        for evaluation in function.get("evaluations", []):
            pricing = evaluation.get("provider_cost", {}).get("pricing")
            if pricing:
                prices[(pricing["version"], pricing["model"])] = pricing
        if not accepted:
            if function.get("reason"):
                lines += ["**Details:** " + _text(function["reason"]), ""]
            continue
        for number, evaluation in enumerate(accepted, 1):
            if len(accepted) > 1:
                lines += [f"#### Numba function for input layout {number}", ""]
            lines += _correctness(evaluation)
            timing = evaluation.get("timing", {})
            if timing.get("mode") != "measured":
                lines += ["Timing evidence is a controlled test fixture; it is not a measured performance claim.", ""]
            for kind, title in (("reference", "Original Python function"), ("candidate", "Numba function")):
                source = _source(evaluation, kind)
                lines += [f"**{title}**", ""]
                if source is not None and kind == "reference":
                    source = _reference_listing(source)
                lines += [_fence(source) if source is not None else "Verified source is unavailable in the retained workspace.", ""]
    if prices:
        lines += ["## Cost estimate basis", "", "Costs are estimates of agent-optimize's API requests, not a reconciled invoice. Reasoning tokens are included in output; cache reads/writes are subsets of input. Missing usage/rates leave the total unavailable.", ""]
        for pricing in prices.values():
            lines += [f"- {_text(pricing['model'])}: {_text(pricing['version'])}, verified {_text(pricing['verified_on'])}. {_text(pricing['scope'])}. [Pricing source]({pricing['source']})."]
        lines.append("")
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
