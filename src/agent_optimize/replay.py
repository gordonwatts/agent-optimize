"""Replay completed captures only after the analysis command has succeeded."""

from pathlib import Path
import sys

from .process import run_worker
from .store import read_json, write_json


def replay_session(directory: Path, child_exit_code: int, *, timeout=None) -> dict:
    state = read_json(directory / "session.json")
    report = {
        "schema_version": 1, "session_id": state["session_id"],
        "session_directory": str(directory.resolve()), "limits": state["limits"],
        "captured_bytes": state["captured_bytes"], "child_exit_code": child_exit_code,
        "functions": [],
    }
    for function_id, entry in state["functions"].items():
        outcome = {
            "function_id": function_id, "identity": entry["identity"],
            "captured_calls": len(entry["captures"]), "skipped": entry["skipped"],
            "replays": [], "status": "reference",
            "reason": "reference replay complete; Python remains in use until an eligible implementation is saved",
        }
        if child_exit_code:
            outcome["reason"] = "analysis command failed; no replay or optimization was started"
        else:
            for call_id in entry["captures"]:
                result_path = directory / "calls" / call_id / "replay.json"
                try:
                    result_path.unlink(missing_ok=True)
                    process = run_worker(
                        [sys.executable, "-m", "agent_optimize.worker", str(directory.resolve()), call_id, str(result_path.resolve())],
                        timeout=timeout,
                    )
                    if process.status not in ("passed", "failed"):
                        replay = {"schema_version": 1, "call_id": call_id, "status": "failed", "reason": f"reference worker {process.status}"}
                        write_json(result_path, replay)
                    elif result_path.exists():
                        replay = read_json(result_path)
                        if process.returncode and replay["status"] == "passed":
                            replay = {"schema_version": 1, "call_id": call_id, "status": "failed", "reason": "reference worker exit status disagrees with its result"}
                            write_json(result_path, replay)
                    else:
                        replay = {"schema_version": 1, "call_id": call_id, "status": "failed", "reason": f"reference worker exited with status {process.returncode} without a result"}
                except OSError as error:
                    replay = {"schema_version": 1, "call_id": call_id, "status": "failed", "reason": f"reference worker could not start: {type(error).__name__}"}
                    write_json(result_path, replay)
                outcome["replays"].append(replay)
            if any(replay["status"] != "passed" for replay in outcome["replays"]):
                outcome["status"] = "failed"
                outcome["reason"] = "frozen reference replay failed; Python remains in use"
            elif not entry["captures"]:
                outcome["status"] = "unsupported" if "unsupported" in entry["skipped"] else "reference"
                outcome["reason"] = "no usable calls captured; inspect skipped-capture diagnostics"
        report["functions"].append(outcome)
    write_json(directory / "report.json", report)
    return report
