"""Portable entry point for replaying a single frozen reference call."""

import argparse
import hashlib
from pathlib import Path
import re

from .experiment import validate_frozen_source
from .serialization import load_value, matches_recording
from .store import read_json, write_json


def replay_call(directory: Path, call_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", call_id):
        raise ValueError("invalid capture identifier")
    call_directory = directory / "calls" / call_id
    capture = read_json(call_directory / "capture.json")
    fingerprint = capture["reference_fingerprint"]
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid reference fingerprint")
    reference_directory = directory / "references" / fingerprint
    metadata = read_json(reference_directory / "reference.json")
    source = (reference_directory / "reference.py").read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != metadata["source_sha256"]:
        raise ValueError("frozen reference source fingerprint mismatch")
    tree, name = validate_frozen_source(source)
    namespace = {"__name__": "agent_optimize_frozen_reference"}
    exec(compile(tree, str(reference_directory / "reference.py"), "exec"), namespace)
    arguments = {name: load_value(value, call_directory) for name, value in capture["arguments"].items()}
    positional = [arguments.pop(name) for name in capture["positional_names"]]
    result = namespace[name](*positional, **arguments)
    matched = matches_recording(result, capture["reference_result"], call_directory)
    return {
        "schema_version": 1, "call_id": call_id,
        "status": "passed" if matched else "failed",
        "reason": "frozen reference matches recorded output" if matched else "frozen reference output differs from recorded output",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("call_id")
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        result = replay_call(args.session.resolve(), args.call_id)
    except Exception as error:
        result = {"schema_version": 1, "call_id": args.call_id, "status": "failed", "reason": f"{type(error).__name__}: {str(error)[:400]}"}
    write_json(args.result, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
