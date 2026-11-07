"""Small JSON/file store for bounded local recording sessions."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

from .records import SCHEMA_VERSION

SESSION_ENV = "AGENT_OPTIMIZE_SESSION"
DEFAULT_CAPTURE_BYTES = 64 * 1024 * 1024
DEFAULT_CALL_LIMIT = 5


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(json_bytes(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported metadata schema version in {path.name}")
    return value


def create_session(root: Path, *, max_bytes=DEFAULT_CAPTURE_BYTES, max_calls=DEFAULT_CALL_LIMIT, provider=None) -> Path:
    if max_bytes <= 0 or max_calls <= 0:
        raise ValueError("recording byte and call limits must be positive")
    directory = root.resolve() / "sessions" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    (directory / "calls").mkdir()
    (directory / "references").mkdir()
    write_json(directory / "session.json", {
        "schema_version": SCHEMA_VERSION,
        "session_id": directory.name,
        "limits": {"max_capture_bytes": max_bytes, "max_calls_per_function": max_calls},
        "captured_bytes": 0,
        "functions": {},
        **({"provider": provider} if provider is not None else {}),
    })
    return directory


@contextmanager
def session_lock(directory: Path):
    """Serialize child processes; a crashed owner leaves a diagnostic lock.

    The parent reads completed checkpoints without acquiring this lock. Waiting
    children fall back to reference-only calls after five seconds.
    """
    path = directory / ".recording.lock"
    deadline = time.monotonic() + 5
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("recording lock unavailable; reference Python retained")
            time.sleep(0.01)
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)
