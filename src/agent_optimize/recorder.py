"""Bounded snapshots around exactly one execution of each reference call."""

from dataclasses import asdict, dataclass
import inspect
from pathlib import Path
import shutil
import sys
import threading
import uuid

from .arguments import bind_supported_arguments, UnsupportedArgumentsError
from .experiment import freeze_reference, function_identity, location_key, UnsupportedReference
from .records import CapturedCall
from .serialization import plan_value, unchanged, UnsupportedValue, write_value
from .store import json_bytes, read_json, session_lock, write_json


@dataclass
class PreparedCall:
    stage: Path
    destination: Path
    record: dict
    plans: dict
    hashes: dict
    frozen: object
    reference_directory: Path
    reference_bytes: int
    remaining: int


class Recorder:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self._lock = threading.RLock()
        self._warned = False

    def _cleanup(self, stage):
        if stage is not None and stage.exists():
            if stage.parent.resolve() != (self.directory / "calls").resolve():
                raise ValueError("partial capture is outside this session's calls directory")
            shutil.rmtree(stage)

    def _skip(self, entry, reason, diagnostic):
        skip = entry["skipped"].setdefault(reason, {"count": 0, "diagnostic": diagnostic[:500]})
        skip["count"] += 1

    def _save(self, state):
        write_json(self.directory / "session.json", state)

    def _warn(self, error):
        if not self._warned:
            print(f"agent-optimize: recording unavailable ({type(error).__name__}); using reference Python", file=sys.stderr)
            self._warned = True

    def _prepare(self, function, args, kwargs, state, entry, key):
        bound = bind_supported_arguments(function, args, kwargs)
        frozen = freeze_reference(function)
        plans = {name: plan_value(value, f"arg{index}") for index, (name, value) in enumerate(bound.arguments.items())}
        call_id = uuid.uuid4().hex
        reference_directory = self.directory / "references" / frozen.fingerprint
        reference_bytes = 0
        if not reference_directory.exists():
            reference_bytes = len(frozen.source.encode("utf-8")) + len(json_bytes(frozen.metadata))
        record = asdict(CapturedCall(
            function_id=key, call_id=call_id, reference_fingerprint=frozen.fingerprint,
            arguments={name: plan.metadata for name, plan in plans.items()},
            positional_names=[name for name, param in bound.signature.parameters.items() if param.kind == inspect.Parameter.POSITIONAL_ONLY],
            reference_result=None,
        ))
        remaining = state["limits"]["max_capture_bytes"] - state["captured_bytes"]
        input_bytes = sum(plan.nbytes for plan in plans.values()) + len(json_bytes(record)) + reference_bytes
        if input_bytes > remaining:
            self._skip(entry, "byte_limit", "complete input buffers and metadata exceed the remaining session byte budget")
            return None
        stage = self.directory / "calls" / f".partial-{call_id}"
        stage.mkdir()
        try:
            hashes = {name: write_value(plan, stage) for name, plan in plans.items()}
        except BaseException:
            self._cleanup(stage)
            raise
        return PreparedCall(stage, self.directory / "calls" / call_id, record, plans, hashes, frozen, reference_directory, reference_bytes, remaining)

    def _finish(self, prepared, result, state, entry):
        for index, (name, plan) in enumerate(prepared.plans.items()):
            if not unchanged(plan, prepared.hashes[name], f"arg{index}"):
                self._skip(entry, "input_mutation", f"argument {name} changed during reference execution; capture discarded")
                return
        output = plan_value(result, "result", output=True)
        prepared.record["reference_result"] = output.metadata
        size = sum(plan.nbytes for plan in prepared.plans.values()) + output.nbytes + len(json_bytes(prepared.record)) + prepared.reference_bytes
        if size > prepared.remaining:
            self._skip(entry, "output_byte_limit", "complete output and metadata exceed the remaining session byte budget; partial capture discarded")
            return
        write_value(output, prepared.stage)
        write_json(prepared.stage / "capture.json", prepared.record)
        if not prepared.reference_directory.exists():
            prepared.reference_directory.mkdir()
            try:
                (prepared.reference_directory / "reference.py").write_bytes(prepared.frozen.source.encode("utf-8"))
                write_json(prepared.reference_directory / "reference.json", prepared.frozen.metadata)
            except BaseException:
                if prepared.reference_directory.parent.resolve() != (self.directory / "references").resolve():
                    raise ValueError("reference directory escaped this session")
                shutil.rmtree(prepared.reference_directory)
                raise
        prepared.stage.replace(prepared.destination)
        state["captured_bytes"] += size
        entry["captures"].append(prepared.record["call_id"])
        try:
            self._save(state)
        except BaseException:
            state["captured_bytes"] -= size
            entry["captures"].pop()
            self._cleanup(prepared.destination)
            if prepared.reference_bytes:
                if prepared.reference_directory.parent.resolve() != (self.directory / "references").resolve():
                    raise ValueError("reference directory escaped this session")
                shutil.rmtree(prepared.reference_directory)
            raise

    def invoke(self, function, args, kwargs):
        """Recorder errors fall back; user exceptions and interrupts propagate."""
        execution = _Execution(function, args, kwargs)
        with self._lock:
            try:
                with session_lock(self.directory):
                    result = self._invoke_locked(function, args, kwargs, execution)
                if execution.called:
                    return result
            except BaseException as error:
                if execution.exception is not None:
                    raise execution.exception.with_traceback(execution.traceback)
                if not isinstance(error, Exception):
                    raise
                self._warn(error)
        if execution.called:
            return execution.result
        return execution.call()

    def _invoke_locked(self, function, args, kwargs, execution):
        state = read_json(self.directory / "session.json")
        identity = function_identity(function)
        key = location_key(identity)
        entry = state["functions"].setdefault(key, {
            "identity": identity, "attempted_calls": 0, "captures": [], "skipped": {},
        })
        prepared = None
        if entry["attempted_calls"] >= state["limits"]["max_calls_per_function"]:
            self._skip(entry, "call_limit", "per-function capture-attempt limit reached before buffer copying")
        else:
            entry["attempted_calls"] += 1
            try:
                prepared = self._prepare(function, args, kwargs, state, entry, key)
            except (UnsupportedArgumentsError, UnsupportedReference, UnsupportedValue) as error:
                self._skip(entry, "unsupported", str(error))
            except Exception as error:
                self._skip(entry, "recording_failure", f"input recording failed: {type(error).__name__}")
        # Persist budgets/diagnostics before the user code runs, even if it exits.
        try:
            self._save(state)
        except Exception:
            if prepared:
                self._cleanup(prepared.stage)
            raise
        if prepared is None:
            # Unsupported/limited outer calls may call other decorated functions.
            # Execute them after releasing the recording lock so those inner
            # functions can independently capture their supported calls.
            return None
        try:
            result = execution.call()
        except BaseException as error:
            if prepared:
                self._cleanup(prepared.stage)
                self._skip(entry, "reference_exception", f"reference raised {type(error).__name__}; capture discarded")
                try:
                    self._save(state)
                except Exception as recording_error:
                    self._warn(recording_error)
            raise
        if prepared:
            try:
                self._finish(prepared, result, state, entry)
            except UnsupportedValue as error:
                self._skip(entry, "unsupported", str(error))
            except Exception as error:
                self._skip(entry, "recording_failure", f"output recording failed: {type(error).__name__}")
            finally:
                self._cleanup(prepared.stage)
            try:
                self._save(state)
            except Exception as error:
                self._warn(error)
        return result


class _Execution:
    """Remember whether user code ran, even if later store cleanup fails."""

    def __init__(self, function, args, kwargs):
        self.function, self.args, self.kwargs = function, args, kwargs
        self.called = False
        self.result = None
        self.exception = None
        self.traceback = None

    def call(self):
        self.called = True
        try:
            self.result = self.function(*self.args, **self.kwargs)
        except BaseException as error:
            self.exception, self.traceback = error, error.__traceback__
            raise
        return self.result


_recorders = {}


def record_call(directory: str, function, args, kwargs):
    path = Path(directory)
    if not path.is_absolute():
        return function(*args, **kwargs)
    recorder = _recorders.setdefault(str(path), Recorder(path))
    return recorder.invoke(function, args, kwargs)
