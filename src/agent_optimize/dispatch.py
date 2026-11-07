"""Ordinary compatible-artifact dispatch with same-call reference fallback."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
from pathlib import Path
import re
import time

import numpy as np
import awkward as ak

from .arguments import bind_supported_arguments, UnsupportedArgumentsError
from .artifacts import persist_quarantine, quarantine_path, validate_manifest
from .candidate import adapt_inputs, compile_kernel, contract_from_dict, wrap_output
from .compatibility import environment_signature
from .experiment import freeze_reference, function_identity, location_key, UnsupportedReference
from .operations import UnsupportedGraph
from .purity import validate_reusable_source
from .serialization import UnsupportedValue
from .store import read_json
from .modes import dispatch_mode

_compiled = {}
_evidence_checked = {}
_quarantined = set()
_fast_loaded = {}
_observer = ContextVar("agent_optimize_dispatch_observer", default=None)
_prospective = ContextVar("agent_optimize_prospective_dispatch", default=False)


@contextmanager
def observe_dispatch(stats, *, allow_prospective=False):
    token = _observer.set(stats)
    prospective_token = _prospective.set(allow_prospective)
    try:
        yield
    finally:
        _observer.reset(token)
        _prospective.reset(prospective_token)


def clear_dispatch_cache():
    _compiled.clear()
    _evidence_checked.clear()
    _quarantined.clear()
    _fast_loaded.clear()


def _quarantine(root, artifact_id, phase, error):
    key = (str(root), artifact_id)
    _quarantined.add(key)
    _compiled.pop(key, None)
    _evidence_checked.pop(key, None)
    try:
        persist_quarantine(root, artifact_id, phase, error)
    except Exception:
        pass  # The in-memory quarantine remains effective in a read-only store.


def _kernel(root, artifact_id, source, inputs, stats):
    key = (str(root), artifact_id)
    kernel = _compiled.get(key)
    if kernel is None:
        started = time.perf_counter_ns()
        kernel = compile_kernel(source)
        import numba

        kernel.compile(tuple(numba.typeof(value) for value in inputs))
        if stats is not None:
            stats["compilation_ns"] = time.perf_counter_ns() - started
        _compiled[key] = kernel
    return kernel


def dispatch_call(function, args, kwargs, root: Path, *, stats=None, mode=None):
    """Return (used, result); the caller executes the reference outside recovery.

    No input recording or candidate generation happens here. Exceptions from
    user references and BaseExceptions such as KeyboardInterrupt are untouched.
    """
    if stats is None:
        stats = _observer.get()
    if stats is not None:
        stats["used"] = False
    mode = dispatch_mode(mode)
    if mode == "fast":
        return _fast_dispatch(function, args, kwargs, root, stats)
    try:
        root = root.resolve()
        function_id = location_key(function_identity(function))
        index_path = root / "accepted" / function_id / "index.json"
        exists = index_path.is_file()
    except Exception:
        return False, None
    if not exists:
        return False, None
    try:
        index = read_json(index_path)
        if not isinstance(index.get("entries"), dict):
            raise ValueError("artifact index entries are invalid")
    except Exception as error:
        # A content-derived identity lets a repaired index be tried again.
        try:
            identity = "index-" + hashlib.sha256(index_path.read_bytes()).hexdigest()[:32]
        except OSError:
            identity = "index-" + function_id
        if (str(root), identity) not in _quarantined:
            _quarantine(root, identity, "index", error)
        return False, None
    try:
        bound = bind_supported_arguments(function, args, kwargs)
        frozen = freeze_reference(function)
        environment = environment_signature()
    except Exception:
        return False, None
    for artifact_id in index["entries"].values():
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
            continue
        key = (str(root), artifact_id)
        if key in _quarantined:
            continue
        phase = "load"
        try:
            if quarantine_path(root, artifact_id).exists():
                continue
            directory = root / "artifacts" / artifact_id
            manifest = read_json(directory / "manifest.json")
            validate_manifest(manifest, artifact_id, allow_prospective=_prospective.get())
            if manifest.get("dispatch_mode", "careful") != "careful":
                continue
            if (
                manifest.get("function_id") != function_id
                or manifest["reference_fingerprint"] != frozen.fingerprint
                or manifest["dependency_fingerprint"] != frozen.metadata["dependency_fingerprint"]
                or manifest["environment"] != environment
            ):
                continue
            evidence_path = directory / "evidence.json"
            stat = evidence_path.stat()
            evidence_key = (stat.st_mtime_ns, stat.st_size, manifest["evidence_sha256"])
            if _evidence_checked.get(key) != evidence_key:
                if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != manifest["evidence_sha256"]:
                    raise ValueError("accepted evaluation evidence is corrupt")
                _evidence_checked[key] = evidence_key
            contract = contract_from_dict(manifest["contract"])
            try:
                inputs = adapt_inputs(contract, bound.arguments)
            except (UnsupportedGraph, UnsupportedValue, ValueError, TypeError):
                continue  # Another call schema may still have a compatible entry.
            source = (directory / "kernel.py").read_text(encoding="utf-8")
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != manifest["source_sha256"]:
                raise ValueError("accepted source is corrupt")
            validate_reusable_source(source)
            phase = "compilation"
            kernel = _kernel(root, artifact_id, source, inputs, stats)
            phase = "execution"
            raw = kernel(*inputs)
            if type(raw) is np.ndarray and any(np.shares_memory(raw, value) for value in [inputs[0], *inputs[1]]):
                raise ValueError("candidate output aliases an input")
            result = wrap_output(contract, raw, len(inputs[0]) - 1)
            if stats is not None:
                stats.update(used=True, artifact_id=artifact_id)
            return True, result
        except Exception as error:
            _quarantine(root, artifact_id, phase, error)
            # Do not try an older candidate after a failure on this call.
            return False, None
    return False, None


def _fast_dispatch(function, args, kwargs, root, stats):
    """Load once, then execute generated adapter/kernel without compatibility checks.

    Code, dependencies, environment and array structure are assumed stable until
    the process exits or clear_dispatch_cache() is explicitly called.
    """
    cache_key = (function, str(root), _prospective.get())
    loaded = _fast_loaded.get(cache_key)
    if loaded is None:
        try:
            root = root.resolve()
            function_id = location_key(function_identity(function))
            index = read_json(root / "accepted" / function_id / "index.json")
            if not isinstance(index.get("entries"), dict):
                return False, None
            signature = inspect.signature(function)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except Exception:
            return False, None
        for artifact_id in reversed(list(index["entries"].values())):
            if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
                continue
            key = (str(root), artifact_id)
            if key in _quarantined or quarantine_path(root, artifact_id).exists():
                continue
            try:
                directory = root / "artifacts" / artifact_id
                manifest = read_json(directory / "manifest.json")
                validate_manifest(manifest, artifact_id, allow_prospective=_prospective.get())
                if manifest.get("dispatch_mode", "careful") != "fast" or manifest.get("function_id") != function_id:
                    continue
                # Select a recorded contract once. Later calls assume its schema.
                contract = contract_from_dict(manifest["contract"])
                try:
                    inputs = adapt_inputs(contract, bound.arguments)
                except (UnsupportedGraph, UnsupportedValue, ValueError, TypeError):
                    continue
                source = (directory / "kernel.py").read_text(encoding="utf-8")
                adapter = (directory / "adapter.py").read_text(encoding="utf-8")
                if hashlib.sha256(source.encode()).hexdigest() != manifest["source_sha256"] or hashlib.sha256(adapter.encode()).hexdigest() != manifest["adapter_sha256"]:
                    raise ValueError("saved generated source is corrupt")
                if hashlib.sha256((directory / "evidence.json").read_bytes()).hexdigest() != manifest["evidence_sha256"]:
                    raise ValueError("saved evaluation evidence is corrupt")
                validate_reusable_source(source)
                # Adapter code is host-generated, not supplied by the model.
                from .fast_adapter import adapter_source
                if adapter != adapter_source(manifest["contract"]):
                    raise ValueError("saved generated adapter is invalid")
                namespace = {}
                exec(compile(adapter, str(directory / "adapter.py"), "exec"), namespace)
                # Compile for the unchecked adapter's buffers, rather than the
                # read-only views used for first-load contract selection.
                inputs = namespace["prepare"](bound.arguments)
                kernel = _kernel(root, artifact_id, source, inputs, stats)
                loaded = (root, artifact_id, signature, namespace["replacement"], kernel)
                _fast_loaded[cache_key] = loaded
                break
            except Exception as error:
                _quarantine(root, artifact_id, "load", error)
                return False, None
        if loaded is None:
            return False, None
    root, artifact_id, signature, replacement, kernel = loaded
    if (str(root), artifact_id) in _quarantined:
        return False, None
    try:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        result = replacement(bound.arguments, kernel)
    except Exception as error:
        _quarantine(root, artifact_id, "fast execution", error)
        _fast_loaded.pop(cache_key, None)
        return False, None
    if stats is not None:
        stats.update(used=True, artifact_id=artifact_id, dispatch_mode="fast")
    return True, result
