import functools
import builtins
import importlib.util
import os
from pathlib import Path
import sys
import types

import awkward as ak
import numpy as np
import pytest

from agent_optimize.evaluator import evaluate_session
from agent_optimize.experiment import _binding_fingerprint, code_fingerprint, freeze_reference, UnsupportedReference
from agent_optimize.recorder import Recorder
from agent_optimize.replay import replay_session
from agent_optimize.store import create_session, read_json
from test_dispatch import accepted, accepted_base, fast_kernel, invoke

SOURCE = """from pathlib import Path
import awkward as ak
from agent_optimize import optimize
marker = Path(__file__).with_suffix('.count')
marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')

@optimize
def ht(events):
    jets = events.Jet
    selected = jets[abs(jets.eta) < CUT_LITERAL]
    return ak.sum(selected.pt, axis=1)
"""


def load(path):
    spec = importlib.util.spec_from_file_location("integrity_case", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def events(eta):
    return ak.Array([{"Jet": [{"pt": 40, "eta": eta}]}])


@pytest.mark.parametrize("wrapped", [False, True])
def test_same_code_closures_with_different_state_are_not_supported_import_bindings(monkeypatch, wrapped):
    original = ak.sum

    def factory(shift):
        def replacement(*args, **kwargs):
            return original(*args, **kwargs) + shift
        return functools.wraps(original)(replacement) if wrapped else replacement

    zero, one = factory(0), factory(1)
    assert code_fingerprint(zero.__code__) == code_fingerprint(one.__code__)
    for replacement in [zero, one]:
        monkeypatch.setattr(ak, "sum", replacement)
        with pytest.raises(UnsupportedReference, match="replaced import callable|installed library source"):
            _binding_fingerprint(ak.sum)


def test_stock_callable_defaults_and_closure_values_change_dependency_fingerprint(monkeypatch):
    original = _binding_fingerprint(ak.sum)
    cells = dict(zip(ak.sum.__code__.co_freevars, ak.sum.__closure__, strict=True))
    implementation = cells["func"].cell_contents
    monkeypatch.setattr(implementation, "__defaults__", (0,))
    assert _binding_fingerprint(ak.sum) != original
    monkeypatch.setattr(implementation, "__defaults__", (None,))
    monkeypatch.setattr(implementation, "__kwdefaults__", {**implementation.__kwdefaults__, "mask_identity": True})
    assert _binding_fingerprint(ak.sum) != original


def test_mutable_defaults_and_replaced_direct_globals_are_rejected(monkeypatch):
    cells = dict(zip(ak.sum.__code__.co_freevars, ak.sum.__closure__, strict=True))
    implementation = cells["func"].cell_contents
    with monkeypatch.context() as patch:
        patch.setattr(implementation, "__defaults__", ([],))
        with pytest.raises(UnsupportedReference, match="mutable/import"):
            _binding_fingerprint(ak.sum)

    original = implementation.__globals__["_impl"]

    def factory():
        def replacement(*args, **kwargs):
            return original(*args, **kwargs)
        return replacement

    monkeypatch.setitem(implementation.__globals__, "_impl", factory())
    with pytest.raises(UnsupportedReference, match="replaced import callable"):
        _binding_fingerprint(ak.sum)


def test_replaced_same_code_closures_never_reuse_an_existing_saved_kernel(accepted, fast_kernel, monkeypatch):
    assert invoke(accepted)["used"]
    original = ak.sum

    def factory(shift):
        def replacement(*args, **kwargs):
            return original(*args, **kwargs) + shift
        return replacement

    zero, one = factory(0), factory(1)
    assert code_fingerprint(zero.__code__) == code_fingerprint(one.__code__)
    for replacement in [zero, one]:
        monkeypatch.setattr(ak, "sum", replacement)
        assert not invoke(accepted)["used"]
    assert len(fast_kernel[1]) == 1


def test_zero_shift_replacement_cannot_be_accepted_then_reused_after_state_change(tmp_path, monkeypatch):
    path = tmp_path / "analysis.py"
    path.write_text(SOURCE.replace("CUT_LITERAL", "2.4"), encoding="utf-8")
    module = load(path)
    original = ak.sum

    def factory(shift):
        def replacement(*args, **kwargs):
            return original(*args, **kwargs) + shift
        return replacement

    monkeypatch.setattr(ak, "sum", factory(0))
    root = tmp_path / "store"
    directory = create_session(root)
    assert ak.to_list(Recorder(directory).invoke(module.ht.__wrapped__, (events(0.0),), {})) == [40]
    report = evaluate_session(directory, replay_session(directory, 0))
    assert report["functions"][0]["captured_calls"] == 0
    assert "unsupported" in report["functions"][0]["skipped"]
    assert not (root / "accepted").exists()
    monkeypatch.setattr(ak, "sum", factory(1))
    assert ak.to_list(module.ht(events(0.0))) == [41]


@pytest.mark.parametrize("change", ["live-code", "file-only", "file-with-preserved-stat"])
def test_source_live_mismatch_with_sample_agreement_cannot_create_an_artifact(tmp_path, change):
    path = tmp_path / "analysis.py"
    path.write_text(SOURCE.replace("CUT_LITERAL", "2.4" if change == "live-code" else "1.0"), encoding="utf-8")
    module = load(path)
    reference = module.ht.__wrapped__
    freeze_reference(reference)  # Seed caches before a same-size/stat source edit.
    if change == "live-code":
        reference.__code__ = reference.__code__.replace(co_consts=tuple(1.0 if value == 2.4 else value for value in reference.__code__.co_consts))
    else:
        stat = path.stat()
        path.write_text(SOURCE.replace("CUT_LITERAL", "2.4"), encoding="utf-8")
        if change == "file-with-preserved-stat":
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert path.stat().st_size == stat.st_size
    # Recorded eta=0 agrees with the stale 2.4 implementation; eta=1.5 diverges.
    assert ak.to_list(reference(events(0.0))) == [40]
    assert ak.to_list(reference(events(1.5))) == [0]
    with pytest.raises(UnsupportedReference, match="source does not match its live executable"):
        freeze_reference(reference)
    root = tmp_path / "store"
    directory = create_session(root)
    assert ak.to_list(Recorder(directory).invoke(reference, (events(0.0),), {})) == [40]
    report = evaluate_session(directory, replay_session(directory, 0))
    assert report["functions"][0]["captured_calls"] == 0
    assert "source does not match" in report["functions"][0]["skipped"]["unsupported"]["diagnostic"]
    assert not (root / "accepted").exists()
    assert path.with_suffix(".count").read_text() == "1"


def test_replaced_builtin_abs_cannot_reuse_a_saved_kernel(accepted, fast_kernel, monkeypatch):
    assert invoke(accepted)["used"]
    values = accepted["module"].make_events()
    values = ak.with_field(values, ak.ones_like(values.Jet.eta) * 3.0, ("Jet", "eta"))
    assert ak.to_list(accepted["module"].ht.__wrapped__(values)) == [0, 0, 0]
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "abs", lambda value: value * 0)
        assert not invoke(accepted, values)["used"]
        assert ak.to_list(accepted["module"].ht(values)) == [0, 0, 153]
    assert len(fast_kernel[1]) == 1
    assert invoke(accepted)["used"]


def test_function_specific_replaced_builtin_is_also_rejected(tmp_path):
    path = tmp_path / "analysis.py"
    path.write_text(SOURCE.replace("CUT_LITERAL", "2.4"), encoding="utf-8")
    module = load(path)
    reference = module.ht.__wrapped__
    globals_ = {**reference.__globals__, "__builtins__": {**builtins.__dict__, "abs": lambda value: value * 0}}
    changed = types.FunctionType(reference.__code__, globals_)
    assert ak.to_list(changed(events(3.0))) == [40]
    with pytest.raises(UnsupportedReference, match="builtin abs was replaced"):
        freeze_reference(changed)


def test_replaced_indirect_reduce_cannot_reuse_a_saved_kernel(accepted, fast_kernel, monkeypatch):
    assert invoke(accepted)["used"]
    original = ak._do.reduce

    def replacement(*args, **kwargs):
        value = original(*args, **kwargs)
        return ak.contents.NumpyArray(np.zeros_like(value.data))

    monkeypatch.setattr(ak._do, "reduce", replacement)
    assert not invoke(accepted)["used"]
    assert ak.to_list(accepted["module"].ht(accepted["module"].make_events())) == [0, 0, 0]
    assert len(fast_kernel[1]) == 1


@pytest.mark.parametrize("replacement", ["builtin", "indirect-reduce"])
def test_mutated_builtin_or_indirect_module_dependency_is_never_accepted(tmp_path, monkeypatch, replacement):
    path = tmp_path / "analysis.py"
    path.write_text(SOURCE.replace("CUT_LITERAL", "2.4"), encoding="utf-8")
    module = load(path)
    if replacement == "builtin":
        monkeypatch.setattr(builtins, "abs", lambda value: value * 0)
        expected = [40]
    else:
        original = ak._do.reduce

        def replace_reduce(*args, **kwargs):
            value = original(*args, **kwargs)
            return ak.contents.NumpyArray(np.zeros_like(value.data))

        monkeypatch.setattr(ak._do, "reduce", replace_reduce)
        expected = [0]
    directory = create_session(tmp_path / "store")
    assert ak.to_list(Recorder(directory).invoke(module.ht.__wrapped__, (events(3.0),), {})) == expected
    report = evaluate_session(directory, replay_session(directory, 0))
    assert report["functions"][0]["captured_calls"] == 0
    assert "unsupported" in report["functions"][0]["skipped"]
    assert not (tmp_path / "store" / "accepted").exists()


def test_changed_sum_reducer_apply_falls_back_without_using_a_stale_saved_kernel(accepted, fast_kernel, monkeypatch):
    assert invoke(accepted)["used"]
    original = ak._reducers.Sum.apply

    def replacement(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        return ak.contents.NumpyArray(np.zeros_like(value.data), backend=value.backend)

    monkeypatch.setattr(ak._reducers.Sum, "apply", replacement)
    assert not invoke(accepted)["used"]
    assert ak.to_list(accepted["module"].ht(accepted["module"].make_events())) == [0, 0, 0]
    assert len(fast_kernel[1]) == 1
    monkeypatch.setattr(ak._reducers.Sum, "apply", original)
    assert invoke(accepted)["used"]


def test_changed_sum_reducer_method_binding_preserves_reference_exception(accepted, fast_kernel, monkeypatch):
    assert invoke(accepted)["used"]
    monkeypatch.setattr(ak._reducers.Sum, "apply", classmethod(ak._reducers.Sum.apply))
    with pytest.raises(AttributeError):
        accepted["module"].ht(accepted["module"].make_events())
    assert len(fast_kernel[1]) == 1


def test_reducer_policy_is_fingerprinted_and_unknown_class_state_is_rejected(monkeypatch):
    original = _binding_fingerprint(ak.sum)
    with monkeypatch.context() as patch:
        patch.setattr(ak._reducers.KernelReducer, "_use32", not ak._reducers.KernelReducer._use32)
        assert _binding_fingerprint(ak.sum) != original
    monkeypatch.setattr(ak._reducers.Sum, "_dtype", property(lambda self: np.float64), raising=False)
    with pytest.raises(UnsupportedReference, match="reducer class structure"):
        _binding_fingerprint(ak.sum)


def test_inherited_reducer_instance_field_descriptor_is_unsupported(monkeypatch):
    monkeypatch.setattr(ak._reducers.KernelReducer, "_dtype", property(lambda self: np.float64), raising=False)
    with pytest.raises(UnsupportedReference, match="reducer class structure"):
        _binding_fingerprint(ak.sum)


def test_reducer_policy_validation_does_not_execute_an_unknown_descriptor(monkeypatch):
    calls = []

    class Descriptor:
        def __get__(self, instance, owner):
            calls.append(1)
            return False

    monkeypatch.setattr(ak._reducers.Sum, "needs_position", Descriptor())
    with pytest.raises(UnsupportedReference, match="policy descriptors"):
        _binding_fingerprint(ak.sum)
    assert calls == []
