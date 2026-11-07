import inspect
import os

import awkward as ak
import pytest

from agent_optimize import optimize


def test_metadata_signature_and_reference_are_preserved():
    def reference(events: ak.Array, threshold: float = 30) -> ak.Array:
        """Sum selected values."""
        return ak.sum(events[events > threshold], axis=1)

    decorated = optimize(reference)
    assert decorated.__name__ == reference.__name__
    assert decorated.__qualname__ == reference.__qualname__
    assert decorated.__doc__ == reference.__doc__
    assert decorated.__module__ == reference.__module__
    assert decorated.__annotations__ == reference.__annotations__
    assert decorated.__wrapped__ is reference
    assert inspect.signature(decorated) == inspect.signature(reference)
    events = ak.Array([[10, 40, 60], [], [31]])
    assert ak.to_list(decorated(events)) == ak.to_list(reference(events))
    assert ak.to_list(decorated(events=events, threshold=50)) == [60, 0, 0]


def test_normal_call_preserves_arbitrary_values_identity_and_side_effects():
    calls = []
    value = object()

    @optimize
    def reference(first, /, *items, option=None, **options):
        calls.append((first, items, option, options))
        return first

    assert reference(value, "extra", option="text", other=[1]) is value
    assert calls == [(value, ("extra",), "text", {"other": [1]})]


def test_reference_exception_is_unchanged():
    failure = RuntimeError("reference failure")

    @optimize
    def reference():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        reference()
    assert caught.value is failure


def test_invalid_normal_call_keeps_python_type_error():
    @optimize
    def reference(first, /, *, second):
        return first + second

    assert reference(1, second=2) == 3
    with pytest.raises(TypeError):
        reference(first=1, second=2)


@pytest.mark.parametrize("override", [False, True])
def test_unavailable_working_directory_does_not_disable_reference_calls(tmp_path, monkeypatch, override):
    @optimize
    def reference(value):
        return value

    with monkeypatch.context() as patch:
        if override:
            patch.setenv("AGENT_OPTIMIZE_ARTIFACT_ROOT", str(tmp_path / "missing artifacts"))
        else:
            patch.delenv("AGENT_OPTIMIZE_ARTIFACT_ROOT", raising=False)
        patch.setattr(os, "getcwd", lambda: (_ for _ in ()).throw(FileNotFoundError("working directory removed")))
        assert reference("original") == "original"
