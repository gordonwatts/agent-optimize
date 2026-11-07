import awkward as ak
import numpy as np
import pytest

from agent_optimize import optimize
from agent_optimize.arguments import UnsupportedArgumentsError, bind_supported_arguments


def test_binding_positional_keyword_defaults_and_parameter_kinds():
    @optimize
    def reference(events, /, threshold=30, *, enabled=True):
        return events

    events = ak.Array([[1, 2], []])
    positional = bind_supported_arguments(reference, (events, 40), {"enabled": False})
    keyword = bind_supported_arguments(reference, (events,), {"threshold": 40, "enabled": False})
    assert positional.arguments.keys() == keyword.arguments.keys()
    assert positional.arguments["events"] is events
    assert keyword.arguments["events"] is events
    assert positional.arguments["threshold"] == keyword.arguments["threshold"] == 40
    assert positional.arguments["enabled"] is keyword.arguments["enabled"] is False
    assert bind_supported_arguments(reference, (events,), {}).arguments["threshold"] == 30


@pytest.mark.parametrize(
    "value", [True, 2, 3.0, float("nan"), np.bool_(False), np.int32(2), np.uint64(3), np.float32(4)]
)
def test_supported_scalar_types(value):
    def reference(value):
        return value

    assert bind_supported_arguments(reference, (value,), {}).arguments["value"] is value


@pytest.mark.parametrize("value", [[1], {"x": 1}, "text", None, 1j, np.array([1]), object()])
def test_unsupported_argument_has_named_diagnostic(value):
    def reference(events, *, threshold):
        return threshold

    with pytest.raises(UnsupportedArgumentsError, match="threshold.*unsupported") as caught:
        bind_supported_arguments(reference, (ak.Array([1]),), {"threshold": value})
    assert caught.value.parameter == "threshold"


@pytest.mark.parametrize("function", [lambda *items: None, lambda **options: None])
def test_variadic_signatures_have_diagnostic(function):
    with pytest.raises(UnsupportedArgumentsError, match="variadic parameters"):
        bind_supported_arguments(function, (), {})


def test_invalid_python_binding_is_not_an_unsupported_diagnostic():
    def reference(events, /, *, threshold):
        return events

    with pytest.raises(TypeError):
        bind_supported_arguments(reference, (), {"events": ak.Array([1]), "threshold": 30})
    with pytest.raises(TypeError):
        bind_supported_arguments(reference, (ak.Array([1]),), {})


def test_custom_scalar_subclasses_remain_outside_optimization_contract():
    class CustomFloat(float):
        pass

    def reference(value):
        return value

    with pytest.raises(UnsupportedArgumentsError, match="CustomFloat"):
        bind_supported_arguments(reference, (CustomFloat(1),), {})
