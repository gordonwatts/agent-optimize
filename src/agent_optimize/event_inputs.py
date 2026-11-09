"""Validated aligned event inputs for multi-array compiled operations.

This is a building block for operations with more than one event-aligned array.
It does not change dispatch or extend the set of accepted computations.
"""

from dataclasses import dataclass

import awkward as ak
import numpy as np

from .operations import UnsupportedGraph
from .serialization import plan_value, UnsupportedValue


@dataclass(frozen=True)
class EventInputContract:
    """Forms and names for numeric arrays sharing an outer event dimension."""

    forms: dict[str, dict]


def _numeric_buffer(layout, name):
    if not isinstance(layout, ak.contents.NumpyArray) or layout.data.ndim != 1:
        raise UnsupportedGraph(f"{name}: expected a flat numeric buffer")
    dtype = layout.data.dtype
    if dtype.kind not in "biuf" or dtype.itemsize > 8 or dtype == np.dtype("float16"):
        raise UnsupportedGraph(f"{name}: unsupported numeric dtype {dtype}")
    result = np.ascontiguousarray(layout.data).view()
    result.setflags(write=False)
    return result


def _buffers(layout, name):
    if isinstance(layout, ak.contents.NumpyArray):
        return {"": _numeric_buffer(layout, name)}
    if isinstance(layout, ak.contents.RecordArray):
        if not layout.fields:
            raise UnsupportedGraph(f"{name}: empty record is unsupported")
        return {field: _numeric_buffer(layout.content(field), f"{name}.{field}")
                for field in layout.fields}
    if isinstance(layout, ak.contents.ListOffsetArray):
        content = layout.content
        if not isinstance(content, ak.contents.RecordArray) or not content.fields:
            raise UnsupportedGraph(f"{name}: expected a list of numeric records")
        offsets = np.ascontiguousarray(layout.offsets.data).view()
        offsets.setflags(write=False)
        fields = {field: _numeric_buffer(content.content(field), f"{name}.{field}")
                  for field in content.fields}
        return {"offsets": offsets, "fields": fields}
    raise UnsupportedGraph(f"{name}: unsupported event layout")


def adapt_event_inputs(contract, arguments):
    """Check all forms and event lengths before exposing read-only buffers."""
    if set(arguments) != set(contract.forms):
        raise UnsupportedGraph("event input names do not match the compiled contract")
    lengths = set()
    buffers = {}
    for name, expected in contract.forms.items():
        array = arguments[name]
        if type(array) is not ak.Array:
            raise UnsupportedGraph(f"{name}: expected a standard Awkward Array")
        try:
            plan_value(array, "array")
        except UnsupportedValue as error:
            raise UnsupportedGraph(f"{name}: {error}") from error
        if array.layout.form.to_dict() != expected:
            raise UnsupportedGraph(f"{name}: input form differs from the compiled contract")
        error = ak.validity_error(array)
        if error:
            raise UnsupportedGraph(f"{name}: invalid input layout: {error[:200]}")
        lengths.add(len(array))
        buffers[name] = _buffers(array.layout, name)
    if len(lengths) != 1:
        raise UnsupportedGraph("event input lengths must match")
    return buffers


def make_event_input_contract(arguments):
    """Capture forms without specializing on event counts or data values."""
    if not arguments:
        raise UnsupportedGraph("at least one event input is required")
    forms = {}
    for name, array in arguments.items():
        if type(array) is not ak.Array:
            raise UnsupportedGraph(f"{name}: expected a standard Awkward Array")
        forms[name] = array.layout.form.to_dict()
    contract = EventInputContract(forms)
    adapt_event_inputs(contract, arguments)
    return contract
