"""Host-owned contracts, read-only adapters, and the first fused Numba loop."""

import ast
from dataclasses import dataclass

import awkward as ak
import numpy as np

from .operations import FilterSumGraph, predicate_is_array, UnsupportedGraph
from .serialization import plan_value


@dataclass(frozen=True)
class InputContract:
    array_argument: str
    schema: dict
    collection_path: tuple[str, ...]
    fields: tuple[str, ...]
    scalar_names: tuple[str, ...]
    scalar_types: dict
    output_kind: str
    output_dtype: str


def contract_from_dict(value):
    return InputContract(
        value["array_argument"], value["schema"], tuple(value["collection_path"]),
        tuple(value["fields"]), tuple(value["scalar_names"]), value["scalar_types"],
        value["output_kind"], value["output_dtype"],
    )


def scalar_type(value):
    metadata = plan_value(value, "scalar").metadata
    if metadata["kind"] != "scalar":
        raise UnsupportedGraph("the first candidate accepts one Awkward array plus numeric/Boolean scalars")
    return {key: metadata[key] for key in ("type", "dtype") if key in metadata}


def collection_layout(array, path):
    value = array
    for field in path:
        if field not in value.fields:
            raise UnsupportedGraph(f"missing collection field: {field}")
        value = value[field]
    if not isinstance(value.layout, ak.contents.ListOffsetArray):
        raise UnsupportedGraph("the compiled collection must use one variable-length ListOffsetArray")
    if not isinstance(value.layout.content, ak.contents.RecordArray):
        raise UnsupportedGraph("the compiled collection must contain numeric records")
    return value.layout


def _predicate_type(node, field_types, scalar_types):
    match node["op"]:
        case "field":
            return "bool" if np.dtype(field_types[node["name"]]).kind == "b" else "number"
        case "scalar":
            descriptor = scalar_types[node["name"]]
            return "bool" if descriptor["type"] == "bool" or np.dtype(descriptor.get("dtype", "int64")).kind == "b" else "number"
        case "literal":
            return "bool" if type(node["value"]) is bool else "number"
        case "abs":
            if _predicate_type(node["value"], field_types, scalar_types) != "number":
                raise UnsupportedGraph("abs requires a numeric selection operand")
            return "number"
        case "compare":
            _predicate_type(node["left"], field_types, scalar_types)
            _predicate_type(node["right"], field_types, scalar_types)
            return "bool"
        case "and" | "or":
            if any(_predicate_type(node[key], field_types, scalar_types) != "bool" for key in ("left", "right")):
                raise UnsupportedGraph("Boolean combinations require comparisons or Boolean fields/scalars")
            return "bool"
        case "not":
            if not predicate_is_array(node["value"]):
                raise UnsupportedGraph("scalar bitwise inversion (~) is unsupported; invert a Boolean array mask instead")
            if _predicate_type(node["value"], field_types, scalar_types) != "bool":
                raise UnsupportedGraph("Boolean inversion requires a Boolean operand")
            return "bool"
    raise UnsupportedGraph("unknown selection operation")


def make_contract(graph: FilterSumGraph, arguments: dict, result_metadata: dict) -> InputContract:
    array = arguments.get(graph.array_argument)
    if type(array) is not ak.Array:
        raise UnsupportedGraph("the graph's array argument must be a standard Awkward Array")
    plan_value(array, "array")  # reject unsupported layouts, backends, and behaviors
    error = ak.validity_error(array)
    if error:
        raise UnsupportedGraph(f"invalid Awkward input layout: {error[:200]}")
    layout = collection_layout(array, graph.collection_path)
    if len(layout) != len(array):
        raise UnsupportedGraph("the selected collection must have one list per input event")
    field_types = {}
    for name in graph.fields:
        if name not in layout.content.fields:
            raise UnsupportedGraph(f"missing numeric selection/sum field: {name}")
        content = layout.content.content(name)
        if not isinstance(content, ak.contents.NumpyArray) or content.data.ndim != 1:
            raise UnsupportedGraph(f"field {name} must be a flat numeric buffer inside the record list")
        dtype = content.data.dtype
        if dtype.kind not in "biuf" or dtype.itemsize > 8 or dtype == np.dtype("float16"):
            raise UnsupportedGraph(f"field {name} dtype {dtype} is unsupported by the first Numba candidate")
        field_types[name] = dtype.str
    scalar_names = tuple(name for name in arguments if name != graph.array_argument)
    types = {name: scalar_type(arguments[name]) for name in scalar_names}
    if not set(graph.scalar_arguments) <= set(scalar_names):
        raise UnsupportedGraph("missing scalar selection argument")
    if _predicate_type(graph.predicate, field_types, types) != "bool":
        raise UnsupportedGraph("the record filter must be Boolean, not an integer index")
    if not predicate_is_array(graph.predicate):
        raise UnsupportedGraph("the record filter must be a Boolean array mask, not scalar indexing")
    if result_metadata["kind"] not in ("awkward", "numpy") or result_metadata["length"] != len(array):
        raise UnsupportedGraph("the reference must return one flat numeric value for every event")
    dtype = np.dtype(result_metadata["dtype"])
    if dtype.kind not in "biuf" or dtype.itemsize > 8 or dtype == np.dtype("float16"):
        raise UnsupportedGraph("the reference reduction dtype is unsupported by Numba")
    contract = InputContract(
        graph.array_argument, array.layout.form.to_dict(), graph.collection_path,
        graph.fields, scalar_names, types, result_metadata["kind"], dtype.str,
    )
    adapt_inputs(contract, arguments)
    return contract


def _readonly(array):
    result = np.ascontiguousarray(array).view()
    result.setflags(write=False)
    return result


def adapt_inputs(contract: InputContract, arguments: dict):
    if set(arguments) != {contract.array_argument, *contract.scalar_names}:
        raise UnsupportedGraph("call argument names do not match the compiled contract")
    array = arguments[contract.array_argument]
    plan_value(array, "array")
    if array.layout.form.to_dict() != contract.schema:
        raise UnsupportedGraph("array fields/nesting/dtypes do not match the compiled contract")
    error = ak.validity_error(array)
    if error:
        raise UnsupportedGraph(f"invalid Awkward input layout: {error[:200]}")
    layout = collection_layout(array, contract.collection_path)
    offsets = _readonly(layout.offsets.data)
    columns = tuple(_readonly(layout.content.content(name).data) for name in contract.fields)
    scalars = []
    for name in contract.scalar_names:
        value = arguments[name]
        if scalar_type(value) != contract.scalar_types[name]:
            raise UnsupportedGraph(f"scalar {name} type does not match the compiled contract")
        if type(value) is int and not -(2**63) <= value < 2**63:
            raise UnsupportedGraph(f"scalar {name} is outside the first candidate's signed 64-bit range")
        scalars.append(value)
    return offsets, columns, tuple(scalars)


def wrap_output(contract: InputContract, raw, event_count: int):
    if type(raw) is not np.ndarray or raw.ndim != 1:
        raise ValueError("candidate kernel must return one flat NumPy array")
    if len(raw) != event_count:
        raise ValueError("candidate kernel dropped or added events")
    if raw.dtype.str != contract.output_dtype:
        raise ValueError(f"candidate kernel dtype {raw.dtype.str} differs from required {contract.output_dtype}")
    return ak.Array(raw) if contract.output_kind == "awkward" else raw


def fused_candidate_source(graph: FilterSumGraph, contract: InputContract) -> str:
    """Handwritten loop template; no fast-math, reassociation, or sample values."""
    def render(node):
        match node["op"]:
            case "field":
                return f"columns[{contract.fields.index(node['name'])}][item]"
            case "scalar":
                return f"scalars[{contract.scalar_names.index(node['name'])}]"
            case "literal":
                return repr(node["value"])
            case "abs":
                return f"abs({render(node['value'])})"
            case "not":
                return f"(not {render(node['value'])})"
            case "compare":
                return f"({render(node['left'])} {node['operator']} {render(node['right'])})"
            case "and" | "or":
                return f"({render(node['left'])} {node['op']} {render(node['right'])})"
        raise UnsupportedGraph("unknown operation in fused candidate")

    dtype = np.dtype(contract.output_dtype).name
    return f"""def kernel(offsets, columns, scalars):
    output = np.empty(len(offsets) - 1, dtype=np.{dtype})
    for event in range(len(offsets) - 1):
        total = np.{dtype}(0)
        for item in range(offsets[event], offsets[event + 1]):
            if {render(graph.predicate)}:
                total += columns[{contract.fields.index(graph.sum_field)}][item]
        output[event] = total
    return output
"""


def compile_kernel(source: str):
    """Executed only inside the candidate worker; the host controls njit flags."""
    if len(source.encode("utf-8")) > 64 * 1024:
        raise ValueError("candidate source exceeds 64 KiB")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("candidate source must contain only def kernel(offsets, columns, scalars)")
    function = tree.body[0]
    if (
        function.name != "kernel" or function.decorator_list or function.returns
        or function.type_params
        or function.args.posonlyargs or function.args.kwonlyargs
        or function.args.vararg or function.args.kwarg or function.args.defaults
        or [arg.arg for arg in function.args.args] != ["offsets", "columns", "scalars"]
        or any(arg.annotation for arg in function.args.args)
    ):
        raise ValueError("candidate signature must be exactly kernel(offsets, columns, scalars)")
    import numba

    namespace = {"np": np}
    exec(compile(tree, "<candidate>", "exec"), namespace)
    return numba.njit(cache=False, fastmath=False, boundscheck=True, error_model="numpy")(namespace["kernel"])
