"""Evaluator-owned deterministic tests preserving the complete input schema."""

from dataclasses import dataclass
import itertools
import math

import awkward as ak
import numpy as np

from .candidate import collection_layout
from .operations import UnsupportedGraph

MAX_SYNTHETIC_ELEMENTS = 1_000_000


@dataclass
class TestCase:
    name: str
    arguments: dict
    recorded_output: dict | None = None
    recorded_directory: str | None = None


def _evaluate(node, fields, scalars):
    match node["op"]:
        case "field":
            return fields[node["name"]]
        case "scalar":
            return scalars[node["name"]]
        case "literal":
            return node["value"]
        case "abs":
            return np.abs(_evaluate(node["value"], fields, scalars))
        case "not":
            return np.logical_not(_evaluate(node["value"], fields, scalars))
        case "and" | "or":
            left = _evaluate(node["left"], fields, scalars)
            right = _evaluate(node["right"], fields, scalars)
            return np.logical_and(left, right) if node["op"] == "and" else np.logical_or(left, right)
        case "compare":
            left, right = _evaluate(node["left"], fields, scalars), _evaluate(node["right"], fields, scalars)
            return {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}[node["operator"]](left, right)
    raise UnsupportedGraph("unknown case-generation operation")


def _constants(node, arguments):
    values = []
    if node["op"] == "literal" and type(node["value"]) is not bool:
        values.append(node["value"])
    elif node["op"] == "scalar":
        value = arguments[node["name"]]
        if not isinstance(value, (bool, np.bool_)):
            values.append(value.item() if isinstance(value, np.generic) else value)
    for key in ("left", "right", "value"):
        if isinstance(node.get(key), dict):
            values.extend(_constants(node[key], arguments))
    return values


def _scalar_variants(value):
    if isinstance(value, (bool, np.bool_)):
        return [type(value)(not value)]
    native = value.item() if isinstance(value, np.generic) else value
    variants = []
    for candidate in [0, native - 1, native + 1]:
        if isinstance(value, np.integer):
            limits = np.iinfo(value.dtype)
            candidate = min(limits.max, max(limits.min, candidate))
        elif type(value) is int:
            candidate = min(2**63 - 1, max(-(2**63), candidate))
        converted = type(value)(candidate)
        if converted != value and not any(converted == previous for previous in variants):
            variants.append(converted)
    return variants


def boundary_values(dtype, constants):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return np.array([False, True], dtype=dtype)
    values = [0, 1, -1]
    for value in constants:
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if not np.isfinite(value):
            continue
        values.extend([value, -value, value - 1, value + 1, -value - 1, -value + 1])
    result = []
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        for value in [*values, limits.min, limits.max]:
            adjacent = {int(value)} if isinstance(value, (int, np.integer)) else {math.floor(value), math.ceil(value)}
            for integer in adjacent:
                integer = min(limits.max, max(limits.min, integer))
                if integer not in result:
                    result.append(integer)
    else:
        limits = np.finfo(dtype)
        for value in values:
            with np.errstate(over="ignore", invalid="ignore"):
                converted = dtype.type(value)
                adjacent = [converted, np.nextafter(converted, dtype.type(-np.inf)), np.nextafter(converted, dtype.type(np.inf))]
            for number in adjacent:
                if np.isfinite(number) and number not in result:
                    result.append(number)
        result.extend([limits.tiny, np.nextafter(dtype.type(0), dtype.type(1)), limits.max, -limits.max])
    return np.asarray(result, dtype=dtype)


def build_schema_array(schema, lengths, collection_path, columns):
    """Build from low-level Forms so empty arrays retain fields and dtypes."""
    event_count = len(lengths)

    def build(form, length, path):
        match form["class"]:
            case "NumpyArray":
                dtype = np.dtype(form["primitive"])
                values = columns.get(path[-1]) if path[:-1] == collection_path else None
                data = np.zeros(length, dtype=dtype) if values is None else np.asarray(values, dtype=dtype)
                if len(data) != length:
                    raise UnsupportedGraph("generated numeric field length differs from its schema")
                return ak.contents.NumpyArray(data)
            case "RecordArray":
                if form["fields"] is None:
                    raise UnsupportedGraph("tuple-record synthetic cases are unsupported")
                contents = [build(content, length, (*path, field)) for field, content in zip(form["fields"], form["contents"], strict=True)]
                return ak.contents.RecordArray(contents, form["fields"], length=length)
            case "ListOffsetArray":
                if length != event_count:
                    raise UnsupportedGraph("nested variable-length lists are unsupported")
                dtype = {"i32": np.int32, "u32": np.uint32, "i64": np.int64}[form["offsets"]]
                offsets = np.empty(event_count + 1, dtype=dtype)
                offsets[0] = 0
                np.cumsum(lengths, out=offsets[1:])
                return ak.contents.ListOffsetArray(ak.index.Index(offsets), build(form["content"], int(offsets[-1]), path))
        raise UnsupportedGraph(f"synthetic cases cannot preserve unsupported layout {form['class']}")

    result = ak.Array(build(schema, event_count, ()))
    if result.layout.form.to_dict() != schema:
        raise UnsupportedGraph("generated case did not preserve the original field/nesting/dtype schema")
    return result


def generate_cases(graph, contract, arguments, *, observed_max_length=0):
    if getattr(graph, "reduction", None) == "argmin":
        from .argmin import generate_cases as argmin_cases
        yield from argmin_cases(graph, contract, arguments, observed_max_length)
        return
    layout = collection_layout(arguments[graph.array_argument], graph.collection_path)
    dtypes = {name: layout.content.content(name).data.dtype for name in graph.fields}
    constants = _constants(graph.predicate, arguments)
    choices = {name: boundary_values(dtype, constants) for name, dtype in dtypes.items()}
    passing = failing = None
    product = itertools.product(*(choices[name] for name in graph.fields))
    for combination in itertools.islice(product, 16_384):
        fields = {name: np.asarray([value], dtype=dtypes[name]) for name, value in zip(graph.fields, combination, strict=True)}
        selected = bool(np.asarray(_evaluate(graph.predicate, fields, arguments)).reshape(-1)[0])
        if selected and passing is None:
            passing = dict(zip(graph.fields, combination, strict=True))
        if not selected and failing is None:
            failing = dict(zip(graph.fields, combination, strict=True))
        if passing is not None and failing is not None:
            break
    if passing is None or failing is None:
        raise UnsupportedGraph("bounded synthetic-case construction could not establish both passing and failing selections")
    # A nonzero passing sum makes Boolean-scalar changes observable as well.
    if not passing[graph.sum_field]:
        for value in choices[graph.sum_field]:
            if value:
                candidate = {**passing, graph.sum_field: value}
                fields = {name: np.asarray([item], dtype=dtypes[name]) for name, item in candidate.items()}
                if bool(np.asarray(_evaluate(graph.predicate, fields, arguments)).reshape(-1)[0]):
                    passing = candidate
                    break
    larger = max(257, observed_max_length + 7)
    if larger > MAX_SYNTHETIC_ELEMENTS:
        raise UnsupportedGraph("sample list length is too large for the bounded mandatory larger-list case")

    def case(name, lengths, *, uniform=passing, overrides=None):
        count = sum(lengths)
        columns = {field: np.full(count, uniform[field], dtype=dtypes[field]) for field in graph.fields}
        if overrides:
            columns.update(overrides)
        values = dict(arguments)
        values[graph.array_argument] = build_schema_array(contract.schema, lengths, graph.collection_path, columns)
        return TestCase(name, values)

    yield case("zero-events", [])
    yield case("empty-lists", [0, 0, 0])
    yield case("varying-lengths", [0, 1, 3, 2, 7, 0])
    yield case("larger-than-recording", [0, larger, 1])
    yield case("all-zero", [0, 1, 4], uniform={field: 0 for field in graph.fields})
    yield case("all-pass", [0, 1, 3, 5])
    yield case("none-pass", [0, 1, 3, 5], uniform=failing)
    for field in graph.fields:
        values = choices[field]
        yield case(f"boundaries-{field}", [len(values), 0], overrides={field: values})
        dtype = dtypes[field]
        if dtype.kind == "f":
            # Include distinct quiet-NaN payloads as well as signs and subnormals.
            bit_type = np.uint32 if dtype.itemsize == 4 else np.uint64
            payloads = np.array([0x7FC00001, 0x7FC00002] if dtype.itemsize == 4 else [0x7FF8000000000001, 0x7FF8000000000002], dtype=bit_type).view(dtype)
            special = np.concatenate([payloads, np.array([np.inf, -np.inf, 0.0, -0.0, np.nextafter(dtype.type(0), dtype.type(1))], dtype=dtype)])
            yield case(f"nan-infinity-signed-zero-{field}", [len(special), 0, 0], overrides={field: special})
    dtype = dtypes[graph.sum_field]
    if dtype.kind == "f":
        magnitude = dtype.type(1e8 if dtype.itemsize == 4 else 1e16)
        difficult = np.array([magnitude, -magnitude, 1, magnitude, 1, -magnitude], dtype=dtype)
        yield case("cancellation-order", [3, 3, 0], overrides={graph.sum_field: difficult})
        limits = np.finfo(dtype)
        extremes = np.array([limits.max, limits.max, -limits.max, limits.tiny, -limits.tiny, np.nextafter(dtype.type(0), dtype.type(1))], dtype=dtype)
        yield case("large-small-difficult-sums", [3, 3, 0], overrides={graph.sum_field: extremes})
        zeros = np.array([-0.0, -0.0, 0.0], dtype=dtype)
        yield case("negative-zero-sums", [2, 1, 0], overrides={graph.sum_field: zeros})
    elif dtype.kind in "iu":
        limits = np.iinfo(dtype)
        difficult = np.array([limits.max, limits.max, limits.min, 1, 0, limits.max], dtype=dtype)
        yield case("integer-overflow-promotion", [3, 3, 0], overrides={graph.sum_field: difficult})
    for scalar in graph.scalar_arguments:
        for number, value in enumerate(_scalar_variants(arguments[scalar])):
            varied_arguments = {**arguments, scalar: value}
            varied_constants = [*constants, *_constants(graph.predicate, varied_arguments)]
            for field in graph.fields:
                boundaries = boundary_values(dtypes[field], varied_constants)
                generated = case(f"scalar-{scalar}-{number}-{field}", [len(boundaries), 0], overrides={field: boundaries})
                generated.arguments[scalar] = value
                yield generated
