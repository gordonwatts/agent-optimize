"""A bounded graph for scoring a jagged collection and selecting its first minimum.

The reference idiom sums the single selected value, giving zero for empty lists.
The compiled loop preserves candidate order, ties, and Awkward's NaN comparisons.
"""

import ast
from dataclasses import dataclass

import numpy as np

from .operations import UnsupportedGraph


@dataclass(frozen=True)
class ArgminGraph:
    array_argument: str
    collection_path: tuple[str, ...]
    score: dict
    value: dict
    fields: tuple[str, ...]
    scalar_arguments: tuple[str, ...]
    reduction: str = "argmin"


def parse_argmin(tree):
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    if sum(isinstance(node, ast.Return) for node in function.body) != 1 or not isinstance(function.body[-1], ast.Return):
        raise UnsupportedGraph("argmin requires one final return")
    arguments = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]]
    modules, constants, bindings = {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif isinstance(node, ast.Assign):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    for node in function.body:
        if isinstance(node, ast.Assign):
            name = node.targets[0].id
            if name in bindings or name in arguments or name in modules or name in constants:
                raise UnsupportedGraph("argmin references may not rebind names")
            bindings[name] = node.value

    def resolve(node, seen=()):
        if isinstance(node, ast.Name) and node.id in bindings:
            if node.id in seen:
                raise UnsupportedGraph("cyclic argmin bindings")
            return resolve(bindings[node.id], (*seen, node.id))
        return node

    def call(node, module, name, keywords):
        node = resolve(node)
        try:
            actual_keywords = {key.arg: ast.literal_eval(key.value) for key in node.keywords} if isinstance(node, ast.Call) else {}
        except (ValueError, TypeError) as error:
            raise UnsupportedGraph("argmin operation keywords must be literals") from error
        return (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and modules.get(node.func.value.id) == module and node.func.attr == name
            and len(node.args) == 1
            and actual_keywords == keywords
        )

    result = resolve(function.body[-1].value)
    if not call(result, "awkward", "sum", {"axis": 1}):
        raise UnsupportedGraph("argmin selection must sum one selected value per event")
    selection = resolve(result.args[0])
    if not isinstance(selection, ast.Subscript):
        raise UnsupportedGraph("argmin selection must index its result expression")
    mask = resolve(selection.slice)
    if not isinstance(mask, ast.Compare) or len(mask.ops) != 1 or not isinstance(mask.ops[0], ast.Eq):
        raise UnsupportedGraph("argmin selection requires local_index == argmin")
    index, minimum = resolve(mask.left), resolve(mask.comparators[0])
    if not call(index, "awkward", "local_index", {"axis": 1}) or not call(minimum, "awkward", "argmin", {"axis": 1, "keepdims": True}):
        raise UnsupportedGraph("use local_index(score, axis=1) == argmin(score, axis=1, keepdims=True)")
    if ast.dump(resolve(index.args[0])) != ast.dump(resolve(minimum.args[0])):
        raise UnsupportedGraph("local_index and argmin must use the same score")
    fields, scalars, roots = set(), set(), set()
    visited = 0

    def expression(node):
        nonlocal visited
        visited += 1
        if visited > 512:
            raise UnsupportedGraph("argmin expression exceeds 512 nodes")
        node = resolve(node)
        if isinstance(node, ast.Name):
            if node.id in constants:
                return literal(constants[node.id])
            if node.id in arguments:
                scalars.add(node.id)
                return {"op": "scalar", "name": node.id}
        if isinstance(node, ast.Constant):
            return literal(node.value)
        if isinstance(node, ast.Attribute):
            path = []
            while isinstance(node, ast.Attribute):
                path.append(node.attr)
                node = resolve(node.value)
            if isinstance(node, ast.Name) and node.id in arguments:
                path.reverse()
                fields.add(path[-1])
                roots.add((node.id, tuple(path[:-1])))
                return {"op": "field", "name": path[-1]}
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Pow):
            # Only square powers are admitted: they have defined native float semantics.
            if isinstance(node.op, ast.Pow) and not (isinstance(node.right, ast.Constant) and node.right.value == 2):
                raise UnsupportedGraph("argmin powers must be squares")
            return {"op": {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Pow: "**"}[type(node.op)], "left": expression(node.left), "right": expression(node.right)}
        if isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1:
                return {"op": "abs", "value": expression(node.args[0])}
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and modules.get(node.func.value.id) == "numpy":
                if node.func.attr == "sqrt" and len(node.args) == 1:
                    return {"op": "sqrt", "value": expression(node.args[0])}
                if node.func.attr == "maximum" and len(node.args) == 2:
                    return {"op": "maximum", "left": expression(node.args[0]), "right": expression(node.args[1])}
        raise UnsupportedGraph(f"unsupported argmin arithmetic: {type(node).__name__}")

    def literal(value):
        if type(value) not in (int, float) or not np.isfinite(value):
            raise UnsupportedGraph("argmin constants must be finite numeric values")
        return {"op": "literal", "value": value}

    score, value = expression(minimum.args[0]), expression(selection.value)
    if len(roots) != 1 or len(fields) > 16:
        raise UnsupportedGraph("argmin fields must come from one collection, with at most 16 fields")
    argument, path = roots.pop()
    if argument in scalars:
        raise UnsupportedGraph("argmin array argument cannot also be a scalar")
    return ArgminGraph(argument, path, score, value, tuple(sorted(fields)), tuple(sorted(scalars)))


def candidate_source(graph, contract):
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
            case "sqrt":
                return f"np.sqrt({render(node['value'])})"
            case "maximum":
                return f"np.maximum({render(node['left'])}, {render(node['right'])})"
            case "+" | "-" | "*" | "**":
                return f"({render(node['left'])} {node['op']} {render(node['right'])})"
        raise UnsupportedGraph("unknown argmin expression")

    dtype = np.dtype(contract.output_dtype).name
    return f"""def kernel(offsets, columns, scalars):
    output = np.empty(len(offsets) - 1, dtype=np.{dtype})
    for event in range(len(offsets) - 1):
        best = offsets[event]
        best_score = np.float64(0)
        for item in range(offsets[event], offsets[event + 1]):
            score = {render(graph.score)}
            if item == offsets[event] or score < best_score:
                best = item
                best_score = score
        total = np.{dtype}(0)
        if offsets[event] < offsets[event + 1]:
            item = best
            total += {render(graph.value)}
        output[event] = total
    return output
"""


def generate_cases(graph, contract, arguments, observed_max_length):
    from .cases import TestCase, boundary_values, build_schema_array, MAX_SYNTHETIC_ELEMENTS
    from .candidate import collection_layout

    layout = collection_layout(arguments[graph.array_argument], graph.collection_path)
    dtypes = {name: layout.content.content(name).data.dtype for name in graph.fields}
    larger = max(257, observed_max_length + 7)
    if larger > MAX_SYNTHETIC_ELEMENTS:
        raise UnsupportedGraph("argmin mandatory case exceeds the synthetic element budget")

    def case(name, lengths, overrides=None):
        count = sum(lengths)
        columns = {field: (np.arange(count, dtype=np.float64) % 13 + 1).astype(dtype) for field, dtype in dtypes.items()}
        columns.update(overrides or {})
        array = build_schema_array(contract.schema, lengths, graph.collection_path, columns)
        return TestCase(name, {**arguments, graph.array_argument: array})

    yield case("zero-events", [])
    yield case("empty-lists", [0, 0])
    yield case("varying-lengths", [0, 1, 3, 7, 2])
    yield case("larger-than-recording", [larger, 0, 1])
    yield case("tied-scores", [4, 0, 2], {field: np.ones(6, dtype=dtype) for field, dtype in dtypes.items()})
    yield case("all-zero", [4, 0, 2], {field: np.zeros(6, dtype=dtype) for field, dtype in dtypes.items()})
    for field, dtype in dtypes.items():
        values = boundary_values(dtype, [0, 172.5])
        yield case(f"boundaries-{field}", [len(values), 0], {field: values})
        payloads = np.array([0x7FF8000000000001, 0x7FF8000000000002], dtype=np.uint64).view(dtype)
        special = np.concatenate([payloads, np.array([np.inf, -np.inf, -0.0, 0.0, 1.0], dtype=dtype)])
        yield case(f"nonfinite-first-{field}", [3, 4, 0], {field: special})
        yield case(f"nonfinite-last-{field}", [3, 4, 0], {field: special[::-1]})
