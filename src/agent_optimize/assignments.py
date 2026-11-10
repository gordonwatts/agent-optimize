"""Ordered Cartesian index assignments with scalar arithmetic and first argmin.

Lower a bounded public Awkward idiom to streaming loops, avoiding candidate
arrays. The graph contains expressions and index predicates, never q16 names.
"""

import ast
from dataclasses import dataclass

import numpy as np

from .operations import UnsupportedGraph


@dataclass(frozen=True)
class AssignmentGraph:
    array_argument: str
    collection_path: tuple
    fields: tuple
    event_fields: tuple
    width: int
    predicate: dict
    expressions: dict
    score: dict
    value: dict
    scalar_arguments: tuple = ()
    reduction: str = "assignment_argmin"


def parse_assignments(tree):
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    args = [a.arg for a in function.args.args]
    if len(args) != 1 or function.args.posonlyargs or function.args.kwonlyargs:
        raise UnsupportedGraph("assignment reduction requires one event-record argument")
    root = args[0]
    modules, constants, bindings = {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update({a.asname or a.name: a.name for a in node.names})
        elif isinstance(node, ast.Assign):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    for node in function.body:
        if isinstance(node, ast.Assign):
            name = node.targets[0].id
            if name in bindings or name in modules or name in constants or name == root:
                raise UnsupportedGraph("assignment references may not rebind names")
            bindings[name] = node.value
    if not isinstance(function.body[-1], ast.Return) or sum(isinstance(n, ast.Return) for n in function.body) != 1:
        raise UnsupportedGraph("assignment reduction needs one final return")

    def resolve(node):
        seen = set()
        while isinstance(node, ast.Name) and node.id in bindings:
            if node.id in seen:
                raise UnsupportedGraph("cyclic assignment bindings")
            seen.add(node.id)
            node = bindings[node.id]
        return node

    def iscall(node, module, name):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and modules.get(node.func.value.id) == module and node.func.attr == name

    def keywords(node, expected):
        try:
            return {k.arg: ast.literal_eval(k.value) for k in node.keywords} == expected
        except (ValueError, TypeError):
            return False

    def path(node):
        result = []
        node = resolve(node)
        while isinstance(node, ast.Attribute):
            result.append(node.attr)
            node = resolve(node.value)
        return tuple(reversed(result)) if isinstance(node, ast.Name) and node.id == root else None

    products = [(name, node) for name, node in bindings.items() if iscall(node, "awkward", "argcartesian")]
    if len(products) != 1:
        raise UnsupportedGraph("use one argcartesian product of one jet-like collection")
    product_name, product = products[0]
    if len(product.args) != 1 or not keywords(product, {"axis": 1}) or not isinstance(product.args[0], ast.Dict):
        raise UnsupportedGraph("argcartesian requires a named dictionary and axis=1")
    mapping = product.args[0]
    slots = [ast.literal_eval(k) for k in mapping.keys]
    paths = [path(v) for v in mapping.values]
    if not 2 <= len(slots) <= 4 or len(set(slots)) != len(slots) or any(type(k) is not str for k in slots) or not paths[0] or any(p != paths[0] for p in paths):
        raise UnsupportedGraph("use 2..4 named slots from the same event collection")
    collection = paths[0]
    selected = [(name, n) for name, n in bindings.items() if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == product_name]
    if len(selected) != 1:
        raise UnsupportedGraph("filter the assignment product once before gathering")
    selected_name, selection = selected[0]
    fields, event_fields, expressions, active = set(), set(), {}, set()

    def expression(node):
        if isinstance(node, ast.Name):
            if node.id in constants:
                return expression(ast.Constant(constants[node.id]))
            if node.id in bindings:
                if node.id in active:
                    raise UnsupportedGraph("cyclic numeric bindings")
                if node.id not in expressions:
                    active.add(node.id)
                    value = expression(bindings[node.id])
                    active.remove(node.id)
                    expressions[node.id] = value
                return {"op": "var", "name": node.id}
        if isinstance(node, ast.Constant) and type(node.value) in (bool, int, float) and np.isfinite(node.value):
            return {"op": "literal", "value": node.value}
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in (product_name, selected_name) and node.attr in slots:
                return {"op": "index", "slot": slots.index(node.attr)}
            base = resolve(node.value)
            if isinstance(base, ast.Subscript) and path(base.value) == collection:
                index = base.slice
                if isinstance(index, ast.Attribute) and isinstance(index.value, ast.Name) and index.value.id == selected_name and index.attr in slots:
                    fields.add(node.attr)
                    return {"op": "field", "name": node.attr, "slot": slots.index(index.attr)}
            p = path(node)
            if p and not p[:len(collection)] == collection:
                event_fields.add(".".join(p))
                return {"op": "event", "name": ".".join(p)}
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
            parts = node.slice.elts
            if len(parts) == 2 and isinstance(parts[0], ast.Slice) and parts[0].lower is None and parts[0].upper is None and parts[0].step is None and isinstance(parts[1], ast.Constant) and parts[1].value is None:
                return expression(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr):
            if isinstance(node.op, ast.Pow) and not (isinstance(node.right, ast.Constant) and node.right.value == 2):
                raise UnsupportedGraph("only square powers are supported")
            return {"op": {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**", ast.BitAnd: "and", ast.BitOr: "or"}[type(node.op)], "left": expression(node.left), "right": expression(node.right)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return {"op": "*", "left": {"op": "literal", "value": -1 if isinstance(node.op, ast.USub) else 1}, "right": expression(node.operand)}
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE):
            return {"op": {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}[type(node.ops[0])], "left": expression(node.left), "right": expression(node.comparators[0])}
        if isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1:
                return {"op": "abs", "value": expression(node.args[0])}
            if iscall(node, "numpy", "sqrt") and len(node.args) == 1:
                return {"op": "sqrt", "value": expression(node.args[0])}
            if iscall(node, "numpy", "maximum") and len(node.args) == 2:
                return {"op": "maximum", "left": expression(node.args[0]), "right": expression(node.args[1])}
            if iscall(node, "awkward", "where") and len(node.args) == 3:
                return {"op": "where", "condition": expression(node.args[0]), "left": expression(node.args[1]), "right": expression(node.args[2])}
        raise UnsupportedGraph(f"unsupported assignment expression: {ast.dump(node)[:120]}")

    result = resolve(function.body[-1].value)
    if not iscall(result, "awkward", "sum") or len(result.args) != 1 or not keywords(result, {"axis": 1}):
        raise UnsupportedGraph("assignment result must sum its single selected value")
    value_selection = resolve(result.args[0])
    if not isinstance(value_selection, ast.Subscript):
        raise UnsupportedGraph("select one value with local_index == argmin")
    mask = resolve(value_selection.slice)
    if not isinstance(mask, ast.Compare) or len(mask.ops) != 1 or not isinstance(mask.ops[0], ast.Eq):
        raise UnsupportedGraph("use first argmin selection")
    index, minimum = resolve(mask.left), resolve(mask.comparators[0])
    if not iscall(index, "awkward", "local_index") or not keywords(index, {"axis": 1}) or not iscall(minimum, "awkward", "argmin") or not keywords(minimum, {"axis": 1, "keepdims": True}) or len(index.args) != 1 or len(minimum.args) != 1 or ast.dump(resolve(index.args[0])) != ast.dump(resolve(minimum.args[0])):
        raise UnsupportedGraph("local_index and first argmin must use the same score")
    predicate = expression(selection.slice)
    score, value = expression(minimum.args[0]), expression(value_selection.value)
    for name, node in bindings.items():
        if name in (product_name, selected_name) or path(node) == collection:
            continue
        resolved = resolve(node)
        if isinstance(resolved, ast.Subscript) and path(resolved.value) == collection and isinstance(resolved.slice, ast.Attribute) and isinstance(resolved.slice.value, ast.Name) and resolved.slice.value.id == selected_name and resolved.slice.attr in slots:
            continue
        expression(ast.Name(id=name, ctx=ast.Load()))
    def value_type(n):
        op = n["op"]
        if op == "var": return value_type(expressions[n["name"]])
        if op == "literal": return "bool" if type(n["value"]) is bool else "number"
        if op in ("field", "event", "index"): return "number"
        if op in ("and", "or"):
            if any(value_type(n[k]) != "bool" for k in ("left", "right")):
                raise UnsupportedGraph("assignment Boolean operations require comparisons")
            return "bool"
        if op == "where":
            if value_type(n["condition"]) != "bool" or any(value_type(n[k]) != "number" for k in ("left", "right")):
                raise UnsupportedGraph("assignment where requires a Boolean condition and numeric branches")
            return "number"
        children = [v for v in n.values() if isinstance(v, dict)]
        if any(value_type(v) != "number" for v in children):
            raise UnsupportedGraph("assignment arithmetic requires numeric operands")
        return "bool" if op in ("==", "!=", "<", "<=", ">", ">=") else "number"
    if value_type(predicate) != "bool" or value_type(score) != "number" or value_type(value) != "number":
        raise UnsupportedGraph("assignment selection requires a Boolean predicate and numeric score/value")
    if set(fields) & set(event_fields):
        raise UnsupportedGraph("event and collection field names must be distinct")
    if len(expressions) > 128 or len(fields) + len(event_fields) > 32:
        raise UnsupportedGraph("assignment expression graph exceeds supported bounds")
    return AssignmentGraph(root, collection, tuple(sorted(fields)), tuple(sorted(event_fields)), len(slots), predicate, expressions, score, value)


def candidate_source(graph, contract):
    def render(n):
        op = n["op"]
        if op == "raw": return n["code"]
        if op == "var": return "v_" + n["name"]
        if op == "literal": return repr(n["value"])
        if op == "index": return f"(item{n['slot']} - offsets[event])"
        if op == "field": return f"columns[{contract.fields.index(n['name'])}][item{n['slot']}]"
        if op == "event": return f"columns[{len(contract.fields) + contract.event_fields.index(n['name'])}][event]"
        if op == "abs": return f"abs({render(n['value'])})"
        if op == "sqrt": return f"np.sqrt({render(n['value'])})"
        if op == "maximum": return f"({render(n['left'])} if np.isnan({render(n['left'])}) else np.maximum({render(n['left'])}, {render(n['right'])}))"
        if op == "where": return f"({render(n['left'])} if {render(n['condition'])} else {render(n['right'])})"
        if op in ("+", "*", "-", "/"):
            # LLVM may commute NaN operands even without fastmath. Preserve the
            # installed NumPy ufunc's payload priority explicitly.
            fn = {"+": np.add, "*": np.multiply, "-": np.subtract, "/": np.divide}[op]
            payloads = np.array([0x7ff8000000000001, 0xfff8000000000002], dtype=np.uint64).view(np.float64)
            priority = "left" if fn(payloads[:1], payloads[1:]).view(np.uint64)[0] == 0x7ff8000000000001 else "right"
            chosen = render(n[priority])
            other = render(n["right" if priority == "left" else "left"])
            return f"(({chosen} + np.float64(0)) if np.isnan({chosen}) else (({other} + np.float64(0)) if np.isnan({other}) else ({render(n['left'])} {op} {render(n['right'])})))"
        return f"({render(n['left'])} {op} {render(n['right'])})"

    # Reconstruct dependency order after JSON serialization (which sorts keys).
    ordered, varying = {}, {}
    def visit(n):
        if n["op"] == "var":
            name = n["name"]
            if name not in ordered:
                varying[name] = visit(graph.expressions[name])
                ordered[name] = graph.expressions[name]
            return varying[name]
        children = [visit(v) for v in n.values() if isinstance(v, dict)]
        return n["op"] in ("index", "field") or any(children)
    for n in (graph.score, graph.value):
        visit(n)
    def expand(n):
        if n["op"] == "var":
            return expand(graph.expressions[n["name"]])
        return {k: expand(v) if isinstance(v, dict) else v for k, v in n.items()}
    def conjuncts(n):
        if n["op"] == "and":
            return [*conjuncts(n["left"]), *conjuncts(n["right"])]
        return [n]
    def slots(n):
        result = {n["slot"]} if n["op"] in ("field", "index") else set()
        for v in n.values():
            if isinstance(v, dict): result.update(slots(v))
        return result
    conditions = {}
    for condition in conjuncts(expand(graph.predicate)):
        depth = max(slots(condition), default=-1)
        conditions.setdefault(depth, []).append(render(condition))
    # Topological bindings retain expression order without exponential expansion.
    lines = ["def kernel(offsets, columns, scalars):", "    output = np.zeros(len(offsets) - 1, dtype=np.float64)", "    for event in range(len(offsets) - 1):", "        found = False", "        best_score = np.float64(0)"]
    temporary = 0
    def emit(node, indent):
        nonlocal temporary
        if node["op"] in ("var", "literal", "index", "field", "event"):
            return render(node)
        shallow = {key: {"op": "raw", "code": emit(value, indent)} if isinstance(value, dict) else value for key, value in node.items()}
        name = f"temporary{temporary}"
        temporary += 1
        lines.append(indent + f"{name} = {render(shallow)}")
        return name
    for name, expr in ordered.items():
        if not varying[name]:
            code = emit(expr, "        ")
            lines.append("        " + f"v_{name} = {code}")
    indent = "        "
    if -1 in conditions:
        lines.append(indent + "if " + " and ".join(conditions[-1]) + ":")
        indent += "    "
    for slot in range(graph.width):
        lines.append(indent + f"for item{slot} in range(offsets[event], offsets[event + 1]):")
        indent += "    "
        if slot in conditions:
            lines.append(indent + "if " + " and ".join(conditions[slot]) + ":")
            indent += "    "
    for name, expr in ordered.items():
        if varying[name]:
            code = emit(expr, indent)
            lines.append(indent + f"v_{name} = {code}")
    lines += [indent + f"score = {render(graph.score)}", indent + "if not found or score < best_score:", indent + "    found = True", indent + "    best_score = score", indent + f"    output[event] = np.float64(0) + {render(graph.value)}"]
    lines.append("    return output")
    return "\n".join(lines) + "\n"


def generate_cases(graph, contract, arguments, observed_max_length):
    from .cases import TestCase, build_schema_array, boundary_values
    # Cartesian workloads grow exponentially; bound generated work, not merely
    # stored input bytes. Recordings retain the caller's existing budgets.
    larger = max(8, observed_max_length + 1)
    if larger ** graph.width > 2_000_000:
        raise UnsupportedGraph("mandatory larger assignment case exceeds the synthetic work budget")

    constants = [0.]
    def collect(node):
        if node.get("op") == "literal" and type(node.get("value")) in (int, float):
            constants.append(node["value"])
        for child in node.values():
            if isinstance(child, dict): collect(child)
    for node in graph.expressions.values(): collect(node)

    def case(name, lengths, special=None):
        count = sum(lengths)
        columns = {f: (np.arange(count, dtype=np.float64) % 7 + 1) for f in graph.fields}
        columns.update({f: np.arange(len(lengths), dtype=np.float64) + 10 for f in graph.event_fields})
        if special:
            field, value = special
            columns[field] = np.full(len(lengths) if field in graph.event_fields else count, value, dtype=np.float64)
        return TestCase(name, {graph.array_argument: build_schema_array(contract.schema, lengths, graph.collection_path, columns)})

    yield case("zero-events", [])
    yield case("empty-lists", [0, 0])
    yield case("varying-lengths", [0, 1, 3, 4, 6])
    yield case("larger-than-recording", [larger, 0, 4])
    for field in (*graph.fields, *graph.event_fields):
        for label, value in (("zero", 0.), ("signed-zero", -0.), ("nan", np.nan), ("positive-infinity", np.inf), ("negative-infinity", -np.inf), ("large", 1e150), ("small", 1e-150)):
            yield case(f"{label}-{field}", [4, 0, 5], (field, value))
        values = boundary_values(np.float64, constants)
        # Many small assignment problems, rather than one exponential product.
        n = (len(values) + 3) // 4
        lengths = [4] * n
        columns = {f: np.ones(4 * n) for f in graph.fields}
        columns.update({f: np.ones(n) for f in graph.event_fields})
        columns[field] = np.resize(values, n if field in graph.event_fields else 4 * n)
        yield TestCase(f"boundaries-{field}", {graph.array_argument: build_schema_array(contract.schema, lengths, graph.collection_path, columns)})
        for reverse in (False, True):
            special = np.array([0x7ff8000000000001, 0x7ff8000000000002, 0xfff8000000000001, 0x8000000000000000, 0, 0x7ff0000000000000, 0xfff0000000000000, 0x3ff0000000000000], dtype=np.uint64).view(np.float64)
            columns[field] = np.resize(special[::-1] if reverse else special, n if field in graph.event_fields else 4 * n)
            yield TestCase(f"nonfinite-order-{field}-{reverse}", {graph.array_argument: build_schema_array(contract.schema, lengths, graph.collection_path, columns)})
