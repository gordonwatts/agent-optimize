"""Parse the first filter-and-sum expression into a small explicit graph."""

import ast
from dataclasses import dataclass
import math

from .experiment import validate_frozen_source


class UnsupportedGraph(ValueError):
    pass


@dataclass(frozen=True)
class FilterSumGraph:
    array_argument: str
    collection_path: tuple[str, ...]
    sum_field: str
    predicate: dict
    fields: tuple[str, ...]
    scalar_arguments: tuple[str, ...]


def graph_from_dict(value):
    if value.get("reduction") == "assignment_argmin":
        from .assignments import AssignmentGraph
        return AssignmentGraph(**{**value, "collection_path": tuple(value["collection_path"]), "fields": tuple(value["fields"]), "event_fields": tuple(value["event_fields"]), "scalar_arguments": tuple(value["scalar_arguments"])})
    if value.get("reduction") == "argmin":
        from .argmin import ArgminGraph
        return ArgminGraph(value["array_argument"], tuple(value["collection_path"]), value["score"], value["value"], tuple(value["fields"]), tuple(value["scalar_arguments"]))
    return FilterSumGraph(
        value["array_argument"], tuple(value["collection_path"]), value["sum_field"],
        value["predicate"], tuple(value["fields"]), tuple(value["scalar_arguments"]),
    )


def predicate_is_array(node):
    """Track broadcasting separately from the Boolean/numeric result type."""
    match node["op"]:
        case "field":
            return True
        case "scalar" | "literal":
            return False
        case "abs" | "not":
            return predicate_is_array(node["value"])
        case "compare" | "and" | "or":
            return predicate_is_array(node["left"]) or predicate_is_array(node["right"])
    raise UnsupportedGraph("unknown selection operation")


def parse_filter_sum(source: str) -> FilterSumGraph:
    tree, name = validate_frozen_source(source)
    if any(isinstance(node, ast.Attribute) and node.attr == "argcartesian" for node in ast.walk(tree)):
        from .assignments import parse_assignments
        return parse_assignments(tree)
    if any(isinstance(node, ast.Attribute) and node.attr == "argmin" for node in ast.walk(tree)):
        from .argmin import parse_argmin
        return parse_argmin(tree)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    if sum(isinstance(node, ast.Return) for node in function.body) != 1 or not isinstance(function.body[-1], ast.Return):
        raise UnsupportedGraph("use one final return containing ak.sum(..., axis=1)")
    modules, imported, constants = {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif isinstance(node, ast.ImportFrom):
            imported.update({alias.asname or alias.name: (node.module, alias.name) for alias in node.names})
        elif isinstance(node, ast.Assign):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    arguments = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]]
    environment = {argument: {"op": "argument", "name": argument} for argument in arguments}

    def literal(value):
        if type(value) is int and not -(2**63) <= value < 2**64:
            raise UnsupportedGraph("selection integer constant is outside the first candidate's 64-bit range")
        if type(value) is float and not math.isfinite(value):
            raise UnsupportedGraph("selection constants must be finite; nonfinite input values are tested separately")
        return {"op": "literal", "value": value}

    def target(call):
        if isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name):
            return modules.get(call.value.id), call.attr
        if isinstance(call, ast.Name):
            return imported.get(call.id, ("builtins", call.id))
        return None, None

    def expression(node):
        if isinstance(node, ast.Name):
            if node.id in environment:
                return environment[node.id]
            if node.id in constants and type(constants[node.id]) in (bool, int, float):
                return literal(constants[node.id])
        elif isinstance(node, ast.Constant) and type(node.value) in (bool, int, float):
            return literal(node.value)
        elif isinstance(node, ast.Attribute):
            return {"op": "field", "base": expression(node.value), "name": node.attr}
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and type(node.slice.value) is str:
                return {"op": "field", "base": expression(node.value), "name": node.slice.value}
            return {"op": "filter", "base": expression(node.value), "predicate": expression(node.slice)}
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Invert):
                return {"op": "not", "value": expression(node.operand)}
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                operand = expression(node.operand)
                if operand["op"] == "literal":
                    return literal(-operand["value"] if isinstance(node.op, ast.USub) else operand["value"])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            return {"op": "and" if isinstance(node.op, ast.BitAnd) else "or", "left": expression(node.left), "right": expression(node.right)}
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            operator = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}.get(type(node.ops[0]))
            if operator:
                return {"op": "compare", "operator": operator, "left": expression(node.left), "right": expression(node.comparators[0])}
        elif isinstance(node, ast.Call):
            module, operation = target(node.func)
            if (module, operation) == ("builtins", "abs") and len(node.args) == 1 and not node.keywords:
                return {"op": "abs", "value": expression(node.args[0])}
            if (module, operation) == ("awkward", "to_numpy") and len(node.args) == 1 and not node.keywords:
                return {"op": "to_numpy", "value": expression(node.args[0])}
            if (module, operation) == ("awkward", "sum") and len(node.args) == 1:
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                if set(keywords) == {"axis"} and isinstance(keywords["axis"], ast.Constant) and keywords["axis"].value == 1:
                    return {"op": "sum", "value": expression(node.args[0]), "axis": 1}
                raise UnsupportedGraph("the first compiled candidate requires exactly ak.sum(..., axis=1)")
        raise UnsupportedGraph(f"unsupported filter/sum expression: {type(node).__name__}")

    for statement in function.body:
        if isinstance(statement, ast.Assign):
            if statement.targets[0].id in arguments:
                raise UnsupportedGraph("reassigning function arguments is unsupported")
            environment[statement.targets[0].id] = expression(statement.value)
    reduction = expression(function.body[-1].value)
    if reduction["op"] == "to_numpy":
        reduction = reduction["value"]
    if reduction["op"] != "sum":
        raise UnsupportedGraph("return a per-event ak.sum(..., axis=1)")
    value = reduction["value"]
    if value["op"] == "field" and value["base"]["op"] == "filter":
        sum_field, selection = value["name"], value["base"]
        collection, predicate = selection["base"], selection["predicate"]
    elif value["op"] == "filter" and value["base"]["op"] == "field":
        sum_field = value["base"]["name"]
        collection, predicate = value["base"]["base"], value["predicate"]
    else:
        raise UnsupportedGraph("sum one numeric field after a single record/list filter")

    def path(node):
        fields = []
        while node["op"] == "field":
            fields.append(node["name"])
            node = node["base"]
        if node["op"] != "argument":
            raise UnsupportedGraph("the selected collection must come directly from an array argument")
        return node["name"], tuple(reversed(fields))

    array_argument, collection_path = path(collection)
    fields, scalars = {sum_field}, set()
    visited = 0

    def normalize(node):
        nonlocal visited
        visited += 1
        if visited > 512:
            raise UnsupportedGraph("the first operation graph is limited to 512 expression nodes")
        match node["op"]:
            case "field":
                root, field_path = path(node)
                if root != array_argument or field_path[:-1] != collection_path:
                    raise UnsupportedGraph("selection fields must belong to the same one-level record collection")
                fields.add(field_path[-1])
                return {"op": "field", "name": field_path[-1]}
            case "argument":
                if node["name"] == array_argument:
                    raise UnsupportedGraph("a selection needs numeric fields, comparisons, or Boolean scalars")
                scalars.add(node["name"])
                return {"op": "scalar", "name": node["name"]}
            case "literal":
                return node
            case "abs" | "not":
                operand = normalize(node["value"])
                if node["op"] == "not" and not predicate_is_array(operand):
                    raise UnsupportedGraph("scalar bitwise inversion (~) is unsupported; invert a Boolean array mask instead")
                return {"op": node["op"], "value": operand}
            case "compare" | "and" | "or":
                result = {"op": node["op"], "left": normalize(node["left"]), "right": normalize(node["right"])}
                if "operator" in node:
                    result["operator"] = node["operator"]
                return result
        raise UnsupportedGraph("nested filters or reductions in a selection are unsupported")

    normalized = normalize(predicate)
    if not predicate_is_array(normalized):
        raise UnsupportedGraph("the record filter must be a Boolean array mask, not scalar indexing")
    if len(fields) > 8:
        raise UnsupportedGraph("the first compiled expression supports at most eight numeric fields")
    return FilterSumGraph(array_argument, collection_path, sum_field, normalized, tuple(sorted(fields)), tuple(sorted(scalars)))
