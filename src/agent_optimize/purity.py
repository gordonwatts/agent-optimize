"""Restrict reusable native kernels to read-only inputs and allocated outputs."""

import ast


def validate_reusable_source(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("reusable candidate must contain only its kernel function")
    function = tree.body[0]
    if function.decorator_list or function.type_params or function.returns:
        raise ValueError("candidate decorators, annotations, and type parameters are unsupported")
    allowed = (
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
        ast.Return, ast.For, ast.If, ast.IfExp, ast.Name, ast.Load, ast.Store,
        ast.Constant, ast.Subscript, ast.Attribute, ast.Call, ast.keyword,
        ast.Tuple, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.BitAnd, ast.BitOr, ast.BitXor, ast.USub, ast.UAdd, ast.Invert,
        ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    numeric = {"bool_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"}
    allocations = {"empty", "zeros"}
    operations = {"sqrt", "maximum", "isnan"}
    parameters = {"offsets", "columns", "scalars"}
    allocated = set()
    locals_ = set()
    for item in ast.walk(function):
        if isinstance(item, ast.Assign):
            if len(item.targets) != 1:
                raise ValueError("candidate chained assignments are unsupported")
            target = item.targets[0]
            if isinstance(target, ast.Name):
                locals_.add(target.id)
                if isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Attribute) and isinstance(item.value.func.value, ast.Name) and item.value.func.value.id == "np" and item.value.func.attr in allocations:
                    allocated.add(target.id)
        elif isinstance(item, ast.For) and isinstance(item.target, ast.Name):
            locals_.add(item.target.id)
    for item in ast.walk(tree):
        if not isinstance(item, allowed):
            raise ValueError(f"candidate side effects or unsupported syntax: {type(item).__name__}")
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id not in locals_ | parameters | {"np", "range", "len", "abs"}:
            raise ValueError(f"candidate external dependency is unsupported: {item.id}")
        if isinstance(item, ast.Attribute):
            if not isinstance(item.value, ast.Name) or item.value.id != "np" or item.attr not in numeric | allocations | operations:
                raise ValueError("candidate object access/methods are unsupported")
        if isinstance(item, ast.Call):
            target = item.func
            if isinstance(target, ast.Name) and target.id in {"range", "len", "abs"}:
                continue
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "np" and target.attr in numeric | allocations | operations:
                continue
            raise ValueError("candidate external calls and side effects are unsupported")
        if isinstance(item, (ast.Assign, ast.AugAssign)):
            target = item.targets[0] if isinstance(item, ast.Assign) else item.target
            if isinstance(target, ast.Name):
                if target.id in parameters | {"np", "range", "len", "abs"}:
                    raise ValueError("candidate may not reassign inputs or host dependencies")
                if target.id in allocated and not (
                    isinstance(item, ast.Assign) and isinstance(item.value, ast.Call)
                    and isinstance(item.value.func, ast.Attribute) and isinstance(item.value.func.value, ast.Name)
                    and item.value.func.value.id == "np" and item.value.func.attr in allocations
                ):
                    raise ValueError("candidate allocated output may not be rebound to an input alias")
            elif isinstance(target, ast.Subscript):
                root = target.value
                while isinstance(root, ast.Subscript):
                    root = root.value
                if not isinstance(root, ast.Name) or root.id not in allocated:
                    raise ValueError("candidate may only write separately allocated output buffers")
            else:
                raise ValueError("candidate assignment target is unsupported")
        if isinstance(item, ast.For) and (not isinstance(item.target, ast.Name) or item.target.id in parameters | allocated | {"np"}):
            raise ValueError("candidate loop may not rebind an input or output buffer")
        if isinstance(item, ast.Return) and (not isinstance(item.value, ast.Name) or item.value.id not in allocated):
            raise ValueError("candidate must return a separately allocated output array")
    if not allocated:
        raise ValueError("candidate has no separately allocated output")
