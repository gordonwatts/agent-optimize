"""Freeze the deliberately restricted reference without importing its script."""

import ast
from dataclasses import dataclass
import dis
import io
import hashlib
import importlib
import inspect
import math
import marshal
import struct
import sys
import sysconfig
from pathlib import Path
import textwrap
import tokenize
import types

from .store import json_bytes

MODULE_SYMBOLS = {
    "awkward": {"sum", "to_numpy", "values_astype", "argmin", "local_index"},
    "numpy": {"sum", "asarray", "abs", "sqrt", "maximum", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "bool_"},
    "math": {"fabs", "isfinite", "isnan", "isinf", "pi", "e", "tau"},
}
BUILTINS = {"abs"}


class UnsupportedReference(ValueError):
    pass


@dataclass(frozen=True)
class FrozenReference:
    identity: dict
    fingerprint: str
    source: str
    metadata: dict


def code_fingerprint(code):
    """Fingerprint executable contents independently of paths/line tables."""
    def payload(value):
        if isinstance(value, types.CodeType):
            return (
                value.co_code, value.co_names, value.co_varnames,
                value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount,
                value.co_freevars, value.co_cellvars, tuple(payload(item) for item in value.co_consts),
            )
        if isinstance(value, tuple):
            return tuple(payload(item) for item in value)
        if isinstance(value, frozenset):
            return tuple(sorted((payload(item) for item in value), key=repr))
        return value
    # Version 3+ encodes object-sharing references. Equal code read from a .pyc
    # and freshly compiled source can have different sharing (notably on 3.13).
    # Version 2 serializes values without those identity-dependent references.
    return hashlib.sha256(marshal.dumps(payload(code), 2)).hexdigest()


_installed_codes = {}


def _source_snapshot(code):
    """Read once and compile without execution; bypass inspect's line cache."""
    path = Path(code.co_filename).resolve()
    try:
        raw = path.read_bytes()
        key = (str(path), hashlib.sha256(raw).hexdigest())
        snapshot = _installed_codes.get(key)
        if snapshot is None:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            source = raw.decode(encoding)
            snapshot = (source, ast.parse(source), compile(source, str(path), "exec", dont_inherit=True))
            _installed_codes[key] = snapshot
    except (OSError, SyntaxError, ValueError) as error:
        raise UnsupportedReference("dependency source cannot be verified") from error
    return snapshot


def _source_code_matches(code, compiled=None):
    """Compile source only: never execute a module to verify its live code."""
    if compiled is None:
        _, _, compiled = _source_snapshot(code)

    def find(current):
        if current.co_name == code.co_name and current.co_firstlineno == code.co_firstlineno and current.co_qualname == code.co_qualname:
            return current
        for value in current.co_consts:
            if isinstance(value, types.CodeType):
                result = find(value)
                if result is not None:
                    return result
        return None

    original = find(compiled)
    return original is not None and code_fingerprint(original) == code_fingerprint(code)


def _trusted_function(value):
    path = Path(value.__code__.co_filename).resolve()
    module = getattr(value, "__module__", "")
    root = module.split(".")[0]
    if root in MODULE_SYMBOLS and root != "math":
        package = importlib.import_module(root)
        directory = Path(package.__file__).resolve().parent
    elif root in sys.stdlib_module_names:
        directory = Path(sysconfig.get_path("stdlib")).resolve()
    else:
        raise UnsupportedReference("replaced import callable is unsupported; use the installed library operation")
    if not path.is_relative_to(directory) or not _source_code_matches(value.__code__):
        raise UnsupportedReference("import callable does not match its installed library source")


def _binding_fingerprint(value):
    """Conservative state fingerprints, with no user callable replacements.

    Stock library wrappers use closures themselves. Their immutable state and
    direct globals are included; arbitrary user functions, mutable defaults,
    and mutable global state are unsupported rather than treated as equivalent.
    """
    if getattr(value, "__module__", "") == "awkward.operations.ak_to_numpy":
        raise UnsupportedReference("ak.to_numpy conversion dependencies are outside the verified subset; convert the result outside the decorated function")
    seen = {}

    def state(item, depth=0):
        if depth > 24 or len(seen) > 256:
            raise UnsupportedReference("import dependency state exceeds the supported verification bound")
        if type(item) is float:
            return {"literal_type": "float", "bits": struct.pack("<d", item).hex()}
        if _immutable(item):
            return {"literal_type": type(item).__name__, "value": repr(item)}
        if item is NotImplemented or item is Ellipsis:
            return {"builtin_constant": repr(item)}
        if type(item) is tuple:
            return {"tuple": [state(value, depth + 1) for value in item]}
        if isinstance(item, types.ModuleType):
            return {"module": item.__name__, "version": getattr(item, "__version__", "stdlib")}
        module = getattr(item, "__module__", type(item).__module__)
        name = getattr(item, "__qualname__", getattr(item, "__name__", type(item).__qualname__))
        result = {"module": module, "name": name}
        if type(item).__module__ == "typing" and not isinstance(item, type):
            return {**result, "typing": repr(item)}
        if type(item).__module__ == "awkward._nplikes.numpy_like" and type(item).__name__ == "NumpyMetadata":
            from awkward._nplikes.numpy_like import NumpyMetadata

            if type(item) is not NumpyMetadata or item is not NumpyMetadata.instance() or item.__dict__:
                raise UnsupportedReference("modified NumPy metadata dependency is unsupported")
            result["attributes"] = {key: state(getattr(item, key), depth + 1) for key in sorted(vars(NumpyMetadata)) if not key.startswith("_")}
            return result
        if isinstance(item, type) and module == "awkward._reducers" and name in {"Sum", "ArgMin"}:
            from awkward._reducers import KernelReducer

            allowed = {
                "name", "preferred_dtype", "needs_position", "__init__", "apply", "axis_none_reducer",
                "__module__", "__doc__", "__annotations__", "__parameters__", "__abstractmethods__",
                "__subclasshook__", "_is_protocol", "_abc_impl",
                "__firstlineno__", "__static_attributes__",
            }
            if name == "ArgMin":
                allowed.add("_dtype_for_kernel")
            if (
                type(item) is not type(KernelReducer) or item.__bases__ != (KernelReducer,)
                or set(vars(item)) - allowed or item.__abstractmethods__
                or item.__new__ is not object.__new__
                or item.__getattribute__ is not object.__getattribute__
                or item.__setattr__ is not object.__setattr__
                or any("_dtype" in vars(base) for base in item.__mro__)
            ):
                raise UnsupportedReference("modified sum reducer class structure is unsupported")
            def member(key):
                return next(vars(base)[key] for base in item.__mro__ if key in vars(base))
            policy = {key: member(key) for key in ("name", "preferred_dtype", "needs_position", "_use32", "_is_protocol")}
            if type(policy["name"]) is not str or any(type(policy[key]) is not bool for key in ("needs_position", "_use32", "_is_protocol")) or not isinstance(policy["preferred_dtype"], type):
                raise UnsupportedReference("modified sum reducer policy descriptors are unsupported")
            result["policy"] = {key: state(value, depth + 1) for key, value in policy.items()}
            result["class_metadata"] = {key: state(vars(item)[key], depth + 1) for key in ("__firstlineno__", "__static_attributes__") if key in vars(item)}
            methods = {}
            class_methods = {"_length_for_kernel", "_dtype_for_kernel", "_promote_integer_rank"}
            for method in ("__init__", "apply", "axis_none_reducer", *sorted(class_methods)):
                # Resolve the raw descriptor without running custom __get__ code.
                descriptor = member(method)
                if name == "ArgMin" and method == "__init__" and descriptor is object.__init__:
                    methods[method] = {"builtin": "object.__init__"}
                    continue
                expected = classmethod if method in class_methods else types.FunctionType
                if type(descriptor) is not expected:
                    raise UnsupportedReference(f"modified sum reducer method binding: {method}")
                function = descriptor.__func__ if expected is classmethod else descriptor
                if not isinstance(function, types.FunctionType):
                    raise UnsupportedReference(f"unsupported sum reducer method: {method}")
                methods[method] = state(function, depth + 1)
            result["methods"] = methods
            return result
        if isinstance(item, types.FunctionType):
            if id(item) in seen:
                return {"function_reference": seen[id(item)]}
            _trusted_function(item)
            seen[id(item)] = len(seen)
            result["code"] = code_fingerprint(item.__code__)
            result["defaults"] = state(item.__defaults__, depth + 1)
            result["keyword_defaults"] = {key: state(value, depth + 1) for key, value in sorted((item.__kwdefaults__ or {}).items())}
            result["closure"] = {key: state(cell.cell_contents, depth + 1) for key, cell in zip(item.__code__.co_freevars, item.__closure__ or (), strict=True)}
            instructions = list(dis.get_instructions(item))
            names = {instruction.argval for instruction in instructions if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}}
            function_builtins = item.__builtins__
            result["globals"] = {key: state(item.__globals__[key] if key in item.__globals__ else function_builtins[key], depth + 1) for key in sorted(names) if key in item.__globals__ or key in function_builtins}
            attributes = {}
            closure = dict(zip(item.__code__.co_freevars, (cell.cell_contents for cell in item.__closure__ or ()), strict=True))
            for index, instruction in enumerate(instructions):
                if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
                    owner = item.__globals__.get(instruction.argval, function_builtins.get(instruction.argval))
                elif instruction.opname == "LOAD_DEREF":
                    owner = closure.get(instruction.argval)
                else:
                    continue
                if not isinstance(owner, types.ModuleType):
                    continue
                parts = [instruction.argval]
                for following in instructions[index + 1:]:
                    if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                        break
                    parts.append(following.argval)
                    owner = getattr(owner, following.argval)
                    attributes[".".join(parts)] = state(owner, depth + 1)
                    if not isinstance(owner, types.ModuleType):
                        break
            result["module_attributes"] = attributes
            return result
        if isinstance(item, (type, types.BuiltinFunctionType)) and module.split(".")[0] in sys.stdlib_module_names | {"numpy", "awkward"}:
            return result
        # NumPy's native dispatchers/ufuncs can wrap Python implementations.
        if module.split(".")[0] == "numpy" and type(item).__module__.startswith("numpy"):
            import numpy as np

            if type(item) is np.ufunc:
                if getattr(np._core.umath, item.__name__, None) is not item:
                    raise UnsupportedReference("replaced NumPy ufunc is unsupported")
                metadata = getattr(item, "__dict__", {})
                if set(metadata) - {"__module__", "__qualname__", "__name__", "__doc__"} or not all(_immutable(value) for value in metadata.values()):
                    raise UnsupportedReference("mutable NumPy ufunc state is unsupported")
                result["metadata"] = metadata
                return result
            wrapped = getattr(item, "__wrapped__", None)
            if wrapped is not None:
                result["wrapped"] = state(wrapped, depth + 1)
            elif getattr(item, "__dict__", None):
                raise UnsupportedReference("mutable import dependency state is unsupported")
            return result
        raise UnsupportedReference(f"unsupported mutable/import dependency state: {type(item).__name__} ({module}.{name})")

    return state(value)


def function_identity(function) -> dict:
    code = function.__code__
    return {
        "module": function.__module__,
        "qualified_name": function.__qualname__,
        "source_file": str(Path(code.co_filename).resolve()),
        "first_line": code.co_firstlineno,
    }


def location_key(identity: dict) -> str:
    return hashlib.sha256(json_bytes(identity)).hexdigest()[:24]


def _immutable(value):
    if type(value) in (type(None), bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
    return type(value) is tuple and all(_immutable(item) for item in value)


def _validate_function(node, modules: dict[str, str], symbols: set[str], constants: set[str]):
    if not isinstance(node, ast.FunctionDef) or node.decorator_list:
        raise UnsupportedReference("only undecorated synchronous reference functions are replayable")
    if node.args.vararg or node.args.kwarg:
        raise UnsupportedReference("variadic reference parameters are unsupported")
    parameters = {argument.arg for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
    locals_ = set(parameters)
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise UnsupportedReference("only local-name assignments are supported; input mutation is unsupported")
            locals_.add(statement.targets[0].id)
        elif isinstance(statement, ast.Return):
            pass
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        else:
            raise UnsupportedReference(f"unsupported reference statement: {type(statement).__name__}")
    allowed = (
        ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.Return, ast.Expr,
        ast.Name, ast.Load, ast.Store, ast.Constant, ast.Attribute, ast.Subscript,
        ast.Slice, ast.Tuple, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Call, ast.keyword, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor, ast.USub, ast.UAdd,
        ast.Invert, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE,
    )
    available = locals_ | set(modules) | symbols | constants | BUILTINS
    for item in ast.walk(node):
        if not isinstance(item, allowed):
            raise UnsupportedReference(f"unsupported reference syntax: {type(item).__name__}")
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id not in available:
            raise UnsupportedReference(f"unsupported dependency: {item.id}")
        if isinstance(item, ast.Attribute):
            if item.attr.startswith("_"):
                raise UnsupportedReference("private attributes and external object internals are unsupported")
            if isinstance(item.value, ast.Name) and item.value.id in modules:
                if item.attr not in MODULE_SYMBOLS[modules[item.value.id]]:
                    raise UnsupportedReference(f"unsupported imported operation: {item.value.id}.{item.attr}")
        if isinstance(item, ast.Call):
            if any(keyword.arg is None for keyword in item.keywords) or any(isinstance(arg, ast.Starred) for arg in item.args):
                raise UnsupportedReference("expanded call arguments are unsupported")
            target = item.func
            if isinstance(target, ast.Name) and target.id in (BUILTINS | symbols) - locals_:
                continue
            if (
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id in modules and target.value.id not in locals_
                and target.attr in MODULE_SYMBOLS[modules[target.value.id]]
            ):
                continue
            raise UnsupportedReference("helper calls, methods, and external side effects are unsupported")


def validate_frozen_source(source: str):
    """Recheck the stored module before any worker executes its code."""
    tree = ast.parse(source)
    modules, symbols, constants = {}, set(), set()
    function = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in MODULE_SYMBOLS:
                    raise UnsupportedReference("unknown import in frozen reference")
                modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in MODULE_SYMBOLS:
                raise UnsupportedReference("unknown import in frozen reference")
            for alias in node.names:
                if alias.name not in MODULE_SYMBOLS[node.module]:
                    raise UnsupportedReference("unknown imported symbol in frozen reference")
                symbols.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise UnsupportedReference("nonliteral global in frozen reference") from error
            if not _immutable(value):
                raise UnsupportedReference("mutable global in frozen reference")
            constants.add(node.targets[0].id)
        elif isinstance(node, ast.FunctionDef) and function is None:
            function = node
        else:
            raise UnsupportedReference("unsupported top-level code in frozen reference")
    if function is None:
        raise UnsupportedReference("frozen reference has no function")
    if function.args.defaults or any(default is not None for default in function.args.kw_defaults):
        raise UnsupportedReference("frozen reference defaults must be bound into the recorded call")
    if function.returns or any(arg.annotation for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]):
        raise UnsupportedReference("frozen reference annotations are unsupported")
    _validate_function(function, modules, symbols, constants)
    return tree, function.name


def freeze_reference(function) -> FrozenReference:
    if function.__closure__:
        raise UnsupportedReference("closures are unsupported; pass captured values as explicit arguments")
    try:
        source_file, module_tree, compiled = _source_snapshot(function.__code__)
        if not _source_code_matches(function.__code__, compiled):
            raise UnsupportedReference("reference source does not match its live executable; reload the module before optimizing")
        definition = next(
            item for item in ast.walk(module_tree)
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == function.__code__.co_name
            and min([item.lineno, *(decorator.lineno for decorator in item.decorator_list)]) == function.__code__.co_firstlineno
        )
        start = min([definition.lineno, *(decorator.lineno for decorator in definition.decorator_list)])
        original_source = textwrap.dedent("".join(source_file.splitlines(keepends=True)[start - 1:definition.end_lineno]))
        tree = ast.parse(original_source)
    except (OSError, TypeError, SyntaxError, StopIteration) as error:
        raise UnsupportedReference("reference source is unavailable; define the function in a Python source file") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise UnsupportedReference("only synchronous Python function definitions are supported")
    node = tree.body[0]
    for decorator in node.decorator_list:
        if not (isinstance(decorator, ast.Name) and decorator.id == "optimize") and not (
            isinstance(decorator, ast.Attribute) and decorator.attr == "optimize"
        ):
            raise UnsupportedReference("additional decorators are unsupported")
    node.decorator_list = []
    node.returns = None
    node.type_params = []
    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        argument.annotation = None
    node.args.defaults = []
    node.args.kw_defaults = [None] * len(node.args.kwonlyargs)
    parameters = {arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
    assigned = {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}
    names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
    modules, symbols, constants, lines, dependencies, builtin_bindings = {}, set(), set(), [], {}, {}
    for name in sorted(names - parameters - assigned):
        if name in BUILTINS and name not in function.__globals__:
            binding = function.__builtins__.get(name)
            if not isinstance(binding, types.BuiltinFunctionType) or binding.__module__ != "builtins" or binding.__name__ != name:
                raise UnsupportedReference(f"builtin {name} was replaced; use the standard builtin operation")
            builtin_bindings[name] = _binding_fingerprint(binding)
            continue
        if name not in function.__globals__:
            raise UnsupportedReference(f"unsupported dependency: {name}")
        value = function.__globals__[name]
        if isinstance(value, types.ModuleType) and value.__name__ in MODULE_SYMBOLS:
            modules[name] = value.__name__
            lines.append(f"import {value.__name__} as {name}")
            dependencies[value.__name__] = getattr(value, "__version__", "stdlib")
        elif _immutable(value):
            constants.add(name)
            lines.append(f"{name} = {value!r}")
        else:
            found = None
            for module_name, choices in MODULE_SYMBOLS.items():
                module = importlib.import_module(module_name)
                for symbol in choices:
                    if value is getattr(module, symbol):
                        found = (module_name, symbol, getattr(module, "__version__", "stdlib"))
                        break
                if found:
                    break
            if found is None:
                raise UnsupportedReference(f"unsupported dependency {name}: helpers and mutable globals are unsupported")
            module_name, symbol, version = found
            lines.append(f"from {module_name} import {symbol} as {name}")
            symbols.add(name)
            dependencies[module_name] = version
    _validate_function(node, modules, symbols, constants)
    source = "\n".join([*lines, ast.unparse(node), ""])
    validate_frozen_source(source)
    identity = function_identity(function)
    bindings = {}
    for item in ast.walk(node):
        if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id in modules:
            bindings[f"{item.value.id}.{item.attr}"] = _binding_fingerprint(getattr(function.__globals__[item.value.id], item.attr))
        elif isinstance(item, ast.Name) and item.id in symbols:
            bindings[item.id] = _binding_fingerprint(function.__globals__[item.id])
    dependency_fingerprint = hashlib.sha256(json_bytes({"bindings": bindings, "dependencies": dependencies, "bindings_source": lines, "builtins": builtin_bindings})).hexdigest()
    body_fingerprint = code_fingerprint(function.__code__)
    fingerprint = hashlib.sha256(json_bytes({"source": source, "original_source": original_source, "dependencies": dependencies, "dependency_fingerprint": dependency_fingerprint, "code_fingerprint": body_fingerprint})).hexdigest()
    metadata = {
        "schema_version": 1, "identity": identity, "fingerprint": fingerprint,
        "function_name": node.name, "dependencies": dependencies,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "dependency_fingerprint": dependency_fingerprint,
        "code_fingerprint": body_fingerprint,
    }
    return FrozenReference(identity, fingerprint, source, metadata)
