"""Argument contract checked only when preparing an optimization call."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import awkward as ak
import numpy as np


class UnsupportedArgumentsError(ValueError):
    """A valid Python call is outside the initial optimization interface."""

    def __init__(self, parameter: str, reason: str) -> None:
        self.parameter = parameter
        self.reason = reason
        super().__init__(f"Cannot optimize argument {parameter!r}: {reason}")


def bind_supported_arguments(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> inspect.BoundArguments:
    """Bind positional/keyword calls to one canonical set of named arguments.

    Accept Awkward Arrays and Python/NumPy Boolean, integer, and floating
    scalars. Layout and computation restrictions belong to later recording
    and experiment checks. Unsupported inputs must never change normal calls.
    Invalid Python calls retain inspect's TypeError rather than being reported
    as an unsupported computation.
    """
    signature = inspect.signature(function)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise UnsupportedArgumentsError(
                name, "variadic parameters (*args and **kwargs) are unsupported"
            )
        value = bound.arguments[name]
        if type(value) in (ak.Array, bool, int, float):
            continue
        if isinstance(value, (np.bool_, np.integer, np.floating)) and type(value) is value.dtype.type:
            continue
        raise UnsupportedArgumentsError(
            name,
            f"{type(value).__name__} is unsupported; use an Awkward Array or a "
            "Boolean, integer, or floating scalar",
        )
    return bound
