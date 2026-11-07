"""The public recording boundary and mode-selected ordinary dispatch."""

from collections.abc import Callable
from functools import wraps
import os
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def optimize(function: Callable[P, R]) -> Callable[P, R]:
    """Mark a function for recording and saved fast/careful implementations.

    Ordinary calls default to cached fast dispatch and never record or search.
    AGENT_OPTIMIZE_MODE=careful opts into repeated compatibility checks.
    An explicit CLI recording session
    snapshots supported inputs around the same reference execution.
    """

    fast_active = False

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal fast_active
        session = os.environ.get("AGENT_OPTIMIZE_SESSION")
        if session:
            from .recorder import record_call

            return record_call(session, function, args, kwargs)
        try:
            from .modes import dispatch_mode
            selected_mode = dispatch_mode()
            root = os.environ.get("AGENT_OPTIMIZE_ARTIFACT_ROOT")
            if not root:
                root = os.path.join(os.getcwd(), ".agent-optimize")
            if (selected_mode == "fast" and fast_active) or os.path.isdir(os.path.join(root, "accepted")):
                from pathlib import Path
                from .dispatch import dispatch_call

                used, result = dispatch_call(function, args, kwargs, Path(root), mode=selected_mode)
                if used:
                    fast_active = selected_mode == "fast"
                    return result
        except Exception:
            # Store discovery/import failures cannot disable the original call.
            # User references execute below, outside this recovery handler.
            pass
        return function(*args, **kwargs)

    return wrapper
