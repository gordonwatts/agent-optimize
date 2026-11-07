"""Select the ordinary-call execution policy."""

import os

MODES = ("fast", "careful")


def dispatch_mode(value=None):
    mode = value if value is not None else os.environ.get("AGENT_OPTIMIZE_MODE", "fast")
    if mode not in MODES:
        raise ValueError("dispatch mode must be fast or careful (AGENT_OPTIMIZE_MODE)")
    return mode
