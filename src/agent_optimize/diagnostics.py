"""Bounded, readable diagnostics for model-authored code failures."""

import json
import re

MAX_DIAGNOSTIC_BYTES = 24 * 1024
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def code_diagnostic(message):
    """Keep compiler detail and paths, excluding terminal styling.

    Bound the encoded JSON size so Unicode escapes cannot overflow the worker
    output envelope. These messages concern compilation/source validation only;
    runtime exceptions and recorded inputs are not passed through this helper.
    """
    text = ANSI.sub("", message)
    def size(value):
        return len(json.dumps(value, ensure_ascii=True).encode("utf-8"))
    if size(text) <= MAX_DIAGNOSTIC_BYTES:
        return text
    notice = "\n[Diagnostic truncated at the 24 KiB JSON size limit.]"
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if size(text[:middle] + notice) <= MAX_DIAGNOSTIC_BYTES:
            low = middle
        else:
            high = middle - 1
    return text[:low] + notice
