import ast

import awkward as ak
from awkward import sum as array_sum
import pytest

from agent_optimize import optimize
from agent_optimize.experiment import freeze_reference, UnsupportedReference, validate_frozen_source

THRESHOLD = 30
MUTABLE = [30]


@optimize
def constant_reference(events, threshold=THRESHOLD):
    selected = events[events > THRESHOLD]
    return array_sum(selected, axis=1)


def helper(events):
    return ak.sum(events, axis=1)


def helper_reference(events):
    return helper(events)


def mutable_reference(events):
    return ak.sum(events[events > MUTABLE[0]], axis=1)


def side_effect_reference(events):
    print("unexpected")
    return events


def control_flow_reference(events):
    if events:
        return events
    return events


def mutation_reference(events):
    events["new"] = events
    return events


def test_frozen_source_has_known_imports_constants_and_no_decorator_or_defaults(monkeypatch):
    frozen = freeze_reference(constant_reference.__wrapped__)
    tree, name = validate_frozen_source(frozen.source)
    assert "from awkward import sum as array_sum" in frozen.source
    assert "THRESHOLD = 30" in frozen.source
    assert "@optimize" not in frozen.source
    namespace = {}
    exec(compile(tree, "frozen.py", "exec"), namespace)
    events = ak.Array([[20, 40], [], [31]])
    assert ak.to_list(namespace[name](events, 30)) == [40, 0, 31]
    assert frozen.identity["qualified_name"] == "constant_reference"
    assert frozen.identity["source_file"].endswith("test_experiment.py")
    assert frozen.identity["first_line"] > 0
    monkeypatch.setitem(constant_reference.__wrapped__.__globals__, "THRESHOLD", 31)
    assert freeze_reference(constant_reference.__wrapped__).fingerprint != frozen.fingerprint


@pytest.mark.parametrize("function, diagnostic", [
    (helper_reference, "dependency"), (mutable_reference, "mutable globals"),
    (side_effect_reference, "print"), (control_flow_reference, "If"),
    (mutation_reference, "input mutation"),
])
def test_unsupported_dependencies_and_statements_have_diagnostics(function, diagnostic):
    with pytest.raises(UnsupportedReference, match=diagnostic):
        freeze_reference(function)


def test_closures_are_rejected():
    threshold = 30

    def reference(events):
        return ak.sum(events[events > threshold], axis=1)

    with pytest.raises(UnsupportedReference, match="closures"):
        freeze_reference(reference)


@pytest.mark.parametrize("source", [
    "import os\ndef f(x):\n    return x\n",
    "def f(x):\n    return __import__('os')\n",
    "X = []\ndef f(x):\n    return x\n",
    "import awkward as ak\ndef f(x):\n    return ak.to_json(x)\n",
    "def f(x):\n    return x.__class__\n",
    "print('side effect')\ndef f(x):\n    return x\n",
])
def test_worker_revalidates_restricted_source(source):
    with pytest.raises(UnsupportedReference):
        validate_frozen_source(source)
