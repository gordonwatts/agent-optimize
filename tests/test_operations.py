from contextlib import nullcontext

import awkward as ak
import numpy as np
import pytest

from agent_optimize.candidate import adapt_inputs, make_contract, wrap_output
from agent_optimize.experiment import freeze_reference, UnsupportedReference
from agent_optimize.operations import parse_filter_sum, UnsupportedGraph
from agent_optimize.serialization import plan_value, UnsupportedValue


def integer_ht(events, threshold=30):
    jets = events.Jet
    selected = jets[(jets.pt > threshold) & (abs(jets.eta) < 2.4)]
    return ak.sum(selected.pt, axis=1)


def floating_ht(events):
    jets = events.Jet
    selected = jets[abs(jets.eta) < 2.4]
    return ak.sum(selected.pt, axis=1)


def numpy_ht(events):
    jets = events["Jet"]
    mask = (jets["pt"] >= 30) | ~(abs(jets["eta"]) <= 2.4)
    return ak.to_numpy(ak.sum(jets["pt"][mask], axis=1))


def make_events(dtype=np.int64, count=5):
    lengths = np.array([2, 0, 3], dtype=np.int64) if count == 5 else np.array([count], dtype=np.int64)
    jets = ak.zip({
        "pt": ak.unflatten(np.asarray(np.arange(count) + 1, dtype=dtype), lengths),
        "eta": ak.unflatten(np.zeros(count, dtype=np.float64), lengths),
        "unused": ak.unflatten(np.zeros(count, dtype=np.int16), lengths),
    })
    return ak.zip({"Jet": jets, "run": np.zeros(len(lengths), dtype=np.uint32)}, depth_limit=1)


def problem(function=integer_ht, dtype=np.int64):
    frozen = freeze_reference(function)
    graph = parse_filter_sum(frozen.source)
    arguments = {"events": make_events(dtype)}
    if function is integer_ht:
        arguments["threshold"] = 30
    result = function(**arguments)
    contract = make_contract(graph, arguments, plan_value(result, "result", output=True).metadata)
    return frozen, graph, contract, arguments


def test_graph_parses_documented_fields_selection_scalars_and_sum():
    _, graph, _, _ = problem()
    assert graph.array_argument == "events"
    assert graph.collection_path == ("Jet",)
    assert graph.sum_field == "pt"
    assert graph.fields == ("eta", "pt")
    assert graph.scalar_arguments == ("threshold",)
    assert graph.predicate["op"] == "and"
    assert graph.predicate["left"]["operator"] == ">"
    assert graph.predicate["right"]["left"]["op"] == "abs"


def test_field_subscripts_boolean_or_inversion_and_numpy_result():
    source = """import awkward as ak
def ht(events):
    jets = events['Jet']
    mask = (jets['pt'] >= 30) | ~(abs(jets['eta']) <= 2.4)
    return ak.to_numpy(ak.sum(jets['pt'][mask], axis=1))
"""
    graph = parse_filter_sum(source)
    arguments = {"events": make_events()}
    contract = make_contract(graph, arguments, plan_value(numpy_ht(**arguments), "result", output=True).metadata)
    assert graph.predicate["op"] == "or"
    assert graph.predicate["right"]["op"] == "not"
    assert contract.output_kind == "numpy"


def test_unverified_to_numpy_conversion_keeps_reference_only():
    with pytest.raises(UnsupportedReference, match="conversion dependencies are outside the verified subset"):
        freeze_reference(numpy_ht)


@pytest.mark.parametrize("mask", [
    "~enabled", "~(threshold > 0)",
    "(jets.pt > 30) & ~enabled", "(jets.pt > 30) & ~(threshold > 0)",
    "~((threshold > 0) & enabled)", "enabled", "threshold > 0", "True",
])
def test_scalar_inversion_and_scalar_indexing_are_rejected(mask):
    source = f"import awkward as ak\ndef ht(events, enabled, threshold):\n    jets = events.Jet\n    return ak.sum(jets[{mask}].pt, axis=1)\n"
    with pytest.raises(UnsupportedGraph, match="scalar bitwise inversion|scalar indexing"):
        parse_filter_sum(source)


@pytest.mark.parametrize("enabled, expected", [(False, [50]), (np.bool_(False), [40])])
def test_scalar_inversion_reference_semantics_are_preserved_and_never_compiled(enabled, expected):
    source = """import awkward as ak
def ht(events, enabled):
    jets = events.Jet
    selected = jets[(jets.pt > 30) & ~enabled]
    return ak.sum(selected.pt, axis=1)
"""
    namespace = {}
    exec(source, namespace)
    events = ak.Array([{"Jet": [{"pt": 10}, {"pt": 40}]}])
    # Python bool inversion makes integer indices; np.bool_ inversion is logical.
    # Conservatively reject both instead of assigning them one graph meaning.
    with pytest.warns(DeprecationWarning) if type(enabled) is bool else nullcontext():
        result = namespace["ht"](events, enabled)
    assert ak.to_list(result) == expected
    with pytest.raises(UnsupportedGraph, match="scalar bitwise inversion"):
        parse_filter_sum(source)


@pytest.mark.parametrize("body", [
    "return ak.sum(events.Jet.pt, axis=1)",
    "jets = events.Jet\n    return ak.sum(jets[jets.pt > 30].pt, axis=0)",
    "jets = events.Jet\n    return ak.sum(jets[jets.pt > 30].pt, axis=1) + 1",
    "jets = events.Jet[:3]\n    return ak.sum(jets[jets.pt > 30].pt, axis=1)",
    "jets = events.Jet\n    selected = jets[jets.pt > 30]\n    return ak.sum(selected[selected.eta < 2.4].pt, axis=1)",
])
def test_unsupported_graphs_have_actionable_rejection(body):
    source = "import awkward as ak\ndef ht(events):\n    " + body + "\n"
    with pytest.raises((UnsupportedGraph, UnsupportedReference)):
        parse_filter_sum(source)


def test_readonly_adapter_preserves_structure_and_allows_different_event_counts_lengths():
    _, graph, contract, arguments = problem()
    offsets, columns, scalars = adapt_inputs(contract, arguments)
    assert scalars == (30,)
    assert not offsets.flags.writeable
    assert all(not column.flags.writeable for column in columns)
    assert len(offsets) - 1 == 3
    changed = {"events": make_events(count=37), "threshold": 50}
    offsets, columns, scalars = adapt_inputs(contract, changed)
    assert len(offsets) - 1 == 1
    assert offsets[-1] == 37
    assert scalars == (50,)
    changed["events"] = ak.with_field(changed["events"], np.ones(1, dtype=np.int64), "run")
    with pytest.raises(UnsupportedGraph, match="schema|fields/nesting/dtypes"):
        adapt_inputs(contract, changed)


def test_integer_scalar_outside_numba_range_is_diagnostic():
    _, _, contract, arguments = problem()
    with pytest.raises(UnsupportedGraph, match="64-bit range"):
        adapt_inputs(contract, {**arguments, "threshold": 2**80})


def test_invalid_offsets_are_rejected_before_native_code():
    _, graph, _, _ = problem()
    jets = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 100], dtype=np.int64)),
        ak.contents.RecordArray([ak.contents.NumpyArray(np.array([1], dtype=np.int64)), ak.contents.NumpyArray(np.array([0.], dtype=np.float64))], ["pt", "eta"]),
    )
    values = ak.Array(ak.contents.RecordArray([jets], ["Jet"]))
    with pytest.raises(UnsupportedGraph, match="invalid Awkward"):
        make_contract(graph, {"events": values, "threshold": 30}, {"kind": "awkward", "length": 1, "dtype": "<i8"})


def test_missing_values_behaviors_and_non_cpu_inputs_are_rejected():
    _, graph, _, _ = problem()
    for events in [ak.Array([{"Jet": [{"pt": None, "eta": 0.}]}]), ak.Array([{"Jet": [{"pt": 1, "eta": 0.}]}], behavior={"custom": object()})]:
        with pytest.raises(UnsupportedValue):
            make_contract(graph, {"events": events, "threshold": 30}, {"kind": "awkward", "length": 1, "dtype": "<i8"})
    values = make_events().layout.to_typetracer(forget_length=True)
    with pytest.raises(UnsupportedValue, match="CPU"):
        make_contract(graph, {"events": ak.Array(values), "threshold": 30}, {"kind": "awkward", "length": 1, "dtype": "<i8"})


@pytest.mark.parametrize("raw, reason", [([1, 2, 3], "flat NumPy"), (np.array([1], dtype=np.int64), "events"), (np.ones(3, dtype=np.float64), "dtype"), (np.ones((3, 1), dtype=np.int64), "flat NumPy")])
def test_host_output_adapter_rejects_wrong_interface(raw, reason):
    _, _, contract, _ = problem()
    with pytest.raises(ValueError, match=reason):
        wrap_output(contract, raw, 3)
