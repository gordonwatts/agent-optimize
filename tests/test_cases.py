import awkward as ak
import numpy as np

from agent_optimize.cases import _constants, _evaluate, boundary_values, generate_cases
from agent_optimize.candidate import make_contract
from agent_optimize.operations import parse_filter_sum
from agent_optimize.serialization import plan_value
from test_operations import floating_ht, problem


def test_mandatory_cases_preserve_complete_schema_types_and_cover_boundaries():
    _, graph, contract, arguments = problem(floating_ht, np.float32)
    cases = {case.name: case for case in generate_cases(graph, contract, arguments, observed_max_length=400)}
    assert {"zero-events", "empty-lists", "varying-lengths", "larger-than-recording", "all-zero", "all-pass", "none-pass", "cancellation-order", "large-small-difficult-sums", "negative-zero-sums"} <= set(cases)
    assert len(cases["zero-events"].arguments["events"]) == 0
    assert ak.to_list(ak.num(cases["empty-lists"].arguments["events"].Jet)) == [0, 0, 0]
    assert max(ak.to_list(ak.num(cases["larger-than-recording"].arguments["events"].Jet))) > 400
    for case in cases.values():
        events = case.arguments["events"]
        assert events.layout.form.to_dict() == contract.schema
        assert events.fields == ["Jet", "run"]
        assert events.Jet.fields == ["pt", "eta", "unused"]
        assert ak.to_numpy(ak.flatten(events.Jet.pt)).dtype == np.dtype("float32")
    eta = ak.to_numpy(ak.flatten(cases["boundaries-eta"].arguments["events"].Jet.eta))
    assert 2.4 in eta
    assert np.nextafter(2.4, -np.inf) in eta
    assert np.nextafter(2.4, np.inf) in eta
    special = ak.to_numpy(ak.flatten(cases["nan-infinity-signed-zero-pt"].arguments["events"].Jet.pt))
    assert np.isnan(special).sum() == 2
    assert np.isposinf(special).any() and np.isneginf(special).any()
    assert ((special == 0) & np.signbit(special)).any()


def test_generated_all_pass_and_none_pass_are_real_selections():
    _, graph, contract, arguments = problem()
    cases = {case.name: case for case in generate_cases(graph, contract, arguments)}
    for name, expected in [("all-pass", True), ("none-pass", False)]:
        jets = cases[name].arguments["events"].Jet
        fields = {field: ak.to_numpy(ak.flatten(jets[field])) for field in graph.fields}
        mask = _evaluate(graph.predicate, fields, arguments)
        assert len(mask) > 0
        assert np.all(mask == expected)


def test_generation_preserves_original_argument_collection_and_field_names():
    source = """import awkward as ak
def energy_sum(records):
    objects = records.Objects
    selected = objects[(objects.weight > 4) & (abs(objects.rapidity) < 1.2)]
    return ak.sum(selected.weight, axis=1)
"""
    events = ak.Array([{"Objects": [{"weight": 8., "rapidity": 0.5}]}])
    graph = parse_filter_sum(source)
    contract = make_contract(graph, {"records": events}, plan_value(ak.Array([8.]), "result", output=True).metadata)
    cases = list(generate_cases(graph, contract, {"records": events}))
    assert cases
    assert all(set(case.arguments) == {"records"} for case in cases)
    assert all(case.arguments["records"].fields == ["Objects"] for case in cases)
    assert all(case.arguments["records"].Objects.fields == ["weight", "rapidity"] for case in cases)


def test_numpy_boolean_constants_are_excluded_and_uint64_boundaries_remain_exact():
    predicate = {"op": "and", "left": {"op": "scalar", "name": "enabled"}, "right": {"op": "compare", "operator": ">", "left": {"op": "field", "name": "pt"}, "right": {"op": "scalar", "name": "threshold"}}}
    threshold = np.uint64(2**64 - 2)
    constants = _constants(predicate, {"enabled": np.bool_(True), "threshold": threshold})
    assert constants == [2**64 - 2]
    values = boundary_values(np.uint64, constants)
    assert threshold in values
    assert np.uint64(2**64 - 3) in values
    assert np.uint64(2**64 - 1) in values
