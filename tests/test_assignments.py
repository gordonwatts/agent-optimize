import json
from dataclasses import asdict
from pathlib import Path

import awkward as ak
import numpy as np

from agent_optimize.candidate import adapt_inputs, compile_kernel, fused_candidate_source, make_contract, wrap_output
from agent_optimize.cases import generate_cases
from agent_optimize.evaluator import compare_results
from agent_optimize.experiment import freeze_reference
from agent_optimize.operations import graph_from_dict, parse_filter_sum
from agent_optimize.serialization import plan_value
from test_argmin import load
from agent_optimize import optimize


@optimize
def pair_assignment(events):
    objects = events.Objects
    indices = ak.argcartesian({"a": objects, "b": objects}, axis=1)
    chosen = indices[indices.a != indices.b]
    first = objects[chosen.a]
    second = objects[chosen.b]
    score = abs(first.x - second.x)
    value = first.x + events.bias[:, None]
    return ak.sum(value[ak.local_index(score, axis=1) == ak.argmin(score, axis=1, keepdims=True)], axis=1)


def test_generic_ordered_pair_ties_empty_and_sliced_events():
    objects = ak.zip({"x": ak.Array([[1., 2., 2.], [], [9.], [5., 7., 6.]])})
    array = ak.zip({"Objects": objects, "bias": ak.Array([10., 20., 30., 40.])}, depth_limit=1)
    reference = pair_assignment.__wrapped__
    source = freeze_reference(reference).source
    graph = parse_filter_sum(source)
    contract = make_contract(graph, {"events": array}, plan_value(reference(array), "r", output=True).metadata)
    kernel = compile_kernel(fused_candidate_source(graph, contract))
    assert ak.to_list(reference(array)) == [12., 0., 0., 45.]
    for value in (array, array[1:], array[:0]):
        actual = wrap_output(contract, kernel(*adapt_inputs(contract, {"events": value})), len(value))
        assert compare_results(reference(value), actual)[0]
    import pytest
    from agent_optimize.operations import UnsupportedGraph
    with pytest.raises(UnsupportedGraph):
        parse_filter_sum(source.replace("indices.a != indices.b", "indices.a & indices.b"))
    with pytest.raises(UnsupportedGraph):
        parse_filter_sum("import numpy as np\n" + source.replace("score =", "unused = np.sqrt(objects.x, out=objects.x)\n    score ="))


def test_argmin_dependency_fingerprint_stable_after_first_construction():
    from agent_optimize.experiment import _binding_fingerprint
    import typing
    from awkward._reducers import ArgMin
    # Both legitimate Protocol initialization states must identify one reference.
    previous = ArgMin.__init__
    try:
        ArgMin.__init__ = typing._no_init_or_replace_init
        before = _binding_fingerprint(ak.argmin)
        ak.argmin(ak.Array([[2., 1.]]), axis=1)
        assert _binding_fingerprint(ak.argmin) == before
    finally:
        ArgMin.__init__ = previous


def test_full_q16_compiled_reconstruction():
    example = load(Path(__file__).resolve().parents[1] / "examples/q16_ttbar.py")
    batch = next(example.synthetic_batches(12))
    array = ak.to_packed(ak.zip(dict(zip(("jets", "lepton", "metx", "mety"), batch)), depth_limit=1))
    reference = example.reconstruct_events.__wrapped__
    frozen = freeze_reference(reference)
    graph = graph_from_dict(json.loads(json.dumps(asdict(parse_filter_sum(frozen.source)), sort_keys=True)))
    expected = reference(array)
    contract = make_contract(graph, {"events": array}, plan_value(expected, "result", output=True).metadata)
    source = fused_candidate_source(graph, contract)
    from agent_optimize.purity import validate_reusable_source
    validate_reusable_source(source)
    kernel = compile_kernel(source)
    actual = wrap_output(contract, kernel(*adapt_inputs(contract, {"events": array})), len(array))
    assert compare_results(expected, actual)[0]
    for case in generate_cases(graph, contract, {"events": array}, observed_max_length=7):
        with np.errstate(all="ignore"):
            expected = reference(**case.arguments)
            actual = wrap_output(contract, kernel(*adapt_inputs(contract, case.arguments)), len(expected))
        assert compare_results(expected, actual)[0], case.name


def test_full_boundary_record_replay_accept_and_dispatch(tmp_path, monkeypatch):
    from agent_optimize.recorder import Recorder
    from agent_optimize.store import create_session
    from agent_optimize.replay import replay_session
    from agent_optimize.evaluator import evaluate_session
    from agent_optimize.dispatch import clear_dispatch_cache, observe_dispatch
    from agent_optimize.search import SearchLimits
    from test_dispatch import controlled
    example = load(Path(__file__).resolve().parents[1] / "examples/q16_ttbar.py")
    batch = next(example.synthetic_batches(5))
    array = ak.to_packed(ak.zip(dict(zip(("jets", "lepton", "metx", "mety"), batch)), depth_limit=1))
    session = create_session(tmp_path)
    expected = Recorder(session).invoke(example.reconstruct_events.__wrapped__, (array,), {})
    report = evaluate_session(session, replay_session(session, 0), timing_provider=controlled, dispatch_mode="fast", search_limits=SearchLimits(proposals=2))
    assert report["functions"][0]["status"] == "optimized", report["functions"]
    monkeypatch.setenv("AGENT_OPTIMIZE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_OPTIMIZE_MODE", "fast")
    clear_dispatch_cache()
    stats = {}
    with observe_dispatch(stats):
        actual = example.reconstruct_events(array)
    assert stats.get("used")
    assert compare_results(expected, actual)[0]
    changed = ak.with_field(array, ak.values_astype(array.metx, np.float32), "metx")
    clear_dispatch_cache()
    stats.clear()
    with observe_dispatch(stats):
        actual = example.reconstruct_events(changed)
    assert not stats.get("used")
    assert compare_results(example.reconstruct_events.__wrapped__(changed), actual)[0]
