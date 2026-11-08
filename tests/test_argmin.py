import importlib.util
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from agent_optimize import optimize
from agent_optimize.candidate import adapt_inputs, compile_kernel, fused_candidate_source, make_contract, wrap_output
from agent_optimize.cases import generate_cases
from agent_optimize.evaluator import compare_results
from agent_optimize.experiment import freeze_reference
from agent_optimize.operations import graph_from_dict, parse_filter_sum, UnsupportedGraph
from agent_optimize.recorder import Recorder
from agent_optimize.serialization import plan_value
from agent_optimize.store import create_session, read_json
from agent_optimize.worker import replay_call


@optimize
def select_value(c):
    score = c.score
    return ak.sum(c.value[ak.local_index(score, axis=1) == ak.argmin(score, axis=1, keepdims=True)], axis=1)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_for(function, array):
    frozen = freeze_reference(function)
    graph = parse_filter_sum(frozen.source)
    arguments = {graph.array_argument: array}
    result = function(array)
    contract = make_contract(graph, arguments, plan_value(result, "result", output=True).metadata)
    kernel = compile_kernel(fused_candidate_source(graph, contract))
    return graph, contract, kernel


def test_first_minimum_empty_ties_nonfinite_and_exact_output():
    scores = ak.Array([[2., 1., 1.], [], [np.nan, 1.], [1., np.nan], [np.inf, np.inf], [-np.inf, 1.]])
    array = ak.zip({"score": scores, "value": ak.Array([[10., 20., 30.], [], [40., 50.], [60., 70.], [80., 90.], [100., 110.]])})
    graph, contract, kernel = compile_for(select_value.__wrapped__, array)
    from dataclasses import asdict
    assert graph_from_dict(asdict(graph)) == graph
    expected = select_value.__wrapped__(array)
    assert ak.to_list(expected) == [20., 0., 40., 60., 80., 100.]
    actual = wrap_output(contract, kernel(*adapt_inputs(contract, {"c": array})), len(array))
    assert compare_results(expected, actual)[0]
    for case in generate_cases(graph, contract, {"c": array}, observed_max_length=3):
        expected = select_value.__wrapped__(**case.arguments)
        actual = wrap_output(contract, kernel(*adapt_inputs(contract, case.arguments)), len(expected))
        assert compare_results(expected, actual)[0], case.name


def test_argmin_recording_and_frozen_replay(tmp_path):
    array = ak.zip({"score": ak.Array([[2., 1.], []]), "value": ak.Array([[10., 20.], []])})
    directory = create_session(tmp_path)
    assert ak.to_list(Recorder(directory).invoke(select_value.__wrapped__, (array,), {})) == [20., 0.]
    captures = list((directory / "calls").glob("*/capture.json"))
    assert len(captures) == 1
    assert replay_call(directory, read_json(captures[0])["call_id"])["status"] == "passed"


def test_argmin_reducer_policy_changes_reference_fingerprint(monkeypatch):
    from awkward._reducers import ArgMin
    before = freeze_reference(select_value.__wrapped__)
    monkeypatch.setattr(ArgMin, "needs_position", not ArgMin.needs_position)
    assert freeze_reference(select_value.__wrapped__).fingerprint != before.fingerprint


def test_q16_matches_supplied_reconstruction_and_compiles_exactly():
    pytest.importorskip("typer")
    root = Path(__file__).resolve().parents[1]
    example = load(root / "examples/q16_ttbar.py")
    original = load(root / "tests/fixtures/q16_reference.py")
    for batch in example.synthetic_batches(19, batch_size=7):
        assert compare_results(np.asarray(ak.to_numpy(original.reconstruct(*batch))), ak.to_numpy(example.reconstruct(*batch)))[0]
    rng = np.random.default_rng(16)
    lengths = [0, 1, 3, 12, 24, 0]
    array = ak.zip({name: ak.unflatten(rng.normal(100., 50., sum(lengths)), lengths)
                    for name in ("he", "hx", "hy", "hz", "le", "lx", "ly", "lz")})
    reference = example.select_ttbar_mass.__wrapped__
    graph, contract, kernel = compile_for(reference, array)
    for case in generate_cases(graph, contract, {"c": array}, observed_max_length=24):
        with np.errstate(all="ignore"):
            expected = reference(**case.arguments)
            actual = wrap_output(contract, kernel(*adapt_inputs(contract, case.arguments)), len(expected))
        assert compare_results(expected, actual)[0], case.name


def test_argmin_rejects_different_score_index_and_non_square_powers():
    source = freeze_reference(select_value.__wrapped__).source
    with pytest.raises(UnsupportedGraph):
        parse_filter_sum(source.replace("ak.local_index(score", "ak.local_index(c.value"))
    with pytest.raises(UnsupportedGraph):
        parse_filter_sum(source.replace("score = c.score", "score = c.score ** 3"))


