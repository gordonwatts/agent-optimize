import awkward as ak
import numpy as np
import pytest

from agent_optimize.event_inputs import adapt_event_inputs, make_event_input_contract
from agent_optimize.operations import UnsupportedGraph


def inputs(lengths):
    count = sum(lengths)
    jets = ak.zip({name: ak.unflatten(np.arange(count, dtype=np.float64) + shift, lengths)
                   for shift, name in enumerate(("px", "py"))})
    lepton = ak.zip({"px": np.arange(len(lengths), dtype=np.float64),
                     "py": np.ones(len(lengths), dtype=np.float64)})
    met = ak.Array(np.arange(len(lengths), dtype=np.float64))
    return {"jets": jets, "lepton": lepton, "met": met}


def test_aligned_arrays_allow_new_event_counts_and_read_only_buffers():
    contract = make_event_input_contract(inputs([4, 5]))
    arrays = inputs([0, 3, 7])
    buffers = adapt_event_inputs(contract, arrays)
    assert buffers["jets"]["offsets"].tolist() == [0, 0, 3, 10]
    assert buffers["jets"]["fields"]["px"].tolist() == list(range(10))
    assert buffers["lepton"]["px"].tolist() == [0, 1, 2]
    assert buffers["met"][""].tolist() == [0, 1, 2]
    for buffer in (buffers["jets"]["offsets"], buffers["jets"]["fields"]["px"],
                   buffers["lepton"]["px"], buffers["met"][""]):
        with pytest.raises(ValueError):
            buffer[0] = 1


def test_rejects_misalignment_changed_form_and_unsupported_layout():
    arrays = inputs([4, 5])
    contract = make_event_input_contract(arrays)
    with pytest.raises(UnsupportedGraph, match="lengths"):
        adapt_event_inputs(contract, {**arrays, "met": arrays["met"][:1]})
    with pytest.raises(UnsupportedGraph, match="form"):
        adapt_event_inputs(contract, {**arrays, "met": ak.values_astype(arrays["met"], np.float32)})
    with pytest.raises(UnsupportedGraph, match="names"):
        adapt_event_inputs(contract, {"jets": arrays["jets"]})
    with pytest.raises(UnsupportedGraph):
        make_event_input_contract({"missing": ak.Array([1.0, None])})


def test_zero_events_and_sliced_jagged_offsets_keep_their_meaning():
    arrays = inputs([2, 4, 3])
    contract = make_event_input_contract(arrays)
    empty = {name: array[:0] for name, array in arrays.items()}
    assert len(adapt_event_inputs(contract, empty)["met"][""]) == 0
    sliced = {name: array[1:] for name, array in arrays.items()}
    buffers = adapt_event_inputs(contract, sliced)
    offsets = buffers["jets"]["offsets"]
    px = buffers["jets"]["fields"]["px"]
    assert [px[offsets[i]:offsets[i + 1]].tolist() for i in range(2)] == [[2., 3., 4., 5.], [6., 7., 8.]]


def test_q16_inputs_share_contract_across_batches():
    import importlib.util
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "q16_ttbar.py"
    spec = importlib.util.spec_from_file_location("q16_example", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batches = list(module.synthetic_batches(7, batch_size=4))
    names = ("jets", "lepton", "metx", "mety")
    first = dict(zip(names, batches[0]))
    second = dict(zip(names, batches[1]))
    contract = make_event_input_contract(first)
    buffers = adapt_event_inputs(contract, second)
    assert len(buffers["jets"]["offsets"]) == 4
    assert len(buffers["lepton"]["px"]) == len(buffers["metx"][""]) == 3
