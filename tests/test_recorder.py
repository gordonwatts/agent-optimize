from concurrent.futures import ThreadPoolExecutor

import awkward as ak
import numpy as np
import pytest

import agent_optimize.recorder as recording
from agent_optimize.experiment import freeze_reference
from agent_optimize.recorder import Recorder
from agent_optimize.serialization import load_value
from agent_optimize.store import create_session, read_json


def identity_reference(events):
    return events


def sum_reference(events, /, threshold=30, *, enabled=True):
    selected = events[(events > threshold) & enabled]
    return ak.sum(selected, axis=1)


def scalar_reference(value, *, enabled=True):
    return np.asarray((value * enabled,))


def missing_field_reference(events):
    return events.missing


def nonflat_reference(events):
    return events


def entry_for(directory):
    state = read_json(directory / "session.json")
    assert len(state["functions"]) == 1
    return state, next(iter(state["functions"].values()))


def test_capture_round_trip_keyword_scalars_and_exact_output_metadata(tmp_path):
    directory = create_session(tmp_path)
    events = ak.Array([[10, 40], [], [31, 50]])
    result = Recorder(directory).invoke(sum_reference, (events,), {"threshold": np.int32(35), "enabled": np.bool_(True)})
    assert ak.to_list(result) == [40, 0, 50]
    state, entry = entry_for(directory)
    capture_directory = directory / "calls" / entry["captures"][0]
    metadata = read_json(capture_directory / "capture.json")
    assert metadata["positional_names"] == ["events"]
    assert metadata["reference_result"]["dtype"] == "<i8"
    assert load_value(metadata["arguments"]["threshold"], capture_directory).dtype == np.dtype("int32")
    assert load_value(metadata["arguments"]["enabled"], capture_directory).dtype == np.dtype("bool")
    assert ak.to_list(load_value(metadata["arguments"]["events"], capture_directory)) == ak.to_list(events)
    assert ak.to_list(load_value(metadata["reference_result"], capture_directory)) == [40, 0, 50]
    actual_size = sum(path.stat().st_size for subdirectory in ["calls", "references"] for path in (directory / subdirectory).rglob("*") if path.is_file())
    assert state["captured_bytes"] == actual_size
    assert not list((directory / "calls").glob(".partial-*"))


def test_numpy_output_scalar_inputs_keep_dtype(tmp_path):
    directory = create_session(tmp_path)
    result = Recorder(directory).invoke(scalar_reference, (np.float32(-0.0),), {"enabled": True})
    assert result.dtype == np.dtype("float32")
    _, entry = entry_for(directory)
    capture_directory = directory / "calls" / entry["captures"][0]
    metadata = read_json(capture_directory / "capture.json")["reference_result"]
    assert metadata["kind"] == "numpy"
    assert metadata["dtype"] == "<f4"
    assert load_value(metadata, capture_directory).tobytes() == result.tobytes()


def test_oversized_inputs_skip_before_any_buffer_conversion_or_copy(tmp_path, monkeypatch):
    directory = create_session(tmp_path, max_bytes=5000)
    events = ak.Array(np.arange(10_000))

    def unexpected(*args, **kwargs):
        raise AssertionError("oversized input reached buffer serialization")

    monkeypatch.setattr(recording, "write_value", unexpected)
    monkeypatch.setattr(ak, "to_buffers", unexpected)
    assert Recorder(directory).invoke(identity_reference, (events,), {}) is events
    state, entry = entry_for(directory)
    assert "byte_limit" in entry["skipped"]
    assert state["captured_bytes"] == 0
    assert not list((directory / "calls").iterdir())


def test_exhausted_call_budget_skips_before_buffer_copy_and_retains_reference(tmp_path, monkeypatch):
    directory = create_session(tmp_path, max_calls=1)
    recorder = Recorder(directory)
    first = ak.Array([1, 2])
    assert recorder.invoke(identity_reference, (first,), {}) is first

    def unexpected(*args, **kwargs):
        raise AssertionError("exhausted call budget reached buffer serialization")

    monkeypatch.setattr(recording, "write_value", unexpected)
    monkeypatch.setattr(ak, "to_buffers", unexpected)
    second = ak.Array([4, 5, 6])
    assert recorder.invoke(identity_reference, (second,), {}) is second
    _, entry = entry_for(directory)
    assert len(entry["captures"]) == 1
    assert entry["skipped"]["call_limit"]["count"] == 1


def test_output_overflow_discards_inputs_and_does_not_copy_output(tmp_path, monkeypatch):
    directory = create_session(tmp_path, max_bytes=14_000)
    events = ak.Array(np.arange(1024, dtype=np.int64))
    writes = []
    original = recording.write_value

    def observed(plan, path):
        writes.append(plan.metadata)
        return original(plan, path)

    monkeypatch.setattr(recording, "write_value", observed)
    assert Recorder(directory).invoke(identity_reference, (events,), {}) is events
    state, entry = entry_for(directory)
    assert len(writes) == 1  # only the pre-execution input snapshot
    assert "output_byte_limit" in entry["skipped"]
    assert state["captured_bytes"] == 0
    assert not list((directory / "calls").iterdir())
    assert not list((directory / "references").iterdir())


def test_observed_mutation_discards_original_snapshot_and_keeps_reference_result(tmp_path, monkeypatch):
    directory = create_session(tmp_path)
    # Bypass static source rejection to exercise the runtime mutation safeguard.
    frozen = freeze_reference(identity_reference)
    monkeypatch.setattr(recording, "freeze_reference", lambda function: frozen)
    source = np.array([1, 2, 3], dtype=np.int64)
    events = ak.Array(source)
    observed_snapshots = []
    original = recording.write_value

    def observed(plan, path):
        hashes = original(plan, path)
        observed_snapshots.append(ak.to_list(load_value(plan.metadata, path)))
        return hashes

    def mutating_reference(events):
        source[1] = 99
        return events

    monkeypatch.setattr(recording, "write_value", observed)
    assert Recorder(directory).invoke(mutating_reference, (events,), {}) is events
    assert observed_snapshots == [[1, 2, 3]]
    assert ak.to_list(events) == [1, 99, 3]
    state, entry = entry_for(directory)
    assert "input_mutation" in entry["skipped"]
    assert not entry["captures"]
    assert state["captured_bytes"] == 0
    assert not list((directory / "calls").iterdir())


def test_reference_exception_is_preserved_and_partial_capture_removed(tmp_path):
    directory = create_session(tmp_path)
    with pytest.raises(AttributeError, match="no field named 'missing'"):
        Recorder(directory).invoke(missing_field_reference, (ak.Array([1, 2]),), {})
    _, entry = entry_for(directory)
    assert "reference_exception" in entry["skipped"]
    assert not entry["captures"]
    assert not list((directory / "calls").iterdir())


@pytest.mark.parametrize("failure", [RuntimeError("original exception"), KeyboardInterrupt(), SystemExit(8)])
def test_original_exception_identity_survives_recording_errors(tmp_path, monkeypatch, failure):
    directory = create_session(tmp_path)
    monkeypatch.setattr(recording, "freeze_reference", lambda function: freeze_reference(identity_reference))
    calls = []

    def failing_reference(events):
        calls.append(1)
        raise failure

    with pytest.raises(type(failure)) as caught:
        Recorder(directory).invoke(failing_reference, (ak.Array([1]),), {})
    assert caught.value is failure
    assert calls == [1]
    assert not list((directory / "calls").iterdir())


def test_post_execution_cleanup_failure_does_not_execute_reference_twice(tmp_path, monkeypatch):
    directory = create_session(tmp_path)
    monkeypatch.setattr(recording, "freeze_reference", lambda function: freeze_reference(identity_reference))
    calls = []

    def observed_reference(events):
        calls.append(1)
        return events

    recorder = Recorder(directory)
    monkeypatch.setattr(recorder, "_cleanup", lambda stage: (_ for _ in ()).throw(OSError("cleanup failure")))
    values = ak.Array([1])
    assert recorder.invoke(observed_reference, (values,), {}) is values
    assert calls == [1]


def test_unsupported_input_and_output_do_not_change_reference(tmp_path):
    directory = create_session(tmp_path)
    recorder = Recorder(directory)
    values = [1, 2]
    assert recorder.invoke(identity_reference, (values,), {}) is values
    result = recorder.invoke(nonflat_reference, (ak.Array([[1], []]),), {})
    assert ak.to_list(result) == [[1], []]
    state = read_json(directory / "session.json")
    assert len(state["functions"]) == 2
    assert all("unsupported" in entry["skipped"] for entry in state["functions"].values())
    assert not list((directory / "calls").iterdir())


def test_separate_recorders_share_session_call_budget_under_concurrency(tmp_path):
    directory = create_session(tmp_path, max_calls=2)

    def call(index):
        value = ak.Array([index])
        return ak.to_list(Recorder(directory).invoke(identity_reference, (value,), {}))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, range(6)))
    assert results == [[index] for index in range(6)]
    _, entry = entry_for(directory)
    assert len(entry["captures"]) == 2
    assert entry["skipped"]["call_limit"]["count"] == 4
