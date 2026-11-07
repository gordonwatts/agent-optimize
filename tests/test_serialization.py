import struct

import awkward as ak
import numpy as np
import pytest

from agent_optimize.serialization import (
    buffer_chunks, CHUNK_BYTES, load_value, matches_recording, plan_value,
    unchanged, UnsupportedValue, write_value,
)


@pytest.mark.parametrize("dtype", [np.int32, np.uint64, np.float32, np.float64, np.bool_])
def test_awkward_round_trip_exact_buffers_and_zero_length_schema(tmp_path, dtype):
    data = np.array([0, 1, 1, 0], dtype=dtype)
    values = ak.zip({"Jet": ak.zip({"pt": ak.unflatten(data, [2, 0, 2])})}, depth_limit=1)
    for index, value in enumerate([values, values[:0]]):
        plan = plan_value(value, f"arg{index}")
        write_value(plan, tmp_path)
        restored = load_value(plan.metadata, tmp_path)
        assert len(restored) == len(value)
        assert restored.layout.form == value.layout.form
        assert ak.to_list(restored) == ak.to_list(value)
        form, length, buffers = ak.to_buffers(restored)
        assert form.to_dict() == plan.metadata["form"]
        original_buffers = ak.to_buffers(value)[2]
        assert sum(buffer.nbytes for buffer in original_buffers.values()) == plan.nbytes
        for key, buffer in ak.to_buffers(value)[2].items():
            assert (tmp_path / plan.metadata["buffers"][key]["file"]).read_bytes() == buffer.tobytes()


@pytest.mark.parametrize("value", [
    True, False, 2**1000, -4, -0.0,
    struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0],
    np.bool_(True), np.int16(-3), np.uint64(2**64 - 1), np.float32(-0.0),
])
def test_scalar_round_trip_preserves_types_dtypes_and_bits(tmp_path, value):
    plan = plan_value(value, "arg0")
    assert plan.nbytes == 0
    restored = load_value(plan.metadata, tmp_path)
    assert type(restored) is type(value)
    if isinstance(value, np.generic):
        assert restored.dtype == value.dtype
        assert restored.tobytes() == value.tobytes()
    elif isinstance(value, float):
        assert struct.pack("<d", restored) == struct.pack("<d", value)
    else:
        assert restored == value


@pytest.mark.parametrize("value", [ak.Array([1, None]), ak.Array([1, "x"]), ak.Array([[[1]]]), ak.Array([]), ak.Array([1j]), ak.Array(["text"])])
def test_unsupported_layout_diagnostics(value):
    with pytest.raises(UnsupportedValue):
        plan_value(value, "arg0")


def test_custom_behaviors_and_parameters_are_rejected():
    with pytest.raises(UnsupportedValue, match="behaviors"):
        plan_value(ak.Array([1], behavior={"custom": object()}), "arg0")
    with pytest.raises(UnsupportedValue, match="parameters"):
        plan_value(ak.with_name(ak.Array([{"x": 1}]), "Custom"), "arg0")
    with pytest.raises(UnsupportedValue, match="attributes"):
        plan_value(ak.Array([1], attrs={"threshold": 30}), "arg0")


@pytest.mark.parametrize("container", [np.asarray, ak.Array])
def test_flat_outputs_preserve_kind_dtype_and_bitwise_values(tmp_path, container):
    bits = np.array([0x80000000, 0x7FC00001, 0x7FC00002], dtype=np.uint32)
    output = container(bits.view(np.float32))
    plan = plan_value(output, "result", output=True)
    write_value(plan, tmp_path)
    restored = load_value(plan.metadata, tmp_path)
    assert type(restored) is type(output)
    assert plan.metadata["dtype"] == "<f4"
    assert matches_recording(restored, plan.metadata, tmp_path)
    changed = container(np.array([0, 0x7FC00001, 0x7FC00002], dtype=np.uint32).view(np.float32))
    assert not matches_recording(changed, plan.metadata, tmp_path)


@pytest.mark.parametrize("output", [3, [1, 2], ak.Array([[1], [2]]), np.ones((2, 2)), np.array([1j])])
def test_nonflat_or_nonnumeric_outputs_are_rejected(output):
    with pytest.raises(UnsupportedValue, match="flat numeric"):
        plan_value(output, "result", output=True)


def test_chunked_write_handles_strides_without_large_temporary_copies(tmp_path):
    values = np.arange(100_000, dtype=np.int64)[::2]
    chunks = list(buffer_chunks(values))
    assert max(map(len, chunks)) <= CHUNK_BYTES
    assert b"".join(chunks) == values.tobytes()
    plan = plan_value(values, "result", output=True)
    write_value(plan, tmp_path)
    assert np.array_equal(load_value(plan.metadata, tmp_path), values)


def test_input_mutation_detects_original_buffer_change(tmp_path):
    source = np.array([1, 2, 3], dtype=np.int64)
    values = ak.Array(source)
    plan = plan_value(values, "arg0")
    hashes = write_value(plan, tmp_path)
    assert unchanged(plan, hashes, "arg0")
    source[1] = 99
    assert not unchanged(plan, hashes, "arg0")
    assert ak.to_list(load_value(plan.metadata, tmp_path)) == [1, 2, 3]
