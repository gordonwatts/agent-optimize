"""Lossless numeric values with preflight sizing and bounded buffer copying."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import awkward as ak
import numpy as np

CHUNK_BYTES = 64 * 1024


class UnsupportedValue(ValueError):
    pass


@dataclass
class ValuePlan:
    metadata: dict
    nbytes: int
    value: object


def _array_description(array: ak.Array):
    if type(array) is not ak.Array:
        raise UnsupportedValue("custom Awkward Array subclasses are unsupported")
    if ak.backend(array) != "cpu":
        raise UnsupportedValue("only CPU-backed Awkward arrays are supported")
    if array.behavior or ak.behavior:
        raise UnsupportedValue("custom Awkward behaviors are unsupported")
    if array.attrs:
        raise UnsupportedValue("custom Awkward attributes are unsupported")
    buffers = {}
    counter = 0

    def visit(layout, depth=0):
        nonlocal counter
        key = f"node{counter}"
        counter += 1
        if layout.parameters:
            raise UnsupportedValue("Awkward layout parameters/custom records are unsupported")
        form = layout.form.to_dict()
        form["form_key"] = key
        if isinstance(layout, ak.contents.NumpyArray):
            if layout.data.dtype.kind not in "biuf" or layout.data.ndim != 1:
                raise UnsupportedValue("only one-dimensional Boolean, integer, or floating buffers are supported")
            buffers[f"{key}-data"] = layout.data.nbytes
        elif isinstance(layout, ak.contents.RecordArray):
            if layout.is_tuple:
                raise UnsupportedValue("tuple records are unsupported")
            form["contents"] = [visit(content, depth) for content in layout.contents]
        elif isinstance(layout, (ak.contents.ListOffsetArray, ak.contents.ListArray)):
            if depth:
                raise UnsupportedValue("only one level of variable-length lists is supported")
            if isinstance(layout, ak.contents.ListOffsetArray):
                buffers[f"{key}-offsets"] = layout.offsets.data.nbytes
            else:
                buffers[f"{key}-starts"] = layout.starts.data.nbytes
                buffers[f"{key}-stops"] = layout.stops.data.nbytes
            form["content"] = visit(layout.content, depth + 1)
        else:
            raise UnsupportedValue(f"unsupported Awkward layout: {type(layout).__name__}")
        return form

    form = visit(array.layout)
    return form, buffers


def plan_value(value, prefix: str, *, output=False) -> ValuePlan:
    """Inspect existing buffers without to_buffers, packing, or copying them."""
    if isinstance(value, ak.Array):
        form, sizes = _array_description(value)
        metadata = {"kind": "awkward", "form": form, "length": len(value)}
        if output:
            if not isinstance(value.layout, ak.contents.NumpyArray):
                raise UnsupportedValue("reference output must be a flat numeric Awkward or NumPy array")
            metadata["dtype"] = value.layout.data.dtype.str
        metadata["buffers"] = {
            key: {"file": f"{prefix}-{key}.bin", "nbytes": int(size)} for key, size in sizes.items()
        }
        return ValuePlan(metadata, sum(sizes.values()), value)
    if output:
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind not in "biuf":
            raise UnsupportedValue("reference output must be a flat numeric Awkward or NumPy array")
        return ValuePlan({
            "kind": "numpy", "dtype": value.dtype.str, "length": len(value),
            "buffers": {"data": {"file": f"{prefix}-data.bin", "nbytes": value.nbytes}},
        }, value.nbytes, value)
    if isinstance(value, np.generic) and value.dtype.kind in "biuf" and type(value) is value.dtype.type:
        metadata = {"kind": "scalar", "type": "numpy", "dtype": value.dtype.str, "hex": value.tobytes().hex()}
    elif type(value) is bool:
        metadata = {"kind": "scalar", "type": "bool", "value": value}
    elif type(value) is int:
        metadata = {"kind": "scalar", "type": "int", "value": str(value)}
    elif type(value) is float:
        metadata = {"kind": "scalar", "type": "float", "hex": struct.pack("<d", value).hex()}
    else:
        raise UnsupportedValue(f"unsupported input type: {type(value).__name__}")
    return ValuePlan(metadata, 0, value)


def buffer_chunks(buffer):
    """Yield at most CHUNK_BYTES of temporary bytes, including strided inputs."""
    array = np.asarray(buffer)
    iterator = np.nditer(
        array, flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"], order="C", buffersize=max(1, CHUNK_BYTES // array.dtype.itemsize),
    )
    for chunk in iterator:
        yield chunk.tobytes(order="C")


def value_buffers(plan: ValuePlan):
    if plan.metadata["kind"] == "awkward":
        form, length, buffers = ak.to_buffers(plan.value, byteorder="<")
        if form.to_dict() != plan.metadata["form"] or length != plan.metadata["length"]:
            raise UnsupportedValue("input layout changed while recording")
        return buffers
    if plan.metadata["kind"] == "numpy":
        return {"data": plan.value}
    return {}


def write_value(plan: ValuePlan, directory: Path) -> dict[str, str]:
    """Copy buffers only after the caller has reserved their full byte budget."""
    hashes = {}
    for key, buffer in value_buffers(plan).items():
        digest = hashlib.sha256()
        description = plan.metadata["buffers"][key]
        written = 0
        with (directory / description["file"]).open("wb") as stream:
            for chunk in buffer_chunks(buffer):
                stream.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != description["nbytes"]:
            raise UnsupportedValue("buffer size changed while recording")
        hashes[key] = digest.hexdigest()
    return hashes


def unchanged(plan: ValuePlan, hashes: dict[str, str], prefix: str) -> bool:
    try:
        current = plan_value(plan.value, prefix)
    except UnsupportedValue:
        return False
    if current.metadata != plan.metadata:
        return False
    for key, buffer in value_buffers(current).items():
        digest = hashlib.sha256()
        for chunk in buffer_chunks(buffer):
            digest.update(chunk)
        if digest.hexdigest() != hashes[key]:
            return False
    return True


def load_value(metadata: dict, directory: Path):
    kind = metadata["kind"]
    if kind == "scalar":
        match metadata["type"]:
            case "numpy":
                return np.frombuffer(bytes.fromhex(metadata["hex"]), dtype=np.dtype(metadata["dtype"]))[0]
            case "bool":
                return metadata["value"]
            case "int":
                return int(metadata["value"])
            case "float":
                return struct.unpack("<d", bytes.fromhex(metadata["hex"]))[0]
        raise ValueError("unknown scalar type")
    buffers = {}
    for key, description in metadata["buffers"].items():
        filename = description["file"]
        if Path(filename).name != filename:
            raise ValueError("buffer filename must be local to the capture")
        data = (directory / filename).read_bytes()
        if len(data) != description["nbytes"]:
            raise ValueError("recorded buffer size mismatch")
        buffers[key] = data
    if kind == "awkward":
        return ak.from_buffers(metadata["form"], metadata["length"], buffers, byteorder="<")
    if kind == "numpy":
        return np.frombuffer(buffers["data"], dtype=np.dtype(metadata["dtype"])).copy()
    raise ValueError(f"unknown recorded value kind: {kind}")


def matches_recording(value, metadata: dict, directory: Path) -> bool:
    prefix = "result"
    plan = plan_value(value, prefix, output=True)
    if plan.metadata != metadata:
        return False
    for key, buffer in value_buffers(plan).items():
        expected = hashlib.sha256()
        with (directory / metadata["buffers"][key]["file"]).open("rb") as stream:
            while chunk := stream.read(CHUNK_BYTES):
                expected.update(chunk)
        actual = hashlib.sha256()
        for chunk in buffer_chunks(buffer):
            actual.update(chunk)
        if actual.digest() != expected.digest():
            return False
    return True
