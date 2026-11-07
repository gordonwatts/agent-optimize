"""Synthetic HEP workloads invoked through the real public decorator.

Run with ``python -m benchmarks.workloads --case quality_regions --events 128
2048``. These are synthetic fixtures, not physics validation or real detector
data. Combinatorics and matching are deliberate unsupported fallback controls.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np

from agent_optimize import optimize


@optimize
def integer_threshold(events, threshold=30):
    jets = events.Jet
    return ak.sum(jets[jets.energy > threshold].energy, axis=1)


@optimize
def central_ht(events, threshold=30.0, eta_limit=2.4):
    jets = events.Jet
    selected = jets[(jets.pt > threshold) & (abs(jets.eta) < eta_limit)]
    return ak.sum(selected.pt, axis=1)


@optimize
def quality_regions(events, threshold=35.0, eta_limit=2.4, btag_limit=0.6):
    jets = events.Jet
    mask = (jets.pt >= threshold) & ((abs(jets.eta) < eta_limit) | (jets.btag > btag_limit)) & ~jets.bad
    return ak.sum(jets[mask].pt, axis=1)


@optimize
def integer_quality(events, minimum_score=3, enabled=True):
    jets = events.Jet
    mask = ((jets.score >= minimum_score) | (jets.energy > 100)) & ~jets.bad & enabled
    return ak.sum(jets[mask].energy, axis=1)


@optimize
def multi_field_regions(events, threshold=40.0, eta_limit=2.5, btag_limit=0.7, mass_limit=15.0):
    jets = events.Jet
    mask = (jets.pt > threshold) & (abs(jets.eta) <= eta_limit) & ((jets.btag >= btag_limit) | (jets.mass > mass_limit)) & ~jets.bad
    return ak.sum(jets[mask].energy, axis=1)


@optimize
def dijet_combinatorics(events, minimum_pair_pt=80.0):
    jets = events.Jet
    pairs = ak.combinations(jets, 2, fields=["left", "right"])
    pair_pt = pairs.left.pt + pairs.right.pt
    return ak.sum(pair_pt[pair_pt > minimum_pair_pt], axis=1)


@optimize
def nearest_jet_matching(events, radius=0.4):
    pairs = ak.cartesian({"jet": events.Jet, "reference": events.RefJet}, axis=1, nested=True)
    delta_eta = pairs.jet.eta - pairs.reference.eta
    delta_phi = (pairs.jet.phi - pairs.reference.phi + np.pi) % (2 * np.pi) - np.pi
    distance_squared = delta_eta * delta_eta + delta_phi * delta_phi
    nearest = ak.min(distance_squared, axis=2, initial=np.inf, mask_identity=False)
    return ak.sum(nearest < radius * radius, axis=1)


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    supported: bool
    function: object


WORKLOADS = {
    item.name: item for item in [
        Workload("integer_threshold", "One-field integer filter and sum", True, integer_threshold),
        Workload("central_ht", "Floating pT sum with threshold and absolute eta", True, central_ht),
        Workload("quality_regions", "Four-field floating sum with Boolean inversion and OR", True, quality_regions),
        Workload("integer_quality", "Integer sum with mixed fields and Boolean scalar", True, integer_quality),
        Workload("multi_field_regions", "Six-field selection and integer sum with four scalar thresholds", True, multi_field_regions),
        Workload("dijet_combinatorics", "Unique dijet pair pT selection; unsupported control", False, dijet_combinatorics),
        Workload("nearest_jet_matching", "Wrapped delta-phi nearest jet matching; unsupported control", False, nearest_jet_matching),
    ]
}


def make_events(event_count, seed=20261106):
    """Stable schemas, zero-length events, variable multiplicities, finite values."""
    if event_count < 0:
        raise ValueError("event_count must be nonnegative")
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 13, size=event_count, dtype=np.int64)
    if event_count:
        lengths[0] = 0
    if event_count > 1:
        lengths[1] = 32

    def objects(counts):
        size = int(counts.sum())
        columns = {
            "pt": rng.uniform(0, 250, size=size),
            "eta": rng.uniform(-4, 4, size=size),
            "phi": rng.uniform(-np.pi, np.pi, size=size),
            "mass": rng.uniform(0, 40, size=size),
            "btag": rng.uniform(0, 1, size=size),
            "bad": rng.random(size=size) < 0.1,
            "score": rng.integers(-2, 10, size=size, dtype=np.int32),
            "energy": rng.integers(0, 500, size=size, dtype=np.int64),
        }
        return ak.zip({name: ak.unflatten(column, counts) for name, column in columns.items()})

    jets = objects(lengths)
    reference_lengths = rng.integers(0, 9, size=event_count, dtype=np.int64)
    if event_count:
        reference_lengths[0] = 0
    return ak.zip({"Jet": jets, "RefJet": objects(reference_lengths), "event_number": np.arange(event_count, dtype=np.int64)}, depth_limit=1)


def array_digest(array):
    form, length, buffers = ak.to_buffers(array)
    digest = hashlib.sha256(form.to_json().encode())
    digest.update(str(length).encode())
    for name, buffer in sorted(buffers.items()):
        digest.update(name.encode())
        digest.update(memoryview(np.ascontiguousarray(buffer)).cast("B"))
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=WORKLOADS, required=True)
    parser.add_argument("--events", nargs="+", type=int, default=[128, 2048])
    parser.add_argument("--seed", type=int, default=20261106)
    options = parser.parse_args(argv)
    if any(count < 0 for count in options.events):
        parser.error("event counts must be nonnegative")
    workload = WORKLOADS[options.case]
    measurements = []
    for event_count in options.events:
        events = make_events(event_count, options.seed)
        result = workload.function(events)
        measurements.append({"events": event_count, "jets": int(ak.sum(ak.num(events.Jet))), "input_sha256": array_digest(events), "result_sha256": array_digest(result), "output_events": len(result)})
    print(json.dumps({"kind": "synthetic-workload", "case": workload.name, "supported_graph": workload.supported, "seed": options.seed, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "measurements": measurements}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
