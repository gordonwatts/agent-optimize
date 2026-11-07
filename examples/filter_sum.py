"""A deterministic, one-level jagged filter-and-sum computation."""

import argparse

import awkward as ak
import numpy as np

from agent_optimize import optimize


@optimize
def ht(events):
    jets = events.Jet
    selected = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
    return ak.sum(selected.pt, axis=1)


def make_events(count: int) -> ak.Array:
    """Build repeatable numeric data, preserving the schema for zero events."""
    lengths = np.arange(count, dtype=np.int64) % 5
    index = np.arange(int(lengths.sum()), dtype=np.int64)
    jets = ak.zip(
        {
            "pt": ak.unflatten((index * 17 % 100).astype(np.float64), lengths),
            "eta": ak.unflatten(((index % 7) - 3).astype(np.float64), lengths),
        }
    )
    return ak.zip({"Jet": jets}, depth_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=1000)
    arguments = parser.parse_args()
    if arguments.max_events < 0:
        parser.error("--max-events must be nonnegative")
    results = ht(make_events(arguments.max_events))
    print(f"events={len(results)} total_ht={float(ak.sum(results)):.1f}")


if __name__ == "__main__":
    main()
