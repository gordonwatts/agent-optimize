"""Measured q16 source experiments; these functions do not use JIT compilation."""

from functools import lru_cache

import awkward as ak
import numpy as np

MW = 80.4
MTOP = 172.5


@lru_cache(maxsize=32)
def assignment_indices(count):
    # The same pair / hadronic-b / leptonic-b ordering as the supplied source.
    return np.asarray([(i, j, k, l) for i in range(count) for j in range(i + 1, count)
                       for k in range(count) if k != i and k != j
                       for l in range(count) if l != i and l != j and l != k],
                      dtype=np.int64).T


@lru_cache(maxsize=32)
def hadronic_indices(count):
    return np.asarray([(i, j, k) for i in range(count) for j in range(i + 1, count)
                       for k in range(count) if k != i and k != j], dtype=np.int64).T


def has_optional_form(node):
    """Detect missing-value forms before conversion could discard NumPy masks."""
    if isinstance(node, dict):
        kind = node.get("class", "").lower()
        return ("option" in kind or "masked" in kind
                or any(has_optional_form(value) for value in node.values()))
    if isinstance(node, list):
        return any(has_optional_form(value) for value in node)
    return False


def grouped_numpy(jets, lepton, metx, mety, *, fallback, flatten_once=True, reuse_subexpressions=False, observe=None):
    """Vectorize equal-multiplicity event groups using existing NumPy/Awkward.

    Nonfinite inputs retain the original reference's NaN/argmin semantics.
    No compiled implementation or learned/saved constants are involved.
    """
    if any(has_optional_form(ak.to_layout(value).form.to_dict()) for value in (jets, lepton, metx, mety)):
        if observe is not None:
            observe["optional_fallback_groups"] = observe.get("optional_fallback_groups", 0) + 1
        return fallback(jets, lepton, metx, mety)
    counts = np.asarray(ak.to_numpy(ak.num(jets, axis=1)))
    flat = ak.flatten(jets, axis=1) if flatten_once else None
    columns = {key: np.asarray(ak.to_numpy(flat[key] if flatten_once else ak.flatten(jets[key], axis=1)))
               for key in ("px", "py", "pz", "e")}
    leptons = {key: np.asarray(ak.to_numpy(lepton[key])) for key in columns}
    mx, my = np.asarray(ak.to_numpy(metx)), np.asarray(ak.to_numpy(mety))
    if any(x.dtype != np.dtype(np.float64) for x in (*columns.values(), *leptons.values(), mx, my)):
        if observe is not None:
            observe["dtype_fallback_groups"] = observe.get("dtype_fallback_groups", 0) + 1
        return fallback(jets, lepton, metx, mety)
    if any(not np.all(np.isfinite(x)) for x in (*columns.values(), *leptons.values(), mx, my)):
        if observe is not None:
            observe["nonfinite_fallback_groups"] = observe.get("nonfinite_fallback_groups", 0) + 1
        return fallback(jets, lepton, metx, mety)
    output = np.zeros(len(counts), dtype=np.float64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for count in np.unique(counts):
        if count < 4:
            continue
        rows = np.flatnonzero(counts == count)
        positions = offsets[rows, None] + np.arange(count)
        jc = {key: value[positions] for key, value in columns.items()}
        px, py, pz, energy = (leptons[key][rows] for key in ("px", "py", "pz", "e"))
        met_x, met_y = mx[rows], my[rows]
        mu = MW * MW / 2.0 + px * met_x + py * met_y
        a = energy * energy - pz * pz
        disc = mu * mu - a * (met_x * met_x + met_y * met_y)
        root = energy * np.sqrt(np.maximum(disc, 0.0))
        plus, minus = (mu * pz + root) / a, (mu * pz - root) / a
        nuz = np.where(abs(plus) <= abs(minus), plus, minus)
        nue = np.sqrt(met_x * met_x + met_y * met_y + nuz * nuz)
        i, j, k, l = assignment_indices(int(count))
        if reuse_subexpressions:
            i, j, k = hadronic_indices(int(count))
            had_index = np.repeat(np.arange(len(i)), int(count) - 3)
            lep_index = np.arange(count)
        else:
            had_index = lep_index = None
        had = {key: (0 + ((0 + value[:, i]) + value[:, j])) + value[:, k]
               for key, value in jc.items()}
        lep = {key: (0 + ((0 + left) + right)[:, None]) + jc[key][:, lep_index if reuse_subexpressions else l]
               for key, left, right in zip(("px", "py", "pz", "e"),
                                            (px, py, pz, energy), (met_x, met_y, nuz, nue), strict=True)}
        def mass(v):
            return np.sqrt(np.maximum(0.0, v["e"] ** 2 - v["px"] ** 2 - v["py"] ** 2 - v["pz"] ** 2))
        had_score, lep_score = abs(mass(had) - MTOP), abs(mass(lep) - MTOP)
        score = had_score[:, had_index] + lep_score[:, l] if reuse_subexpressions else had_score + lep_score
        # Degenerate finite four-vectors may create NaN/inf intermediates.
        if not np.all(np.isfinite(score)):
            if observe is not None:
                observe["degenerate_fallback_groups"] = observe.get("degenerate_fallback_groups", 0) + 1
            return fallback(jets, lepton, metx, mety)
        best = np.argmin(score, axis=1)
        event = np.arange(len(rows))
        best_had = had_index[best] if reuse_subexpressions else best
        best_lep = l[best] if reuse_subexpressions else best
        combined = {key: (0 + had[key][event, best_had]) + lep[key][event, best_lep] for key in had}
        selected_mass = mass(combined)
        if not np.all(np.isfinite(selected_mass)):
            if observe is not None:
                observe["degenerate_fallback_groups"] = observe.get("degenerate_fallback_groups", 0) + 1
            return fallback(jets, lepton, metx, mety)
        output[rows] = selected_mass
    if observe is not None:
        observe["optimized_groups"] = observe.get("optimized_groups", 0) + 1
        observe["optimized_events"] = observe.get("optimized_events", 0) + len(counts)
    return ak.mask(ak.Array(output), counts >= 4)
