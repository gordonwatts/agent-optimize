"""Unmodified numerical reconstruction from the supplied q16 script."""

import awkward as ak
import numpy as np

MW = 80.4
MTOP = 172.5


def mass(v):
    return np.sqrt(np.maximum(0.0, v.e ** 2 - v.px ** 2 - v.py ** 2 - v.pz ** 2))


def add(*vectors):
    return ak.zip({component: sum(v[component] for v in vectors)
                   for component in ("px", "py", "pz", "e")})


def neutrino_pz(lepton, metx, mety):
    # Solve (p_l + p_nu)^2 = m_W^2 with massless neutrino and measured MET.
    px, py, pz, energy = lepton.px, lepton.py, lepton.pz, lepton.e
    mu = MW * MW / 2.0 + px * metx + py * mety
    a = energy * energy - pz * pz
    disc = mu * mu - a * (metx * metx + mety * mety)
    root = energy * np.sqrt(np.maximum(disc, 0.0))
    plus = (mu * pz + root) / a
    minus = (mu * pz - root) / a
    return ak.where(abs(plus) <= abs(minus), plus, minus)


def reconstruct(jets, lepton, metx, mety):
    """Return one mass per event using jagged arrays of jet assignments.

    Inputs have shapes events * var * {px,py,pz,e} for jets and
    events * {px,py,pz,e} for the lepton. Candidate order matches the original
    pair/b-had/b-lep loops, so argmin also preserves first-candidate tie breaking.
    Call on bounded event batches: assignment counts grow as n*(n-1)*(n-2)*(n-3)/2.
    """
    nuz = neutrino_pz(lepton, metx, mety)
    nu = ak.zip({"px": metx, "py": mety, "pz": nuz,
                 "e": np.sqrt(metx * metx + mety * mety + nuz * nuz)})
    lep_w = add(lepton, nu)
    indexed = ak.with_field(jets, ak.local_index(jets, axis=1), "index")
    pairs = ak.combinations(indexed, 2, fields=["j1", "j2"], axis=1)
    hadronic = ak.cartesian({"w": pairs, "b": indexed}, axis=1)
    hadronic = hadronic[(hadronic.b.index != hadronic.w.j1.index)
                        & (hadronic.b.index != hadronic.w.j2.index)]
    candidates = ak.cartesian({"had": hadronic, "blep": indexed}, axis=1)
    candidates = candidates[(candidates.blep.index != candidates.had.w.j1.index)
                            & (candidates.blep.index != candidates.had.w.j2.index)
                            & (candidates.blep.index != candidates.had.b.index)]
    thad = add(add(candidates.had.w.j1, candidates.had.w.j2), candidates.had.b)
    tlep = add(lep_w[:, np.newaxis], candidates.blep)
    score = abs(mass(thad) - MTOP) + abs(mass(tlep) - MTOP)
    best = ak.argmin(score, axis=1, keepdims=True)
    return ak.firsts(mass(add(thad, tlep))[best])
