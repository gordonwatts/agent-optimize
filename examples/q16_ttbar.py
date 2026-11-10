#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "agent-optimize @ git+https://github.com/gordonwatts/agent-optimize.git@efb7dbd23923ebd27c16fef23f2a90906b0de743",
#   "typer>=0.12",
#   "func_adl_servicex_xaodr25",
#   "servicex>=3.0",
#   "jinja2",
#   "uproot>=5",
#   "awkward>=2",
#   "numpy",
#   "matplotlib",
# ]
# ///
"""Run IRIS-HEP q16 as a single, self-contained uv script.

    uv run --script q16_ttbar.py
    uv run --script q16_ttbar.py --n-files 2 --output-dir plots
    uv run --script q16_ttbar.py --input-file delivered.root
    agent-optimize run q16_ttbar.py --n-files 1

uv installs the dependencies declared above; no project or benchmark runner is
needed. Fetching data requires your own ServiceX endpoint and credentials in
servicex.yaml (in the working directory or a standard ServiceX config location).
Credentials are deliberately not embedded. One dataset file is processed by
default; --n-files 0 requests the full dataset. --input-file can be repeated to
reuse ROOT files previously produced by this script's ServiceX query, without
contacting ServiceX. It does not accept untransformed DAOD files.

Only jets with pT > 30 GeV and electrons/muons with pT > 25 GeV are selected.
Events must have exactly one such lepton and at least four such jets. All
quantities are in GeV. The neutrino uses the smaller-|pz| W-constraint solution
(the real part for a negative discriminant); all distinct four-jet assignments
are tried, minimizing the sum of absolute deviations of both top masses from
172.5 GeV. No b-tagging or additional event cuts are applied.
Awkward builds and scores the jet assignments across batches of 128 events;
there are no Python loops over events or assignments in the reconstruction.

This preserves the generated benchmark's reconstruction. Its one-file mean was
1537.14 GeV, which FAILED q16's 2477.6205 GeV (+/-2%) reference comparison.
That result is historical context from the supplied script; physics agreement
with the reference is still unresolved. Synthetic mode validates execution only.

The @optimize boundary covers neutrino reconstruction, ordered jet assignment,
candidate construction, scoring and winner selection. Delivery, I/O, event
packing and plots remain outside the compiled boundary. See docs/case-study-q16.md for the WSL recording workflow. Outputs are plot_1.png and the unweighted METRIC line.
"""

from pathlib import Path

import awkward as ak
import numpy as np
import typer
from agent_optimize import optimize

app = typer.Typer(add_completion=False)
DID = "user.zmarshal:user.zmarshal.301333_OpenData_v1_p6026_2024-04-23"
MW = 80.4
MTOP = 172.5


def build_query():
    from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE

    base = FuncADLQueryPHYSLITE()
    collections = base.Select(lambda e: {
        "jets": e.Jets().Where(lambda j: j.pt() / 1000.0 > 30.0),
        "electrons": e.Electrons().Where(lambda x: x.pt() / 1000.0 > 25.0),
        "muons": e.Muons().Where(lambda x: x.pt() / 1000.0 > 25.0),
        "met": e.MissingET().First(),
    })
    return collections.Select(lambda c: {
        "jet_px": c.jets.Select(lambda j: j.p4().Px() / 1000.0),
        "jet_py": c.jets.Select(lambda j: j.p4().Py() / 1000.0),
        "jet_pz": c.jets.Select(lambda j: j.p4().Pz() / 1000.0),
        "jet_e": c.jets.Select(lambda j: j.p4().E() / 1000.0),
        "ele_px": c.electrons.Select(lambda x: x.p4().Px() / 1000.0),
        "ele_py": c.electrons.Select(lambda x: x.p4().Py() / 1000.0),
        "ele_pz": c.electrons.Select(lambda x: x.p4().Pz() / 1000.0),
        "ele_e": c.electrons.Select(lambda x: x.p4().E() / 1000.0),
        "mu_px": c.muons.Select(lambda x: x.p4().Px() / 1000.0),
        "mu_py": c.muons.Select(lambda x: x.p4().Py() / 1000.0),
        "mu_pz": c.muons.Select(lambda x: x.p4().Pz() / 1000.0),
        "mu_e": c.muons.Select(lambda x: x.p4().E() / 1000.0),
        "met_px": c.met.mpx() / 1000.0,
        "met_py": c.met.mpy() / 1000.0,
    })


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
    """Pack aligned event inputs; construction and scoring share one boundary."""
    events = ak.to_packed(ak.zip({"jets": jets, "lepton": lepton, "metx": metx, "mety": mety}, depth_limit=1))
    return reconstruct_events(events)


@optimize
def reconstruct_events(events):
    """Reconstruct via ordered distinct jet assignments and first-minimum scoring."""
    jets = events.jets
    px = events.lepton.px
    py = events.lepton.py
    pz = events.lepton.pz
    energy = events.lepton.e
    metx = events.metx
    mety = events.mety
    mu = MW * MW / 2.0 + px * metx + py * mety
    a = energy * energy - pz * pz
    disc = mu * mu - a * (metx * metx + mety * mety)
    root = energy * np.sqrt(np.maximum(disc, 0.0))
    plus = (mu * pz + root) / a
    minus = (mu * pz - root) / a
    nuz = ak.where(abs(plus) <= abs(minus), plus, minus)
    nue = np.sqrt(metx * metx + mety * mety + nuz * nuz)
    indices = ak.argcartesian({"i": jets, "j": jets, "k": jets, "l": jets}, axis=1)
    valid = (indices.i < indices.j) & (indices.k != indices.i) & (indices.k != indices.j) & (indices.l != indices.i) & (indices.l != indices.j) & (indices.l != indices.k)
    chosen = indices[valid]
    j1 = jets[chosen.i]
    j2 = jets[chosen.j]
    bh = jets[chosen.k]
    bl = jets[chosen.l]
    h_e = (0 + ((0 + j1.e) + j2.e)) + bh.e
    l_e = (0 + ((0 + energy) + nue)[:, None]) + bl.e
    h_px = (0 + ((0 + j1.px) + j2.px)) + bh.px
    l_px = (0 + ((0 + px) + metx)[:, None]) + bl.px
    h_py = (0 + ((0 + j1.py) + j2.py)) + bh.py
    l_py = (0 + ((0 + py) + mety)[:, None]) + bl.py
    h_pz = (0 + ((0 + j1.pz) + j2.pz)) + bh.pz
    l_pz = (0 + ((0 + pz) + nuz)[:, None]) + bl.pz
    mh = np.sqrt(np.maximum(0.0, h_e ** 2 - h_px ** 2 - h_py ** 2 - h_pz ** 2))
    ml = np.sqrt(np.maximum(0.0, l_e ** 2 - l_px ** 2 - l_py ** 2 - l_pz ** 2))
    score = abs(mh - MTOP) + abs(ml - MTOP)
    combined = np.sqrt(np.maximum(0.0, ((0 + h_e) + l_e) ** 2 - ((0 + h_px) + l_px) ** 2 - ((0 + h_py) + l_py) ** 2 - ((0 + h_pz) + l_pz) ** 2))
    return ak.sum(combined[ak.local_index(score, axis=1) == ak.argmin(score, axis=1, keepdims=True)], axis=1)

@optimize
def select_ttbar_mass(c):
    """Fuse mass scoring and first-minimum selection; empty lists return zero.

    Keep the numerical reference self-contained so frozen replay never imports
    ServiceX or re-executes the analysis. The event selection ensures nonempty
    candidate lists in the real analysis.
    """
    mh = np.sqrt(np.maximum(0.0, c.he ** 2 - c.hx ** 2 - c.hy ** 2 - c.hz ** 2))
    ml = np.sqrt(np.maximum(0.0, c.le ** 2 - c.lx ** 2 - c.ly ** 2 - c.lz ** 2))
    score = abs(mh - MTOP) + abs(ml - MTOP)
    combined = np.sqrt(np.maximum(0.0, (c.he + c.le) ** 2 - (c.hx + c.lx) ** 2
                                  - (c.hy + c.ly) ** 2 - (c.hz + c.lz) ** 2))
    return ak.sum(combined[ak.local_index(score, axis=1)
                           == ak.argmin(score, axis=1, keepdims=True)], axis=1)


def synthetic_batches(count, batch_size=128):
    """Deterministic selected events for offline workflow checks, not physics."""
    rng = np.random.default_rng(16)
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        lengths = rng.integers(4, 8, size=size)
        pt = rng.uniform(31.0, 300.0, int(lengths.sum()))
        phi = rng.uniform(-np.pi, np.pi, len(pt))
        eta = rng.uniform(-2.5, 2.5, len(pt))
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        jets = ak.zip({key: ak.unflatten(value, lengths) for key, value in {
            "px": px, "py": py, "pz": pz, "e": np.sqrt(px*px + py*py + pz*pz + 25.0),
        }.items()})
        lpx, lpy, lpz = rng.uniform(26.0, 100.0, size=(3, size))
        lepton = ak.zip({"px": lpx, "py": lpy, "pz": lpz,
                         "e": np.sqrt(lpx*lpx + lpy*lpy + lpz*lpz)})
        yield jets, lepton, ak.Array(rng.normal(0.0, 40.0, size)), ak.Array(rng.normal(0.0, 40.0, size))



def four_vectors(arrays, prefix):
    return ak.zip({component: arrays[f"{prefix}_{component}"]
                   for component in ("px", "py", "pz", "e")})


def read_events(files):
    import uproot

    # ServiceX's ROOT tree is named `tree`; use its sole TTree if that changes.
    branches = ["jet_px", "jet_py", "jet_pz", "jet_e", "ele_px", "ele_py", "ele_pz", "ele_e",
                "mu_px", "mu_py", "mu_pz", "mu_e", "met_px", "met_py"]
    for filename in files:
        with uproot.open(filename) as root_file:
            tree = root_file["tree"] if "tree" in root_file else next(
                root_file[k] for k in root_file.keys() if hasattr(root_file[k], "arrays")
            )
            # Bound memory while vectorizing both events and their assignments.
            for arrays in tree.iterate(branches, step_size=128, library="ak"):
                jets = four_vectors(arrays, "jet")
                leptons = ak.concatenate(
                    [four_vectors(arrays, "ele"), four_vectors(arrays, "mu")], axis=1
                )
                selected = (ak.num(leptons, axis=1) == 1) & (ak.num(jets, axis=1) >= 4)
                if ak.any(selected):
                    yield (jets[selected], ak.flatten(leptons[selected], axis=1),
                           arrays.met_px[selected], arrays.met_py[selected])


@app.command()
def main(
    output_dir: Path = typer.Option(Path("."), help="Directory for plot_1.png."),
    n_files: int = typer.Option(1, min=0, help="Dataset file count; 0 requests all files."),
    servicex_config: Path | None = typer.Option(
        None, exists=True, dir_okay=False, readable=True,
        help="Optional ServiceX config path (useful for a Windows config from WSL).",
    ),
    synthetic_events: int = typer.Option(0, min=0, help="Offline synthetic selected events; no ServiceX."),
    input_file: list[Path] | None = typer.Option(
        None, "--input-file", exists=True, dir_okay=False, readable=True,
        help="Previously delivered query ROOT file; repeat for multiple files.",
    ),
) -> None:
    """Fetch q16 data (or reuse delivered files), reconstruct ttbar mass, and plot."""
    if synthetic_events and input_file:
        raise typer.BadParameter("Use either --synthetic-events or --input-file")
    if synthetic_events:
        files = ["synthetic"]
    elif input_file:
        files = [str(p) for p in input_file]
    else:
        from servicex import Sample, ServiceXSpec, dataset, deliver

        result = deliver(ServiceXSpec(Sample=[Sample(
            Name="ttbar_semileptonic", Dataset=dataset.Rucio(DID),
            NFiles=None if n_files == 0 else n_files, Query=build_query()
        )]), **({"config_path": servicex_config} if servicex_config else {}))
        files = [str(p) for p in result["ttbar_semileptonic"]]
    if not files:
        raise RuntimeError("ServiceX delivered no files")
    batches = synthetic_batches(synthetic_events) if synthetic_events else read_events(files)
    chunks = [ak.to_numpy(reconstruct(*batch)) for batch in batches]
    if not chunks:
        raise RuntimeError("No events passed the requested baseline selections")
    n_selected = sum(len(chunk) for chunk in chunks)
    masses = np.concatenate(chunks)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    # Include the 3 TeV signal region and every reconstructed entry.
    plt.hist(masses, bins=100, range=(0, max(5000.0, float(masses.max()) * 1.01)),
             histtype="step", linewidth=1.8)
    plt.xlabel(r"Reconstructed $m_{t\bar{t}}$ [GeV]")
    plt.ylabel("Events")
    plt.title("Semileptonic top-pair reconstruction")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_1.png", dpi=150)
    plt.close()
    print(f"METRIC: avg_entries_per_event={len(masses) / n_selected:.6g} mean={float(np.mean(masses)):.6g}", flush=True)
    typer.echo(f"Reconstructed {len(masses)} selected events from {len(files)} file(s).")
    typer.echo(f"Saved {output_dir / 'plot_1.png'}")


if __name__ == "__main__":
    app()
