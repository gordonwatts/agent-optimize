"""Optional analysis-dependency checks, exercised by the Linux q16 CI job."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")
pytest.importorskip("matplotlib")
pytest.importorskip("typer")
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/q16_ttbar.py"


def example():
    spec = importlib.util.spec_from_file_location("q16_io", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transformed_root_input_preserves_values_and_numeric_layout(tmp_path):
    module = example()
    jets, lepton, mx, my = next(module.synthetic_batches(4))
    branches = {f"jet_{field}": jets[field] for field in jets.fields}
    for field in lepton.fields:
        branches[f"ele_{field}"] = ak.singletons(lepton[field])
        branches[f"mu_{field}"] = ak.singletons(lepton[field])[:, :0]
    branches.update(met_px=mx, met_py=my)
    filename = tmp_path / "delivered.root"
    with uproot.recreate(filename) as output:
        output.mktree("tree", branches)
    batches = list(module.read_events([filename]))
    assert len(batches) == 1
    assert np.array_equal(ak.to_numpy(module.reconstruct(*batches[0])), ak.to_numpy(module.reconstruct(jets, lepton, mx, my)))


def test_standalone_synthetic_cli_creates_plot_without_servicex(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--synthetic-events", "4", "--output-dir", str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "METRIC: avg_entries_per_event=1" in result.stdout
    assert (tmp_path / "plot_1.png").stat().st_size > 0
