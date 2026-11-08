"""Opt-in real uv resolution tests, including a dependency absent from the host."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

pytestmark = [pytest.mark.uv, pytest.mark.skipif(os.environ.get("AGENT_OPTIMIZE_UV_TESTS") != "1", reason="set AGENT_OPTIMIZE_UV_TESTS=1 for real uv resolution")]
ROOT = Path(__file__).resolve().parents[1]


def dependency_wheel(directory):
    path = directory / "script_only_dep-1.0.0-py3-none-any.whl"
    files = {
        "script_only_dep/__init__.py": 'MESSAGE = "dependency provided by the script header"\n',
        "script_only_dep-1.0.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: script-only-dep\nVersion: 1.0.0\n",
        "script_only_dep-1.0.0.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files["script_only_dep-1.0.0.dist-info/RECORD"] = "".join(f"{name},,\n" for name in [*files, "script_only_dep-1.0.0.dist-info/RECORD"])
    with zipfile.ZipFile(path, "w") as wheel:
        for name, content in files.items():
            wheel.writestr(name, content)
    return path


def test_uv_standalone_and_script_launcher_share_declared_environment(tmp_path):
    uv = shutil.which("uv")
    assert uv, "uv must be installed for this integration test"
    assert importlib.util.find_spec("script_only_dep") is None
    wheel = dependency_wheel(tmp_path)
    analysis = tmp_path / "analysis with spaces.py"
    requirements = [f"agent-optimize @ {ROOT.as_uri()}", f"script-only-dep @ {wheel.as_uri()}"]
    analysis.write_text(
        '# /// script\n# requires-python = ">=3.13,<3.14"\n'
        f'# dependencies = {json.dumps(requirements)}\n# ///\n'
        'import json, sys\nfrom pathlib import Path\n'
        'import awkward as ak\nimport numpy as np\nimport script_only_dep\n'
        'from agent_optimize import optimize\n'
        'Path("executions").open("a").write("x")\n'
        'print(json.dumps({"python": sys.executable, "version": sys.version_info[:2], "args": sys.argv[1:], "dependency": script_only_dep.MESSAGE}), flush=True)\n'
        '@optimize\ndef total(events):\n'
        '    jets = events.Jet\n'
        '    return ak.sum(jets[jets.pt > 30].pt, axis=1)\n'
        'events = ak.zip({"Jet": ak.zip({"pt": ak.unflatten(np.array([20., 40., 50.]), [3])})}, depth_limit=1)\n'
        'print(ak.to_list(total(events)), flush=True)\n', encoding="utf-8",
    )
    forwarded = ["--json", "an argument with spaces", "", "--", "-x"]
    ordinary = subprocess.run([uv, "run", "--script", str(analysis), *forwarded], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert ordinary.returncode == 0, ordinary.stderr
    ordinary_info = json.loads(ordinary.stdout.splitlines()[0])
    assert ordinary_info["version"] == [3, 13]
    assert ordinary_info["args"] == forwarded
    assert ordinary_info["dependency"] == "dependency provided by the script header"
    assert (tmp_path / "executions").read_text() == "x"
    optimized = subprocess.run([sys.executable, "-m", "agent_optimize", "run", "--json", "--max-request-bytes", "1", str(analysis), *forwarded], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert optimized.returncode == 0, optimized.stderr
    report = json.loads(optimized.stdout)
    child_info = next(json.loads(line) for line in optimized.stderr.splitlines() if line.startswith('{"python":'))
    assert child_info["version"] == [3, 13]
    assert child_info["args"] == forwarded
    assert child_info["dependency"] == ordinary_info["dependency"]
    assert report["python_executable"] == child_info["python"]
    assert (tmp_path / "executions").read_text() == "xx"
    function = report["functions"][0]
    assert function["captured_calls"] == 1
    assert function["replays"][0]["status"] == "passed"
    assert function["replays"][0]["python_executable"] == child_info["python"]
    assert function["evaluations"][0]["status"] == "request_limit"
    assert importlib.util.find_spec("script_only_dep") is None

    # Adding a generic capability extra must retain the header's local library
    # source, rather than trying to fetch an unrelated package from the index.
    provider = subprocess.run([uv, "run", "--no-project", "--python", "3.13", "--with-requirements", str(analysis), "--with", "agent-optimize[openai]", "python", "-c", "import openai, pydantic; print('provider-sdk-ready')"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert provider.returncode == 0, provider.stderr
    assert provider.stdout.strip() == "provider-sdk-ready"


def test_uv_conflicting_dependencies_fail_before_analysis(tmp_path):
    path = tmp_path / "conflict.py"
    path.write_text('# /// script\n# dependencies = ["numpy==2.2.0", "numpy==2.3.0"]\n# ///\nfrom pathlib import Path\nPath("ran").touch()\n', encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "agent_optimize", "run", "--json", str(path)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["error"]["phase"] == "configuration"
    assert not (tmp_path / "ran").exists()
    assert not (tmp_path / ".agent-optimize").exists()
