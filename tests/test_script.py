import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_optimize import script
from agent_optimize.cli import main


def write_script(tmp_path, header="", body="raise AssertionError('metadata reading executed code')\n"):
    path = tmp_path / "analysis with spaces.py"
    path.write_text(header + body, encoding="utf-8", newline="")
    return path


def header(dependencies='"agent-optimize==0.1.0"', python=">=3.13,<3.14"):
    return f'# /// script\n# requires-python = "{python}"\n# dependencies = [{dependencies}]\n# ///\n'


def test_reads_metadata_without_importing_analysis_and_handles_crlf(tmp_path):
    path = write_script(tmp_path, header('"agent-optimize>=0.1", "missing-analysis-dependency"').replace("\n", "\r\n"))
    assert script.read_metadata(path) == {"requires-python": ">=3.13,<3.14", "dependencies": ["agent-optimize>=0.1", "missing-analysis-dependency"]}
    assert script.read_metadata(write_script(tmp_path)) == {}


@pytest.mark.parametrize("content,diagnostic", [
    (header() + "\n" + header(), "multiple"),
    ('# /// script\n# dependencies = [\n# ///\n', "TOML"),
    ('# /// script\n# dependencies = "numpy"\n# ///\n', "list"),
    ('# /// script\n# dependencies = [3]\n# ///\n', "strings"),
    ('# /// script\n# requires-python = 3.13\n# ///\n', "requires-python"),
    ('# /// script\n# tool = "bad"\n# ///\n', "table"),
    ('# /// script\n# [tool.uv]\n# index-url = "https://example.invalid"\n# ///\n', r"\[tool.uv\]"),
])
def test_bad_metadata_is_actionable(tmp_path, content, diagnostic):
    with pytest.raises(script.ScriptConfigurationError, match=diagnostic):
        script.read_metadata(write_script(tmp_path, content))


def test_launch_preserves_declared_package_python_arguments_cwd_and_exit(tmp_path, monkeypatch):
    path = write_script(tmp_path, header('"agent_optimize @ git+https://example.invalid/repo.git@revision", "numpy==2.3.0"'))
    calls = []
    monkeypatch.setattr(script.shutil, "which", lambda name: "uv executable")
    monkeypatch.setattr(script, "installed_requirement", lambda: pytest.fail("must honor the script's declared library source"))
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=7, stdout=None)
    monkeypatch.setattr(script.subprocess, "run", run)
    forwarded = [str(path), "--json", "spaces and ; shell $(syntax)", "", "--", "-x"]
    arguments = ["run", "--verification-groups", "2", *forwarded]
    assert script.launch_script(arguments, forwarded) == 7
    command, kwargs = calls[0]
    assert command[:7] == ["uv executable", "run", "--no-project", "--python", ">=3.13,<3.14", "--with-requirements", str(path)]
    assert command[7:] == ["python", "-c", script.CONTROLLER, "3", "run", "--verification-groups", "2", *forwarded]
    assert kwargs == {"check": False, "stdout": None}


def test_plain_script_adds_installed_library_and_uv_is_required(tmp_path, monkeypatch):
    path = write_script(tmp_path)
    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    with pytest.raises(script.ScriptConfigurationError, match="uv on PATH"):
        script.launch_script(["run", str(path)], [str(path)])
    monkeypatch.setattr(script.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(script, "installed_requirement", lambda: "agent-optimize @ file:///a/local/checkout")
    calls = []
    monkeypatch.setattr(script.subprocess, "run", lambda command, **kwargs: (calls.append(command) or SimpleNamespace(returncode=0)))
    assert script.launch_script(["run", str(path)], [str(path)]) == 0
    assert calls[0][5:7] == ["--with", "agent-optimize @ file:///a/local/checkout"]


def test_lockfile_does_not_silently_get_ignored(tmp_path):
    path = write_script(tmp_path, header())
    path.with_name(path.name + ".lock").touch()
    with pytest.raises(script.ScriptConfigurationError, match="lockfiles"):
        script.launch_script(["run", str(path)], [str(path)])


def test_openai_extra_keeps_script_package_source(tmp_path, monkeypatch):
    path = write_script(tmp_path, header('"agent-optimize @ git+https://example.invalid/repo.git@pinned"'))
    monkeypatch.setattr(script.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(script, "installed_requirement", lambda: pytest.fail("script source must win"))
    calls = []
    monkeypatch.setattr(script.subprocess, "run", lambda command, **kwargs: (calls.append(command) or SimpleNamespace(returncode=0)))
    arguments = ["run", "--provider", "openai", str(path)]
    assert script.launch_script(arguments, [str(path)], provider="openai") == 0
    assert calls[0][5:9] == ["--with-requirements", str(path), "--with", "agent-optimize[openai]"]


def test_controller_runs_script_once_and_shares_interpreter_without_bootstrap(tmp_path):
    path = write_script(tmp_path, body="import sys\nfrom pathlib import Path\nPath('once').open('a').write('x')\nprint(sys.executable)\nprint(repr(sys.argv[1:]))\n")
    prefix = ["run", "--json"]
    result = subprocess.run([sys.executable, "-c", script.CONTROLLER, str(len(prefix)), *prefix, str(path), "--json", "a b", ""], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["python_executable"] == sys.executable
    assert sys.executable in result.stderr
    assert "['--json', 'a b', '']" in result.stderr
    assert (tmp_path / "once").read_text() == "x"
    assert "resolving script environment" not in result.stderr


def test_cli_separates_optimizer_json_from_script_json(tmp_path, monkeypatch):
    path = write_script(tmp_path)
    seen = []
    monkeypatch.setattr(script, "launch_script", lambda args, command, **kwargs: (seen.append(kwargs["json_mode"]) or 0))
    assert main(["run", str(path), "--json"]) == 0
    assert main(["run", "--json", str(path), "--json"]) == 0
    assert seen == [False, True]


def test_json_uv_resolution_failure_emits_one_configuration_result(tmp_path, monkeypatch, capsys):
    path = write_script(tmp_path, header())
    monkeypatch.setattr(script.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(script.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b""))
    assert main(["run", "--json", str(path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["phase"] == "configuration"
    assert "uv could not prepare" in result["error"]["message"]
    assert not (tmp_path / ".agent-optimize").exists()


@pytest.mark.parametrize("origin,expected", [
    (None, "agent-optimize==0.2.3"),
    ({"url": "file:///a/local/checkout", "dir_info": {"editable": True}}, "agent-optimize @ file:///a/local/checkout"),
    ({"url": "https://example.invalid/repo.git", "vcs_info": {"vcs": "git", "commit_id": "abc123"}, "subdirectory": "python"}, "agent-optimize @ git+https://example.invalid/repo.git@abc123#subdirectory=python"),
])
def test_fallback_package_requirement_uses_installation_provenance(monkeypatch, origin, expected):
    monkeypatch.setattr(script.metadata, "distribution", lambda name: SimpleNamespace(version="0.2.3", read_text=lambda name: json.dumps(origin) if origin else None))
    assert script.installed_requirement() == expected
