"""Resolve a script's inline requirements before starting the controller.

Only metadata is read here. The analysis itself is executed once by the normal
recording command, with the same interpreter used by all evaluation workers.
"""

from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tokenize
import tomllib


# Canonical PEP 723 matcher; normalize newlines before matching.
METADATA = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
PACKAGE = re.compile(r"^\s*agent[-_.]+optimize(?=\s|\[|@|[<>=!~;]|$)", re.I)

# Fixed code, with every user-supplied argument forwarded separately via argv.
# The explicit command form prevents another bootstrap in the resolved process.
CONTROLLER = (
    "import sys; from agent_optimize.cli import main; "
    "n = int(sys.argv[1]); "
    "raise SystemExit(main(sys.argv[2:2+n] + ['--', sys.executable] + sys.argv[2+n:]))"
)


class ScriptConfigurationError(ValueError):
    pass


def read_metadata(path):
    try:
        with tokenize.open(path) as stream:
            source = stream.read()
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ScriptConfigurationError(f"could not read Python script: {path}") from error
    matches = [match for match in re.finditer(METADATA, source) if match["type"] == "script"]
    if len(matches) > 1:
        raise ScriptConfigurationError("script contains multiple inline script metadata blocks")
    if not matches:
        return {}
    content = "".join(line[2:] if line.startswith("# ") else line[1:]
                      for line in matches[0]["content"].splitlines(keepends=True))
    try:
        value = tomllib.loads(content)
    except tomllib.TOMLDecodeError as error:
        raise ScriptConfigurationError("script has invalid inline TOML metadata") from error
    dependencies = value.get("dependencies", [])
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise ScriptConfigurationError("script dependencies must be a list of requirement strings")
    if "requires-python" in value and (not isinstance(value["requires-python"], str) or not value["requires-python"].strip()):
        raise ScriptConfigurationError("script requires-python must be a nonempty version requirement string")
    if not isinstance(value.get("tool", {}), dict):
        raise ScriptConfigurationError("script tool configuration must be a TOML table")
    if value.get("tool", {}).get("uv"):
        raise ScriptConfigurationError("script-aware launch currently supports dependencies and requires-python; use explicit command mode for [tool.uv] configuration")
    return value


def installed_requirement():
    """Reuse the launching installation only when the script does not name us."""
    distribution = metadata.distribution("agent-optimize")
    direct = distribution.read_text("direct_url.json")
    if direct:
        origin = json.loads(direct)
        url = origin["url"]
        if "vcs_info" in origin:
            vcs = origin["vcs_info"]
            url = f"{vcs['vcs']}+{url}@{vcs['commit_id']}"
        if origin.get("subdirectory"):
            url += f"#subdirectory={origin['subdirectory']}"
        return f"agent-optimize @ {url}"
    return f"agent-optimize=={distribution.version}"


def launch_script(arguments, command, *, json_mode=False, provider="mock"):
    path = Path(command[0]).resolve()
    if not path.is_file():
        raise ScriptConfigurationError(f"Python script does not exist: {path}")
    config = read_metadata(path)
    if path.with_name(path.name + ".lock").exists():
        raise ScriptConfigurationError("script-aware launch does not yet consume script lockfiles; use explicit command mode with a locked environment")
    uv = shutil.which("uv")
    if uv is None:
        raise ScriptConfigurationError("script-aware launch requires uv on PATH; install uv or use run -- <command> with an existing environment")
    launch = [uv, "run", "--no-project", "--python", config.get("requires-python", f"{sys.version_info.major}.{sys.version_info.minor}")]
    if config:
        launch += ["--with-requirements", str(path)]
    if not any(PACKAGE.match(requirement) for requirement in config.get("dependencies", [])):
        launch += ["--with", installed_requirement()]
    if provider == "openai":
        launch += ["--with", "agent-optimize[openai]"]
    prefix = list(arguments[:len(arguments) - len(command)])
    launch += ["python", "-c", CONTROLLER, str(len(prefix)), *prefix, str(path), *command[1:]]
    print(f"agent-optimize: resolving script environment with uv: {path}", file=sys.stderr)
    try:
        # Capture only the controller's JSON stream. uv diagnostics remain on
        # stderr; analysis stdout is redirected by the inner controller.
        result = subprocess.run(launch, check=False, stdout=subprocess.PIPE if json_mode else None)
    except OSError as error:
        raise ScriptConfigurationError("could not start uv for script dependency resolution") from error
    if json_mode:
        if result.stdout:
            sys.stdout.write(result.stdout.decode("utf-8"))
        else:
            # No controller ran, so main can emit the usual structured error.
            raise ScriptConfigurationError(f"uv could not prepare the script environment (exit {result.returncode}); inspect the dependency diagnostic above")
    elif result.returncode:
        print(f"agent-optimize: script environment/controller exited {result.returncode}; inspect the diagnostic above", file=sys.stderr)
    return result.returncode
