from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_documentation_navigation_and_local_links_exist():
    config = tomllib.loads((DOCS / "zensical.toml").read_text(encoding="utf-8"))["project"]
    targets = [value for entry in config["nav"] for value in entry.values()]
    assert {"index.md", "ux-first-round.md", "implementation-notes.md"} <= set(targets)
    assert (DOCS / config["site_dir"]).resolve() == DOCS / "_build"
    for target in targets:
        source = DOCS / target
        assert source.is_file()
        for link in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            parsed = urlsplit(link)
            if not parsed.scheme and parsed.path:
                assert (source.parent / unquote(parsed.path)).is_file(), f"missing link {link} in {target}"


def test_documented_zensical_build_succeeds_without_warnings():
    result = subprocess.run([sys.executable, "-m", "zensical", "build", "--config-file", "docs/zensical.toml", "--strict"], cwd=ROOT, capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (DOCS / "_build" / "index.html").is_file()
    assert (DOCS / "_build" / "ux-first-round" / "index.html").is_file()
    assert (DOCS / "_build" / "implementation-notes" / "index.html").is_file()
