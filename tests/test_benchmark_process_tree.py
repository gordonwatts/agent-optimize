"""Real offline child/grandchild tests for bounded benchmark execution."""

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarks.process_tree import run


GRANDCHILD = """
from pathlib import Path
import sys
import time
heartbeat = Path(sys.argv[1])
while True:
    with heartbeat.open('a') as stream:
        stream.write('x')
    time.sleep(0.02)
"""

PARENT = """
from pathlib import Path
import subprocess
import sys
import time
heartbeat = Path(sys.argv[1])
child = subprocess.Popen([sys.executable, '-c', sys.argv[2], str(heartbeat)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.monotonic() + 5
while not heartbeat.exists():
    if time.monotonic() > deadline:
        raise RuntimeError('grandchild did not start')
    time.sleep(0.01)
if sys.argv[3] == 'wait':
    time.sleep(60)
"""


@pytest.mark.parametrize("mode", ["wait", "exit"])
def test_managed_command_cleans_up_grandchild_on_timeout_and_normal_exit(tmp_path, mode):
    heartbeat = tmp_path / "grandchild-heartbeat.txt"
    command = [sys.executable, "-c", PARENT, str(heartbeat), GRANDCHILD, mode]
    start = time.monotonic()
    if mode == "wait":
        with pytest.raises(subprocess.TimeoutExpired):
            run(command, timeout=2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        result = run(command, timeout=10, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert result.returncode == 0, result.stderr
    assert time.monotonic() - start < 8
    assert heartbeat.exists(), "the grandchild must actually run before cleanup"
    # A process still alive after the launcher exits would continue this
    # heartbeat. Check after OS termination has had time to settle.
    time.sleep(0.1)
    stopped = heartbeat.read_text()
    time.sleep(0.2)
    assert heartbeat.read_text() == stopped


def test_process_wrapper_preserves_exit_status_and_output():
    result = run([sys.executable, "-c", "import sys; print('output'); sys.exit(3)"], timeout=10, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 3
    assert result.stdout.strip() == "output"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object bootstrap")
def test_windows_launcher_does_not_run_command_if_job_attachment_fails(monkeypatch):
    from benchmarks import process_tree

    monkeypatch.setattr(sys, "argv", ["process_tree", "--windows-child", "missing-job", "--", "command-must-not-run"])
    def fail_attachment(name):
        raise OSError("test attachment failure")
    monkeypatch.setattr(process_tree, "_join_windows_job", fail_attachment)
    monkeypatch.setattr(process_tree.subprocess, "call", lambda *args, **kwargs: pytest.fail("command ran without cleanup protection"))
    assert process_tree.main() == 125


def test_posix_timeout_kills_new_process_group_before_collecting_output(monkeypatch):
    from benchmarks import process_tree

    calls = []
    class Process:
        pid = 2468
        stdout = None
        stderr = None
        attempts = 0
        def communicate(self, timeout):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired(["command"], timeout)
            assert calls == [(2468, 9)]
            return "partial", ""
        def wait(self, timeout):
            return -9

    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(process_tree, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: calls.append((pid, sig))))
    monkeypatch.setattr(process_tree, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(process_tree.subprocess, "Popen", popen)
    with pytest.raises(subprocess.TimeoutExpired) as error:
        run(["command"], timeout=2)
    assert error.value.output == "partial"
    assert calls == [(2468, 9)]
