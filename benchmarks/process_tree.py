"""Bounded benchmark execution with descendant cleanup on Windows and POSIX.

Windows uses a kill-on-close Job Object. A small launcher joins the job before
starting the CLI, so no CLI or provider can run outside its lifetime. POSIX/WSL
uses a new session and kills its process group. Only trusted benchmark commands
are launched; this is resource cleanup, not a sandbox for arbitrary programs.
"""

import os
import signal
import subprocess
import sys
import uuid


class _WindowsJob:
    def __init__(self, name):
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong), ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t), ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_operations", "write_operations", "other_operations", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IOCounters), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _join_windows_job(name):
    import ctypes
    from ctypes import wintypes

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    api.OpenJobObjectW.restype = wintypes.HANDLE
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    api.AssignProcessToJobObject.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    job = api.OpenJobObjectW(0x1, False, name)  # JOB_OBJECT_ASSIGN_PROCESS
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not api.AssignProcessToJobObject(job, api.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        # The parent keeps the sole lifetime handle. Closing it kills this
        # launcher and every descendant, including a still-running provider.
        api.CloseHandle(job)


def run(command, *, timeout=None, **kwargs):
    """Like subprocess.run(check=False), but always clean up descendants."""
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    kwargs.pop("check", None)
    job = None
    process = None
    tree_stopped = False
    if os.name == "nt":
        name = "Local\\agent-optimize-benchmark-" + uuid.uuid4().hex
        job = _WindowsJob(name)
        launch = [sys.executable, "-m", "benchmarks.process_tree", "--windows-child", name, "--", *command]
    else:
        launch = command
        kwargs["start_new_session"] = True

    def stop_tree():
        nonlocal tree_stopped
        if process is None or tree_stopped:
            return
        tree_stopped = True
        if os.name == "nt":
            # Kill the launcher even if it has not yet opened/entered the job.
            # Closing the job then kills any children it already started.
            try:
                process.kill()
            except OSError:
                pass
            job.close()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        process = subprocess.Popen(launch, **kwargs)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stop_tree()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                # Inherited pipes outside the managed tree cannot hold the
                # runner indefinitely; retain output collected so far.
                stdout, stderr = error.output, error.stderr
                for stream in (process.stdout, process.stderr):
                    if stream:
                        stream.close()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        stop_tree()
        if process is not None:
            process.wait(timeout=10)
        if job is not None:
            job.close()


def main():
    if os.name != "nt" or len(sys.argv) < 5 or sys.argv[1] != "--windows-child" or sys.argv[3] != "--":
        raise SystemExit("This module is an internal benchmark process launcher")
    try:
        _join_windows_job(sys.argv[2])
    except OSError:
        print("benchmark: could not establish descendant cleanup; command not started", file=sys.stderr)
        return 125
    return subprocess.call(sys.argv[4:])


if __name__ == "__main__":
    raise SystemExit(main())
