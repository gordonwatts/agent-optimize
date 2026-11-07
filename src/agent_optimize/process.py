"""Portable worker execution with deadlines and bounded diagnostic output."""

from dataclasses import dataclass
import os
import subprocess
import threading

from .store import SESSION_ENV


@dataclass(frozen=True)
class ProcessResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str


def worker_environment(environment=None):
    cleaned = {}
    for key, value in (os.environ if environment is None else environment).items():
        upper = key.upper()
        if (
            upper == SESSION_ENV or upper.startswith(("OPENAI_", "AZURE_OPENAI_", "ANTHROPIC_", "COHERE_", "GEMINI_"))
            or upper.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
            or upper in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HUGGING_FACE_HUB_TOKEN"}
        ):
            continue
        cleaned[key] = value
    return cleaned


def run_worker(command, *, timeout=30, environment=None, max_output_bytes=64 * 1024, input_data=None) -> ProcessResult:
    if timeout <= 0 or max_output_bytes <= 0:
        raise ValueError("worker timeout and output limit must be positive")
    try:
        process = subprocess.Popen(
            command, env=worker_environment(environment), stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as error:
        return ProcessResult("startup_error", None, "", f"{type(error).__name__}: worker could not start")
    buffers = [bytearray(), bytearray()]
    lock = threading.Lock()
    exceeded = threading.Event()

    def drain(stream, index):
        while chunk := stream.read(4096):
            with lock:
                remaining = max_output_bytes - sum(map(len, buffers))
                buffers[index].extend(chunk[:remaining])
                overflow = len(chunk) > remaining
            if overflow:
                exceeded.set()
                try:
                    process.kill()
                except OSError:
                    pass
        stream.close()

    threads = [threading.Thread(target=drain, args=(stream, index), daemon=True) for index, stream in enumerate([process.stdout, process.stderr])]
    if input_data is not None:
        def supply():
            try:
                process.stdin.write(input_data)
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threads.append(threading.Thread(target=supply, daemon=True))
    for thread in threads:
        thread.start()
    timed_out = False

    def kill():
        try:
            process.kill()
        except OSError:
            pass

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill()
        process.wait()
    except BaseException:
        kill()
        process.wait()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1)
    status = "output_limit" if exceeded.is_set() else "timeout" if timed_out else "passed" if process.returncode == 0 else "failed"
    return ProcessResult(status, process.returncode, *(bytes(buffer).decode("utf-8", errors="replace") for buffer in buffers))
