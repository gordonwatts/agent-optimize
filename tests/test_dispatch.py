import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest

from agent_optimize import optimize
from agent_optimize.artifacts import manifest_seal
import agent_optimize.dispatch as dispatch
from agent_optimize.dispatch import clear_dispatch_cache, observe_dispatch
from agent_optimize.evaluator import evaluate_session
from agent_optimize.performance import measure_evaluation
from agent_optimize.recorder import Recorder
from agent_optimize.replay import replay_session
from agent_optimize.store import create_session, read_json, write_json

SOURCE = """import awkward as ak
import numpy as np
from agent_optimize import optimize
CUT = 2.4

@optimize
def ht(events, threshold=30):
    jets = events.Jet
    selected = jets[(jets.pt > threshold) & (abs(jets.eta) < CUT)]
    return ak.sum(selected.pt, axis=1)

def make_events(lengths=(2, 0, 3)):
    lengths = np.asarray(lengths, dtype=np.int64)
    count = int(lengths.sum())
    jets = ak.zip({
        'pt': ak.unflatten(np.arange(count, dtype=np.int64) * 17 % 100, lengths),
        'eta': ak.unflatten(np.zeros(count, dtype=np.float64), lengths),
    })
    return ak.zip({'Jet': jets}, depth_limit=1)
"""


def load_module(path):
    spec = importlib.util.spec_from_file_location("accepted_analysis", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def controlled(evaluation):
    return {
        "schema_version": 1, "status": "passed", "mode": "controlled-test-fixture",
        "note": "synthetic acceptance evidence; not a measured performance claim",
        "recordings": [{"case_id": identifier, "batches": [
            {"reference_ns": [1000] * 5, "candidate_ns": [500] * 5} for _ in range(3)
        ]} for identifier in evaluation["recorded_case_ids"]],
    }


@pytest.fixture(scope="module")
def accepted_base(tmp_path_factory):
    directory = tmp_path_factory.mktemp("accepted base with spaces")
    path = directory / "analysis.py"
    path.write_text(SOURCE, encoding="utf-8")
    module = load_module(path)
    root = directory / ".agent-optimize"
    session = create_session(root)
    Recorder(session).invoke(module.ht.__wrapped__, (module.make_events(),), {})
    report = evaluate_session(session, replay_session(session, 0), timing_provider=controlled, dispatch_mode="careful")
    evaluation = next(evaluation for evaluation in reversed(report["functions"][0]["evaluations"]) if evaluation["status"] == "optimized")
    assert evaluation["status"] == "optimized", evaluation
    return {"root": root, "source": path, "session": session, "evaluation": evaluation, "function_id": report["functions"][0]["function_id"]}


@pytest.fixture
def accepted(accepted_base, tmp_path, monkeypatch):
    root = tmp_path / "store with spaces"
    shutil.copytree(accepted_base["root"] / "artifacts", root / "artifacts")
    shutil.copytree(accepted_base["root"] / "accepted", root / "accepted")
    accepted_base["source"].write_text(SOURCE, encoding="utf-8")
    module = load_module(accepted_base["source"])
    clear_dispatch_cache()
    monkeypatch.setenv("AGENT_OPTIMIZE_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("AGENT_OPTIMIZE_MODE", "careful")
    monkeypatch.delenv("AGENT_OPTIMIZE_SESSION", raising=False)
    value = {**accepted_base, "root": root, "module": module, "artifact_id": accepted_base["evaluation"]["artifact_id"]}
    yield value
    accepted_base["source"].write_text(SOURCE, encoding="utf-8")
    clear_dispatch_cache()


@pytest.fixture
def fast_kernel(monkeypatch):
    compiled, calls = [], []

    class Kernel:
        def __init__(self, source):
            namespace = {"np": np}
            exec(source, namespace)
            self.function = namespace["kernel"]

        def compile(self, signature):
            pass

        def __call__(self, *args):
            calls.append(1)
            return self.function(*args)

    def compile_source(source):
        compiled.append(1)
        return Kernel(source)

    monkeypatch.setattr(dispatch, "compile_kernel", compile_source)
    return compiled, calls


def invoke(case, events=None):
    events = case["module"].make_events() if events is None else events
    stats = {}
    with observe_dispatch(stats):
        result = case["module"].ht(events)
    assert ak.to_list(result) == ak.to_list(case["module"].ht.__wrapped__(events))
    return stats


def test_controlled_acceptance_is_labelled_and_ordinary_dispatch_reuses_kernel(accepted, fast_kernel):
    compiled, calls = fast_kernel
    assert invoke(accepted)["used"] is True
    assert invoke(accepted)["used"] is True
    assert len(compiled) == 1 and len(calls) == 2
    manifest = read_json(accepted["root"] / "artifacts" / accepted["artifact_id"] / "manifest.json")
    assert manifest["timing"]["mode"] == "controlled-test-fixture"
    assert manifest["correctness"]["status"] == "passed"
    evidence = read_json(accepted["root"] / "artifacts" / accepted["artifact_id"] / "evidence.json")
    assert "not a measured performance claim" in evidence["timing"]["note"]


def test_real_compilation_artifact_is_used_by_a_fresh_ordinary_python_process(accepted):
    program = """import importlib.util, json, sys
import awkward as ak
from agent_optimize.dispatch import observe_dispatch
spec = importlib.util.spec_from_file_location('accepted_analysis', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
stats = {}
with observe_dispatch(stats):
    result = module.ht(module.make_events((0, 7, 1, 0)))
print(json.dumps({'used': stats.get('used'), 'values': ak.to_list(result)}))
"""
    result = subprocess.run([sys.executable, "-c", program, str(accepted["source"])], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["used"] is True
    assert json.loads(result.stdout)["values"] == [0, 238, 0, 0]
    assert not list(accepted["root"].glob("sessions"))


def test_event_count_and_jagged_lengths_are_not_specialized(accepted, fast_kernel):
    for lengths in [(0, 0), (1, 100, 2, 0), (0,)]:
        assert invoke(accepted, accepted["module"].make_events(lengths))["used"]


def test_changed_constants_and_restored_constants_do_not_reuse_stale_compatibility(accepted, fast_kernel):
    assert invoke(accepted)["used"]
    accepted["module"].CUT = 1.5
    assert not invoke(accepted)["used"]
    accepted["module"].CUT = 2.4
    assert invoke(accepted)["used"]


def test_changed_source_and_executable_code_fall_back(accepted, fast_kernel):
    assert invoke(accepted)["used"]
    path = accepted["source"]
    path.write_text(SOURCE.replace("jets.pt > threshold", "jets.pt >= threshold"), encoding="utf-8")
    assert not invoke(accepted)["used"]
    path.write_text(SOURCE, encoding="utf-8")
    reference = accepted["module"].ht.__wrapped__
    # Change executable contents without changing the owning source file.
    reference.__code__ = reference.__code__.replace(co_names=tuple("num" if name == "sum" else name for name in reference.__code__.co_names))
    assert not invoke(accepted)["used"]
    assert len(fast_kernel[1]) == 1


def test_different_schema_and_scalar_types_fall_back_without_quarantining_valid_artifact(accepted, fast_kernel):
    events = accepted["module"].make_events()
    changed = ak.with_field(events, ak.values_astype(events.Jet.pt, np.float64), ("Jet", "pt"))
    assert not invoke(accepted, changed)["used"]
    stats = {}
    with observe_dispatch(stats):
        accepted["module"].ht(events, threshold=np.int32(30))
    assert not stats["used"]
    assert not (accepted["root"] / "quarantine").exists()
    assert invoke(accepted)["used"]


def test_environment_change_falls_back(accepted, fast_kernel, monkeypatch):
    original = dispatch.environment_signature
    monkeypatch.setattr(dispatch, "environment_signature", lambda: {**original(), "python": "changed"})
    assert not invoke(accepted)["used"]
    assert not (accepted["root"] / "quarantine").exists()


def test_changed_import_binding_and_numpy_error_policy_invalidate_compatibility(accepted, fast_kernel, monkeypatch):
    original = ak.sum

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(ak, "sum", replacement)
    assert not invoke(accepted)["used"]
    monkeypatch.setattr(ak, "sum", original)
    assert invoke(accepted)["used"]
    with np.errstate(invalid="raise"):
        assert not invoke(accepted)["used"]


def test_artifact_removal_restores_reference_calls_even_with_a_compiled_cache(accepted, fast_kernel):
    assert invoke(accepted)["used"]
    index = accepted["root"] / "accepted" / accepted["function_id"] / "index.json"
    index.unlink()
    assert not invoke(accepted).get("used", False)
    assert len(fast_kernel[1]) == 1


@pytest.mark.parametrize("filename", ["manifest.json", "kernel.py", "evidence.json"])
def test_corrupt_artifacts_fall_back_and_are_quarantined_by_identity(accepted, fast_kernel, filename):
    assert invoke(accepted)["used"]
    path = accepted["root"] / "artifacts" / accepted["artifact_id"] / filename
    path.write_text("corrupt", encoding="utf-8")
    assert not invoke(accepted)["used"]
    assert not invoke(accepted)["used"]
    diagnostic = read_json(accepted["root"] / "quarantine" / f"{accepted['artifact_id']}.json")
    assert diagnostic["phase"] == "load"
    assert len(fast_kernel[1]) == 1


@pytest.mark.parametrize("phase", ["load", "compilation", "execution"])
def test_recoverable_failures_fall_back_on_same_call_and_are_not_retried(accepted, fast_kernel, monkeypatch, phase):
    calls = []
    if phase == "load":
        original = dispatch.read_json

        def failure(path):
            if path.name == "manifest.json":
                calls.append(1)
                raise OSError("injected load failure")
            return original(path)

        monkeypatch.setattr(dispatch, "read_json", failure)
    elif phase == "compilation":
        def failure(source):
            calls.append(1)
            raise RuntimeError("injected compilation failure")
        monkeypatch.setattr(dispatch, "compile_kernel", failure)
    else:
        assert invoke(accepted)["used"]

        def failure(*args):
            calls.append(1)
            raise RuntimeError("injected execution failure")
        dispatch._compiled[(str(accepted["root"].resolve()), accepted["artifact_id"])] = failure
    assert not invoke(accepted)["used"]
    assert not invoke(accepted)["used"]
    assert calls == [1]
    assert (str(accepted["root"].resolve()), accepted["artifact_id"]) not in dispatch._compiled


def test_unwritable_quarantine_store_keeps_effective_memory_quarantine(accepted, fast_kernel, monkeypatch):
    attempts = []

    def failure(source):
        attempts.append(1)
        raise RuntimeError("compilation failure")

    monkeypatch.setattr(dispatch, "compile_kernel", failure)
    monkeypatch.setattr(dispatch, "persist_quarantine", lambda *args: (_ for _ in ()).throw(PermissionError("read only")))
    assert not invoke(accepted)["used"]
    assert not invoke(accepted)["used"]
    assert attempts == [1]
    assert not (accepted["root"] / "quarantine").exists()


def test_user_interrupts_are_not_quarantined_or_suppressed(accepted, fast_kernel, monkeypatch):
    monkeypatch.setattr(dispatch, "compile_kernel", lambda source: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        accepted["module"].ht(accepted["module"].make_events())
    assert not (accepted["root"] / "quarantine").exists()


def test_fallback_reference_exceptions_propagate(accepted, monkeypatch):
    failure = RuntimeError("original reference failure")

    @optimize
    def reference():
        raise failure

    monkeypatch.setattr(dispatch, "dispatch_call", lambda *args, **kwargs: (False, None))
    with pytest.raises(RuntimeError) as caught:
        reference()
    assert caught.value is failure


def test_real_timings_record_full_path_and_cold_costs_without_asserting_speedup(accepted_base):
    evaluation = read_json(Path(accepted_base["evaluation"]["workspace"]) / "evaluation.json")
    reference = next((accepted_base["session"] / "references").glob("*/reference.json"))
    timing = measure_evaluation(evaluation, read_json(reference), accepted_base["function_id"])
    assert timing["status"] == "passed", timing
    assert timing["mode"] == "measured"
    assert timing["compilation_ns"] > 0
    assert timing["first_use_ns"] >= timing["compilation_ns"]
    assert "ordinary decorator" in timing["path"]
    assert len(timing["recordings"]) == len(evaluation["recorded_case_ids"])
    assert all(len(record["batches"]) == 3 for record in timing["recordings"])
    assert all(len(batch["candidate_ns"]) == 5 for record in timing["recordings"] for batch in record["batches"])
