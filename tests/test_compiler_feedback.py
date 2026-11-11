import json

from agent_optimize.diagnostics import MAX_DIAGNOSTIC_BYTES, code_diagnostic
from agent_optimize.evaluator import evaluate_session
from agent_optimize.generator import propose
from agent_optimize.recorder import Recorder
from agent_optimize.replay import replay_session
from agent_optimize.search import SearchLimits, feedback
from agent_optimize.store import create_session
from test_dispatch import controlled
from test_operations import integer_ht, problem


def test_compiler_diagnostic_keeps_detail_paths_and_tail_without_other_worker_metadata():
    diagnostic = "TypingError: compiler failure\nFile C:/project/candidate.py\n" + "compiler detail\n" * 80 + "abs(array(float64, 1d, C))"
    result = feedback({"status": "rejected", "reason": "private runtime detail", "candidate_worker": {
        "phase": "compilation", "compiler_diagnostic": diagnostic,
        "case_id": "private-recording", "case_name": "private-values", "outputs": ["private-output"]}}, [])
    assert result["phase"] == "compilation"
    assert result["compiler_diagnostic"] == diagnostic
    assert "correctness checks" not in result["diagnostic"]
    assert all(value not in json.dumps(result) for value in ("private-recording", "private-values", "private-output", "private runtime detail"))


def test_diagnostic_removes_terminal_styling_and_bounds_escaped_json():
    assert code_diagnostic("\x1b[31mTypingError\x1b[0m: detail") == "TypingError: detail"
    result = code_diagnostic("TypingError: " + "é" * 30_000)
    assert "truncated" in result
    assert len(json.dumps(result, ensure_ascii=True).encode("utf-8")) <= MAX_DIAGNOSTIC_BYTES


def test_runtime_worker_errors_are_not_forwarded_as_compiler_feedback():
    result = feedback({"status": "rejected", "candidate_worker": {
        "phase": "execution", "reason": "private input 8877665544332211",
        "compiler_diagnostic": "private input 8877665544332211"}}, [])
    assert "8877665544332211" not in json.dumps(result)
    assert "compiler_diagnostic" not in result


def test_source_validation_feedback_preserves_the_validator_message():
    result = feedback({"status": "invalid_source", "validation_diagnostic": "ValueError: candidate object access/methods are unsupported"}, [])
    assert result["phase"] == "source_validation"
    assert "object access/methods" in result["validation_diagnostic"]


def test_real_numba_compiler_error_reaches_the_next_model_request(tmp_path):
    session = create_session(tmp_path / "artifacts")
    Recorder(session).invoke(integer_ht, (), problem()[3])
    seen = []
    def provider(public, history):
        seen.append(history)
        if not history:
            return {"status": "proposal", "source": "def kernel(offsets, columns, scalars):\n    output = np.zeros(len(offsets) - 1, dtype=np.int64)\n    values = abs(columns[0])\n    for event in range(len(offsets) - 1):\n        output[event] = values[offsets[event]]\n    return output\n",
                    "description": "deliberate array abs compilation error", "extra_cases": []}
        diagnostic = history[-1]["feedback"]["compiler_diagnostic"]
        assert "abs" in diagnostic and "array" in diagnostic and "TypingError" in diagnostic
        assert history[-1]["feedback"]["phase"] == "compilation"
        return propose(public, history)
    report = evaluate_session(session, replay_session(session, 0), proposer=provider,
                              timing_provider=controlled, search_limits=SearchLimits(proposals=2))
    evaluations = report["functions"][0]["evaluations"]
    assert [e["status"] for e in evaluations] == ["rejected", "optimized"]
    assert evaluations[0]["candidate_worker"]["phase"] == "compilation"
    assert len(evaluations[0]["provider_feedback"]["compiler_diagnostic"]) > 500
    assert len(seen) == 2
