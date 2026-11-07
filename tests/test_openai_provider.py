"""Offline coverage: no credential lookup or API traffic in the fake tests."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import time

import pytest

from agent_optimize import cli, openai_provider as live, search
from agent_optimize.generator import propose as mock_propose
from agent_optimize.process import ProcessResult, run_worker, worker_environment
from agent_optimize.provider import KEY_ENV, MODELS, ProviderSettings, resolve_api_key
from agent_optimize.store import json_bytes, read_json, write_json
from test_search import group, harness


SECRET = "offline-private-sentinel-8877665544332211"


def request():
    return {"schema_version": 1, "problem": search.provider_problem(group()), "history": []}


def completed():
    return NS(status="completed", output=[], output_parsed=live.Proposal.model_validate(mock_propose(request()["problem"], [])),
              usage=NS(input_tokens=12, output_tokens=18, total_tokens=30,
                       input_tokens_details=NS(cached_tokens=2), output_tokens_details=NS(reasoning_tokens=4)))


def fake_client(response=None, error=None):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})
            self.responses = self
            self.transport = kwargs["http_client"]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def parse(self, **kwargs):
            calls.append({"parse": kwargs})
            for hook in self.transport.event_hooks["request"]:
                hook(None)
            if error:
                raise error
            return response or completed()
    return Client, calls


def test_key_precedence_cli_environment_project_home_and_no_openai_fallback(tmp_path):
    project, home = tmp_path / "project", tmp_path / "home"
    project.mkdir(); home.mkdir()
    (project / ".env").write_text(f"export {KEY_ENV}='project-key'\n")
    (home / ".env").write_text(f'{KEY_ENV}="home-key"\n')
    options = {"project_root": project, "home": home}
    assert resolve_api_key("cli-key", environment={KEY_ENV: "env-key"}, **options) == "cli-key"
    assert resolve_api_key(environment={KEY_ENV: "env-key"}, **options) == "env-key"
    assert resolve_api_key(environment={}, **options) == "project-key"
    (project / ".env").unlink()
    assert resolve_api_key(environment={}, **options) == "home-key"
    (home / ".env").unlink()
    with pytest.raises(ValueError, match=KEY_ENV):
        resolve_api_key(environment={"OPENAI_API_KEY": "must-not-use"}, **options)


@pytest.mark.parametrize("model", MODELS)
def test_direct_sdk_parse_schema_settings_usage_and_identity(model):
    client, calls = fake_client()
    settings = ProviderSettings("openai", model, 1.25, 1234)
    response = live.propose(request(), SECRET, settings, client_factory=client)
    assert response["status"] == "proposal"
    assert response["provider_usage"] == {"input_tokens": 12, "output_tokens": 18, "total_tokens": 30, "cached_input_tokens": 2, "reasoning_output_tokens": 4}
    assert response["provider_request_count"] == 1
    assert calls[0]["client"]["max_retries"] == 0
    assert calls[0]["client"]["api_key"] == SECRET
    assert calls[0]["client"]["base_url"] == "https://api.openai.com/v1"
    parsed = calls[1]["parse"]
    assert parsed["text_format"] is live.Proposal and parsed["model"] == model
    assert parsed["timeout"] == 1.25 and parsed["max_output_tokens"] == 1234 and parsed["store"] is False
    assert SECRET not in parsed["input"] and SECRET not in json.dumps(response)
    assert settings.identity["model"] == model and SECRET not in json.dumps(settings.identity)
    schema = live.Proposal.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"status", "source", "description", "extra_cases"}
    assert all(definition["additionalProperties"] is False for definition in schema["$defs"].values())


@pytest.mark.parametrize("state,expected", [("refusal", "refused"), ("incomplete", "incomplete"), ("failed", "error"), ("missing", "malformed")])
def test_response_states_and_usage_are_distinct(state, expected):
    response = completed()
    if state == "refusal":
        response.output = [NS(content=[NS(type="refusal", refusal=SECRET)])]
    elif state == "missing":
        response.output_parsed = None
    else:
        response.status = state
    client, _ = fake_client(response)
    actual = live.propose(request(), SECRET, ProviderSettings("openai"), client_factory=client)
    assert actual["status"] == expected and actual["provider_usage"]["total_tokens"] == 30
    assert SECRET not in json.dumps(actual)


@pytest.mark.parametrize("exception,expected", [("AuthenticationError", "error"), ("RateLimitError", "error"), ("APIConnectionError", "error"), ("APITimeoutError", "provider_timeout"), ("LengthFinishReasonError", "incomplete"), ("ValidationError", "malformed")])
def test_sdk_errors_are_fixed_codes_without_exception_messages(exception, expected):
    error = type(exception, (Exception,), {})(SECRET)
    client, _ = fake_client(error=error)
    actual = live.propose(request(), SECRET, ProviderSettings("openai"), client_factory=client)
    assert actual["status"] == expected and actual["provider_request_count"] == 1
    assert SECRET not in json.dumps(actual)


@pytest.mark.parametrize("key", [SECRET, 'private-"quoted\\key-ñ'])
def test_credential_echo_is_rejected_before_persistence(key):
    response = completed()
    response.output_parsed.description = key
    client, _ = fake_client(response)
    assert live.propose(request(), key, ProviderSettings("openai"), client_factory=client)["status"] == "error"


def test_private_stdin_transport_and_total_deadline_cap(tmp_path, monkeypatch):
    calls = []
    settings = ProviderSettings("openai", timeout=30)
    def worker(command, **kwargs):
        calls.append((command, kwargs))
        assert SECRET not in " ".join(command)
        assert SECRET not in (tmp_path / "request.json").read_text()
        private = json.loads(kwargs["input_data"])
        assert private["key"] == SECRET
        assert 0 < private["settings"]["timeout"] <= kwargs["timeout"] <= 0.5
        write_json(tmp_path / "response.json", {"schema_version": 1, "response": {"status": "incomplete", "provider_usage": {"total_tokens": 3}, "provider_request_count": 1}})
        return ProcessResult("passed", 0, SECRET, SECRET)
    monkeypatch.setattr(search, "run_worker", worker)
    response, failure = search.request_proposal(request(), tmp_path, search.SearchLimits(), time.monotonic() + .5, provider=settings, api_key=SECRET)
    assert failure == "incomplete" and response["provider_usage"] == {"total_tokens": 3}
    assert calls and all(SECRET not in path.read_text() for path in tmp_path.rglob("*.json"))


def test_real_subprocess_private_pipe_and_scrubbed_environment():
    import sys
    code = "import os, sys, json; payload = sys.stdin.buffer.read(); print(json.dumps({'bytes': len(payload), 'key_in_environment': 'AGENT_OPTIMIZE_OAI_API_KEY' in os.environ}))"
    result = run_worker([sys.executable, "-c", code], input_data=SECRET.encode(),
                        environment={**os.environ, KEY_ENV: SECRET}, timeout=5)
    assert result.status == "passed" and SECRET not in result.stdout + result.stderr
    assert json.loads(result.stdout) == {"bytes": len(SECRET.encode()), "key_in_environment": False}


def test_request_and_response_size_limits_and_worker_timeout(tmp_path, monkeypatch):
    calls = []
    def worker(*args, **kwargs):
        calls.append(1)
        return ProcessResult("timeout", None, SECRET, SECRET)
    monkeypatch.setattr(search, "run_worker", worker)
    opts = {"provider": ProviderSettings("openai"), "api_key": SECRET}
    assert search.request_proposal(request(), tmp_path, search.SearchLimits(request_bytes=1), time.monotonic()+2, **opts) == (None, "request_limit")
    assert not calls
    assert search.request_proposal(request(), tmp_path, search.SearchLimits(), time.monotonic()+2, **opts) == (None, "provider_timeout")
    assert calls == [1]
    assert search.request_proposal(request(), tmp_path, search.SearchLimits(response_bytes=1), time.monotonic()+2, proposer=lambda *args: mock_propose(*args)) == (None, "response_limit")


def test_provider_identity_usage_history_privacy_and_checkpoints(tmp_path, monkeypatch):
    directory, outcome, report, _ = harness(tmp_path, monkeypatch)
    settings = ProviderSettings("openai", MODELS[1])
    seen = []
    private = group()
    private["cases"][0].arguments["threshold"] = 8877665544332211
    def provider(problem, history):
        seen.append({"problem": problem, "history": history})
        return {**mock_propose(problem, history), "provider_usage": {"total_tokens": 30}, "provider_request_count": 1}
    search.search_groups({"one": private}, outcome, directory, report, search.SearchLimits(proposals=2), time.monotonic()+30,
                         provider=settings, api_key=SECRET, proposer=provider)
    assert seen[1]["history"][0]["source"] == mock_propose(seen[0]["problem"], [])["source"]
    assert "8877665544332211" not in json.dumps(seen)
    assert all(item["generator"] == settings.identity and item["provider_usage"]["total_tokens"] == 30 and item["provider_request_count"] == 1 for item in outcome["evaluations"])
    assert outcome["search"]["provider"] == settings.identity
    assert SECRET not in json.dumps(report)


def test_live_identity_and_usage_survive_real_correctness_and_artifact(tmp_path):
    from agent_optimize.evaluator import evaluate_session
    from agent_optimize.recorder import Recorder
    from agent_optimize.replay import replay_session
    from agent_optimize.store import create_session
    from test_dispatch import controlled
    from test_operations import integer_ht, problem
    directory = create_session(tmp_path / "artifacts")
    Recorder(directory).invoke(integer_ht, (), problem()[3])
    settings = ProviderSettings("openai", "gpt-6-luna")
    def provider(problem, history):
        return {**mock_propose(problem, [{"feedback": "correct this offline fixture"}]),
                "provider_usage": {"total_tokens": 30}, "provider_request_count": 1}
    report = evaluate_session(directory, replay_session(directory, 0), provider=settings, api_key=SECRET,
                              proposer=provider, timing_provider=controlled, search_limits=search.SearchLimits(proposals=1))
    evaluation = report["functions"][0]["evaluations"][0]
    assert evaluation["status"] == "optimized"
    manifest = read_json(Path(evaluation["artifact_path"]))
    assert manifest["generator"] == report["provider"] == settings.identity
    assert manifest["provider_usage"] == {"total_tokens": 30} and manifest["provider_request_count"] == 1
    assert SECRET not in json.dumps(report) and SECRET not in json.dumps(manifest)


def test_project_root_lookup_from_subdirectory(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / ".env").write_text(f"{KEY_ENV}=project-key")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    assert resolve_api_key(environment={}, home=child) == "project-key"


def test_default_mock_and_child_environment_excludes_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    cleaned = worker_environment({KEY_ENV: SECRET, "OPENAI_API_KEY": SECRET, "PATH": "kept"})
    assert cleaned == {"PATH": "kept"}
    seen = []
    def child(command, **kwargs):
        seen.append(kwargs["env"])
        return NS(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", child)
    result = cli._run(["run", "--artifact-root", str(tmp_path / "artifacts"), "--", "analysis"])
    assert result["provider"]["kind"] == "mock"
    assert KEY_ENV not in seen[0] and SECRET not in json.dumps(result)


@pytest.mark.parametrize("options", [{"timeout": float("nan")}, {"timeout": 0}, {"max_output_tokens": 0}, {"max_output_tokens": True}, {"max_output_tokens": 128001}, {"kind": "unknown"}, {"model": "unknown"}])
def test_provider_limits_reject_invalid_configuration(options):
    with pytest.raises(ValueError):
        ProviderSettings(**options)


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("AGENT_OPTIMIZE_LIVE_TESTS") != "1", reason="real API calls are opt-in")
def test_opt_in_live_proposal(tmp_path):
    model = os.environ.get("AGENT_OPTIMIZE_LIVE_MODEL", "gpt-6-luna")
    key = resolve_api_key()
    response, failure = search.request_proposal(request(), tmp_path, search.SearchLimits(), time.monotonic()+30,
                                               provider=ProviderSettings("openai", model), api_key=key)
    assert failure is None and response["status"] == "proposal"
    assert response["provider_request_count"] == 1
