from copy import deepcopy
from decimal import Decimal

import pytest

from agent_optimize.usage import PRICING, estimate_cost, summarize_usage, usage_line


PROVIDER = {"kind": "openai", "model": "gpt-6.1-sol"}
USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100,
         "cached_input_tokens": 200, "cache_write_input_tokens": 100, "reasoning_output_tokens": 80}


def attempt(key="one", usage=None, provider=None, **extra):
    u = dict(USAGE if usage is None else usage)
    p = dict(PROVIDER if provider is None else provider)
    return {"attempt_id": key, "generator": p, "provider_usage": u, "provider_request_count": 1,
            "provider_cost": estimate_cost(u, p, 1), **extra}


def report(*evaluations, pending=None):
    function = {"function_id": "f", "identity": {"qualified_name": "function"}, "evaluations": list(evaluations)}
    if pending: function["pending_attempt"] = pending
    return {"functions": [function]}


def test_cost_charges_disjoint_cache_buckets_and_output_once():
    cost = estimate_cost(USAGE, PROVIDER, 1)
    # 700 ordinary, 200 read, 100 write; reasoning is part of the 100 output.
    assert Decimal(cost["amount_usd"]) == Decimal("0.00267")
    assert cost["pricing"]["verified_on"] == "2026-11-10"
    assert cost["pricing"]["rates_per_million"]["cache_write"] == "2.50"
    assert estimate_cost(USAGE, PROVIDER, 1, service_tier="flex")["status"] == "unavailable"
    assert estimate_cost(USAGE, {**PROVIDER, "model": "unknown"}, 1)["status"] == "unavailable"


def test_long_context_and_historical_price_snapshot():
    large = {**USAGE, "input_tokens": 300000, "total_tokens": 300100}
    cost = estimate_cost(large, PROVIDER, 1)
    assert cost["pricing"]["long_context"]
    assert cost["pricing"]["rates_per_million"]["input"] == "4"
    assert cost["pricing"]["rates_per_million"]["output"] == "15.0"
    old = attempt()
    rates = deepcopy(PRICING)
    rates["rates"]["gpt-6.1-sol"]["input"] = "20"
    assert estimate_cost(USAGE, PROVIDER, 1, pricing=rates)["amount_usd"] != old["provider_cost"]["amount_usd"]
    assert summarize_usage(report(old))["cost"]["estimated_total_usd"] == old["provider_cost"]["amount_usd"]


@pytest.mark.parametrize("usage", [{}, {"input_tokens": 1}, {**USAGE, "cached_input_tokens": 1001}, {**USAGE, "output_tokens": True}, {**USAGE, "total_tokens": 0}])
def test_missing_and_inconsistent_counts_have_no_fake_cost(usage):
    assert estimate_cost(usage, PROVIDER, 1)["amount_usd"] is None


def test_deduplicated_function_model_and_run_totals_include_failures():
    first = attempt(status="rejected")
    second = attempt("two", provider={"kind": "openai", "model": "gpt-6-luna"}, status="refused")
    r = report(first, deepcopy(first))
    r["functions"].append({"function_id": "g", "evaluations": [second]})
    r["artifacts"] = [first]  # persisted copies are not new usage
    s = summarize_usage(r)
    assert s["attempts"] == s["requests"]["total"] == 2
    assert s["tokens"]["total_tokens"] == 2200
    assert s["tokens"]["reasoning_output_tokens"] == 160
    assert len(s["models"]) == 2
    assert s["functions"][0]["attempts"] == 1


def test_pending_timeout_is_partial_and_known_usage_survives_evaluation_interrupt():
    done = attempt()
    pending = {"attempt_id": "pending", "generator": PROVIDER, "provider_request_count": None}
    s = summarize_usage(report(done, pending=pending))
    assert s["tokens"]["total_tokens"] == 1100
    assert not s["coverage"]["total_tokens"]["complete"]
    assert s["cost"]["estimated_total_usd"] is None
    assert s["cost"]["known_estimated_usd"] == done["provider_cost"]["amount_usd"]
    assert "partial" in usage_line(s)
    assert s["requests"]["total"] is None
    priced_pending = attempt("pending", status="evaluating")
    s = summarize_usage(report(done, pending=priced_pending))
    assert s["cost"]["status"] == "estimated"
    assert s["tokens"]["total_tokens"] == 2200


def test_mock_zero_legacy_unknown_and_non_attempt_errors():
    mock = {"attempt_id": "mock", "generator": {"kind": "mock"}, "provider_request_count": 0}
    s = summarize_usage(report(mock, {"status": "unsupported"}))
    assert s["tokens"]["total_tokens"] == 0
    assert s["cost"]["estimated_total_usd"] == "0"
    legacy = {"generator": PROVIDER, "provider_usage": USAGE, "provider_request_count": 1}
    s = summarize_usage(report(legacy))
    assert s["cost"]["status"] == "unavailable"  # never reprice old evidence


@pytest.mark.parametrize("phase", ["request", "evaluation", "refused"])
def test_search_checkpoints_known_or_unknown_usage_without_paid_calls(tmp_path, monkeypatch, phase):
    import time
    from agent_optimize import search, performance
    from agent_optimize.provider import ProviderSettings
    from agent_optimize.generator import propose
    from agent_optimize.store import read_json
    from test_search import harness, group
    directory, outcome, run_report, _ = harness(tmp_path, monkeypatch)
    def provider(problem, history):
        if phase == "request": raise KeyboardInterrupt()
        return {**propose(problem, history), "status": "refused" if phase == "refused" else "proposal",
                "provider_usage": USAGE, "provider_request_count": 1}
    def interrupt(*args, **kwargs): raise KeyboardInterrupt()
    if phase == "evaluation": monkeypatch.setattr(performance, "complete_evaluation", interrupt)
    def run():
        search.search_groups({"g": group()}, outcome, directory, run_report, search.SearchLimits(proposals=1), time.monotonic() + 30,
                             proposer=provider, provider=ProviderSettings("openai"))
    if phase == "refused":
        run()
    else:
        with pytest.raises(KeyboardInterrupt): run()
    checkpoint = read_json(directory / "report.json")
    summary = checkpoint["usage_summary"]
    assert summary["attempts"] == 1
    if phase == "request":
        assert summary["cost"]["estimated_total_usd"] is None
        assert summary["tokens"]["total_tokens"] is None
    else:
        assert summary["tokens"]["total_tokens"] == 1100
        assert summary["cost"]["estimated_total_usd"] == "0.00267"


def test_provider_collects_cache_writes_as_an_input_subset():
    from types import SimpleNamespace as NS
    from agent_optimize.openai_provider import usage_summary
    u = usage_summary(NS(input_tokens=100, output_tokens=50, total_tokens=150,
                         input_tokens_details=NS(cached_tokens=20, cache_write_tokens=30), output_tokens_details=NS(reasoning_tokens=40)))
    assert u["cache_write_input_tokens"] == 30
    assert u["total_tokens"] == 150


def test_contradictory_request_count_does_not_erase_reported_tokens():
    a = attempt(provider_request_count=0)
    a["provider_cost"] = estimate_cost(USAGE, PROVIDER, 0)
    s = summarize_usage(report(a))
    assert s["tokens"]["total_tokens"] == 1100
    assert s["cost"]["estimated_total_usd"] is None
    assert "1,100" in usage_line(s)


@pytest.mark.parametrize("mode", ["malformed", "response_limit"])
def test_unusable_proposal_keeps_billable_usage(tmp_path, mode):
    import time
    from agent_optimize.search import request_proposal, SearchLimits
    response = {"status": "proposal", "source": None if mode == "malformed" else "x" * 2000,
                "description": "invalid", "provider_usage": USAGE, "provider_request_count": 1}
    retained, failure = request_proposal({"schema_version": 1, "problem": {}, "history": []}, tmp_path,
                                        SearchLimits(response_bytes=1000), time.monotonic() + 10,
                                        proposer=lambda *args: response)
    assert failure == mode
    assert retained["provider_usage"] == USAGE
    assert retained["provider_request_count"] == 1
