"""Checkpoint-safe API accounting. Tokens are reported; costs are estimates."""

from collections import defaultdict
from decimal import Decimal

TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens")
PRICING = {
    "version": "openai-standard-2026-11-10", "verified_on": "2026-11-10",
    "currency": "USD", "unit_tokens": 1_000_000,
    "source": "https://developers.openai.com/api/docs/pricing",
    "cache_source": "https://developers.openai.com/api/docs/guides/prompt-caching",
    "scope": "OpenAI direct API, Standard tier, default 30-minute cache writes; excludes taxes/credits and regional premiums",
    "rates": {
        "gpt-6.1-sol": {"input": "2", "cached_input": "0.10", "cache_write": "2.50", "output": "10"},
        "gpt-6.1-terra": {"input": "1", "cached_input": "0.05", "cache_write": "1.25", "output": "5"},
        "gpt-6-luna": {"input": "0.10", "cached_input": "0.01", "cache_write": "0.125", "output": "0.50"},
    },
}


def valid_count(value):
    return type(value) is int and value >= 0


def normalize_usage(raw):
    return {k: raw[k] for k in TOKEN_FIELDS if valid_count(raw.get(k))} if type(raw) is dict else {}


def estimate_cost(usage, provider, request_count, *, service_tier="default", pricing=None):
    """Freeze applicable rates into each attempt; never reprice saved attempts."""
    pricing = PRICING if pricing is None else pricing
    base = {"schema_version": 1, "currency": "USD", "amount_usd": None, "status": "unavailable"}
    if provider.get("kind") == "mock" or request_count == 0 and not any(normalize_usage(usage).values()):
        return {**base, "status": "not_applicable", "amount_usd": "0", "reason": "No API request"}
    if request_count == 0:
        return {**base, "reason": "Positive usage conflicts with a zero request count"}
    model = provider.get("model")
    if provider.get("kind") != "openai" or model not in pricing["rates"]:
        return {**base, "reason": "No verified pricing for provider/model"}
    if service_tier != "default":
        return {**base, "reason": "No verified pricing for returned service tier"}
    u = normalize_usage(usage)
    required = ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens")
    if any(k not in u for k in required):
        return {**base, "reason": "Incomplete usage, including cache-read/write counts"}
    i, o, c, w = (u[k] for k in required)
    if c + w > i or u.get("reasoning_output_tokens", 0) > o or u.get("total_tokens", i + o) != i + o:
        return {**base, "reason": "Inconsistent provider token counts"}
    # Model pages document the full-request long-context surcharge.
    long = i > 272_000
    rates = {k: Decimal(v) * (Decimal("1.5") if k == "output" and long else 2 if long else 1) for k, v in pricing["rates"][model].items()}
    amount = ((i - c - w) * rates["input"] + c * rates["cached_input"] + w * rates["cache_write"] + o * rates["output"]) / pricing["unit_tokens"]
    snapshot = {k: v for k, v in pricing.items() if k != "rates"}
    snapshot.update(model=model, rates_per_million={k: str(v) for k, v in rates.items()}, service_tier=service_tier, cache_write_ttl="30m", long_context=long,
                    model_source=f"https://developers.openai.com/api/docs/models/{model}")
    return {**base, "status": "estimated", "amount_usd": format(amount, "f"), "pricing": snapshot}


def _summary(attempts):
    def no_request(a):
        return a.get("generator", {}).get("kind") == "mock" or a.get("provider_request_count") == 0 and not any(normalize_usage(a.get("provider_usage")).values())
    tokens, coverage = {}, {}
    for field in TOKEN_FIELDS:
        values = []
        for a in attempts:
            if no_request(a):
                values.append(0)
            elif field in normalize_usage(a.get("provider_usage")):
                values.append(a["provider_usage"][field])
        tokens[field] = sum(values) if values or not attempts else None
        coverage[field] = {"reported_attempts": len(values), "complete": len(values) == len(attempts)}
    counts = [a.get("provider_request_count") for a in attempts if valid_count(a.get("provider_request_count"))]
    known_cost, priced = Decimal(0), 0
    for a in attempts:
        cost = a.get("provider_cost") or {}
        if no_request(a):
            priced += 1
        elif cost.get("status") == "estimated" and cost.get("currency") == "USD":
            try:
                amount = Decimal(cost["amount_usd"])
                if amount.is_finite() and amount >= 0:
                    known_cost += amount
                    priced += 1
            except (ValueError, KeyError, ArithmeticError):
                pass
    complete = priced == len(attempts)
    return {"attempts": len(attempts), "tokens": tokens, "coverage": coverage,
            "requests": {"known_count": sum(counts), "total": sum(counts) if len(counts) == len(attempts) else None, "unknown_attempts": len(attempts) - len(counts)},
            "cost": {"currency": "USD", "status": "not_applicable" if complete and len(counts) == len(attempts) and sum(counts) == 0 else "estimated" if complete else "partial" if priced else "unavailable",
                     "estimated_total_usd": format(known_cost, "f") if complete else None,
                     "known_estimated_usd": format(known_cost, "f"), "unpriced_attempts": len(attempts) - priced}}


def summarize_usage(report):
    """Aggregate attempt records only, excluding artifact copies of that evidence."""
    all_attempts, functions, models = {}, [], defaultdict(dict)
    for index, function in enumerate(report.get("functions", [])):
        fid = function.get("function_id", str(index))
        records = {}
        for n, evaluation in enumerate(function.get("evaluations", [])):
            # Graph/setup errors are not provider attempts.
            if not any(k in evaluation for k in ("generator", "provider_usage", "provider_request_count")):
                continue
            evaluation = {**evaluation, "generator": evaluation.get("generator", function.get("search", {}).get("provider", report.get("provider", {})))}
            key = evaluation.get("attempt_id", f"{fid}/legacy-{n}")
            records[key] = evaluation
        pending = function.get("pending_attempt")
        if pending:
            records.setdefault(pending["attempt_id"], pending)
        # Legacy partial checkpoints can state more proposals than retained evidence.
        for n in range(max(0, function.get("search", {}).get("proposals", 0) - len(records))):
            key = f"{fid}/missing-{n}"
            records[key] = {"generator": function.get("search", {}).get("provider", report.get("provider", {}))}
        functions.append({"function_id": fid, "name": function.get("identity", {}).get("qualified_name", fid), **_summary(list(records.values()))})
        for key, value in records.items():
            all_attempts.setdefault(key, value)
            generator = value.get("generator", {})
            models[(generator.get("kind", "unknown"), generator.get("model", "none"))].setdefault(key, value)
    return {"schema_version": 1, **_summary(list(all_attempts.values())), "functions": functions,
            "models": [{"provider": p, "model": m, **_summary(list(records.values()))} for (p, m), records in sorted(models.items())]}


def update_usage(report):
    report["usage_summary"] = summarize_usage(report)
    return report


def usage_line(summary):
    if summary["cost"]["status"] == "not_applicable":
        return "API usage: 0 tokens; cost USD 0 (no API requests)."
    total = summary["tokens"]["total_tokens"]
    complete = summary["coverage"]["total_tokens"]["complete"]
    tokens = f"{total:,}" if total is not None else "unavailable"
    if not complete and total is not None:
        tokens += " known (partial)"
    cost = summary["cost"]
    dollars = f"USD {cost['estimated_total_usd']} estimated" if cost["estimated_total_usd"] is not None else f"unavailable total (USD {cost['known_estimated_usd']} known estimate; {cost['unpriced_attempts']} unpriced attempts)"
    return f"API usage: {tokens} tokens; cost {dollars}."
