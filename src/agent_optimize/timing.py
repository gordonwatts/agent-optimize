"""Conservative eligibility from repeated complete replacement timings."""

import math
import statistics

MIN_BATCHES = 3
MIN_SAMPLES = 5
IMPROVEMENT = 0.10


def assess_eligibility(timing, expected_cases):
    expected = set(expected_cases)
    recordings = timing.get("recordings", [])
    if not expected or len(recordings) != len(expected) or {record.get("case_id") for record in recordings} != expected:
        return {"eligible": False, "reason": "insufficient timing evidence for every retained recording"}
    summaries, speedups = [], []
    for recording in recordings:
        batches = recording.get("batches", [])
        if len(batches) < MIN_BATCHES:
            return {"eligible": False, "reason": "insufficient timing batches"}
        batch_summaries = []
        for batch in batches:
            reference, candidate = batch.get("reference_ns", []), batch.get("candidate_ns", [])
            if len(reference) < MIN_SAMPLES or len(candidate) < MIN_SAMPLES or not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in [*reference, *candidate]):
                return {"eligible": False, "reason": "insufficient or invalid timing samples"}
            left, right = statistics.median(reference), statistics.median(candidate)
            if right > left * 1.01:
                return {"eligible": False, "reason": "a retained recording has a meaningful timing regression"}
            if right > left * (1 - IMPROVEMENT):
                return {"eligible": False, "reason": "less than 10% median improvement in at least one batch"}
            reference_low = statistics.quantiles(reference, n=4, method="inclusive")[0]
            candidate_high = statistics.quantiles(candidate, n=4, method="inclusive")[2]
            if candidate_high > reference_low * (1 - IMPROVEMENT):
                return {"eligible": False, "reason": "timing uncertainty crosses the 10% improvement threshold"}
            speedups.append(left / right)
            batch_summaries.append({"reference_median_ns": left, "candidate_median_ns": right, "reference_q25_ns": reference_low, "candidate_q75_ns": candidate_high})
        summaries.append({"case_id": recording["case_id"], "batches": batch_summaries})
    return {"eligible": True, "reason": "at least 10% repeatable improvement on every retained recording", "minimum_batch_speedup": min(speedups), "recordings": summaries}
