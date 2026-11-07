import copy

import pytest

from agent_optimize.timing import assess_eligibility


def timing(candidate=80, reference=100):
    return {"mode": "controlled-test-fixture", "recordings": [
        {"case_id": case, "batches": [{"reference_ns": [reference] * 5, "candidate_ns": [candidate] * 5} for _ in range(3)]}
        for case in ["recording-a", "recording-b"]
    ]}


def test_ten_percent_threshold_is_inclusive_and_every_batch_must_pass():
    assert assess_eligibility(timing(90), {"recording-a", "recording-b"})["eligible"]
    assert not assess_eligibility(timing(90.1), {"recording-a", "recording-b"})["eligible"]
    record = timing()
    record["recordings"][0]["batches"][2]["candidate_ns"] = [95] * 5
    assert not assess_eligibility(record, {"recording-a", "recording-b"})["eligible"]


def test_regression_on_one_recording_is_ineligible():
    record = timing()
    record["recordings"][1]["batches"][1]["candidate_ns"] = [120] * 5
    result = assess_eligibility(record, {"recording-a", "recording-b"})
    assert not result["eligible"] and "regression" in result["reason"]


def test_uncertainty_can_reject_an_improved_median():
    record = timing()
    record["recordings"][0]["batches"][0]["candidate_ns"] = [50, 60, 80, 95, 100]
    result = assess_eligibility(record, {"recording-a", "recording-b"})
    assert not result["eligible"] and "uncertainty" in result["reason"]


@pytest.mark.parametrize("alter", [
    lambda record: record["recordings"].pop(),
    lambda record: record["recordings"][0]["batches"].pop(),
    lambda record: record["recordings"][0]["batches"][0]["reference_ns"].pop(),
    lambda record: record["recordings"][0]["batches"][0]["candidate_ns"].__setitem__(0, float("nan")),
    lambda record: record["recordings"][0]["batches"][0]["candidate_ns"].__setitem__(0, 0),
])
def test_insufficient_or_invalid_evidence_is_never_eligible(alter):
    record = timing()
    alter(record)
    assert not assess_eligibility(record, {"recording-a", "recording-b"})["eligible"]
