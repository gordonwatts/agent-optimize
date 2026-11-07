import json
from dataclasses import asdict

import pytest

from agent_optimize.records import (
    SCHEMA_VERSION,
    CapturedCall,
    EvaluationFailure,
    FinalOutcome,
    UnsupportedFunction,
)


@pytest.mark.parametrize(
    "record",
    [
        CapturedCall("module.ht", "call-1", {"threshold": 30}, {"dtype": "float64"}),
        UnsupportedFunction("module.ht", "unsupported argument", "threshold"),
        EvaluationFailure("module.ht", "candidate-1", "correctness", "wrong dtype"),
        FinalOutcome("module.ht", "reference", "No eligible improvement"),
    ],
)
def test_records_have_json_metadata_and_schema_version(record):
    metadata = json.loads(json.dumps(asdict(record)))
    assert metadata == asdict(record)
    assert metadata["schema_version"] == SCHEMA_VERSION == 1
    assert metadata["function_id"] == "module.ht"
