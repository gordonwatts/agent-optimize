"""Plain, versioned metadata records shared by the implementation modules.

These records describe local artifacts; array values will live in separate
binary buffers. Use dataclasses.asdict for JSON serialization. Version 1 is
the initial schema, and consumers must reject unknown schema versions.
"""

from dataclasses import dataclass, field
from typing import Literal

SCHEMA_VERSION = 1
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]


@dataclass(frozen=True)
class CapturedCall:
    function_id: str
    call_id: str
    arguments: dict[str, JSONValue]
    reference_result: dict[str, JSONValue] | None
    reference_fingerprint: str = ""
    positional_names: list[str] = field(default_factory=list)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)


@dataclass(frozen=True)
class UnsupportedFunction:
    function_id: str
    reason: str
    parameter: str | None = None
    schema_version: int = field(default=SCHEMA_VERSION, init=False)


@dataclass(frozen=True)
class EvaluationFailure:
    function_id: str
    candidate_id: str
    phase: str
    reason: str
    schema_version: int = field(default=SCHEMA_VERSION, init=False)


@dataclass(frozen=True)
class FinalOutcome:
    function_id: str
    status: Literal["reference", "unsupported", "failed", "ineligible", "optimized"]
    reason: str
    artifact_path: str | None = None
    schema_version: int = field(default=SCHEMA_VERSION, init=False)
