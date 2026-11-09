"""Deterministic local mock; no credentials, network, or evaluator access."""

IDENTITY = {"kind": "mock", "identity": "filter-sum-feedback-v1"}


def propose(problem, history):
    from .candidate import contract_from_dict, fused_candidate_source
    from .operations import graph_from_dict

    source = fused_candidate_source(graph_from_dict(problem["graph"]), contract_from_dict(problem["contract"]))
    if not history:
        if "output[event] = total" in source:
            source = source.replace("output[event] = total", "output[event] = total + 1")
        else:
            source = source.replace("    return output", "    for event in range(len(output)):\n        output[event] += 1\n    return output")
    return {
        "status": "proposal", "source": source,
        "description": "Fused filter and sum; correct the output offset after feedback." if history else "Initial fused loop with an intentionally incorrect output offset.",
        "extra_cases": [{"kind": "lengths", "lengths": [0, 2, 5]}],
    }


def main():
    import sys
    from pathlib import Path
    from .store import read_json, write_json, json_bytes

    request_path, response_path = map(Path, sys.argv[1:3])
    limit = int(sys.argv[3])
    request = read_json(request_path)
    response = propose(request["problem"], request["history"])
    if len(json_bytes(response)) > limit:
        response = {"status": "response_limit"}
    write_json(response_path, {"schema_version": 1, "response": response})


if __name__ == "__main__":
    main()
