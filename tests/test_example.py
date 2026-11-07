from pathlib import Path
import runpy

import awkward as ak


def test_example_is_deterministic_and_preserves_zero_event_schema():
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "filter_sum.py"))
    make_events = namespace["make_events"]
    ht = namespace["ht"]
    events = make_events(12)
    empty = make_events(0)
    assert ak.to_list(events) == ak.to_list(make_events(12))
    assert events.layout.form == empty.layout.form
    assert ak.to_list(ht(empty)) == []
    assert ak.to_list(ht(events)) == ak.to_list(ht.__wrapped__(events))
