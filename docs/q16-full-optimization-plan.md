# Full q16 reconstruction: work in progress

Issue #2 extends the optimizer boundary to include candidate construction and
scoring. The current q16 example optimizes only scoring. This branch is not yet
a complete implementation and does not claim a full-reconstruction speedup.

## Current foundation

`event_inputs.py` validates a set of event-aligned, standard Awkward arrays with
flat numeric, numeric-record, or jagged numeric-record layouts. It checks forms,
validity and outer lengths and exposes read-only buffer views. It does not yet
participate in recording, evaluation, generation, saved dispatch or fallback.
The forms do not specialize on event counts or jagged lengths. Unsupported
layouts remain rejected. This interface is provisional until integrated.

## Remaining work

- Represent and validate multiple inputs throughout the serialized contract,
  worker protocol, timing, generated cases, saved artifacts and normal dispatch.
  Reject mismatched schemas and lengths before kernel execution and preserve
  reference fallback and existing artifact compatibility.
- Safely capture or replace reference helper dependencies and extend the generic
  operation representation to cover the needed arithmetic, broadcasting,
  records, indexing, combinations, Cartesian products and filters. Do not admit
  arbitrary code or trust generated candidates to define their own tests.
- Implement reusable candidate generation for the accepted operation subset,
  preserving event/assignment order, first-minimum ties, expression order,
  output kind and dtype, and the reference's behavior on empty and nonfinite
  cases. Reject unsupported cases rather than guessing their semantics.
- Build mandatory independent correctness cases including empty and sliced
  layouts, changing event and jet counts, ties, boundaries and nonfinite values;
  compare against the frozen full reference bit for bit where required. Verify
  recording, replay, persistence, applicability, failure and fallback paths.
- Measure the entire warm full-reconstruction replacement path against the
  supplied baseline, including dispatch and first-use cost. State workloads and
  excluded ServiceX delivery, ROOT reading and plotting. Obtain representative
  transformed data if accessible; do not present synthetic results as real-data
  evidence.

The prior case study reported all 7,552 values bitwise equal to the supplied
script, while its 1537.14 GeV mean disagreed with the 2477.6205 GeV physics
benchmark beyond 2%. Software equivalence is not physics validation. The cause
remains unresolved; this work makes no claim to resolve it.
