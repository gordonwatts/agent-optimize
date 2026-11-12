# q16 raw-source optimization experiment

This experiment improves the supplied Python/Awkward reconstruction independently
of Numba compilation. It uses the cached delivered ROOT file, with 7,552 selected
events and the original 235 reconstruction calls. Loading ROOT data, ServiceX and
plotting are outside the reconstruction timing. The earlier approximately six
seconds measured this numerical reconstruction work, not the complete analysis.

Final isolated measurements on Windows 11, Python 3.13.14, Awkward 2.14.0 and
NumPy 2.5.4 used three alternating warm passes per implementation:

| Implementation, original 235 groups | Median reconstruction time |
| --- | ---: |
| Supplied Python/Awkward | 5.704917 s |
| Only move the final mass calculation after selection | 5.955669 s |
| Grouped NumPy, flatten each column separately | 0.466855 s |
| Grouped NumPy, flatten the records once | 0.371083 s |
| Grouped NumPy, also reuse hadronic/leptonic calculations | **0.328130 s** |
| Fresh saved Numba implementation, including normal input packing/dispatch | **0.140171 s** |

The best raw-source candidate is **17.4× faster** than the supplied raw source;
the saved compiled implementation is **40.7× faster** on the same groups. Every
one of the 7,552 present output values matches bit for bit. Both grouped NumPy
variants used their optimized path for every real input group, with no fallback.

The first raw NumPy pass with empty assignment-index caches took 0.353486 s.
The first saved-Numba pass took 4.132640 s, including 3.983183 s of compilation.
These are first numerical passes after data loading, not complete fresh-process
startup timings. The raw candidate therefore has a useful short-run advantage;
the compiled implementation wins after compilation is amortized.

Combining eight neighboring input groups produces 30 reconstruction calls:

| Implementation, 30 groups | Reconstruction | Group preparation | Total |
| --- | ---: | ---: | ---: |
| Supplied Python/Awkward | 0.939914 s | 0.027185 s | 0.967099 s |
| Best grouped NumPy | 0.058024 s | 0.027185 s | 0.085210 s |
| Saved Numba | 0.031613 s | 0.027185 s | 0.058798 s |

The preparation cost is charged once for each full pass in the Total column.
It must not be omitted when comparing these results with the original grouping.
Reading larger ROOT groups directly can avoid this extra concatenation step;
that is a separate application-level change. Larger groups also increase the
raw reference's temporary-array memory needs.

The candidate in `benchmarks/q16_raw_candidates.py` groups events by jet count,
extracts NumPy columns once, caches the ordered assignment indices by jet count,
and vectorizes the arithmetic. Its second variant calculates each hadronic
four-vector/mass before expanding the leptonic-b choice, and calculates each
leptonic four-vector/mass once per jet. Both preserve the original expression
order and first-assignment tie breaking. They calculate the final combined mass
only for the selected assignment. They use no JIT or additional dependency.

The optimized path handles the selected float64 event records in this analysis.
Other numeric dtypes, nonfinite inputs or nonfinite intermediate results run the original reference.
The benchmark records these branches explicitly, so successful exceptional-input
checks do not imply that NumPy independently reproduces Awkward's NaN selection
rules. Empty inputs, fewer than four jets and exact score ties are also checked.
The oracle checks output shape, dtype, missing-value positions and the exact bits
of present values. It does not require the same internal Awkward buffer layout.
The final run passed 174 synthetic comparisons, including float32, optional jet
records, optional leptons, optional lepton energy, nonfinite and extreme finite
energies, subnormal energies, signed zero and ties. Across the two grouped
variants the exceptional tests recorded 48 nonfinite-input fallbacks, 16
degenerate-intermediate fallbacks, two dtype fallbacks and six optional-schema
fallbacks; 54 exceptional groups used the optimized path. Intentional max-float
tests produce overflow warnings in both the reference and candidate.

One simpler source change—selecting the winner before calculating its final mass
while retaining all other Awkward expressions—was slower in the exploratory
measurements and is retained as a rejected experiment. Profiling showed that
Awkward broadcasting and layout handling dominate the supplied source; reducing
arithmetic alone did not reduce that overhead.

Run the comparison from the repository root with the q16 environment:

```powershell
.venv/Scripts/python.exe -m benchmarks.q16_raw_optimization --input-file .agent-optimize/q16-delivered.root --artifact-root .agent-optimize/q16-scheduled-final --repeats 3 --no-profile --output docs/evidence/q16-raw-optimization.json
```

The artifact root must contain a compatible accepted reconstruction. The
benchmark fails if the saved Numba implementation is not observed. Omitting
`--artifact-root` compares the raw candidates alone. First raw passes clear the
assignment-index caches; warm passes reuse them. The saved implementation's
first pass includes Numba compilation. Omitting `--no-profile` also collects
separate cProfile runs after the reported warm timings.

`--coalesce 8` is a separate batching experiment: it combines eight neighboring
input groups without changing the supported numerical source. Its preparation
time is reported separately. Correctness is still checked against results from
the original 235 groups. This distinguishes algorithm changes from reduced call
overhead and allows the existing translated source to benefit from batching.

```powershell
.venv/Scripts/python.exe -m benchmarks.q16_raw_optimization --input-file .agent-optimize/q16-delivered.root --artifact-root .agent-optimize/q16-scheduled-final --coalesce 8 --repeats 3 --no-profile --output docs/evidence/q16-raw-coalesced.json
```

The grouped NumPy implementation is a reviewable source candidate, not a new
default reconstruction boundary. The benchmark attempts the real reference
freezer and the assignment parser and preserves their diagnostics. Python loops,
comprehensions and the NumPy grouping helpers fall outside the current supported
translation subset. A raw speedup therefore does not establish a translated
speedup. The existing supported Awkward source and saved dispatch remain intact.
The actual freezer attempt reported `UnsupportedReference: unsupported dependency:
any`; the direct assignment-parser probe reported that it requires one
event-record argument. No model-generation attempt was made for an unsupported
reference.

The [final original-group evidence](../evidence/q16-raw-optimization.json) and
[coalesced evidence](../evidence/q16-raw-coalesced.json) retain timing samples,
source hashes, environment, branch observations and the accepted artifact ID.
The input file SHA-256 is
`bd3ecc50194dc836bf0e8f9f7201fc000309644fc9ad0f8f3cc4dea5e3bc5a50`.
Observed jet counts ranged from four to fourteen. Cached index arrays for all
these multiplicities occupy 1,251,360 bytes; this is not a process peak-RSS
measurement. Independent per-implementation peak RSS was not collected here.

[Exploratory evidence](../evidence/q16-raw-exploratory.json) retains the preceding
five-pass measurements, separate cProfile results and complete source snapshots
before the optional-schema guard. The guard was then added and the final
three-pass measurements rerun. The initial algorithm changes were hypotheses
from source inspection; the subsequent profiles confirmed the bookkeeping cost.

This follows an adapted profile/change/check/measure workflow inspired by
`pfackeldey/auto-opt`, rather than executing its Analysis Grand Challenge example
unchanged. The workloads, environment and strict bitwise oracle are specific to
this repository. No model API requests are needed for this raw-source experiment;
agent effort and unavailable token accounting must not be reported as a measured
zero-cost optimization search.

Software equivalence to the supplied source does not resolve its independent
physics validation discrepancy.
