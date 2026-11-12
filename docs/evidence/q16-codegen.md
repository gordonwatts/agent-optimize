# q16 assignment-loop scheduling experiment

On the same 7,552 selected events, scheduling score expressions at their first
required loop depth and calculating the final mass only when the best score
changes reduced native kernel time. Every real-data result retained the original
floating-point bits. The existing assignment tests also passed, including their
generated nonfinite, signed-zero, tie and empty-input cases (4 tests, 38.78 s).

The historical generator comes from commit `9c9004a`; both generators receive the
same parsed reference, input contract and installed dependencies. No model API
requests were made. This changes the generic assignment compiler, not the q16
physics calculation.

## Full-file warm measurements

| Path | Historical generator | Scheduled generator |
| --- | ---: | ---: |
| Native kernel, preadapted inputs | 25.083 ms | 16.235 ms |
| Full reconstruction with checked input adapter | 212.766 ms | 203.220 ms |
| Full reconstruction with generated fast adapter | 139.692 ms | 130.315 ms |

These are medians of five alternating complete-file passes over the original
235 input groups. The supplied Python reconstruction took 5.741240 s. First full
passes with the checked adapter, including compilation, took 3.728169 s and
3.593442 s. Fast-adapter measurements invoke the generated replacement directly;
they exclude dispatch lookup and are not presented as saved-dispatch timings.

Packing, adaptation and output wrapping dominate this many-small-groups case.
The native kernel improved by 1.55×; the direct fast-adapter full reconstruction
improved by 1.07×. Timing variation limits the precision of these ratios.

## Larger input groups

Concatenating each eight adjacent input groups produces 30 groups containing
exactly the same events. The supplied Python reconstruction then took 0.984554 s.
Across three alternating full-file passes:

| Path | Historical generator | Scheduled generator |
| --- | ---: | ---: |
| Native kernel, preadapted inputs | 26.079 ms | 13.882 ms |
| Full reconstruction with checked input adapter | 49.181 ms | 40.194 ms |
| Full reconstruction with generated fast adapter | 41.919 ms | 30.394 ms |

Grouping is a separate workload configuration, not a compiler optimization.
Within the larger-group configuration, the direct fast-adapter improvement was
1.38×. Neither measurement includes ServiceX, ROOT reading or plotting.

## Saved-dispatch confirmation

A fresh local-generator run saved the scheduled replacement after all 108
verification cases passed. It first rejected the deliberately wrong mock
candidate. The two recorded groups account for two checks; 106 further cases
exercise generated inputs. No model requests, tokens or API costs were incurred.
Recording the analysis took 9.850 s, generation/checking/compilation/timing took
29.143 s, and the complete command took 39.402 s. Accepted compilation took
3.763 s.

The ordinary saved-dispatch benchmark then confirmed replacement use and exact
agreement on all 7,552 events in the original 235 groups. Three alternating
passes gave medians of 5.827037 s for the supplied Python, 5.730112 s for the
revised Python and 0.140905 s for saved dispatch: 41.35× over supplied Python.
The first complete dispatched pass, including compilation, took 3.944911 s.
This comparison includes packing and ordinary dispatch, excludes ROOT reading,
ServiceX and plotting, and is separate from the direct-adapter table above.

## Reproduction

```powershell
.venv/Scripts/python.exe -m benchmarks.q16_codegen --input-file .agent-optimize/q16-delivered.root
.venv/Scripts/python.exe -m benchmarks.q16_codegen --input-file .agent-optimize/q16-delivered.root --coalesce-groups 8 --repeats 3 --output docs/evidence/q16-codegen-coalesced.json
.venv/Scripts/python.exe -m pytest tests/test_assignments.py -q -p no:cacheprovider
.venv/Scripts/agent-optimize.exe run --json --provider mock --verification-groups 2 --max-attempts 2 --artifact-root .agent-optimize/q16-scheduled-final -- .venv/Scripts/python.exe -B examples/q16_ttbar.py --input-file .agent-optimize/q16-delivered.root --events-per-read 128 --output-dir .agent-optimize/q16-scheduled-final-plots
.venv/Scripts/python.exe benchmarks/q16_reconstruction.py --input-file .agent-optimize/q16-delivered.root --artifact-root .agent-optimize/q16-scheduled-final --repeats 3
```

The JSON files retain raw samples, source/data hashes, the resolved historical
commit and environment versions. The original reference returns option arrays;
its selected real-data values are compared after conversion to NumPy. Both
compiled implementations additionally pass the strict container-and-bits check
against the current numeric reference. This validates software preservation;
the independent physics benchmark discrepancy remains unresolved.
