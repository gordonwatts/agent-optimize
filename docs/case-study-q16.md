# Case study: full q16 reconstruction

The q16 analysis reconstructs one top-pair mass per selected event. On one cached
ServiceX file with 7,552 selected events, the saved Numba replacement took
**0.0502 seconds warm**, compared with **1.0158 seconds for the supplied Python**
under the same batching: a **20.2× warm speedup**. Its first use took
**3.9442 seconds including compilation**. These are reconstruction timings,
excluding ServiceX delivery, ROOT reading, imports and plotting.

## What changed

The supplied analysis selects exactly one electron or muon above 25 GeV and at
least four jets above 30 GeV. It solves the neutrino momentum from the W-mass
constraint, tries distinct four-jet assignments, and selects the first assignment
minimizing `abs(m_had - 172.5) + abs(m_lep - 172.5)`.

We changed how that calculation is expressed and executed:

1. The supplied Python builds pairs and candidate records with Awkward operations.
   Its numerical code is preserved in
   [`tests/fixtures/q16_reference.py`](https://github.com/gordonwatts/agent-optimize/blob/main/tests/fixtures/q16_reference.py).
2. The rewritten Python in
   [`examples/q16_ttbar.py`](https://github.com/gordonwatts/agent-optimize/blob/main/examples/q16_ttbar.py)
   packs the inputs into an event record and places the entire numerical
   calculation in `@optimize reconstruct_events`. It uses explicit jet indices
   and arithmetic that the compiler recognizes. This Python function remains
   the correctness reference for generated code; rewriting it alone does not
   make it faster, as the table below shows.
3. The generated Numba replacement streams the same ordered assignments instead
   of constructing candidate arrays. The compiler now reuses arithmetic at the
   earliest applicable loop depth and calculates the final mass only when the
   winning score changes. Normal calls use this saved replacement when its
   compatibility checks pass, otherwise they use the rewritten Python reference.

The script also reads 1,024 ROOT entries at a time before event selection,
instead of 128. For this file that reduces reconstruction calls from 235 to 30,
avoiding repeated Python, input-packing and dispatch overhead. Larger groups
can use more memory on the Python path; `--events-per-read 128` retains the
smaller groups. The selected events, their order and their numeric inputs are
unchanged. None of these changes alter the physics calculation.

## Current full-reconstruction measurements

Measured November 13 on Windows, Python 3.13.14, Awkward 2.14.0, NumPy 2.5.4
and Numba 0.68.1. Every row uses the same 7,552 events in the same 30 groups.
Warm times are medians of three alternating full-file passes.

| Implementation | Warm full-file reconstruction | First use including compilation |
| --- | ---: | ---: |
| Supplied Python: preserved `q16_reference.reconstruct` | 1.015773 s | Not separately measured |
| Rewritten Python: `q16_ttbar.reconstruct`, with its Python `reconstruct_events` | 1.307562 s | Not separately measured |
| Rewritten function using the saved Numba replacement during normal calls | 0.050214 s | 3.944235 s |

Both Numba figures in the last row include input packing, normal dispatch and
output construction. The first-use figure measures the first complete pass
through the **Numba replacement**, including its lazy compilation; it is not an
optimization-search duration or a first-use measurement of either Python row.
Thus this short file benefits in warm execution, while compilation makes the
first Numba pass slower than the measured warm Python passes.

All reconstructed values match the supplied Python bit for bit. A separate
reader check verified that using 1,024-entry reads preserves the selected input
values, dtypes, jet-list lengths and event order from 128-entry reads.

## Run and reproduce

The example owns its Python 3.13 environment through inline uv metadata, including
agent-optimize from the head of `main`. Ordinary execution fetches one ServiceX file
and writes `plot_1.png`:

```bash
uv run --script examples/q16_ttbar.py --n-files 1 --output-dir plots/q16
```

`--n-files 0` requests the whole dataset. To reuse already transformed query
output without ServiceX, use `--input-file delivered.root` instead.

Install the generic CLI directly from GitHub; no repository checkout is needed:

```bash
uv tool install git+https://github.com/gordonwatts/agent-optimize.git@main
```

Then record inputs, generate and check a replacement, and save it:

```bash
agent-optimize run --report --artifact-root .agent-optimize/q16 --verification-groups 2 examples/q16_ttbar.py --input-file delivered.root --output-dir plots/q16
```

The default generator is local and makes no model API calls. A replacement must
pass exact correctness and measured speed checks before it is saved. Model-based
generation is optional; see the [provider guide](live-provider.md).

In a prepared environment with the example's dependencies, reproduce the table
using the compatible saved artifact:

```bash
python -m benchmarks.q16_reconstruction --input-file delivered.root --artifact-root .agent-optimize/q16 --events-per-read 1024 --repeats 3
```

The benchmark checks the complete result against the supplied Python, verifies
that saved dispatch was used, and prints raw samples plus the separate first-use
time. Ordinary later script runs reuse the artifact when
`AGENT_OPTIMIZE_ARTIFACT_ROOT` points to that same directory.

## Correctness and limits

Exact agreement preserves the supplied software, not an independently validated
physics result. The observed mean remains 1537.1414 GeV, outside q16's stated
2477.6205 GeV reference by more than its 2% tolerance. That discrepancy remains
unresolved. Synthetic checks additionally cover nonfinite values, signed zeros,
ties, changing lengths and empty inputs.

Historical [Windows/WSL full-reconstruction evidence](evidence/q16-full-reconstruction.json)
uses the older 235-group setup and should not be compared as a compiler-only
speedup against this 30-group table. The [local-generator report](examples/q16-report.md),
[live-model report](examples/q16-live-report.md) and
[rejected-candidate diagnosis](examples/q16-full-data-check.md) retain separate
search, usage and correctness evidence. The live-model trial did not produce an
accepted replacement; the timings above use the successful local generator.
