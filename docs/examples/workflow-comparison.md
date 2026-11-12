# auto-opt and agent-optimize: measured comparison

This investigation compares an adapted auto-opt workflow with agent-optimize on
the same in-memory Awkward workloads. It also improves the raw q16 reconstruction
and checks whether that source can be translated. Work is tracked in
[issue #9](https://github.com/gordonwatts/agent-optimize/issues/9).
The [short test log](q16-test-log.md) is suitable for pasting into a run log.

## What was compared

[auto-opt](https://github.com/pfackeldey/auto-opt/tree/56cbad7d561097037eefaecc5440268d83c6b9a3)
contains a Markdown optimization instruction template and a configuration for a
different Analysis Grand Challenge application. It has no executable optimizer
or published benchmark results in that revision. Its loop asks an agent to
profile, change source, check correctness, measure, keep improvements and retain
failed attempts.

We adapted that loop to this repository's existing environment and strict output
checks. We did not run the separate AGC application, its pixi setup, or its
requirement to attempt three custom C implementations. The shared checkout and
curated benchmark sources replace auto-opt's per-attempt branch/commit layout.
The Numba loops are explicit source changes written during this investigation;
they are not functions emitted by an executable auto-opt package.
The initial raw and loop candidates were hypothesis-driven; profiles were then
collected to check the bottlenecks and guide further changes. This differs from
the template's requirement to profile before the first edit. We retain the
candidate sources and measurements together rather than claiming a literal run
of every upstream procedural instruction.

The agent-optimize side uses its deterministic local generator, three proposals
per supported function, and two captured input groups. This tests its current
translation and reuse capabilities without additional paid requests. It is not
an equal-model, equal-token-budget contest between autonomous agents. The
desktop agent's optimization token usage and cost are unavailable here.

## Shared-workload method

`benchmarks/workflow_comparison.py` uses 8,192 synthetic events with variable jet
counts and empty collections. Each candidate is checked bit for bit across
three seeds and six input sizes, plus nonfinite/signed-zero inputs and different
scalar settings. The further matching candidate also checks radius boundaries,
wrapped phi and large/nonfinite coordinates. These are software checks, not
detector or physics validation.

Warm timings alternate implementation order and include the complete function,
ordinary saved dispatch where applicable, and output wrapping. They exclude
imports, generation, and compilation. Fresh subprocess measurements separately
record the first function call and peak process resident memory, including
imports and JIT allocation. The memory figures describe whole-process peaks;
they do not isolate temporary kernel arrays.

The matching baseline calculates all distances and their minimum. The first
candidate uses streaming Numba loops. The second tests whether a match exists,
rejects pairs outside the eta radius before calculating phi, and stops at the
first passing pair. It preserves this workload's returned match count, rather
than returning a nearest-distance array.

Reproduce generation, then ordinary execution from the repository environment:

```text
agent-optimize run --json --artifact-root .agent-optimize/workflow-comparison --provider mock --verification-groups 2 --max-attempts 3 -- python -m benchmarks.workflow_comparison --record-only
python -m benchmarks.workflow_comparison --artifact-root .agent-optimize/workflow-comparison
```

Use the same Python environment for both commands. The benchmark writes machine
readable timings and provenance to `docs/evidence/workflow-comparison.json` and
profiles to the artifact directory. The retained generation summary is
`docs/evidence/workflow-comparison-generation.json`.

## Shared-workload results

Medians of 11 alternating warm measurements, in milliseconds:

| Workload | Original Python | Agent-written loop | agent-optimize ordinary call |
| --- | ---: | ---: | ---: |
| Filter and sum | 1.669 | 0.351 | 0.287, saved replacement |
| Dijet combinations | 7.186 | 0.387 | 7.980, Python fallback |
| Nearest-jet match count | 13.147 | 0.773, eta pruning and early exit | 13.885, Python fallback |

The initial streaming matching loop took 1.323 ms. Pruning and early exit improved
it further. All three initial loop candidates passed 19 input checks and four
scalar-setting checks each; the further matching candidate additionally passed
its boundary and scalar checks. The ordinary agent-optimize calls also match.

For filter-and-sum, agent-optimize was slightly faster than the hand-written loop
in this run. Its current front end could not capture the other two sources,
where explicit source changes provided improvements. These results establish a
coverage difference on these workloads, not a universal ranking.

The mock recording/search took 18.508 s overall, with 16.539 s in generation,
checking, compilation and timing. It made zero API requests. Initial compilation
in separate fresh processes took roughly 0.3–0.4 s for the source loops. Whole
process peaks were approximately 98–109 MB for Python/fallback and 145–147 MB
for JIT implementations. Warm speedups therefore come with startup and process
memory costs. All raw samples and exact first-call/memory figures are retained
in the JSON evidence.

## q16 findings

The [raw-source experiment](q16-raw-optimization.md) retains the attempted
algorithms, profiling evidence, rejected change, exact-output checks and the
unsupported translation diagnostics. Improving raw source can help substantially
without making that source compatible with the current compiler.

The [compiler experiment](../evidence/q16-codegen.md) separately measures the
historical and scheduled generators with the same supported source. It includes
kernel-only, checked-adapter, fast-adapter and actual saved-dispatch evidence.
These boundaries must stay distinct when interpreting speedups.

The delivered file contains 7,552 selected events. Its original approximately
six-second baseline measures warm reconstruction over 235 small function calls,
including candidate construction and selection. It excludes ServiceX, ROOT
reading and plotting. Batching therefore matters alongside arithmetic.

The generated Numba source and the raw NumPy source both preserve the measured
software outputs. This does not resolve the original analysis's independent
physics benchmark discrepancy.

## Actual ROOT reader and CLI

The example now exposes `--events-per-read`, defaulting to 1,024 ROOT entries
before selection. The historical helper/benchmark default remains 128 for
reproducibility. Use `--events-per-read 128` on the example for smaller groups
and lower reference-path memory use.

The actual 1,024-entry reader produced 30 groups. A control pass with 128-entry
reads verified every selected numeric input and dtype, jet-list length, field
and event order exactly. Every reconstructed output also matched the supplied
Python reference bit for bit. Five alternating warm passes measured:

| Implementation, 30 actual ROOT groups | Median time |
| --- | ---: |
| Supplied Python reconstruction | 0.958031 s |
| Current Python reference | 1.208838 s |
| Ordinary saved dispatch | 0.032197 s |

The first dispatched full pass, including compilation, took 3.629156 s. On this
short workload, that first pass is slower than either Python reference at the
larger group size. The pure NumPy experiment avoids JIT startup; the compiled
path has the best measured warm runtime. These timings exclude ROOT reading,
ServiceX and plotting. Direct larger reads require no concatenation preparation,
unlike the separately reported coalescing experiment.

```text
python -m benchmarks.q16_reconstruction --input-file .agent-optimize/q16-delivered.root --artifact-root .agent-optimize/q16-scheduled-final --events-per-read 1024 --repeats 5
python examples/q16_ttbar.py --input-file .agent-optimize/q16-delivered.root --output-dir .agent-optimize/q16-read1024-cli-plots
```

The ordinary CLI completed on the same file, reported 7,552 selected events and
wrote `plot_1.png`. For that CLI check, `AGENT_OPTIMIZE_ARTIFACT_ROOT` pointed to
the same compatible saved artifact. The retained measurement is
`docs/evidence/q16-read1024-dispatch.json`.

## Capabilities and practical differences

| Aspect | Adapted auto-opt workflow | agent-optimize |
| --- | --- | --- |
| Optimization scope | Agent can change algorithms, batching and source structure | Captures and translates supported decorated function expressions |
| Correctness | Run must configure and enforce an oracle; template suggests numeric tolerance | Frozen-reference checks require exact output structure, dtype and floating bits |
| Unsupported source | Agent may rewrite it or try another implementation | Retains Python with a diagnostic |
| Reuse | Kept source edits become the implementation | Saved artifact with applicability checks and reference fallback |
| Effort evidence here | Explicit candidates and measured outcomes; agent billing unavailable | Recorded generation duration, attempts, checks, token/cost coverage |

The broader editing scope is useful for reducing work before compilation, as
the raw q16 experiments show. It also places more responsibility on the chosen
verification and benchmark boundaries. The template alone does not provide
agent-optimize's frozen inputs, automatic rejection, artifact validation or
fallback behavior.

The most useful combination is to profile and improve the source first, then
translate supported boundaries and measure the full replacement path. A faster
kernel is insufficient when packing and dispatch dominate. Results from the
adapted workflow do not establish that upstream auto-opt will produce the same
edits, use less money, or outperform agent-optimize with equal model resources.
