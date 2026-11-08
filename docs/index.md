# agent-optimize user guide

Decorate an Awkward computation, run your analysis once through `agent-optimize`,
and let it check whether a compiled replacement is eligible for local reuse.
Your readable Python function remains the reference and fallback.

The default generator is a deterministic **local mock**. It deliberately proposes
an incorrect loop before responding to feedback with a corrected loop. The
recording, Numba compilation, independent correctness checks, timing, storage,
and ordinary dispatch are real. Mock mode needs no credential or network access.
An optional OpenAI provider uses the official SDK and strict structured proposals
when explicitly selected; the independent evaluator owns correctness and eligibility.

## Run a script with inline dependencies

Install the generic optimizer once and put uv on PATH. For this GitHub-distributed
package, a tool installation can use:

```sh
uv tool install git+https://github.com/gordonwatts/agent-optimize.git
agent-optimize run --max-calls 2 examples/q16_ttbar.py --n-files 1
```

No `--script` flag is needed on `agent-optimize`. Optimizer options go before the
script path. Everything after the path is forwarded to the script, including
its own `--json`, separators, empty arguments, and arguments containing spaces.

The launcher reads the inline metadata without importing the analysis. It asks
uv to select `requires-python` and resolve `dependencies`, then starts the
controller in that environment. The controller executes the analysis once and
launches replay and evaluation workers with the same Python interpreter.
The report and replay results record `python_executable` for diagnosis.

The script's declared `agent-optimize` dependency controls the library version
and source. It is not replaced with the launching installation. If the script
does not declare the package, the launcher adds its own installed source/version.
Conflicting or unavailable requirements fail before the analysis starts.
`--provider openai` additionally requests the generic `agent-optimize[openai]`
extra in that environment, while retaining the declared package version/source.

The first resolution may download packages or Python; uv caches resolved
environments and packages. No packages are installed into the active interpreter.
Normal execution stays `uv run --script analysis.py`; the script must declare
`agent-optimize` if it imports it. Normal execution reuses saved replacements
when available and does not begin an optimization search.

This initial launcher consumes `dependencies` and `requires-python`. It rejects
script lockfiles and `[tool.uv]` configuration rather than silently ignoring them.
For those features, prepare a suitable environment with uv and use explicit
command mode below. Relative dependency paths follow uv's requirements-file
semantics. Use absolute file URLs for local sources when launching from a
different working directory.

## Install and run

Use Python 3.12 or 3.13 and a separate environment on each platform. From the checkout's
root directory on native Windows:

```powershell
python3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test,docs]"
.\.venv\Scripts\agent-optimize run -- .\.venv\Scripts\python examples/filter_sum.py --max-events 1000
.\.venv\Scripts\python examples/filter_sum.py --max-events 1000
```

On Ubuntu:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,docs]'
.venv/bin/agent-optimize run -- .venv/bin/python examples/filter_sum.py --max-events 1000
.venv/bin/python examples/filter_sum.py --max-events 1000
```

Substitute the path to your Python 3.12 interpreter if `python3.12` is unavailable.
The example prints `events=1000 total_ht=64115.0`. It may legitimately remain
ineligible: dispatch and conversion overhead can exceed any kernel improvement.
No measured speedup is promised.

Everything after `run --` is the existing analysis command, forwarded as separate
arguments with spaces and script options preserved. The command runs the analysis
once; later evaluations replay only captured function calls. Imports and ordinary
calls never start a search or record data.

## Select an optional live provider

Install `pip install -e '.[openai]'`, configure `AGENT_OPTIMIZE_OAI_API_KEY`, and
select live generation explicitly:

```text
agent-optimize run --provider openai --model gpt-6.1-sol --provider-seconds 75 --max-output-tokens 4096 -- python analysis.py
```

The supported models are `gpt-6.1-sol`, `gpt-6.1-terra`, and `gpt-6-luna`.
Credential lookup is `--api-key` > `AGENT_OPTIMIZE_OAI_API_KEY` in the environment
> project root `.env` > user home `~/.env`. There is no `OPENAI_API_KEY` fallback,
and configuring a credential does not enable live mode. Files named `.env` are
ignored by Git. Live calls incur API usage. See the
[provider guide](live-provider.md) for setup, privacy, SDK settings, and exact bounds.

The ordinary test suite uses fake clients without credentials. The real-API test
is skipped unless `AGENT_OPTIMIZE_LIVE_TESTS=1` is explicitly set; then run
`python -m pytest tests/test_openai_provider.py::test_opt_in_live_proposal -q`.
It defaults to `gpt-6-luna`, overridable with `AGENT_OPTIMIZE_LIVE_MODEL`.

## Measured synthetic comparisons

On November 6, 2026, native Windows trials compared all three models using one
seed/repeat and two proposals per supported case. The seven-case small suite
used 1,000 events; a `multi_field_regions` follow-up used 100,000 events.
Luna and Sol each passed correctness for 12/12 proposals across both batches;
Terra passed 11/12, with one source rejected before execution. No candidate was
saved: the complete dispatch path was slower than the reference on these
recordings. Both unsupported controls retained Python with zero provider calls.

Across proposal medians, reference versus complete replacement took 1.48–5.11
versus 50.98–59.04 ms/call in the small suite and 22.57–24.34 versus 57.98–60.49
ms/call in the larger follow-up. Complete dispatch includes repeated source and
dependency verification; these timings do not isolate the generated kernel.
These are synthetic, single-machine observations; some early
trials overlapped a test suite. They do not establish a model ranking or a speedup.
The README's measured-results table contains per-case counts, timing definitions,
usage, budgets, and exact saved evidence identifiers.

List cases with `python -m benchmarks.compare_models --list`. An offline smoke run:

```text
python -m benchmarks.compare_models --provider mock --cases integer_threshold dijet_combinatorics nearest_jet_matching --events 16 --max-proposals 2
```

For live comparison, explicitly pass `--provider openai` and repeat `--model`:

```text
python -m benchmarks.compare_models --provider openai --model gpt-6-luna --model gpt-6.1-terra --model gpt-6.1-sol --events 1000 --max-proposals 2
```

Each model/case/repeat gets a fresh artifact root and retains settings, source/input
digests, reports, and `comparison.json`. Override the budgets explicitly to match
the measured README runs; runner defaults differ from those trials.

## Mark the computation

```python
import awkward as ak
from agent_optimize import optimize

@optimize
def ht(events, threshold=30):
    jets = events.Jet
    selected = jets[(jets.pt > threshold) & (abs(jets.eta) < 2.4)]
    return ak.sum(selected.pt, axis=1)
```

During the explicit optimization command every decorated call executes its original
Python function exactly once. Its metadata and `ht.__wrapped__` remain available.
Multiple encountered functions and repeated calls are reported independently.

Compiled reuse currently targets one standard CPU Awkward array containing a
one-level `ListOffsetArray` of numeric records, with supported Python/NumPy numeric
or Boolean scalar arguments. The expression supports field access, comparisons,
`abs`, Boolean array masks combined with `&`, `|`, or `~`, filtering, and a per-event
`ak.sum(..., axis=1)`. Scalar inversion/indexing is unsupported. Integer/Boolean
fields up to 64 bits and float32/float64 fields are checked conservatively.

Unsupported layouts, missing values, unions, nested lists, custom behaviors,
variadic parameters, arbitrary helpers/control flow, mutable globals, side effects,
or unsupported dependency state retain Python with diagnostics. A function must
have source in a file, and its live code must match that source. The current subset
excludes `ak.to_numpy` inside decorated functions; convert outside the decorated
function when needed. Recording can preserve flat NumPy results through supported
dependencies such as `np.asarray`, but automatic reuse targets the unconverted
Awkward reduction result.

## Read the result

Human progress and per-function summaries go to stderr; ordinary child output stays
on its original streams. Summaries include captures, replay failures, skipped-call
diagnostics, proposal outcomes, search stop reasons, saved manifest paths, and a
next action. Measured timing reports separate compilation and first use from the
warmed full replacement path, and list reference/replacement times and the minimum
batch speedup for each retained recording. Controlled acceptance-test timings are
explicitly labelled and are never presented as measured speedups.

For scripts and coding agents, add `--json` before the separator:

```text
agent-optimize run --json -- python examples/filter_sum.py --max-events 1000
```

This noninteractive mode writes **one JSON object** to stdout. Child stdout,
child stderr, and progress go directly to stderr, so noisy script output cannot
corrupt the JSON. A script's own `--json` after `--` is forwarded unchanged.

The result has `schema_version: 1`, `status`, `exit_code`, `child_exit_code`,
`functions`, and `next_action`. Once a session exists it also includes its absolute
directory, artifact root, report path, recording/search limits, captured bytes,
per-function diagnostics, evaluations, timing evidence, and accepted artifact paths.
Provider identity/settings, allowlisted token usage, HTTP request counts, and fixed
provider-error codes accompany live evaluations; an interrupted provider may have
an unknown request count rather than a claimed zero.
Configuration/launch/local failures include `error` with `phase`, `kind`, and
`message`. Final report-storage problems add `report_error`. Fields unavailable
before session creation are omitted; `child_exit_code` is null before a child exits.
The final `report.json` contains the same result when it can be written.

| Outcome | JSON status | Exit code |
| --- | --- | --- |
| Analysis succeeded; search completed or found unsupported/ineligible functions | `completed` | `0` |
| Analysis failed; no replay/search started | `child_failed` | Child's nonzero status |
| Invalid command/configuration, launch, or local report failure | `error` | `2`, preserving a child's existing nonzero status |
| User interrupted command/replay/search | `interrupted` | `130` |

An interrupt preserves completed evaluation checkpoints and saved implementations.
A new invocation starts a new session; it does not resume a partial search. Help
commands display help rather than a JSON run result.

## Bounds and local data

Defaults are 64 MiB retained capture bytes per session and five capture attempts
per function. Unsupported/skipped calls count as attempts. Inputs are preflighted
before copying and never silently truncated; output overflow or observed mutation
discards that capture. The byte count includes buffers, capture JSON, and frozen
reference files; it excludes report bookkeeping and interpreter/reference-function
allocations. Copying uses bounded temporary chunks. These bounds do not limit the
analysis function's own memory use.

Search defaults are three proposals per function shared across input contracts,
120 seconds per function, 30 seconds per worker capped by remaining time, and
300 seconds total search time. Host preparation and comparison check the same
deadline between bounded units of work. Requests are limited to 128 KiB, responses
to 64 KiB, and extra-case proposals to eight. The evaluator alone builds additional
`zeros`, bounded `lengths`, or known-field `boundaries` cases; mandatory cases remain
in force. Effective limits are printed before the command starts. See
`agent-optimize run --help` for the corresponding overrides.

Provider requests contain source, graph, schema, signature, constraints, independent
synthetic examples, and sanitized feedback. Captured arrays/scalars and raw worker
diagnostics stay local. No recorded values or recording-derived counterexamples
are sent to either provider. Source and literal constants are included. Analysis
and worker environments remove provider credentials; workers also remove the
recording marker. Process separation contains worker failures; it is not a sandbox
for hostile code.

## Correctness, timing, and fallback

Reference and candidate execute in separate workers against evaluator-owned tests.
Container, structure, length, and dtype must match; integer/Boolean values compare
exactly and floating outputs compare by bits, including signed zero and NaN payloads.
Fast-math is disabled. Every retained recording and its scalar configuration is
checked alongside deterministic schema-preserving empties, varied/larger lists,
all/no passes, boundaries/adjacent values, NaNs/infinities, cancellation, extreme
values, and difficult sums. Construction is capped at 512 cases per contract and
one million elements in a larger-list case. A failed or unsupported check retains
Python. Passing finite tests is empirical evidence, not a proof of equivalence.

For eligibility, three timing batches on every retained recording each need at
least five samples and a median improvement of at least 10%. The candidate's
75th percentile must also beat the reference's 25th percentile by at least 10%.
Regression, uncertainty, or insufficient evidence makes a candidate ineligible.
Timing measures the selected dispatch mode, conversion, kernel work, and output wrapping;
compilation is recorded separately. Small recorded workloads do not establish
performance on larger or different data.

Fast is the default dispatch mode. The generated `adapter.py` and `kernel.py` are
loaded once and cached for subsequent ordinary calls. Initial load validates
artifact integrity and selects a recorded input contract; the hot path binds
arguments, extracts buffers, executes the kernel and constructs the output.
Source, dependencies and environment are not checked, even on first load. Fast
mode assumes these and the input schema/types remain valid; otherwise it can
silently produce incorrect results. Regenerate artifacts after such changes.
Restart the process or call `clear_dispatch_cache()` after changing/removing saved
artifacts; running fast code does not watch the store for changes.

Select `--dispatch-mode careful` during optimization and
`AGENT_OPTIMIZE_MODE=careful` during ordinary execution to opt into repeated
compatibility checks. The environment variable also sets the CLI default; an
explicit CLI option wins. The two modes store independent timing evidence and
selection entries; neither consumes artifacts accepted only for the other mode.
Both modes retain all pre-acceptance correctness tests. Ordinary careful calls
reuse only artifacts matching the current reference/source,
verified dependencies/constants, full input schema/dtypes and scalar types, and
Python/Awkward/NumPy/Numba/LLVM/platform/CPU signature. Event counts, variable list
lengths, maxima, and observed values are not fixed by recordings. Unknown,
incompatible, missing, or corrupt artifacts fall back to Python. A recoverable
artifact load/compilation/execution failure quarantines that identity, invalidates
its cache, and runs the reference on the current call. User interrupts and reference
exceptions propagate. Ordinary in-process dispatch cannot recover a native crash.

Each run writes `.agent-optimize/sessions/<id>/` with recordings, frozen references,
request/response histories, evaluations, and `report.json`. Completed evaluations
are checkpointed before the next request or accepted-index update. Accepted source,
manifests, and evidence live in `artifacts/`, selection indexes in `accepted/`, and
quarantine diagnostics in `quarantine/`. Deleting `.agent-optimize/` restores
reference-only behavior after restarting or clearing the fast cache. To choose another store, pass `--artifact-root`; set
`AGENT_OPTIMIZE_ARTIFACT_ROOT` to that same root for ordinary later calls.

## Verify and build documentation

With the environment active, these commands work from the checkout root on both
native Windows and Ubuntu under WSL2:

```text
python -m pytest
python -m zensical build --config-file docs/zensical.toml
```

Generated HTML goes to ignored `docs/_build/`. Add `--strict` to fail on documentation
warnings. Python 3.12.11, Awkward 2.14.0, NumPy 2.5.4, Numba 0.68.1, pytest 9.2.0,
and Zensical 0.1.0 are the current test environment. Through step 5, all 278 tests
passed separately on native Windows and Ubuntu under WSL2. Step 6 passed all
296 tests and the documentation build on both native Windows and Ubuntu under
WSL2. Final native Windows example/ordinary, space-containing paths, and timeout
checks also passed; the real example remained ineligible. See
[platform verification](verification.md) for the separate environment results and
source-manifest method. Ubuntu passed 296 tests in 91.45 seconds and its
documentation build; its space-containing 1,000-event example produced the
expected result and remained ineligible. The Linux suite's worker crash and
timeout tests passed with temporary files inside the Linux workspace, and source
bytes matched the synchronized Windows verification snapshot. Both platform
verification gates passed.

The live-stage Ubuntu offline suite passed 343 tests with two skips; separate
opted-in live tests passed on Windows (20.00 seconds) and Ubuntu WSL2 (12.65 seconds).
These later checkpoints supplement the historical step-6 counts above; see
[platform verification](verification.md) for scope.

The [first round workflow](ux-first-round.md) describes the intended user experience;
[implementation notes](implementation-notes.md) include the design and implemented
optional provider. The [source conversation](source-conversation.md) preserves the
exploratory discussion that preceded this implementation.
