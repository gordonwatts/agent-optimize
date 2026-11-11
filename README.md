# agent-optimize

Decorate an Awkward computation, record an analysis once, and evaluate bounded
proposals with independent correctness checks and measured eligibility.
Compatible saved implementations are reused automatically with Python fallback.
The command runs your script once, checks the compiled numerical kernel against
recorded and generated inputs, and saves it only after repeatable improvement.
The default generator is a deterministic local mock exercising the real Numba
evaluation and feedback loop. It needs no credential or network access. An
explicit `--provider openai` enables the optional official OpenAI SDK integration.

For a ServiceX analysis with a larger dependency set, see the
[q16 WSL2 case study](docs/case-study-q16.md) and
[standalone example](examples/q16_ttbar.py). The script declares its own analysis
dependencies in its inline metadata; there is no analysis-specific package extra.
The library supports Python 3.12 and 3.13; q16 uses 3.13.
Its decorated boundary covers neutrino reconstruction, ordered jet assignment,
candidate construction, mass scoring and first-minimum selection.

## Share what a run did

```sh
agent-optimize run --report analysis.py
```

`--report` writes a Markdown file in the current directory, named after the
analysis/command plus a session identifier. The path is printed to stderr and
included as `markdown_report_path` in JSON. Existing reports are never overwritten.
The analysis runs once; report generation reads its retained evidence.

`--verification-groups 2` saves up to two real-data groups per function for
correctness checks and timing; these appear as verification groups 1 and 2 in
the report. `--max-attempts` limits replacement-code attempts per function and
defaults to three.

The report starts with original-versus-replacement function timings and the
speedup, followed by the elapsed time and API cost needed to obtain the result.
It explains the saved verification groups, which can each contain many events, and
distinguishes recorded inputs from generated correctness checks. Each optimized
function shows its frozen Python reference, a visible `@optimize` marker and
the saved Numba code. Compilation and first use are included in optimization time.
Unsupported or unsuccessful functions explain why Python remains in use.
See the [real q16 sample report](docs/examples/q16-report.md). The
[live model trial report](docs/examples/q16-live-report.md) records real token
usage and partial cost evidence from a ServiceX run; no Numba function was
accepted in that trial. Generate an offline
q16 example with:

```sh
agent-optimize run --report examples/q16_ttbar.py --synthetic-events 128
```

Function timings include the overhead of calling the replacement through
`@optimize` and describe the saved verification groups. Other analysis work is outside
that comparison. Total elapsed time excludes Python environment preparation.
Controlled test timings are labeled rather than presented as measurements.
If an explicitly requested report cannot be written, the run reports an error
(exit 2 unless the analysis already failed with its own nonzero status).

Usage totals are available in human output and `usage_summary` in the durable
JSON report, even without `--report`. They cover agent-optimize's model requests,
including rejected/refused/incomplete attempts. Unknown usage remains unknown;
partial totals show their coverage. Mock/no-request runs report zero API usage.
Input, output and total tokens are separate from cached-read, cache-write and
reasoning detail counts, which must not be counted again as additional tokens.

Costs are USD **estimates**, not invoices. Each priced attempt retains a verified
rate snapshot and source/date, so later pricing updates do not reprice history.
The initial profile uses OpenAI direct API Standard rates, default 30-minute
cache writes and the documented long-context adjustment. It excludes credits,
taxes and regional premiums. Missing usage/cache details, unsupported models or
service tiers leave the total estimate unavailable while preserving known costs.
Requests explicitly select the Standard (`default`) service tier. Pricing is
checked in rather than fetched during normal runs; rate updates require reviewing
the [official pricing](https://developers.openai.com/api/docs/pricing),
[cache accounting](https://developers.openai.com/api/docs/guides/prompt-caching)
and model pages linked in each snapshot.

## Script-owned environments

With uv on PATH, pass a Python script directly to resolve its inline dependencies
and Python requirement before recording:

```text
agent-optimize run --verification-groups 2 examples/q16_ttbar.py --n-files 1
```

Optimizer options go before the script; all following arguments belong to the
script. The controller, analysis, and replay/evaluation workers use the resolved
interpreter. A script's declared `agent-optimize` requirement, including its pin
or Git source, is honored. Without that declaration the launcher's installed
package source/version is used. `--provider openai` adds the generic OpenAI extra.

Ordinary execution remains:

```text
uv run --script examples/q16_ttbar.py --n-files 1
```

Ordinary execution can reuse saved replacements; it does not start a search.
The existing `agent-optimize run -- <command> [arguments]` form executes a command
in its supplied environment. See the [user guide](docs/index.md) for scope and
configuration details.

## Fast execution and optional careful checks

The default execution mode is now **fast**. The optimization command still runs
all reference replay and generated correctness cases, then measures the actual
ordinary fast path before saving a candidate. Each saved artifact includes
`kernel.py` and a host-generated `adapter.py` that extracts buffers and constructs
the output without compatibility checks on every call.

```text
agent-optimize run --dispatch-mode fast -- python analysis.py
python analysis.py
```

The first call loads and checks artifact integrity, chooses a compatible recorded
input contract, and compiles the kernel. Later calls use the cached generated
adapter and kernel. **Fast mode assumes the function, dependencies, environment,
and input structure/types stay valid.** It does not detect changed source or
dependencies, even in a fresh process. After changing them, regenerate artifacts.
After changing array schemas/types, regenerate or select careful mode; an unchecked
call can silently produce a wrong result. Values, event counts, jagged lengths,
and scalar values may vary within the established contract. Removing artifacts
on disk does not evict code already cached in a running process; restart or call
`agent_optimize.dispatch.clear_dispatch_cache()` to reload the store.

For repeated applicability/integrity checks, optimize with `--dispatch-mode careful`
and set `AGENT_OPTIMIZE_MODE=careful` when running the script normally. The mode
environment variable also supplies the CLI default; an explicit CLI option wins.
Fast and careful artifacts have independent timing evidence and index entries;
one mode does not silently reuse an artifact measured in the other mode. Missing
or unusable artifacts retain Python. A recoverable candidate exception also falls
back to Python and quarantines the artifact. Incorrect results without an exception
are not detected during fast dispatch.

### Fast code running in a fresh process

On November 7, 2026, the mock generator produced and saved fast artifacts for
three synthetic workloads on Windows. A separate ordinary Python process then
loaded each artifact, confirmed the generated implementation actually ran, and
compared its output with the reference. The warm measurements below are medians
of 21 alternating calls. The 100,000-event run reused the artifact recorded
with 1,000 events.

| Workload | Events | Reference | Fast ordinary call | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Integer threshold and sum | 1,000 | 1.299 ms | 0.110 ms | 11.9× |
| Central jet HT | 1,000 | 2.337 ms | 0.150 ms | 15.6× |
| Multi-field selection | 1,000 | 4.059 ms | 0.184 ms | 22.0× |
| Multi-field selection | 100,000 | 19.465 ms | 6.606 ms | 2.9× |

First use in the fresh processes took about 0.8–0.9 seconds, including Numba
compilation; it is excluded from the warm measurements. These are illustrative
synthetic inputs, not a general performance guarantee. A separate Ubuntu WSL2
run also saved and reused the 1,000-event multi-field implementation: 4.196 ms
reference versus 0.205 ms fast (20.5×), with 0.530 seconds on first use.
An additional native Windows run with the live `gpt-6-luna` provider generated
and saved a candidate that a fresh process used with an exact result match:
4.162 ms reference versus 0.193 ms fast (21.6×), with 0.876 seconds on first
use. Live model results can vary between runs.

To reproduce the multi-field example from the repository root on Windows, using
the installed Python environment:

```powershell
python -m agent_optimize run --dispatch-mode fast --artifact-root .agent-optimize/fast-demo -- python -m benchmarks.fast_in_situ --case multi_field_regions --events 1000
$env:AGENT_OPTIMIZE_ARTIFACT_ROOT = (Resolve-Path .agent-optimize/fast-demo).Path
$env:AGENT_OPTIMIZE_MODE = 'fast'
python -m benchmarks.fast_in_situ --case multi_field_regions --events 1000
```

The final command reports whether the generated artifact was used, whether its
result matched the reference, first-use cost, and warm timings. Saved `adapter.py`
and `kernel.py` files can be inspected under the artifact root's `artifacts/`
directory. The example uses the mock provider and makes no API requests.

## Measured model comparison — November 6, 2026

These are historical measurements of the careful path, before fast ordinary
dispatch was introduced. They do not describe the current default fast mode.

Two saved native Windows comparisons used synthetic HEP-like arrays, seed
`20261106`, one repeat, and two sequential proposals per supported model/case.
The small suite used 1,000 events with 5,938 jets; the larger follow-up used
100,000 events with 600,016 jets. These are generated fixtures, not detector data
or physics validation. Of 36 proposals, 35 passed the configured correctness
checks; **none passed timing eligibility or was saved**.

Counts below are correctness-passing proposals / requested proposals. “None”
means no implementation was saved for any of the three models.

| Workload | Events | gpt-6-luna | gpt-6.1-terra | gpt-6.1-sol | Saved, all models |
| --- | ---: | ---: | ---: | ---: | --- |
| `integer_threshold` | 1,000 | 2/2 | 2/2 | 2/2 | None |
| `central_ht` | 1,000 | 2/2 | 2/2 | 2/2 | None |
| `quality_regions` | 1,000 | 2/2 | 2/2 | 2/2 | None |
| `integer_quality` | 1,000 | 2/2 | 1/2 | 2/2 | None |
| `multi_field_regions` | 1,000 | 2/2 | 2/2 | 2/2 | None |
| `dijet_combinatorics` (unsupported control) | 1,000 | 0 proposals | 0 proposals | 0 proposals | None; Python fallback verified |
| `nearest_jet_matching` (unsupported control) | 1,000 | 0 proposals | 0 proposals | 0 proposals | None; Python fallback verified |
| `multi_field_regions` | 100,000 | 2/2 | 2/2 | 2/2 | None |

Terra's second `integer_quality` proposal was rejected as `invalid_source` before
correctness execution. All other proposals passed finite checks and were
`ineligible` because the complete replacement path regressed. The unsupported
controls retained Python and made zero provider requests for every model.

| Measured workload | Reference, ms/call | Complete replacement, ms/call |
| --- | ---: | ---: |
| `integer_threshold`, 1,000 events | 1.48–1.58 | 51.37–57.61 |
| `central_ht`, 1,000 events | 2.43–2.57 | 50.98–56.18 |
| `quality_regions`, 1,000 events | 3.62–4.01 | 51.44–59.04 |
| `integer_quality`, 1,000 events | 3.22–3.39 | 52.19–56.03 |
| `multi_field_regions`, 1,000 events | 4.36–5.11 | 51.87–57.42 |
| `multi_field_regions`, 100,000 events | 22.57–24.34 | 57.98–60.49 |

### Exploratory timings without compatibility checks

On November 7, 2026, we remeasured the previously correctness-passing candidates
from those comparisons on the same Windows machine. This experiment bypassed
dispatch and all compatibility checks; it prepared array buffers on each call,
ran the compiled kernel, and constructed the Awkward output. It did not install
any implementation or change ordinary dispatch. Compilation was excluded. Each
candidate was checked against its reference on the recorded input before timing.

| Workload | Events | Reference, median ms/call | Unchecked replacement, median ms/call | Median paired speedup |
| --- | ---: | ---: | ---: | ---: |
| `integer_threshold` | 1,000 | 1.373 | 0.064 | 21.5× |
| `central_ht` | 1,000 | 2.340 | 0.069 | 33.6× |
| `quality_regions` | 1,000 | 3.729 | 0.081 | 46.2× |
| `integer_quality` | 1,000 | 3.161 | 0.074 | 41.8× |
| `multi_field_regions` | 1,000 | 4.390 | 0.084 | 51.8× |
| `multi_field_regions` | 100,000 | 22.244 | 6.359 | 3.8× |

Each result is the median across five or six candidates from the three models;
each candidate's timing is the median of three batches of seven measurements.
Speedup is the median of the per-candidate reference/replacement ratios, so it
need not equal the ratio of the displayed medians. The inputs are synthetic,
the sample is small, and these are potential speedups without safety checks, not
speedups currently delivered by the library. The standalone script is
`python -m benchmarks.unchecked_speed`; pass the two local `comparison.json`
paths shown below and `--output` followed by a destination. Its saved local
measurements are in `.agent-optimize/benchmarks/unchecked-speed-20261107.json`.

In the full-dispatch table above, each range spans the proposal/recording medians across models; each such value
is the median of three batch medians, with five samples per batch. These are
full ordinary-dispatch timings, including compatibility verification, adaptation,
kernel work, and output construction. They are not isolated kernel timings.
Compilation was separate: about 435–611 ms in the small suite and 528–541 ms in the
larger follow-up; first replacement use was about 0.92–1.21 seconds overall.

The complete replacement path includes repeated verification of live library
dependencies, source, and bindings, with Windows path resolution and disassembly.
Inspection identified this verification as a source of fixed dispatch overhead.
The measured table shows that complete dispatch exceeds reference costs on these
workloads; it does not isolate kernel speed. Faster generated kernels alone do
not remove compatibility checks. Changed-code and dependency invalidation remain
enabled.

| Model, both saved comparisons combined | Correctness passes | HTTP requests | Reported input tokens | Reported output tokens | Reported total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-6-luna` | 12/12 | 12 | 28,445 | 11,407 | 39,852 |
| `gpt-6.1-terra` | 11/12 | 12 | 28,469 | 9,849 | 38,318 |
| `gpt-6.1-sol` | 12/12 | 12 | 28,494 | 6,397 | 34,891 |

Usage is the API-reported total for these attempts, including reasoning in output
usage, with retries disabled. It is not a billing estimate. The trials used
Windows 11/AMD64, Python 3.12.11, Awkward 2.14.0, NumPy 2.5.4, Numba 0.68.1,
and OpenAI SDK 2.54.0. A Windows test suite ran concurrently during some early
trials. This small, sequential, single-seed comparison is illustrative; it cannot
establish a model ranking, general correctness, larger-workload performance, or
reproducible benchmark guarantees.

Both batches used 75 seconds per provider request, 4,096 maximum output tokens,
two proposals, and no retries. The small suite used 180 seconds/function,
210 seconds/search, 40 seconds/worker, and a 300-second outer run limit; the
larger follow-up used 240/270/60/360 seconds respectively. Raw settings, source
hashes, input/result digests, usage, and timing samples are in the ignored local
evidence files `.agent-optimize/benchmarks/20261106T161559683343Z/comparison.json`
(16:15:59–16:28:27 UTC) and
`.agent-optimize/benchmarks/20261106T162855353601Z/comparison.json`
(16:28:55–16:31:42 UTC), both on November 6, 2026. These local files are not
distributed with the repository. To rerun the experiment, see below.

## Continuous integration and published docs

GitHub Actions runs the offline test suite on Windows, macOS, and Linux with
Python 3.12 on every push and pull request. This includes the strict Zensical
documentation build. CI excludes live API tests and needs no API credentials.
The workflow can also be started manually from the Actions tab.

After all three test jobs and the documentation build succeed on `main`, the
workflow publishes [the documentation](https://gordonwatts.github.io/agent-optimize/)
to GitHub Pages. In the repository's **Settings → Pages → Build and deployment**,
select **GitHub Actions** as the source before the first deployment. Pull requests
and other branches build and check the docs without publishing them.

## Install and run

### Run from GitHub with `uvx`

The repository, Python package, and command are all named `agent-optimize`.
Run the command directly from GitHub with [uv](https://docs.astral.sh/uv/):

```sh
uvx --from git+https://github.com/gordonwatts/agent-optimize.git agent-optimize --help
```

`uvx` installs the command in its own isolated environment. Your analysis
script also needs `agent-optimize` and its own dependencies in the Python
environment that runs the script. For example, `uv run` can supply this
package to the analysis without modifying an existing environment:

```sh
uvx --from git+https://github.com/gordonwatts/agent-optimize.git agent-optimize run -- uv run --with git+https://github.com/gordonwatts/agent-optimize.git python analysis.py
```

Add the analysis's other dependencies to its project or to `uv run` as needed.
The subsequent ordinary script run must use an environment containing the same
package, for example `uv run --with git+https://github.com/gordonwatts/agent-optimize.git python analysis.py`.
To use the optional OpenAI provider with `uvx`, add `--with openai` before
`--from` and select `--provider openai` on the `agent-optimize run` command.
The GitHub URL is required for these commands until the package is separately
published to a package index.

Use Python 3.12 on native Windows or Ubuntu. Create a separate environment on
each platform. From the repository root on Windows:

```powershell
python3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test,docs]"
.\.venv\Scripts\agent-optimize run -- .\.venv\Scripts\python examples/filter_sum.py --max-events 1000
.\.venv\Scripts\python examples/filter_sum.py --max-events 1000
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m zensical build --config-file docs/zensical.toml
```

Use the path to your Python 3.12 interpreter if `python3.12` is unavailable.
On Ubuntu:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,docs]'
.venv/bin/agent-optimize run -- .venv/bin/python examples/filter_sum.py --max-events 1000
.venv/bin/python examples/filter_sum.py --max-events 1000
.venv/bin/python -m pytest
.venv/bin/python -m zensical build --config-file docs/zensical.toml
```

Everything after `run --` is forwarded as an argument list, including spaces
inside quoted arguments and script flags. The command inherits the current
directory and output streams; credential variables are removed from its environment.
Its exit status is returned; command
configuration or launch errors return 2. A failing script is reported without
starting replay or compilation. A successful analysis command returns 0 even when
a function is unsupported, replay fails, or the compiled candidate is rejected;
inspect the per-function report. User interruption returns 130 and preserves
completed evaluations and saved implementations.

Use `agent-optimize run --json -- python analysis.py ...` for one versioned JSON
result on stdout. Child stdout/stderr and progress go to stderr in this mode.
The result includes `schema_version: 1`, status, exit codes, per-function outcomes,
diagnostics, timing evidence, report/artifact paths, and the next action. Configuration
and command errors also produce a JSON result; help commands display normal help.
The [user guide](docs/index.md) describes the result schema, exit codes, and limits.

```python
import awkward as ak
from agent_optimize import optimize

@optimize
def ht(events, threshold=30):
    jets = events.Jet
    return ak.sum(jets[jets.pt > threshold].pt, axis=1)
```

Ordinary calls preserve function metadata and the original
function at `ht.__wrapped__`. Saved implementations use the selected fast or
careful policy described above; otherwise the reference runs. Ordinary calls
never record inputs, generate proposals, or start an optimization search.
During an explicit recording session, every decorated call also executes its
original reference exactly once. Unsupported calls and recording failures retain
their reference results; reference exceptions and user interrupts propagate.

The initial recording contract accepts standard CPU Awkward arrays and standard
Python or NumPy Boolean, integer, and floating scalars. Fixed positional,
positional-only, and keyword-only parameters bind consistently, including defaults.
Arrays may have numeric record fields and one level of variable-length lists;
typed zero-length arrays preserve their schema. Reference outputs must be flat
numeric Awkward or NumPy arrays. Scalar types, array dtypes, signed zero, and NaN
bits are preserved. Missing values, unions, untyped empties, nested lists, custom
behaviors/attributes/subclasses, variadic parameters, NumPy array inputs, Python
containers, strings, and complex inputs receive diagnostics.

Replay extracts a restricted standalone function with known Awkward/NumPy/math
imports and simple immutable constants. It accepts local assignments, field
access, indexing, scalar arithmetic/comparisons, Boolean combinations, `abs`,
and a small whitelist of pure imported operations including `ak.sum`.
Closures, helpers, mutable globals, input assignments, side effects, arbitrary
control flow, and unsupported syntax are rejected for recording. References must
be defined in a source file. A single source snapshot is compiled without
execution to verify that it matches the live
function before capture or reuse. Stale files and hot code replacements require
reloading the module. Imported Python callables must match their installed library
source; replacement wrappers and unsupported mutable dependency state retain
Python. Verified library closure/default state, actual builtin bindings, and
direct global/module attribute dependencies are fingerprinted.
The supported sum reducer also verifies its constructor, apply/dtype helper
methods, method bindings, and numeric policy; modified reducer classes fall back.
The current verified subset excludes `ak.to_numpy` inside decorated functions:
its general coercion dependencies exceed this first implementation. Such calls
retain their Python result with a diagnostic. Convert outside the decorated
function when NumPy output is needed. NumPy-output recording through supported
dependencies such as `np.asarray` remains available; automatic compiled reuse
currently targets the unconverted Awkward reduction result.
Replay validates the frozen source and its hash, then executes it in a fresh worker without importing the original analysis
script. It compares complete output metadata and bytes with the captured result.

Each run creates an absolute session directory under the invoking directory's
`.agent-optimize/sessions/`. It contains versioned session metadata, complete
input/output buffers, frozen reference modules, replay results, and `report.json`.
The report lists effective limits, retained bytes, per-function captures,
skipped-capture reasons, replay outcomes, parsed graphs, candidate correctness,
raw timing samples, workload sizes, eligibility, and artifact paths. Evaluation
workspaces under each session's `experiments/` directory retain the frozen
reference, candidate source, generated inputs, and case results. Completed
evaluations are checkpointed before accepted indexes are changed. Accepted source,
versioned manifests, and evidence live under `artifacts/`; atomic indexes under
`accepted/` select current implementations. Deleting `.agent-optimize/` removes
local recordings and saved implementations and restores reference-only calls.
Mock mode contacts no provider; live generation requires explicit selection.

Defaults are 64 MiB retained capture bytes per session and five capture attempts
per function. Attempts include unsupported or skipped calls. Before copying input
buffers the recorder checks both limits and reserves room for complete inputs
and metadata. Output overflow discards the partial capture. Observed input
mutation also discards it. These controls are available before `--`:

```text
agent-optimize run --artifact-root "artifacts with spaces" --max-capture-bytes 1048576 --verification-groups 2 -- python analysis.py
```

For an ordinary run with a custom artifact root, set
`AGENT_OPTIMIZE_ARTIFACT_ROOT` to the same directory. The default is the current
directory's `.agent-optimize/` on each invocation.

The byte count covers retained binary buffers, capture JSON, and frozen reference
files. It excludes session/report bookkeeping and interpreter/library allocations.
Buffers are written incrementally with bounded temporary chunks; inputs are never
silently truncated or packed into a full extra copy. This bounds snapshot storage
and copying, not the reference function's own allocations. A session lock shares
budgets between recording processes. A crashed lock owner causes later calls to
fall back after a bounded wait. Replay and correctness checks have no execution
time limit. Worker output is bounded to 64 KiB, and provider credentials and the recording
marker are removed from worker environments. Process separation is not a sandbox
for hostile code.

The first compiled expression is the documented `events.Jet` filter-and-sum:
field access, numeric comparisons, `abs`, elementwise Boolean `&`, `|`, and `~`,
one filter, and `ak.sum(..., axis=1)`. Inversion `~` requires a Boolean array mask;
scalar inversion and
scalar indexing are unsupported because their Python and NumPy semantics differ.
Field and argument names come from the
reference. The implementation supports one CPU array with a one-level
`ListOffsetArray` of numeric records and numeric/Boolean scalar arguments.
It accepts integer/Boolean fields up to 64 bits and float32/float64 fields;
unsupported layouts, dependencies, expressions, and dtypes retain Python with
an explanation. It preserves the reference's flat Awkward or NumPy container.

The host validates buffers and supplies read-only offsets/columns to a generated
fused loop. Numba uses bounds checks and strict arithmetic with fast-math disabled. Reference and
candidate run in separate workers against independently frozen requirements.
Container, structure, length, and exact dtype must match before values are checked.
Integers and Booleans compare exactly; floating outputs compare bitwise, including
signed zero and NaN payloads. Reduction promotion and order are checked against
the installed Awkward implementation. A candidate that fails any check is rejected.

Deterministic cases include schema-preserving zero events, empty lists, varied
and larger list lengths, all/no passes, zero values, exact selection boundaries
and adjacent representable values, NaNs/infinities, signed zero, cancellation,
large/small values, and difficult sums. Every retained recording is checked, and
each recorded scalar configuration receives boundary cases. Bounded changes to
numeric and Boolean scalar arguments also check for candidates that hard-code
one recorded threshold. Construction is
bounded to 512 cases per contract and one million elements in a generated
larger-list case; graphs have at most eight referenced fields and 512 expression
nodes. Unsupported case construction is reported instead of weakening checks.

Correctness checks are empirical evidence, not a proof of equivalence. A passing
correctness result alone does not establish eligibility. For each retained
recording the tool warms both paths, compiles before steady-state measurement,
calibrates repeated calls, and takes five samples in each of three batches with
alternating reference/candidate order. It measures the ordinary replacement path,
including the selected mode's dispatch, conversion, kernel execution, and output
construction, against a direct frozen-reference call. Compilation and first-use
costs are reported separately.

Every batch on every retained recording must improve its median by at least 10%.
The candidate's upper quartile must also be at least 10% below the reference's
lower quartile; uncertainty, regression or missing evidence keeps Python in use.
There is no function, worker or overall optimization time limit. Small workloads may provide no eligible improvement. Timing claims apply
only to the measured workloads. Tests use explicitly labelled
`controlled-test-fixture` timing records to prove acceptance paths without making
performance claims; ordinary CLI runs use measured timings.

In careful mode, saved implementations match the current source/executable contents, immutable
constants and import bindings, full field/nesting/dtype contract, scalar types,
and Python/Awkward/NumPy/Numba/LLVM/platform/CPU environment (including NumPy's
error policy). Event counts, variable list lengths, and observed values are not
fixed by recordings. Unknown, invalid, incompatible, missing, or corrupt artifacts
fall back to the reference. A saved artifact can compile on its first ordinary
use; importing the analysis module never compiles or searches. Fast mode performs
contract selection only on first load and skips dynamic reference/dependency and
environment matching entirely, as described above.

Reusable source is restricted to the fixed numeric computation interface, read-only
inputs, and separately allocated output. A recoverable load, compilation, or
execution failure quarantines that artifact, clears its cached kernel, and executes
the reference for the same call. Later calls skip it until a new accepted identity
replaces it. Quarantine diagnostics are stored under `quarantine/` when writable;
an in-memory quarantine still prevents retries when the store is read-only.
Reference exceptions and user interrupts propagate. Worker isolation contains
native crashes during evaluation; ordinary in-process dispatch cannot recover a
native process crash.

Metadata records carry `schema_version: 1`; recorded array buffers are stored
separately. `.agent-optimize/` and compiled caches are ignored by Git.

Search defaults to a deterministic local mock, with no API key, SDK, or network
calls. It proposes an intentionally wrong loop first, then a corrected loop after feedback.
The evaluator owns the reference, mandatory tests, timing, and acceptance. Provider
requests contain source, graph, schema, signature, constraints, independent synthetic
examples, and sanitized diagnostics. Recorded arrays, scalars, worker exceptions,
and recording-derived counterexamples never enter provider requests or feedback.
Local request/response files are retained alongside each proposal's evaluation.

The default is three replacement-code attempts per function, shared across its
input contracts. Model requests have a fixed 60-second timeout; local checking,
timing and the overall optimization have no time limits. Requests are limited to 128 KiB, responses
to 64 KiB, and proposed extra cases to eight per proposal. Extra cases use only
`zeros`, bounded `lengths`, or a known field's `boundaries`; the evaluator builds
them without executing provider test code and retains every mandatory test.
Each completed proposal and the session report are atomically checkpointed before
another proposal is requested. Interruptions and later failures preserve earlier
completed evaluations and saved artifacts. Attempt count, byte limits,
and case count can be overridden with the corresponding `run --help` options;
effective limits are printed before the command starts.

## Optional OpenAI generation and comparison

Install `pip install -e '.[openai]'` and select `--provider openai` explicitly:

```text
agent-optimize run --provider openai --model gpt-6.1-sol --max-output-tokens 4096 -- python analysis.py
```

Supported IDs are `gpt-6.1-sol`, `gpt-6.1-terra`, and `gpt-6-luna`. Configure
`AGENT_OPTIMIZE_OAI_API_KEY` in the environment or an ignored `.env` file.
Credential precedence is `--api-key` > the named environment variable > project
root `.env` > user home `~/.env`; `OPENAI_API_KEY` alone is not used. A key's
presence never enables live mode. The official SDK uses `responses.parse` with
a Pydantic schema, retries disabled, a fixed 60-second model-request timeout, bounded output
tokens, the official endpoint, and `store=False`. Reports/manifests identify the
model and settings and retain allowlisted usage. Refusal, incomplete output,
malformed output, transport failure, and timeout are reported separately.

Reference source and literal constants, schema/graph, synthetic examples, and
sanitized feedback/prior proposals may be sent. Captured buffers and scalar
values remain local. Credentials use a private pipe to the provider process and
are removed from analysis and evaluation environments. Process separation does
not restrict trusted analysis code's normal filesystem access. Request-byte
limits cover the local problem/history envelope, not the full HTTP request;
response-byte limits apply after SDK parsing, not during download. See the
[provider guide](docs/live-provider.md) for details.

The ordinary test suite is offline. To run the single real-API test, explicitly
set `AGENT_OPTIMIZE_LIVE_TESTS=1` in the test process and run
`python -m pytest tests/test_openai_provider.py::test_opt_in_live_proposal -q`.
It needs a configured credential, incurs API usage, and defaults to `gpt-6-luna`;
`AGENT_OPTIMIZE_LIVE_MODEL` overrides that model. Unset the opt-in flag afterward.

List the synthetic comparison cases with `python -m benchmarks.compare_models
--list`. A small offline smoke run uses:

```text
python -m benchmarks.compare_models --provider mock --cases integer_threshold dijet_combinatorics nearest_jet_matching --events 16 --max-attempts 2
```

Reproduce the small live comparison settings with:

```text
python -m benchmarks.compare_models --provider openai --model gpt-6-luna --model gpt-6.1-terra --model gpt-6.1-sol --events 1000 --seed 20261106 --repeats 1 --max-attempts 2 --max-output-tokens 4096
```

For the larger follow-up, add `--cases multi_field_regions`, set `--events
100000`, and change function/search/worker/run seconds to `240`, `270`, `60`,
and `360`. The runner calls the real CLI, uses a fresh artifact root per
model/case/repeat, checks unsupported fallback controls, and saves `comparison.json`
plus reports and workload digests under `.agent-optimize/benchmarks/`. It requires
each selected model to reach candidate evaluation; a completed command alone
does not establish correctness or eligibility. The default comparison is mock;
live calls require explicit `--provider openai` and model selection.

## Verification

The native Windows environment uses Python 3.12.11, Awkward 2.14.0,
NumPy 2.5.4, Numba 0.68.1, pytest 9.2.0, and Zensical 0.1.0 installed.
Steps 1 and 2 passed all 103 tests on both native Windows and Ubuntu under WSL2.
Step 3 passed all 161 tests on native Windows and Ubuntu under WSL2.
Step 4 passed all 223 tests on native Windows and Ubuntu under WSL2.
Step 5 passed all 278 tests on native Windows and Ubuntu under WSL2.
Step 6 passed all 296 tests and a documentation build on both native Windows and
Ubuntu under WSL2. Final native Windows example/ordinary, space-containing paths,
and timeout checks also passed. The real 1,000-event example retained Python:
the complete replacement path was slower on that workload. See
[platform verification](docs/verification.md) for versions, measured results,
and the source-manifest method. Ubuntu passed 296 tests in 91.45 seconds and its
documentation build; the space-containing 1,000-event example also produced the
expected result and retained Python. The Linux suite included worker crash and
timeout checks with temporary files inside the Linux workspace. Source byte
comparisons matched the synchronized Windows verification snapshot.
Build documentation from the checkout root with
`python -m zensical build --config-file docs/zensical.toml`; HTML is generated in
ignored `docs/_build/`. These step-6 counts describe the earlier offline
checkpoint. Live-stage verification on November 6 also passed the Windows
offline suite (342 passed, one live test skipped, with later focused additions),
the Ubuntu WSL2 offline suite (343 passed, two skipped), and the explicitly
opted-in live provider test separately on Windows (one passed in 20.00 seconds)
and Ubuntu WSL2 (one passed in 12.65 seconds). See
[platform verification](docs/verification.md) for the scope of these checkpoints.
The full intended workflow is described in
[the UX](docs/ux-first-round.md) and [implementation notes](docs/implementation-notes.md).
