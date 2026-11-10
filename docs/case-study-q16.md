# Case study: semileptonic top-pair reconstruction in WSL2

The [q16 example](https://github.com/gordonwatts/agent-optimize/blob/main/examples/q16_ttbar.py)
starts with a standalone ServiceX analysis. It selects events with exactly one
electron or muon above 25 GeV and at least four jets above 30 GeV, solves the
neutrino longitudinal momentum using the W-mass constraint, and tries every
distinct four-jet assignment. The winning assignment minimizes
`abs(m_had - 172.5) + abs(m_lep - 172.5)`. The output is one reconstructed
top-pair mass per selected event and `plot_1.png`.

The analysis has two different costs: constructing the candidate assignments,
and scoring those candidates. The current example optimizes both stages, together with neutrino reconstruction.
ServiceX delivery, ROOT reading, event packing and plotting remain outside the
decorated function. The earlier scoring-only result is retained below as historical
evidence of the smaller boundary.

## Ordinary execution in Ubuntu on WSL2

The script's inline metadata declares Typer, the PHYSLITE query builder, ServiceX,
Jinja2, Uproot, Awkward, NumPy, Matplotlib, and the ordinary `agent-optimize`
package. The library has no q16 dependency extra. The script requests Python
3.13 and pins its library dependency to a Git revision for repeatability.

```bash
uv run --script examples/q16_ttbar.py --n-files 1 --output-dir plots/q16
```

uv prepares the script environment. This ordinary run uses the reference or a
saved replacement and does not start an optimization search. A downloaded copy
can likewise run as `uv run --script q16_ttbar.py`.

ServiceX normally discovers `servicex.yaml` or `.servicex` in the working
directory, its parents, or the Linux home directory. A configuration in the
Windows home directory may need an explicit path:

```bash
uv run --script examples/q16_ttbar.py \
  --servicex-config /mnt/c/Users/YOUR_NAME/servicex.yaml --output-dir plots/q16
```

Credentials and endpoint settings stay in that local configuration. `--n-files 0`
requests the full dataset; the default requests one file. Previously delivered
files can be reused with repeated `--input-file` arguments. These must contain
the query's transformed branches, rather than untransformed DAOD data.

## Record, evaluate, and save a replacement

Install the generic CLI once, with uv available on PATH:

```bash
uv tool install git+https://github.com/gordonwatts/agent-optimize.git
agent-optimize run \
  --artifact-root .agent-optimize/q16 --max-calls 2 \
  --worker-seconds 60 --function-seconds 240 --search-seconds 300 \
  examples/q16_ttbar.py --n-files 1 --output-dir plots/q16
```

Optimizer options go before the script path. Analysis options follow it; no
`--script` flag or command separator is required. Add `--servicex-config PATH`
to the analysis arguments if needed. When developing an unmerged checkout,
install the CLI with `uv tool install --editable .` instead of the GitHub URL.
The script still controls the version of the library used for its analysis.

The launcher reads `requires-python` and passes the inline dependencies to uv,
then starts the controller in the resolved environment. Analysis and all replay,
correctness, compilation, and timing workers use that interpreter. The script's
library pin is honored; no q16-specific installation or active-environment
modification is needed. The [user guide](index.md) describes current metadata
support and the explicit command form for already prepared environments.

The two recorded calls are complete batches, each containing up to 128 selected
events. Capture and search budgets still apply. Later batches execute the
reference during recording without additional captures.

The deterministic local provider proposes a Numba loop, checks the frozen
reference and mandatory cases, and measures ordinary dispatch before deciding
whether to save it. The first mock proposal deliberately has an output error
to exercise rejection and feedback. A replacement is saved only after passing
the correctness and performance gates. Inspect `report.json` for the outcome.

To try the optional OpenAI provider, add `--provider openai` before the script
path. The launcher requests the generic OpenAI extra in the same environment.
See the [provider guide](live-provider.md) for credentials and request limits.
This case study's measurements use the local provider.

Reuse a saved replacement from the same working directory and artifact root:

```bash
AGENT_OPTIMIZE_ARTIFACT_ROOT="$PWD/.agent-optimize/q16" \
  uv run --script examples/q16_ttbar.py --n-files 1 --output-dir plots/q16
```

Fast dispatch is the default. `AGENT_OPTIMIZE_MODE=careful` enables repeated
compatibility checks. On a shared Windows/WSL checkout, uv's script environments
are separate from a project `.venv`; no Windows environment needs to be replaced.

## The numerical boundary

`reconstruct` packs the jets, lepton and missing momentum into one event record.
The decorated `reconstruct_events` contains neutrino reconstruction, ordered jet
assignment, top-vector arithmetic, scoring and first-minimum winner selection.
Packing is included in the full-reconstruction benchmark.

The reusable library recognizes a bounded `ak.argcartesian` idiom: two to four
named index slots from one jagged numeric-record collection, a Boolean index
filter, numeric gathers, event-level fields broadcast with `[:, None]`, arithmetic,
`abs`, `sqrt`, `maximum`, `where`, and a first-argmin reduction. It is independent
of q16 field names and constants. The current contract requires float64 numeric
fields/output, no scalar arguments, at most 32 referenced fields and 128 numeric
bindings. Unsupported forms and expressions retain the Python reference.

The compiler streams assignments in the same order as the reference, applies
index constraints as soon as their slots are available, and hoists event-only
arithmetic outside the assignment loops. It allocates the output without
materializing candidate arrays. Equal scores retain the first candidate; empty
candidate lists yield zero. The real analysis selects at least four jets.

Fast-math remains disabled. Explicit NaN operand selection preserves the installed
NumPy arithmetic policy when LLVM would otherwise commute operands or replace
subtraction with negation. Mandatory cases check NaN signs and payloads, infinities,
signed zeros, ties, boundaries, changing event/jet counts, and empty inputs. Frozen
reference verification covers the added stock Awkward operations and their
supported dependency state. Code, dependency, schema and environment compatibility,
quarantine, and reference fallback remain part of the existing workflow.

## Verification and scope

### Shareable run report

`agent-optimize run --report examples/q16_ttbar.py --synthetic-events 128`
generates a Markdown summary with timings, usage/cost, correctness results and
the frozen Python and saved Numba source. The analysis executes once.
[This real-data sample](examples/q16-report.md) was generated from the cached
7,552-event q16 file using the local mock provider, with two recorded calls and
108 correctness cases for the accepted proposal. Its comparison table is for
those recorded batches; the full-reconstruction measurements below are separate.
It reports no model calls or API charges. Reference agreement does not establish
physics validity; the mean discrepancy described below remains unresolved.

### Full reconstruction (November 9, 2026)

The full boundary was recorded and evaluated on one delivered real-data file in
native Windows and Ubuntu under WSL2. Both recorded batches replayed successfully;
the intentionally wrong proposal was rejected and a corrected proposal passed
all mandatory cases and the measured acceptance gate. No OpenAI calls were made.
A fresh process observed saved dispatch and matched all 7,552 masses against the
supplied reconstruction bit for bit, on both systems.

The full benchmark includes event packing, dispatch, neutrino reconstruction,
jet assignment, candidate construction, scoring and result construction. It
excludes ServiceX delivery, ROOT reading and plotting. Warm timings are medians
of three passes in alternating order; first-process cost is reported separately.
These are workstation observations, not a general speedup guarantee.

| Environment | Original full reconstruction | Saved full replacement | Warm speedup | First full replacement pass |
| --- | ---: | ---: | ---: | ---: |
| Windows, Python 3.13.14 | 5.929 s | 0.156 s | 38.13× | 4.108 s |
| Ubuntu WSL2, Python 3.13.15 | 5.476 s | 0.146 s | 37.57× | 3.631 s |

Both environments used Awkward 2.14.0, NumPy 2.5.4 and Numba 0.68.1.
[Raw timing and validation evidence](evidence/q16-full-reconstruction.json)
records the three full passes, measured recording batches and environment versions.
Each corrected proposal passed 108 correctness cases, including the two recordings.
The native Windows suite passed 387 tests (3 skipped); Ubuntu passed 386
(4 skipped). A source-file hash comparison verified the shared implementation.
The standalone uv workflow also recorded, replayed, checked and saved a full
replacement. CI covers Python 3.12/3.13 on Windows, Linux and macOS.

The observed mean remains 1537.1414 GeV. Bitwise preservation of the supplied
software does not resolve its disagreement with the 2477.6205 GeV physics
benchmark beyond the 2% tolerance. No explanation has been established here.

### Historical scoring-only WSL2 result

On November 8, 2026, an Ubuntu WSL2 run used Python 3.13.18, Awkward 2.14.0,
NumPy 2.5.4, and Numba 0.68.1. ServiceX delivered one file. The local provider
recorded two calls and replayed both successfully. A corrected proposal passed
33 correctness cases and the performance gate, then was saved for fast dispatch.
No OpenAI calls were made.

| Measurement | Reference | Saved replacement | Speedup |
| --- | ---: | ---: | ---: |
| Scoring, recorded batch: 24 events / 8,856 candidates | 13.670 ms | 0.162 ms | 83.46× minimum across timing batches |
| Full reconstruction: 7,552 events / 235 batches | 10.791 s | 7.269 s | 1.48× |

The scoring comparison includes ordinary fast dispatch. The full reconstruction
comparison uses the supplied, unmodified reconstruction as its baseline; the
revised reference without saved dispatch took 10.448 seconds. Full reconstruction
timings are medians of three warm passes in alternating order. ROOT reading,
ServiceX delivery, and plotting are excluded. The first full pass in a fresh
process took 10.798 seconds, including initial compilation and loading.

That fresh process observed saved dispatch and checked all 7,552 reconstructed
masses against the supplied reference, including their exact floating-point bits.
The smaller full-reconstruction speedup reflects the candidate-building work
outside the optimized boundary. These are illustrative measurements on one
workstation, rather than a general performance guarantee.

To repeat the full-data check with a transformed ROOT file:

```bash
uv run --no-project --python 3.13 --with-requirements examples/q16_ttbar.py \
  python benchmarks/q16_reconstruction.py \
  --input-file delivered.root --artifact-root .agent-optimize/q16 --repeats 3
```

The benchmark fails if saved dispatch is unavailable or any reconstructed value
differs from the supplied reference. Keep the artifact root used during recording.

### Automated coverage

Tests compare the rewritten reconstruction with the supplied numerical reference
bit for bit on deterministic synthetic events. Compiled scoring and full assignment reconstruction are checked on
recorded calls and independent cases covering zero events, empty lists, varying
and larger candidate counts, ties, boundaries, NaN payloads, infinities, and signed
zeros. Optional analysis tests also read a generated transformed ROOT tree and
exercise the plotting CLI. CI covers Python 3.12 and 3.13 on Linux, macOS, and
Windows, plus actual uv environment resolution on Windows and Linux and the q16
script-owned analysis dependencies on Linux.

Python 3.13 exposed two source-verification issues: equal code objects can have
different internal sharing after loading bytecode, and classes have additional
compiler metadata. Fingerprints now ignore object sharing while retaining code
values; reducer validation includes the additional metadata and the ArgMin
reducer. Existing replacement and mutation checks remain in place.

For an offline execution check:

```bash
agent-optimize run \
  --artifact-root .agent-optimize/q16-synthetic --max-calls 2 \
  examples/q16_ttbar.py --synthetic-events 256 \
  --output-dir plots/q16-synthetic
```

Synthetic events exercise the software and do not validate the physics. The
supplied script reported a one-file mean of 1537.14 GeV, outside q16's stated
2477.6205 GeV reference by more than its 2% tolerance. The WSL ServiceX run also
reconstructed 7,552 selected events with a mean of 1537.14 GeV. Agreement with
that benchmark remains unresolved; optimizing the existing reconstruction does
not resolve it.
