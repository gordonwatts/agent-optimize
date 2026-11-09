# First-round implementation plan

## Current project work (November 9, 2026)

Keep backlog items in GitHub issues. Gordon prefers Typer for Python CLIs.
Implementation and draft PRs for issues #2 and #3 are authorized. Keep PRs in
draft until Gordon explicitly approves; do not mark ready, merge or deploy.

- [Issue #3](https://github.com/gordonwatts/agent-optimize/issues/3): Typer
  migration is in [draft PR #4](https://github.com/gordonwatts/agent-optimize/pull/4).
  Windows Python 3.13: 383 tests passed, 4 skipped; two opt-in uv tests passed.
- [Issue #2](https://github.com/gordonwatts/agent-optimize/issues/2): full q16
  reconstruction is in [draft PR #5](https://github.com/gordonwatts/agent-optimize/pull/5),
  based on merged PR #1 (`62d3c59`). Generic ordered Cartesian assignment lowering
  now covers neutrino arithmetic, candidate construction, jet matching and scoring
  in one compiled boundary. Recording, replay, correctness, persistence and saved
  dispatch are integrated. Independent pair-assignment tests also exercise reuse.
  Full Windows/Ubuntu regression runs and final performance evidence are in progress.

A delivered real ROOT file is available locally under ignored `.agent-optimize/`.
The fresh-process full comparison matched all 7,552 masses bit for bit. The mean
remains 1537.14 GeV, outside the physics benchmark's 2477.6205 GeV ±2% target;
the reason is unresolved. Software preservation does not establish physics validity.
No extended physics investigation is part of this implementation.

## Goal and boundaries

Deliver the workflow in `docs/ux-first-round.md`, following `docs/implementation-notes.md`: decorate a function, run `agent-optimize run -- python analysis.py ...` once, evaluate locally after the script exits, and automatically retain an eligible implementation for ordinary subsequent calls. Finish one real Awkward filter-and-sum example before adding other expressions. Initial generation is deterministic mock generation; no API key, SDK dependency, or live request is needed.

Use a small Python package with six responsibilities: decorator, recorder, experiment builder, generator, evaluator, and artifact store. Keep these as modules, with a separate worker entry point and CLI; do not introduce a service, database, plugin framework, or general tracing system. Use standard-library argument parsing, JSON metadata, binary array buffers, and subprocesses. Declare Awkward, NumPy, and Numba dependencies and separate test/documentation dependencies in `pyproject.toml`. Choose and verify one Python minor version supported by the installed versions on both platforms, then document it and the tested dependency versions.

## 1. Establish the package and runnable example

1. Add `src/agent_optimize/`, the `agent-optimize` console entry point, `tests/`, and a small deterministic `examples/filter_sum.py` accepting `--max-events`.
2. Implement `@optimize` as an ordinary reference-preserving wrapper first. Preserve function metadata and support positional/keyword binding consistently. Define a clear initial interface: Awkward array arguments plus supported numeric/Boolean scalars; reject unsupported argument kinds with a diagnostic during optimization.
3. Add shared result records for captured calls, unsupported functions, evaluation failures, and final outcomes. Keep the durable metadata versioned; use plain dataclasses rather than an elaborate object hierarchy.
4. Ignore `.agent-optimize/` and generated compiled caches. Add a short README with the exact install and example commands.

Verification: ordinary decorated calls equal undecorated calls, imports and normal calls neither record nor generate candidates, and CLI argument forwarding preserves arguments after `--`, including spaces and script flags.

## 2. Record and replay without rerunning the script

1. Have the CLI create an absolute local session directory and launch the supplied command as an argument list with a session identifier in its environment. Default the artifact root to the invoking working directory's `.agent-optimize/`. During a session, decorated functions always execute their reference implementation.
2. Serialize complete supported inputs with Awkward Forms, lengths, and named binary buffers. Serialize scalars and supported flat numeric reference outputs with exact dtype information. Record the original values before execution; reject observed input mutation and never save a truncated input.
3. Enforce finite per-session byte and per-function call limits, recording why a capture was skipped. Proposed initial defaults: 64 MiB captured bytes per session and 5 calls per function. Check call limits and preflight supported layout/buffer sizes against the remaining byte budget before copying buffers; avoid materializing or concatenating a full oversized capture. Account for metadata and outputs, reserve the input budget, and discard an incomplete capture if the output exceeds what remains. Write buffers incrementally and keep transient copying bounded where practical; document that this bounds recording overhead, not the reference function's own allocations. Include effective limits in the report and preserve the original function's exceptions.
4. Extract source, known imports, and simple immutable constants into a standalone reference module. Parse the restricted source before replay; reject closures, helpers, mutable globals, side effects, unsupported AST nodes, and unsupported layouts. Do not import the analysis script. Identify functions by source location and qualified name, with a content fingerprint for invalidation.
5. After successful script exit, replay frozen references directly in worker processes from recorded inputs. Preserve the child's nonzero exit status and report it without starting an optimization search on a failed script.

Verification: round-trip awkward buffers and scalar/keyword arguments, exact output metadata, bounded capture skips, zero-length schema preservation, unsupported dependency diagnostics, input mutation rejection, and reference exceptions. Instrument serialization to confirm oversized inputs and exhausted call budgets skip before buffer copying; cover output overflow and cleanup of partial captures. An integration fixture writes a marker outside the decorated function; replay must leave its count at one. Ensure two decorated functions and repeated calls are independently recorded. Unsupported cases still return their reference results.

## 3. Parse and evaluate the first compiled candidate

1. Parse field access, comparisons, `abs`, Boolean combinations, filtering, and `ak.sum(..., axis=1)` into a small explicit operation graph. Begin with the documented `events.Jet` example. Reject unsupported nesting, non-CPU arrays, missing values, unions, custom behaviors, arbitrary control flow, and unsupported dependencies with actionable reasons.
2. Define the fixed candidate interface and a minimal host-owned adapter: validated numeric buffers/offsets and scalars in, one flat array out. Build a handwritten fused Numba loop for the example. The adapter owns input conversion and output construction so the replacement returns the same container kind as the reference.
3. Run reference and candidate in separate subprocess workers using portable module entry points, timeouts, bounded output, and explicit status records. Clean worker environments of provider credentials and do not inherit the recording-session marker. Terminate timed-out workers and preserve concise diagnostics. This is process isolation, not a sandbox for hostile code.
4. Freeze reference code and test requirements outside candidate control. Validate output container, structure, length, and exact dtype before value comparison. Compare integers and Booleans exactly and floating-point values by their bit patterns, including signed zero and NaN payload differences. Disable fast-math and reassociation.
5. Construct deterministic test inputs from the parsed contract and selection constants: zero events, empty lists, varying list lengths including lengths larger than the sample, zero values, all/no passes, exact boundaries and adjacent representable values, NaNs/infinities, cancellation, large/small magnitudes, and difficult sums. Preserve original field names, nesting, and dtypes, including empty arrays.
6. Check recorded calls and every generated case. Confirm actual Awkward reduction order and promotion against Numba behavior; do not assume an imperative accumulator is bitwise equivalent. If the candidate cannot satisfy strict semantics for a contract, reject it and retain Python. Correctness rejection is an expected outcome.

Verification: real Numba compilation of a correct candidate, rejection of wrong boundaries, dropped events, wrong dtype/container, sample-value specialization, changed signed zero, changed summation order, and crashes/timeouts. Include integer and floating contracts and schema-preserving empties. A valid integer filter-and-sum candidate provides the first guaranteed acceptance-path fixture without weakening floating checks.

## 4. Measure eligibility and save reusable artifacts

1. Warm reference and candidate; compile before steady-state measurements. Measure the entire prospective replacement path, including applicability checks, dispatch, conversion, kernel execution, and output construction. Record compilation and first-use cost separately.
2. Repeat timings across every retained recording using alternating order and a simple documented summary. Proposed rule: three timing batches, at least 10% improvement in each batch's median on each retained recording; treat timing uncertainty or meaningful regression as ineligible. Calibrate batch duration without letting measurement exceed the experiment deadline. Record workload sizes and raw timing samples.
3. Store accepted source, frozen reference fingerprint, dependency/constants fingerprint, input contract, environment signature, correctness outcomes, timing results, and generator identity. Atomically checkpoint every completed candidate evaluation, including rejection/failure diagnostics, before requesting the next proposal. Write accepted manifests atomically only after all requirements pass. Preserve completed evaluation records and earlier accepted artifacts if a later experiment fails or is interrupted.
4. Implement normal wrapper dispatch using artifacts compatible with the current reference, relevant constants/import dependencies, field/nesting/dtype/fixed-dimension contract, and Python/Awkward/NumPy/Numba/platform/CPU signature. Do not specialize on event count, jagged lengths, maximum observed length, or values. Validate applicability before execution; unknown, invalid, missing, corrupt, or incompatible artifacts fall back to Python.
5. On saved-artifact load, compilation, or recoverable execution failure, quarantine that artifact by identity, invalidate its dispatch cache entry, and run the reference for the current call. Persist a small local failure diagnostic when possible; if the store is unwritable, keep the quarantine in memory. Later calls skip quarantined artifacts until a new accepted artifact replaces them. Do not catch user interrupts or suppress exceptions from the fallback reference. Make retry safe by restricting generated code to the fixed computation interface, read-only inputs, and separately allocated output; reject candidate code that mutates inputs or has external side effects before acceptance. Native crashes are contained during worker evaluation; ordinary in-process dispatch cannot recover a process crash and must not be described as such a guarantee.
6. Keep compatibility cache entries safe under source changes and different call signatures. Ordinary dispatch never starts a search or records inputs. Deleting `.agent-optimize/` restores reference-only behavior.

Verification: controlled timing samples test 10% threshold, repeatability, a regression on one retained recording, and insufficient timing evidence. Real timing reports verify measurement plumbing without asserting speedup. End-to-end compilation/evaluation with controlled acceptance timing saves an artifact, uses it in a fresh ordinary Python process, and demonstrates fallback after source/constants/schema/environment changes or artifact removal/corruption. Inject load/compilation/runtime exceptions to verify same-call reference fallback, quarantine, no repeat candidate execution, and propagation of reference exceptions; test unwritable quarantine storage and reject input-mutating candidates. Changed event counts and jagged lengths remain applicable. Keep controlled timing fixtures explicit in reports so they cannot be mistaken for measured performance claims.

## 5. Add bounded mock generation and feedback

1. Implement `generator.propose(problem, history)` and deterministic mock responses carrying candidate source, description, and declarative additional edge-case proposals. Separate provider data from evaluator-owned requirements. Record provider identity as mock and retain local request/response histories.
2. Supply source, operation graph, schemas, required signature, constraints, synthetic examples, and prior diagnostics. Never include recorded buffers or values in provider requests or failure feedback; use evaluator-generated counterexamples when needed.
3. Let the evaluator construct and validate proposed extra cases from a bounded supported vocabulary. Treat invalid proposals as diagnostics, never as executable test-building code. Retain all mandatory cases regardless of provider output.
4. Add a sequential loop: propose, build/check/measure, checkpoint the completed evaluation, save if eligible, feed back concise failures, and stop at a limit. Do not request the next proposal before the checkpoint succeeds. Proposed defaults: 3 proposals per function, 120 seconds per function, 30 seconds per worker capped by remaining time, and a 300-second total search deadline. Track requests and response-size limits; bound proposed cases. Expose effective limits before search; offer only controls needed to override these bounds.
5. Model responses must not edit project files, install dependencies, change the reference/tests, or decide acceptance. Handle malformed responses, refusal/incomplete status, provider errors, invalid source, and limit exhaustion as distinct outcomes with reference fallback.

Verification: deterministic incorrect-then-correct feedback flow, invalid/refused/incomplete/error responses, all deadline/count/size limits, empty capture, edge-case proposal rejection, and interruption. Interrupt during the next proposal and verify prior candidate evaluations and accepted artifacts remain readable; inject checkpoint failure and confirm no further proposal is requested. Inspect serialized provider requests and diagnostics with distinctive captured-data sentinels to confirm raw recordings remain local. All tests run without credentials or service access.

## 6. Finish the user command and documentation

1. Connect the complete recording/replay/search/save sequence to `agent-optimize run -- ...`. Handle all encountered functions and report per-function outcomes, measured timings, speedup when meaningful, skipped captures, failures, and artifact locations. Show reference fallback and the next useful action when no eligible improvement exists.
2. Implement `--json` as one versioned JSON result on stdout. Route child stdout/stderr and progress separately to stderr so arbitrary script output cannot corrupt JSON. Keep this mode noninteractive. Document exit codes for child failure, command/configuration failure, and completed searches with unsupported/ineligible functions.
3. Add Zensical configuration and a landing page under `docs/`, including the UX and implementation notes in navigation. Set generated HTML to ignored `docs/_build/`. Document one portable build command, `python -m zensical build --config-file docs/zensical.toml`, and verify it against the installed CLI.
4. Document installation, example, artifacts and deletion, mock mode, supported subset, strict correctness and empirical limits, recording bounds, timing interpretation, fallback/invalidation, and platform verification commands. Clearly distinguish implemented behavior from later live integration.

Verification: subprocess CLI integration exercises the exact human and JSON commands, argument forwarding, several functions, no usable captures, unsupported function, successful persistence, child failure, and interrupted search. Parse JSON in the presence of noisy child output. Build documentation with no missing local navigation targets.

## 7. Verify native Windows and Ubuntu under WSL2

1. On native Windows, create a repository-local Python environment, install the package plus test and documentation extras, record interpreter/dependency versions, and run `python -m pytest` and the documented Zensical build command. Run the example command and a subsequent ordinary invocation in a temporary artifact directory.
2. Check the installed Ubuntu WSL2 distribution and create/use `~/code/agent-optimizer`. Copy the current source changes, including uncommitted implementation files, into that Linux-side checkout. Exclude Windows environments, caches, recordings, compiled output, and `.git` internals; do not execute tests from `/mnt/c` or `/mnt/d`. Compare source file hashes or an exported source manifest so the Linux run verifiably tests the same implementation.
3. Create a separate Linux Python environment and install dependencies there. Run the same full test suite, documentation build, and end-to-end example/ordinary-dispatch checks with all artifacts and caches inside the Linux workspace. Verify paths containing spaces and process timeout behavior on both systems.
4. Record native Windows and Ubuntu results separately with OS/Python/dependency versions, source revision or manifest, test counts, documentation result, and real example performance/acceptance outcome. Real example speedup may legitimately be ineligible; correctness and automatic acceptance-path tests must still pass. Fix platform failures and repeat affected checks before reporting completion. If Ubuntu/WSL2 is unavailable, report that verification as blocked rather than claiming the MVP is fully verified.

## Completion and deferred work

The MVP is complete when both environments pass the same tests and documentation build, the documented command records and replays without rerunning the script, real Numba candidates are independently checked, controlled acceptance tests prove persistence/dispatch/invalidation, and ordinary calls always have reference fallback. Report measured performance honestly without promising a speedup for every workload.

Defer live OpenAI integration and prompt tuning until the user supplies an API key and both platform gates pass. Later use the OpenAI SDK directly with structured responses and explicit refusal/incomplete handling, preserving the same evaluator and command. Also defer Rust/GPU backends, provider/backend frameworks, arbitrary Awkward expressions, general tracing, distributed execution, visual interfaces, and manually managed experiment workflows.
