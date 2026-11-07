# First round implementation notes

Implement the [first round UX](ux-first-round.md): mark a function, run the script on small real input through one command, and automatically save an eligible improvement. Start with a complete filter-and-sum example before expanding the supported computations.

These notes retain the design rationale and implemented provider choices. The [user guide](index.md) documents the supported subset, default mock mode, optional OpenAI generation, command behavior, and verification status. The SDK is an optional runtime dependency; [the provider guide](live-provider.md) gives configuration and exact bounds.

## Architecture

Keep six internal components with separate responsibilities:

- **Decorator:** execute the reference, record calls during an optimization session, or dispatch to an accepted implementation.
- **Recorder:** save bounded snapshots of arguments, reference results, and function dependencies.
- **Experiment builder:** assemble reproducible reference execution, generated tests, and benchmarks.
- **Candidate generator:** propose implementations through a bounded model-driven loop.
- **Evaluator:** build, check, benchmark, and decide eligibility independently of the generator.
- **Artifact store:** retain recordings, reports, implementations, and applicability conditions under `.agent-optimize/`.

The command launches the existing script with a recording session in its environment. Decorated functions record their reference execution; optimization later replays those functions without rerunning the surrounding script.

## Supported computation

Use Numba as the first compilation backend. Generate fused imperative loops; do not attempt to compile arbitrary chains of `ak.*` calls directly. Rust and other backends can follow without changing the UX. Avoid a backend plugin framework initially.

Support CPU-backed Awkward arrays with one level of variable-length lists containing numeric records. Initially support field access, scalar comparisons, `abs`, Boolean combinations, filtering, and per-event sums returning flat numeric arrays.

Parse the supported Python subset into a small operation graph. Defer general tracing, custom behaviors, missing values, unions, combinations, arbitrary control flow, and unsupported dependencies. Unsupported functions continue using Python with an explanation.

## Recording and replay

Store Awkward inputs as Forms, lengths, and binary buffers using `ak.to_buffers` and `ak.from_buffers`. Record supported scalar arguments and reference outputs as well. Bound captured bytes and calls; report skipped captures without silently truncating inputs.

Treat recorded shapes and types as the intended structural contract: fields, nesting, numeric dtypes, and fixed dimensions. The actual values may change. Event counts and variable-length dimensions are not fixed by the sample and must support zero entries.

Extract the reference function into a standalone module with known imports and simple immutable constants. Do not import the original script to replay it. Initially reject closures, arbitrary helper dependencies, mutable globals, input mutation, and external side effects.

## Correctness and reuse

The evaluator owns the frozen reference and acceptance requirements. The generator cannot change them.

Require matching output container kind, nesting, length, and numeric dtype. Compare integers and Booleans exactly and floating-point values bitwise, including signed zero. Do not introduce numerical tolerances or relaxed floating-point transformations.

Evaluate recorded calls and generated cases within the supported contract. The LLM must look for edge cases absent from the recording and propose additional cases. The evaluator independently constructs those inputs and compares the candidate with the reference; successful replay of the sample alone is insufficient.

Required cases include zero events, empty lists within events, all-zero values, all or no values passing a selection, exact selection boundaries, varying list lengths, NaNs, infinities, and difficult sums. Preserve the structural schema when constructing empty cases. Generate these required cases deterministically even while model calls are mocked. Passing checks is empirical evidence, not a proof of equivalence.

Specialize on structure and types, not observed values, maximum list lengths, or data distributions. Fast dispatch is now the default at user request: generate `adapter.py` beside `kernel.py`, validate integrity/select a recorded contract once on load, then cache and call the unchecked adapter/kernel. Fast mode does not validate current source, dependencies or environment, even on first load, and assumes subsequent input schemas/types remain valid. Changes can silently give incorrect results; regenerate artifacts after changes. Process restart or explicit cache clearing reloads saved code but does not revalidate its reference. `--dispatch-mode careful` plus `AGENT_OPTIMIZE_MODE=careful` enables the original repeated reference/dependency/schema/environment checks. Mode-specific index entries and timing evidence coexist; neither mode consumes the other's artifacts. Correctness evaluation remains mandatory in both modes.

## Performance and candidate generation

Measure the complete replacement call, including dispatch, conversions, and output construction. Compile before steady-state measurements, warm both implementations, and repeat measurements. Report compilation and first-use costs separately.

Initially require a repeatable improvement of at least 10% with no material regression across retained recorded cases. Performance claims apply to measured workloads.

Give the model source, the operation graph, schemas, synthetic examples, and diagnostics. Keep recorded data buffers local, including when preparing failure reports. Generate candidates within a fixed computation interface; the generator does not edit the project, install dependencies, or control the evaluator.

Run candidates in separate worker processes with time limits. This contains failures but is not a security boundary for arbitrary code. Keep searches bounded in time, candidate count, and model usage.

## Model invocation

The default implementation uses deterministic local responses behind a small proposal interface. It requires no API key and makes no live requests. It exercises candidate generation and feedback while running the real recording, compilation, evaluation, and artifact infrastructure.

Explicit `--provider openai` enables the official OpenAI SDK in a dedicated, deadline-bounded provider process. It uses `responses.parse` with a strict Pydantic proposal schema, fixed official endpoint, retries disabled, bounded output tokens, and `store=False`. Provider-specific objects stay outside the evaluator. Coding-agent runners and provider frameworks remain deferred.

The application owns the loop: request one candidate, evaluate it locally, return concise feedback, and repeat within the search budget. Start with sequential generation and evaluation. The model proposes source and additional edge cases; it cannot change the reference, remove or weaken required tests, alter limits, or control the acceptance decision.

Requests include the reference source, operation graph, input schemas, required candidate signature, correctness constraints, and prior evaluation results. Use synthetic examples and counterexamples where useful; recorded data buffers remain local.

Use `client.responses.parse(...)` with a Pydantic response schema containing candidate source, a short description, and proposed additional edge cases. Handle refusals, incomplete responses, and invalid candidates explicitly. See the [official structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).

Live generation supports `gpt-6.1-sol`, `gpt-6.1-terra`, and `gpt-6-luna`; the CLI model default is `gpt-6.1-sol` only after live mode is explicitly selected. Credential precedence is `--api-key` > `AGENT_OPTIMIZE_OAI_API_KEY` in the environment > project root `.env` > user home `~/.env`. A configured key does not select live mode and `OPENAI_API_KEY` alone is not used. Identity, model/settings, allowlisted usage, and HTTP request counts accompany experiments and accepted manifests. The small saved comparison does not establish a preferred model.

Retain prompts, responses, usage, candidate source, and evaluation results locally for inspection. The key enters only the provider process through private stdin; analysis/evaluation environments omit credential variables. SDK failures become fixed diagnostic codes rather than raw HTTP exceptions. No key enters request files or candidate workers. This is process/data-flow isolation, not a restriction on trusted analysis code's filesystem access.

The synthetic comparison runner invokes the public CLI with fresh stores per model/case/repeat. The November 6 results reported in the README passed correctness for 35 of 36 proposals and saved none after measured dispatch regressions. Unsupported combinations/matching controls fell back without provider calls. The source/dependency fingerprint checks remain enabled; repeated verification contributes fixed dispatch overhead, which faster model-generated kernels alone do not remove. These illustrative single-seed Windows observations do not establish a model ranking or a general speedup.

## Platforms and testing

Support both Ubuntu Linux and native Windows. During development, run the same test suite in the Ubuntu distribution under WSL2 and directly on the local Windows platform. Passing on only one platform is insufficient. Report results for each environment separately.

For WSL2, work in a Linux-side checkout under `~/code/agent-optimizer` or a similar directory under `~/code`. Keep its Python environment, test files, generated artifacts, build output, and caches in that workspace rather than running from a Windows-mounted checkout under `/mnt/c` or `/mnt/d`. Ensure the Linux checkout includes the current changes being tested.

Use separate Python environments and platform-specific compiled caches for Windows and Ubuntu. Keep paths, command launching, and worker processes portable; do not depend on a particular shell or Linux-only process behavior.

The ordinary tests run without credentials or access to the OpenAI service. They cover fake SDK responses, credential precedence, private transport, errors/refusal/incomplete output, search limits, and fallback, alongside real local Numba compilation, recording/replay, persistence, dispatch, and invalidation. Controlled timing fixtures test acceptance rules without making ordinary tests depend on noisy speedups. The real provider test is separately opt-in with `AGENT_OPTIMIZE_LIVE_TESTS=1` and configured credentials.

## Documentation

Use [Zensical](https://zensical.org/) to build the Markdown documentation. Keep documentation sources and configuration under `docs/`, with generated HTML under `docs/_build/`, excluded from Git. Provide one documented build command that works on both native Windows and Ubuntu under WSL2, and verify the documentation build in both environments.

Include the [first round UX](ux-first-round.md) and these implementation notes in the documentation navigation. The UX document defines the user-facing workflow; these notes define the implementation choices behind it.

## Build order

1. Record and replay reference calls after the script exits.
2. Evaluate a hand-written fused candidate, including rejection of incorrect candidates.
3. Save accepted artifacts and verify ordinary dispatch, invalidation, and fallback.
4. Add the bounded generation loop using the mocked OpenAI provider.
5. Connect the complete workflow to the agreed command and concise human or JSON output.
6. Verify the infrastructure, test suite, and Zensical documentation build on local Windows and Ubuntu under WSL2.
7. Add optional live model calls after platform verification; retain offline default/tests and record bounded synthetic comparisons. This stage is implemented; further prompt tuning and support expansion need separate evidence.
