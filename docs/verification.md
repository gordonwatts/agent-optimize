# Platform verification

The implementation defaults to deterministic mock generation; ordinary tests need
no credentials or live service. Optional OpenAI generation is covered by fake-client
tests plus a separately opted-in live test. Acceptance tests use explicitly labelled controlled timings
to exercise persistence and fresh-process dispatch. Real example timings below
are measured results and may remain ineligible.

## Optional provider checkpoint — November 6, 2026

The live-stage Windows offline checkpoint passed 342 tests with one live test
skipped; subsequent focused coverage also passed. The Ubuntu WSL2 offline
checkpoint passed 343 tests with two skips and a strict documentation build.
The separately opted-in `test_opt_in_live_proposal` passed on native Windows
(one test, 20.00 seconds) and Ubuntu WSL2 (one test, 12.65 seconds). The WSL2
live test read the authorized Windows home credential locally and passed it only
in the test child's environment; it did not copy a credential file into the
Linux checkout. Neither platform printed the key.

These are verification checkpoints, not speedup evidence. The earlier step-6
platform records below retain their original counts and source-snapshot scope.
The README separately reports two actual, native Windows synthetic model
comparisons from November 6: 35/36 correctness-passing proposals, no saved
implementation, and verified fallback for both unsupported controls. The
comparison used one seed/repeat and some early trials overlapped the Windows
test suite; its timings are illustrative observations, not benchmark guarantees.

## Earlier step-6 native Windows checkpoint — November 6, 2026

| Component | Verified value |
| --- | --- |
| OS | Windows 11, build 10.0.26300, AMD64 |
| Python | 3.12.11, repository-local `.venv` |
| Package | agent-optimize 0.1.0, editable installation with test/docs extras |
| Awkward / NumPy | 2.14.0 / 2.5.4 |
| Numba / llvmlite | 0.68.1 / 0.50.0 |
| pytest / Zensical | 9.2.0 / 0.1.0 |
| Full suite | 296 passed in 259.05 seconds |
| Documentation | Strict build passed without warnings |
| Package dependencies | `pip check` passed |

This earlier step-6 suite covers independent real Numba compilation and strict correctness,
controlled acceptance, persistence, fresh-process ordinary dispatch, compatibility
changes, quarantine/fallback, paths with spaces, and process timeouts. Application
source subsequently changed for the optional provider stage above. Documentation-only status and
navigation updates receive the local-link checks and strict documentation build;
they do not trigger another full suite.

The standalone verification copied `examples/filter_sum.py` to a script whose
path contains spaces and used a separate artifact directory with spaces. It ran
the normal JSON command on 1,000 events, followed by a fresh ordinary invocation
with `AGENT_OPTIMIZE_ARTIFACT_ROOT` pointing at the same store. Both completed
successfully and produced `events=1000 total_ht=64115.0`. The ordinary invocation
created no recording session.

The mock's initial proposal was rejected for correctness. Its two corrected
proposals passed correctness but were ineligible because the complete replacement
path regressed on the retained recording of 1,000 events and 2,000 elements:

| Corrected proposal | Reference ms/call | Replacement ms/call | Compilation ms | First replacement use ms |
| --- | ---: | ---: | ---: | ---: |
| 2 | 2.437 | 59.978 | 495.661 | 1078.551 |
| 3 | 2.378 | 57.692 | 481.995 | 1089.950 |

Call times are medians of batch medians, with compilation excluded from steady
state. The replacement includes applicability/dependency checks, conversion,
kernel work, and output construction. No measured improvement was saved; the
ordinary example retained Python. These are local observations for this workload,
not a general performance benchmark. Controlled acceptance evidence remains
separate from these measured results.

A worker script with spaces in its path, sleeping for 60 seconds, was stopped by
the configured 0.2-second timeout. The result was `timeout`, with a terminated
process and approximately 0.219 seconds observed elapsed time on Windows.

Versioned native evidence, exact commands, local report paths, and raw timing
records are retained under ignored `.agent-optimize/verification/`. The analysis
reports retain the full measurement samples in their evaluation evidence.

## Earlier step-6 Ubuntu WSL2 checkpoint — November 6, 2026

| Component | Verified value |
| --- | --- |
| OS | Ubuntu 22.04 under WSL2, Linux 6.18.54.1 |
| Python | 3.12.9, separate Linux environment |
| Awkward / NumPy | 2.14.0 / 2.5.4 |
| Numba | 0.68.1 |
| Full suite | 296 passed in 91.45 seconds |
| Documentation | Build passed with no issues |

The Linux-side run also exercised the 1,000-event example using a path with spaces
and obtained `events=1000 total_ht=64115.0`. A subsequent ordinary invocation also
printed `events=1000 total_ht=64115.0`. The optimization command's proposal outcomes were
`rejected`, `ineligible`, and `ineligible`. The corrected replacements regressed
on the recorded workload, so no measured improvement was saved:

| Corrected proposal | Reference ms/call | Replacement ms/call |
| --- | ---: | ---: |
| 2 | 2.377 | 18.695 |
| 3 | 2.442 | 19.957 |

These are local timing summaries of the full replacement path, separate from
controlled acceptance tests. The final Linux suite used `TMPDIR` and pytest's
base temporary directory inside the Linux workspace. Its worker crash and timeout
tests passed as part of the full suite. Source byte comparisons of `src/`,
`tests/`, `examples/`, `docs/`, and the root README, plan, package configuration,
and ignore file matched the synchronized Windows verification snapshot, excluding
generated caches and build output. Each platform used its own environment,
recordings, artifacts, and caches.

## Source identity and reproduction

At the earlier verification checkpoint, Git HEAD was baseline commit
`c6305581d42eb241468c3d40dfb78d87b81c36fe`, and the implementation under test
included uncommitted changes. That baseline alone therefore does not identify
those tested files. For a new comparison, synchronize the current source into
the Linux-side checkout and compare a fresh source manifest.

The earlier step-6 native Windows inventory is retained at
`.agent-optimize/verification/windows-7c87cc7677b44141a61a7e829634d78e/source-manifest.json`,
with command/version/timing evidence in the adjacent `windows-verification.json`.
That manifest records the Windows files and their hashes. The separately verified
byte comparison described above established the match for the synchronized
Windows/Linux verification snapshot. This historical manifest does not identify
the later optional-provider implementation or these final documentation changes.
Refresh the inventory and comparison as described below before claiming a match
for the current tree.

The portable manifest includes these root files:

- `.gitignore`, `README.md`, `plan.md`, and `pyproject.toml`;
- Python files under `src/`, `tests/`, `examples/`, and `benchmarks/`;
- Markdown, TOML, and PDF sources under `docs/`.

Exclude environments, `.git` internals, Python/compiled caches, recordings,
artifacts, generated HTML, and documentation caches. Sort repository-relative
paths using `/`; record each file's byte length and SHA-256 over its raw bytes.
Do not normalize line endings during transfer. Serialize
`{"schema_version": 1, "files": [...]}` as UTF-8 JSON with sorted keys, compact
separators, and one final newline. Each file entry has `path`, `nbytes`, and
`sha256`. The SHA-256 of that complete JSON is the aggregate manifest identifier.
The manifest is stored outside its source set under ignored
`.agent-optimize/verification/`, avoiding a self-referential hash.

Windows and Linux manifests must have the same paths, sizes, and hashes. Build
outputs and installed environments are verified independently rather than copied.
If final documentation results change after a manifest is produced, synchronize
those files and regenerate both manifests before reporting the final comparison.

With the local environment active, reproduce the gates from the checkout root:

```text
python -m pytest
python -m zensical build --config-file docs/zensical.toml --strict
```

Use a Linux-side checkout under `~/code/agent-optimizer` for WSL2, with its own
environment and artifact directories. Do not run the Linux checks from `/mnt/c`
or `/mnt/d`. The [user guide](index.md) gives installation and example commands.
