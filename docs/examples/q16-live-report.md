# Optimization report: q16\_ttbar

**Result:** Completed. **0 of 1 functions optimized.**

Command used for this run:

```text
agent-optimize run --json --report --provider openai --model gpt-6.1-sol --verification-groups 2 --max-output-tokens 8192 --artifact-root .agent-optimize/q16-live-servicex-180 -- C:/Users/gwatts/.codex/worktrees/q16-reconstruction/agent-optimize/.venv/Scripts/python.exe -B C:/Users/gwatts/.codex/worktrees/q16-reconstruction/agent-optimize/examples/q16_ttbar.py --n-files 1 --output-dir .agent-optimize/q16-live-servicex-180-plots
```

## Function speedup

Each row is a different saved verification group. The Events column gives the number of events in that group. The original Python function and Numba function are timed using the same events, and repeated measurements reuse those events.

| Function / verification group | Events | Original Python function | Numba function | Conservative speedup |
| --- | ---: | ---: | ---: | ---: |

No Numba function with measured timing evidence was saved.

## Time and API cost to obtain the result

| Phase | Elapsed time |
| --- | ---: |
| Run analysis once and save verification groups | 11.947 s |
| Generate, check, compile and time Numba functions | 125.329 s |
| Total elapsed time | 137.705 s |

The total covers the analysis and optimization work, plus coordination overhead. Preparing the Python environment is excluded. The analysis executes once; subsequent checks rerun only the saved function inputs.

API usage: 26,778 tokens; cost USD 0.0999033 estimated.

| Code generator / model | Reported tokens | Estimated API cost |
| --- | ---: | ---: |
| openai / gpt-6.1-sol | 26778 | USD 0.0999033 |

## Checks and saved code

Correctness here means exact agreement with the original Python function, including floating-point bit patterns.

### Function reconstruct\_events

No Numba function was saved; the original Python function remains in use.

Collected 2 verification groups (out of 235 total function calls).

**Details:** no eligible implementation saved; Python remains in use

## Cost estimate basis

Costs are estimates of agent-optimize's API requests, not a reconciled invoice. Reasoning tokens are included in output; cache reads/writes are subsets of input. Missing usage/rates leave the total unavailable.

- gpt-6.1-sol: openai-standard-2026-11-10, verified 2026-11-10. OpenAI direct API, Standard tier, default 30-minute cache writes; excludes taxes/credits and regional premiums. [Pricing source](https://developers.openai.com/api/docs/pricing).
