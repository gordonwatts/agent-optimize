# First round user interface

The first round should make optimizing an Awkward computation simple: mark a function, run the script on a small amount of real input, and let the tool handle the experiment. Extra controls can follow when new use cases require them.

This document defines the first-round interface. The current implementation defaults to a deterministic local mock generator and supports the restricted subset described in the [user guide](index.md). Explicit `--provider openai` enables the optional [live provider](live-provider.md). The broader motivation is in [the source conversation](source-conversation.md).

## Mark a function

The user keeps the readable Python implementation and adds a decorator:

```python
import awkward as ak
from agent_optimize import optimize


@optimize
def ht(events):
    jets = events.Jet
    selected = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
    return ak.sum(selected.pt, axis=1)
```

The decorated function is the optimization boundary. For this first round, it must compute its result from its arguments without modifying inputs or relying on mutable external state. The original Python remains the reference implementation.

## Run one command

The user runs their existing script through the tool:

```text
agent-optimize run -- python analysis.py --max-events 1000
```

Everything after `--` is the user's existing command. In this example, `--max-events` belongs to the script; the user uses their normal way of selecting a small amount of input.

The tool automatically:

1. Runs the script with decorated functions executing their reference Python implementations.
2. Records arguments and reference results for the decorated functions actually called.
3. Creates experiments and measures repeatable reference baselines.
4. Generates candidate implementations for the local CPU.
5. Builds, checks, and benchmarks the candidates.
6. Saves a compatible implementation when it passes the required checks and provides a measured improvement.

The search replays the recorded function calls directly. It does not repeatedly execute the whole script or repeat its file loading, plotting, and other surrounding work.

The user does not need to name an experiment, write an input factory, choose a candidate, or issue a separate acceptance command. Multiple decorated functions encountered during the run are handled automatically.

The default mock requires no credentials or live service. Optional live generation requires explicit selection and a configured key, resolved from the command line, named environment variable, project `.env`, then home `~/.env`, in that order. Searches have bounded defaults for time, candidate count, output tokens, and local request/response sizes. The tool shows effective limits before searching and stops when a limit is reached.

## Read a short result

The tool shows readable progress and ends with an outcome for each function. For example:

```text
ht: optimized implementation saved
    Faster on your recorded sample
    Passed configured correctness checks

Next: run your script normally.
```

The actual result includes measured timings and a speedup when available. If no eligible improvement is found, it says that Python will continue to be used. A failure identifies the affected function and the smallest useful next action.

Passing checks is evidence against the reference, not a proof of equivalence. The first round uses strict result comparison without silently relaxing numerical requirements. Evaluation also includes supported generated edge cases beyond the recorded sample. Performance claims apply to the measured workload; a small sample does not establish performance on larger inputs.

## Run normally afterward

```text
python analysis.py
```

Ordinary calls default to fast mode: load a saved fast implementation once, then execute its generated adapter/kernel without repeated compatibility checks. This assumes the function, dependencies, environment, and input schemas/types remain valid. Changed code or schemas can silently produce incorrect results; regenerate artifacts after changes. Missing or unusable artifacts retain Python. Select `--dispatch-mode careful` when optimizing and `AGENT_OPTIMIZE_MODE=careful` when running normally for repeated matching of the reference, dependencies, input structure and environment. Each mode only reuses artifacts timed and accepted for that same mode.

Importing the module or calling a decorated function normally never starts an agent search, contacts a model provider, or records input data. Careful mode detects changed code/dependencies; fast mode deliberately trusts saved code.

Recordings, reports, and generated implementations live in a local `.agent-optimize/` directory. Recording real inputs is an explicit effect of the optimization command. Provider requests and feedback exclude captured buffers and scalar values; synthetic examples are independent of the recordings. Removing the directory leaves the original Python usable.

## Keep exceptional cases simple

Recording has bounds on stored bytes and calls. The tool reports skipped captures clearly and does not silently truncate nested inputs. If no usable calls were captured, it explains that there is nothing to optimize.

Unsupported computations retain their Python implementation with an explanation. An interrupted search preserves completed evaluation results and any already accepted implementation.

## Let agents use the same interface

A coding agent uses the same command with structured output:

```text
agent-optimize run --json -- python analysis.py --max-events 1000
```

This mode runs without interactive prompts and returns structured outcomes, diagnostics, and artifact locations. Script output and progress stay separate from the structured result. Missing configuration produces an actionable error. The reference, evaluation requirements, and automatic acceptance rules are the same as for a human.

Separate recording commands, manual candidate submission, custom experiments, target selection, and visual interfaces can be added when needed. None is required for the first-round workflow.
