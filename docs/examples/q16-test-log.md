# q16 optimization test log — November 12, 2026

- Same cached ROOT file: 7,552 selected events. All retained implementations match
  the supplied reconstruction bit for bit on this file.
- The approximately six-second Python baseline covers reconstruction over 235
  small calls. It excludes ServiceX, ROOT reading and plotting.
- Same 235 groups: supplied Python **5.705 s**, improved raw NumPy **0.328 s**
  (**17.4×**), saved Numba **0.140 s**. Raw candidates passed **174** synthetic
  and edge comparisons, including explicit missing-value fallback.
- Raw NumPy experiments reduce repeated assignment and mass calculations without
  JIT. The grouped implementation remains experimental: its source is currently
  unsupported by the translator, and exceptional inputs use reference fallback.
- Compiler scheduling hoists reusable arithmetic and calculates the final mass
  only when the winning assignment changes. Fresh generation passed 108/108
  checks; ordinary saved dispatch was observed on all real-data events.
- Larger read groups reduce repeated call overhead. The q16 CLI exposes
  `--events-per-read`, default 1,024. Actual ROOT reading yielded 30 groups with
  exactly unchanged selected inputs: **0.0322 s warm saved dispatch**, **3.629 s
  first pass including compilation**. Use 128 for historical smaller groups.
- The latest live-model report records 143,327 tokens and an estimated
  $0.5311853. That verifies accounting from real API responses, not an invoice
  reconciliation. The new local compiler/raw experiments made no API calls.
- Exact software preservation does not resolve the separate physics benchmark
  discrepancy.

Detailed timings and reproducible commands:
[comparison](workflow-comparison.md), [raw source](q16-raw-optimization.md),
[compiler](../evidence/q16-codegen.md).
