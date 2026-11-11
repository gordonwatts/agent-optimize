# q16 complete-file comparison of the LLM-generated Numba function

The last candidate from the ten-attempt compiler-feedback trial was tested on every selected event in the complete delivered ROOT file, despite its two synthetic-test failures. The retained candidate source was verified by SHA-256 before execution. This check made no model API requests.

## Full real-data results

| Check | Result |
| --- | ---: |
| Selected events | 7,552 |
| Function input groups | 235 |
| Shape and output dtype | Match |
| Results matching bit for bit | 7,547 |
| Finite results with different bits | 5 |
| Largest absolute difference | 6.82121e-13 GeV |
| Largest relative difference | 5.49324e-16 |
| Largest representable-float distance | 4 ULP |
| Revised Python versus supplied Python | Bit-for-bit match on every event |

The five real-data differences are small floating-point rounding differences. The two recorded verification groups did not contain these events. Each of three repeated full-file passes reproduced the same outputs.

## Warm full-reconstruction timings

| Implementation | Median elapsed time |
| --- | ---: |
| Supplied original Python reconstruction | 5.780842 s |
| Revised Python reference | 5.656156 s |
| LLM-generated Numba function, direct checked adapter | 0.245314 s |

Measured speedup over the supplied Python reconstruction: **23.57×**.

These are medians of three complete-file passes, alternating measurement order. They include input packing, candidate input adaptation and output wrapping, but exclude ROOT reading, ServiceX, plotting and first compilation. The first candidate call, including compilation, took 1.125677 s. The candidate was invoked through a checked adapter for this diagnostic benchmark; no saved-dispatch artifact was installed.

## What the synthetic failures mean

Both failing cases mix NaNs, infinities and signed zeros in the energy field, in different orders. Each contains 16 outputs. In each case, eight NaN outputs have different sign/payload bits. Both implementations produce NaN in the same positions, and all finite outputs in these cases match exactly. The failures disappear if NaN bit patterns are ignored.

The current strict bitwise acceptance rule still rejects this candidate. This diagnostic continued through the complete real-data comparison and warm timing measurements and retained both the synthetic failures and the real-data rounding differences.

## Provenance

- Candidate source SHA-256: `767d91f6b0029c66f1b006308c7283a31d567641a27ab47a09e482b31cbf31f5`
- Delivered input SHA-256: `bd3ecc50194dc836bf0e8f9f7201fc000309644fc9ad0f8f3cc4dea5e3bc5a50`
- Candidate source and retained outputs came from the completed ten-attempt live compiler-feedback trial.
