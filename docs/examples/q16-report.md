# Optimization report: q16\_ttbar

**Result:** Completed. **1 of 1 functions optimized.**

## Run at a glance

| Phase | Elapsed time |
| --- | ---: |
| Analysis (recording run) | 10.570 s |
| Replay, correctness and optimization | 30.652 s |
| Controller total | 41.719 s |

Controller timings exclude uv environment preparation. The analysis executes once during recording.

API usage: 0 tokens; cost USD 0 (no API requests).

Provider attempts: 2. HTTP requests: 0.

Generator: deterministic local mock. No model calls were made.

| Provider / model | Reported tokens | Estimated API cost |
| --- | ---: | ---: |
| mock / none | 0 | USD 0 |

## Timing comparison

Measurements below compare function calls on recorded batches, including ordinary dispatch. They are not whole-analysis speedups. Correctness means agreement with the Python reference, not independent validation of the underlying analysis.

| Function / batch | Events | Python reference | Replacement | Minimum batch speedup |
| --- | ---: | ---: | ---: | ---: |
| reconstruct\_events / case-000 | 24 | 24.639 ms | 0.174 ms | 137.87x |
| reconstruct\_events / case-001 | 24 | 23.921 ms | 0.149 ms | 154.25x |

## Functions

### reconstruct\_events

**Status:** optimized. eligible fast implementation saved; run the script normally to reuse it

Recorded calls: 2. Proposals attempted: 2.

Attempt outcomes: 1 rejected, 1 optimized.

Not recorded: 233 calls. per-function capture-attempt limit reached before buffer copying. These calls still used the Python reference.

#### Saved replacement 1

Correctness: **108 cases passed**. Timing and compatibility gates passed.

Compilation: 3805.181 ms. First replacement use: 3817.745 ms. These are separate from warm measurements.

**Python reference**

```python
MTOP = 172.5
MW = 80.4
import awkward as ak
import numpy as np
def reconstruct_events(events):
    """Reconstruct via ordered distinct jet assignments and first-minimum scoring."""
    jets = events.jets
    px = events.lepton.px
    py = events.lepton.py
    pz = events.lepton.pz
    energy = events.lepton.e
    metx = events.metx
    mety = events.mety
    mu = MW * MW / 2.0 + px * metx + py * mety
    a = energy * energy - pz * pz
    disc = mu * mu - a * (metx * metx + mety * mety)
    root = energy * np.sqrt(np.maximum(disc, 0.0))
    plus = (mu * pz + root) / a
    minus = (mu * pz - root) / a
    nuz = ak.where(abs(plus) <= abs(minus), plus, minus)
    nue = np.sqrt(metx * metx + mety * mety + nuz * nuz)
    indices = ak.argcartesian({'i': jets, 'j': jets, 'k': jets, 'l': jets}, axis=1)
    valid = (indices.i < indices.j) & (indices.k != indices.i) & (indices.k != indices.j) & (indices.l != indices.i) & (indices.l != indices.j) & (indices.l != indices.k)
    chosen = indices[valid]
    j1 = jets[chosen.i]
    j2 = jets[chosen.j]
    bh = jets[chosen.k]
    bl = jets[chosen.l]
    h_e = 0 + (0 + j1.e + j2.e) + bh.e
    l_e = 0 + (0 + energy + nue)[:, None] + bl.e
    h_px = 0 + (0 + j1.px + j2.px) + bh.px
    l_px = 0 + (0 + px + metx)[:, None] + bl.px
    h_py = 0 + (0 + j1.py + j2.py) + bh.py
    l_py = 0 + (0 + py + mety)[:, None] + bl.py
    h_pz = 0 + (0 + j1.pz + j2.pz) + bh.pz
    l_pz = 0 + (0 + pz + nuz)[:, None] + bl.pz
    mh = np.sqrt(np.maximum(0.0, h_e ** 2 - h_px ** 2 - h_py ** 2 - h_pz ** 2))
    ml = np.sqrt(np.maximum(0.0, l_e ** 2 - l_px ** 2 - l_py ** 2 - l_pz ** 2))
    score = abs(mh - MTOP) + abs(ml - MTOP)
    combined = np.sqrt(np.maximum(0.0, (0 + h_e + l_e) ** 2 - (0 + h_px + l_px) ** 2 - (0 + h_py + l_py) ** 2 - (0 + h_pz + l_pz) ** 2))
    return ak.sum(combined[ak.local_index(score, axis=1) == ak.argmin(score, axis=1, keepdims=True)], axis=1)
```

**Numba replacement**

```python
def kernel(offsets, columns, scalars):
    output = np.zeros(len(offsets) - 1, dtype=np.float64)
    for event in range(len(offsets) - 1):
        found = False
        best_score = np.float64(0)
        v_energy = columns[4][event]
        v_metx = columns[8][event]
        v_mety = columns[9][event]
        v_px = columns[5][event]
        v_py = columns[6][event]
        temporary0 = ((80.4 + np.float64(0)) if np.isnan(80.4) else ((80.4 + np.float64(0)) if np.isnan(80.4) else (80.4 * 80.4)))
        temporary1 = ((temporary0 + np.float64(0)) if np.isnan(temporary0) else ((2.0 + np.float64(0)) if np.isnan(2.0) else (temporary0 / 2.0)))
        temporary2 = ((v_metx + np.float64(0)) if np.isnan(v_metx) else ((v_px + np.float64(0)) if np.isnan(v_px) else (v_px * v_metx)))
        temporary3 = ((temporary2 + np.float64(0)) if np.isnan(temporary2) else ((temporary1 + np.float64(0)) if np.isnan(temporary1) else (temporary1 + temporary2)))
        temporary4 = ((v_mety + np.float64(0)) if np.isnan(v_mety) else ((v_py + np.float64(0)) if np.isnan(v_py) else (v_py * v_mety)))
        temporary5 = ((temporary4 + np.float64(0)) if np.isnan(temporary4) else ((temporary3 + np.float64(0)) if np.isnan(temporary3) else (temporary3 + temporary4)))
        v_mu = temporary5
        v_pz = columns[7][event]
        temporary6 = ((v_energy + np.float64(0)) if np.isnan(v_energy) else ((v_energy + np.float64(0)) if np.isnan(v_energy) else (v_energy * v_energy)))
        temporary7 = ((v_pz + np.float64(0)) if np.isnan(v_pz) else ((v_pz + np.float64(0)) if np.isnan(v_pz) else (v_pz * v_pz)))
        temporary8 = ((temporary6 + np.float64(0)) if np.isnan(temporary6) else ((temporary7 + np.float64(0)) if np.isnan(temporary7) else (temporary6 - temporary7)))
        v_a = temporary8
        temporary9 = ((v_mu + np.float64(0)) if np.isnan(v_mu) else ((v_mu + np.float64(0)) if np.isnan(v_mu) else (v_mu * v_mu)))
        temporary10 = ((v_metx + np.float64(0)) if np.isnan(v_metx) else ((v_metx + np.float64(0)) if np.isnan(v_metx) else (v_metx * v_metx)))
        temporary11 = ((v_mety + np.float64(0)) if np.isnan(v_mety) else ((v_mety + np.float64(0)) if np.isnan(v_mety) else (v_mety * v_mety)))
        temporary12 = ((temporary11 + np.float64(0)) if np.isnan(temporary11) else ((temporary10 + np.float64(0)) if np.isnan(temporary10) else (temporary10 + temporary11)))
        temporary13 = ((temporary12 + np.float64(0)) if np.isnan(temporary12) else ((v_a + np.float64(0)) if np.isnan(v_a) else (v_a * temporary12)))
        temporary14 = ((temporary9 + np.float64(0)) if np.isnan(temporary9) else ((temporary13 + np.float64(0)) if np.isnan(temporary13) else (temporary9 - temporary13)))
        v_disc = temporary14
        temporary15 = (v_disc if np.isnan(v_disc) else np.maximum(v_disc, 0.0))
        temporary16 = np.sqrt(temporary15)
        temporary17 = ((temporary16 + np.float64(0)) if np.isnan(temporary16) else ((v_energy + np.float64(0)) if np.isnan(v_energy) else (v_energy * temporary16)))
        v_root = temporary17
        temporary18 = ((v_pz + np.float64(0)) if np.isnan(v_pz) else ((v_mu + np.float64(0)) if np.isnan(v_mu) else (v_mu * v_pz)))
        temporary19 = ((v_root + np.float64(0)) if np.isnan(v_root) else ((temporary18 + np.float64(0)) if np.isnan(temporary18) else (temporary18 + v_root)))
        temporary20 = ((temporary19 + np.float64(0)) if np.isnan(temporary19) else ((v_a + np.float64(0)) if np.isnan(v_a) else (temporary19 / v_a)))
        v_plus = temporary20
        temporary21 = ((v_pz + np.float64(0)) if np.isnan(v_pz) else ((v_mu + np.float64(0)) if np.isnan(v_mu) else (v_mu * v_pz)))
        temporary22 = ((temporary21 + np.float64(0)) if np.isnan(temporary21) else ((v_root + np.float64(0)) if np.isnan(v_root) else (temporary21 - v_root)))
        temporary23 = ((temporary22 + np.float64(0)) if np.isnan(temporary22) else ((v_a + np.float64(0)) if np.isnan(v_a) else (temporary22 / v_a)))
        v_minus = temporary23
        temporary24 = abs(v_plus)
        temporary25 = abs(v_minus)
        temporary26 = (temporary24 <= temporary25)
        temporary27 = (v_plus if temporary26 else v_minus)
        v_nuz = temporary27
        temporary28 = ((v_metx + np.float64(0)) if np.isnan(v_metx) else ((v_metx + np.float64(0)) if np.isnan(v_metx) else (v_metx * v_metx)))
        temporary29 = ((v_mety + np.float64(0)) if np.isnan(v_mety) else ((v_mety + np.float64(0)) if np.isnan(v_mety) else (v_mety * v_mety)))
        temporary30 = ((temporary29 + np.float64(0)) if np.isnan(temporary29) else ((temporary28 + np.float64(0)) if np.isnan(temporary28) else (temporary28 + temporary29)))
        temporary31 = ((v_nuz + np.float64(0)) if np.isnan(v_nuz) else ((v_nuz + np.float64(0)) if np.isnan(v_nuz) else (v_nuz * v_nuz)))
        temporary32 = ((temporary31 + np.float64(0)) if np.isnan(temporary31) else ((temporary30 + np.float64(0)) if np.isnan(temporary30) else (temporary30 + temporary31)))
        temporary33 = np.sqrt(temporary32)
        v_nue = temporary33
        for item0 in range(offsets[event], offsets[event + 1]):
            for item1 in range(offsets[event], offsets[event + 1]):
                if ((item0 - offsets[event]) < (item1 - offsets[event])):
                    for item2 in range(offsets[event], offsets[event + 1]):
                        if ((item2 - offsets[event]) != (item0 - offsets[event])) and ((item2 - offsets[event]) != (item1 - offsets[event])):
                            for item3 in range(offsets[event], offsets[event + 1]):
                                if ((item3 - offsets[event]) != (item0 - offsets[event])) and ((item3 - offsets[event]) != (item1 - offsets[event])) and ((item3 - offsets[event]) != (item2 - offsets[event])):
                                    temporary34 = ((columns[0][item0] + np.float64(0)) if np.isnan(columns[0][item0]) else ((0 + np.float64(0)) if np.isnan(0) else (0 + columns[0][item0])))
                                    temporary35 = ((columns[0][item1] + np.float64(0)) if np.isnan(columns[0][item1]) else ((temporary34 + np.float64(0)) if np.isnan(temporary34) else (temporary34 + columns[0][item1])))
                                    temporary36 = ((temporary35 + np.float64(0)) if np.isnan(temporary35) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary35)))
                                    temporary37 = ((columns[0][item2] + np.float64(0)) if np.isnan(columns[0][item2]) else ((temporary36 + np.float64(0)) if np.isnan(temporary36) else (temporary36 + columns[0][item2])))
                                    v_h_e = temporary37
                                    temporary38 = ((columns[1][item0] + np.float64(0)) if np.isnan(columns[1][item0]) else ((0 + np.float64(0)) if np.isnan(0) else (0 + columns[1][item0])))
                                    temporary39 = ((columns[1][item1] + np.float64(0)) if np.isnan(columns[1][item1]) else ((temporary38 + np.float64(0)) if np.isnan(temporary38) else (temporary38 + columns[1][item1])))
                                    temporary40 = ((temporary39 + np.float64(0)) if np.isnan(temporary39) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary39)))
                                    temporary41 = ((columns[1][item2] + np.float64(0)) if np.isnan(columns[1][item2]) else ((temporary40 + np.float64(0)) if np.isnan(temporary40) else (temporary40 + columns[1][item2])))
                                    v_h_px = temporary41
                                    temporary42 = ((columns[2][item0] + np.float64(0)) if np.isnan(columns[2][item0]) else ((0 + np.float64(0)) if np.isnan(0) else (0 + columns[2][item0])))
                                    temporary43 = ((columns[2][item1] + np.float64(0)) if np.isnan(columns[2][item1]) else ((temporary42 + np.float64(0)) if np.isnan(temporary42) else (temporary42 + columns[2][item1])))
                                    temporary44 = ((temporary43 + np.float64(0)) if np.isnan(temporary43) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary43)))
                                    temporary45 = ((columns[2][item2] + np.float64(0)) if np.isnan(columns[2][item2]) else ((temporary44 + np.float64(0)) if np.isnan(temporary44) else (temporary44 + columns[2][item2])))
                                    v_h_py = temporary45
                                    temporary46 = ((columns[3][item0] + np.float64(0)) if np.isnan(columns[3][item0]) else ((0 + np.float64(0)) if np.isnan(0) else (0 + columns[3][item0])))
                                    temporary47 = ((columns[3][item1] + np.float64(0)) if np.isnan(columns[3][item1]) else ((temporary46 + np.float64(0)) if np.isnan(temporary46) else (temporary46 + columns[3][item1])))
                                    temporary48 = ((temporary47 + np.float64(0)) if np.isnan(temporary47) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary47)))
                                    temporary49 = ((columns[3][item2] + np.float64(0)) if np.isnan(columns[3][item2]) else ((temporary48 + np.float64(0)) if np.isnan(temporary48) else (temporary48 + columns[3][item2])))
                                    v_h_pz = temporary49
                                    temporary50 = (v_h_e ** 2)
                                    temporary51 = (v_h_px ** 2)
                                    temporary52 = ((temporary50 + np.float64(0)) if np.isnan(temporary50) else ((temporary51 + np.float64(0)) if np.isnan(temporary51) else (temporary50 - temporary51)))
                                    temporary53 = (v_h_py ** 2)
                                    temporary54 = ((temporary52 + np.float64(0)) if np.isnan(temporary52) else ((temporary53 + np.float64(0)) if np.isnan(temporary53) else (temporary52 - temporary53)))
                                    temporary55 = (v_h_pz ** 2)
                                    temporary56 = ((temporary54 + np.float64(0)) if np.isnan(temporary54) else ((temporary55 + np.float64(0)) if np.isnan(temporary55) else (temporary54 - temporary55)))
                                    temporary57 = (0.0 if np.isnan(0.0) else np.maximum(0.0, temporary56))
                                    temporary58 = np.sqrt(temporary57)
                                    v_mh = temporary58
                                    temporary59 = ((v_energy + np.float64(0)) if np.isnan(v_energy) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_energy)))
                                    temporary60 = ((v_nue + np.float64(0)) if np.isnan(v_nue) else ((temporary59 + np.float64(0)) if np.isnan(temporary59) else (temporary59 + v_nue)))
                                    temporary61 = ((temporary60 + np.float64(0)) if np.isnan(temporary60) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary60)))
                                    temporary62 = ((columns[0][item3] + np.float64(0)) if np.isnan(columns[0][item3]) else ((temporary61 + np.float64(0)) if np.isnan(temporary61) else (temporary61 + columns[0][item3])))
                                    v_l_e = temporary62
                                    temporary63 = ((v_px + np.float64(0)) if np.isnan(v_px) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_px)))
                                    temporary64 = ((v_metx + np.float64(0)) if np.isnan(v_metx) else ((temporary63 + np.float64(0)) if np.isnan(temporary63) else (temporary63 + v_metx)))
                                    temporary65 = ((temporary64 + np.float64(0)) if np.isnan(temporary64) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary64)))
                                    temporary66 = ((columns[1][item3] + np.float64(0)) if np.isnan(columns[1][item3]) else ((temporary65 + np.float64(0)) if np.isnan(temporary65) else (temporary65 + columns[1][item3])))
                                    v_l_px = temporary66
                                    temporary67 = ((v_py + np.float64(0)) if np.isnan(v_py) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_py)))
                                    temporary68 = ((v_mety + np.float64(0)) if np.isnan(v_mety) else ((temporary67 + np.float64(0)) if np.isnan(temporary67) else (temporary67 + v_mety)))
                                    temporary69 = ((temporary68 + np.float64(0)) if np.isnan(temporary68) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary68)))
                                    temporary70 = ((columns[2][item3] + np.float64(0)) if np.isnan(columns[2][item3]) else ((temporary69 + np.float64(0)) if np.isnan(temporary69) else (temporary69 + columns[2][item3])))
                                    v_l_py = temporary70
                                    temporary71 = ((v_pz + np.float64(0)) if np.isnan(v_pz) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_pz)))
                                    temporary72 = ((v_nuz + np.float64(0)) if np.isnan(v_nuz) else ((temporary71 + np.float64(0)) if np.isnan(temporary71) else (temporary71 + v_nuz)))
                                    temporary73 = ((temporary72 + np.float64(0)) if np.isnan(temporary72) else ((0 + np.float64(0)) if np.isnan(0) else (0 + temporary72)))
                                    temporary74 = ((columns[3][item3] + np.float64(0)) if np.isnan(columns[3][item3]) else ((temporary73 + np.float64(0)) if np.isnan(temporary73) else (temporary73 + columns[3][item3])))
                                    v_l_pz = temporary74
                                    temporary75 = (v_l_e ** 2)
                                    temporary76 = (v_l_px ** 2)
                                    temporary77 = ((temporary75 + np.float64(0)) if np.isnan(temporary75) else ((temporary76 + np.float64(0)) if np.isnan(temporary76) else (temporary75 - temporary76)))
                                    temporary78 = (v_l_py ** 2)
                                    temporary79 = ((temporary77 + np.float64(0)) if np.isnan(temporary77) else ((temporary78 + np.float64(0)) if np.isnan(temporary78) else (temporary77 - temporary78)))
                                    temporary80 = (v_l_pz ** 2)
                                    temporary81 = ((temporary79 + np.float64(0)) if np.isnan(temporary79) else ((temporary80 + np.float64(0)) if np.isnan(temporary80) else (temporary79 - temporary80)))
                                    temporary82 = (0.0 if np.isnan(0.0) else np.maximum(0.0, temporary81))
                                    temporary83 = np.sqrt(temporary82)
                                    v_ml = temporary83
                                    temporary84 = ((v_mh + np.float64(0)) if np.isnan(v_mh) else ((172.5 + np.float64(0)) if np.isnan(172.5) else (v_mh - 172.5)))
                                    temporary85 = abs(temporary84)
                                    temporary86 = ((v_ml + np.float64(0)) if np.isnan(v_ml) else ((172.5 + np.float64(0)) if np.isnan(172.5) else (v_ml - 172.5)))
                                    temporary87 = abs(temporary86)
                                    temporary88 = ((temporary87 + np.float64(0)) if np.isnan(temporary87) else ((temporary85 + np.float64(0)) if np.isnan(temporary85) else (temporary85 + temporary87)))
                                    v_score = temporary88
                                    temporary89 = ((v_h_e + np.float64(0)) if np.isnan(v_h_e) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_h_e)))
                                    temporary90 = ((v_l_e + np.float64(0)) if np.isnan(v_l_e) else ((temporary89 + np.float64(0)) if np.isnan(temporary89) else (temporary89 + v_l_e)))
                                    temporary91 = (temporary90 ** 2)
                                    temporary92 = ((v_h_px + np.float64(0)) if np.isnan(v_h_px) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_h_px)))
                                    temporary93 = ((v_l_px + np.float64(0)) if np.isnan(v_l_px) else ((temporary92 + np.float64(0)) if np.isnan(temporary92) else (temporary92 + v_l_px)))
                                    temporary94 = (temporary93 ** 2)
                                    temporary95 = ((temporary91 + np.float64(0)) if np.isnan(temporary91) else ((temporary94 + np.float64(0)) if np.isnan(temporary94) else (temporary91 - temporary94)))
                                    temporary96 = ((v_h_py + np.float64(0)) if np.isnan(v_h_py) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_h_py)))
                                    temporary97 = ((v_l_py + np.float64(0)) if np.isnan(v_l_py) else ((temporary96 + np.float64(0)) if np.isnan(temporary96) else (temporary96 + v_l_py)))
                                    temporary98 = (temporary97 ** 2)
                                    temporary99 = ((temporary95 + np.float64(0)) if np.isnan(temporary95) else ((temporary98 + np.float64(0)) if np.isnan(temporary98) else (temporary95 - temporary98)))
                                    temporary100 = ((v_h_pz + np.float64(0)) if np.isnan(v_h_pz) else ((0 + np.float64(0)) if np.isnan(0) else (0 + v_h_pz)))
                                    temporary101 = ((v_l_pz + np.float64(0)) if np.isnan(v_l_pz) else ((temporary100 + np.float64(0)) if np.isnan(temporary100) else (temporary100 + v_l_pz)))
                                    temporary102 = (temporary101 ** 2)
                                    temporary103 = ((temporary99 + np.float64(0)) if np.isnan(temporary99) else ((temporary102 + np.float64(0)) if np.isnan(temporary102) else (temporary99 - temporary102)))
                                    temporary104 = (0.0 if np.isnan(0.0) else np.maximum(0.0, temporary103))
                                    temporary105 = np.sqrt(temporary104)
                                    v_combined = temporary105
                                    score = v_score
                                    if not found or score < best_score:
                                        found = True
                                        best_score = score
                                        output[event] = np.float64(0) + v_combined
    return output
```

## Next step

Run the script normally to reuse the saved fast-mode implementations.
