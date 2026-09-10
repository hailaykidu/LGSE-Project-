# Table 2 -- rankings and the LGSE-FOCUS comparison

## Note on interpretation

**The authoritative Table 2 results are those reported in the published LGSE paper.** The results below are reproduced using the repository's implementation and documented choices. They should be read as supporting reproduction evidence, not as replacement or reinterpretation of the published values.

For complete context on the relationship between published results and repository artifacts, see `IMPLEMENTATION_NOTES.md` Section 3: "Table 2 reproducibility."

## Results from this implementation

Mean +/- sample sd over seeds 42-46. Metric: accuracy for TC (Table 2 'AC'), F1 for NER and QA.

The paper specifies a learned projection matrix W but does not specify how W is obtained (Sec. 4.1). To implement the pipeline, this repository uses orthogonal Procrustes alignment to instantiate W (see `data/alignment/W_*.json`). The paper defines the regularization coefficient lambda but does not provide a numerical value or selection procedure; to implement the pipeline, this repository uses lambda = 1.0. These are results from this implementation, obtained using those implementation choices.

## amharic / ner (f1)

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLM-R (default) | 68.97 | 1.94 | 5 | - | 68.86 | 71.33 | 66.61 | 70.43 | 67.63 |
| 2 | +LAPT | 68.65 | 1.34 | 5 | - | 69.41 | 66.50 | 69.21 | 69.86 | 68.25 |
| 3 | +FOCUS+LAPT | 68.43 | 2.81 | 5 | - | 66.21 | 72.06 | 70.32 | 65.30 | 68.28 |
| 4 | +LGSE+LAPT | 67.73 | 2.27 | 5 | - | 69.05 | 70.48 | 67.62 | 64.45 | 67.05 |
| 5 | +Random+LAPT | 67.23 | 1.46 | 5 | - | 67.43 | 65.06 | 69.14 | 67.51 | 67.01 |

**LGSE - FOCUS = -0.70** (67.73 +/- 2.27 vs 68.43 +/- 2.81)

- Primary (LGSE > FOCUS): **NOT MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: XLM-R (default) > +LAPT > +FOCUS+LAPT > +LGSE+LAPT > +Random+LAPT

## amharic / qa (f1)

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLM-R (default) | 59.50 | 1.48 | 5 | - | 59.84 | 61.35 | 59.94 | 57.30 | 59.08 |
| 2 | +FOCUS+LAPT | 59.23 | 1.98 | 5 | - | 58.26 | 59.33 | 56.45 | 60.65 | 61.47 |
| 3 | +LAPT | 59.21 | 2.74 | 5 | - | 59.04 | 55.30 | 58.42 | 62.71 | 60.58 |
| 4 | +LGSE+LAPT | 58.67 | 0.79 | 5 | - | 58.42 | 58.06 | 57.86 | 59.32 | 59.67 |
| 5 | +Random+LAPT | 58.62 | 2.48 | 5 | - | 59.82 | 58.41 | 54.62 | 59.03 | 61.23 |

**LGSE - FOCUS = -0.57** (58.67 +/- 0.79 vs 59.23 +/- 1.98)

- Primary (LGSE > FOCUS): **NOT MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: XLM-R (default) > +FOCUS+LAPT > +LAPT > +LGSE+LAPT > +Random+LAPT

## amharic / tc (accuracy)

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLM-R (default) | 62.28 | 4.56 | 5 | - | 54.17 | 65.22 | 64.00 | 64.00 | 64.00 |
| 2 | +LAPT | 62.10 | 8.86 | 5 | - | 50.00 | 56.52 | 68.00 | 72.00 | 64.00 |
| 3 | +Random+LAPT | 62.07 | 8.63 | 5 | - | 54.17 | 52.17 | 72.00 | 68.00 | 64.00 |
| 4 | +FOCUS+LAPT | 61.17 | 11.68 | 5 | - | 50.00 | 47.83 | 72.00 | 72.00 | 64.00 |
| 5 | +LGSE+LAPT | 60.47 | 7.97 | 5 | - | 54.17 | 52.17 | 60.00 | 72.00 | 64.00 |

**LGSE - FOCUS = -0.70** (60.47 +/- 7.97 vs 61.17 +/- 11.68)

- Primary (LGSE > FOCUS): **NOT MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: XLM-R (default) > +LAPT > +Random+LAPT > +FOCUS+LAPT > +LGSE+LAPT

## tigrinya / ner (f1)

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLM-R (default) | 73.96 | 0.51 | 5 | - | 73.08 | 74.29 | 74.31 | 73.98 | 74.12 |
| 2 | +FOCUS+LAPT | 73.65 | 0.86 | 5 | - | 74.73 | 74.43 | 73.08 | 73.06 | 72.95 |
| 3 | +Random+LAPT | 73.49 | 1.12 | 5 | - | 74.03 | 74.54 | 71.66 | 73.98 | 73.24 |
| 4 | +LGSE+LAPT | 73.26 | 0.90 | 5 | - | 74.03 | 71.83 | 73.91 | 72.97 | 73.54 |
| 5 | +LAPT | 72.84 | 2.21 | 5 | - | 69.17 | 75.01 | 73.62 | 73.74 | 72.67 |

**LGSE - FOCUS = -0.39** (73.26 +/- 0.90 vs 73.65 +/- 0.86)

- Primary (LGSE > FOCUS): **NOT MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: XLM-R (default) > +FOCUS+LAPT > +Random+LAPT > +LGSE+LAPT > +LAPT

## tigrinya / qa (f1)

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLM-R (default) | 49.20 | 2.54 | 5 | - | 51.28 | 50.34 | 50.11 | 49.49 | 44.80 |
| 2 | +Random+LAPT | 48.41 | 2.42 | 5 | - | 49.30 | 50.18 | 50.50 | 44.66 | 47.40 |
| 3 | +LAPT | 47.77 | 2.03 | 5 | - | 49.67 | 48.67 | 45.02 | 49.26 | 46.24 |
| 4 | +LGSE+LAPT | 47.38 | 4.65 | 5 | - | 49.16 | 51.24 | 51.66 | 41.87 | 42.95 |
| 5 | +FOCUS+LAPT | 44.63 | 13.48 | 5 | - | 49.41 | 54.23 | 48.76 | 20.82 | 49.91 |

**LGSE - FOCUS = +2.75** (47.38 +/- 4.65 vs 44.63 +/- 13.48)

- Primary (LGSE > FOCUS): **MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: XLM-R (default) > +Random+LAPT > +LAPT > +LGSE+LAPT > +FOCUS+LAPT

## tigrinya / tc (accuracy)

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +Random+LAPT | 37.30 | 10.14 | 5 | - | 32.43 | 33.78 | 32.43 | 32.43 | 55.41 |
| 2 | XLM-R (default) | 37.03 | 5.78 | 5 | - | 43.24 | 31.08 | 37.84 | 41.89 | 31.08 |
| 3 | +LGSE+LAPT | 37.03 | 5.78 | 5 | - | 45.95 | 35.14 | 39.19 | 33.78 | 31.08 |
| 4 | +FOCUS+LAPT | 34.32 | 3.25 | 5 | - | 36.49 | 37.84 | 32.43 | 29.73 | 35.14 |
| 5 | +LAPT | 34.05 | 5.69 | 5 | - | 41.89 | 28.38 | 37.84 | 32.43 | 29.73 |

**LGSE - FOCUS = +2.70** (37.03 +/- 5.78 vs 34.32 +/- 3.25)

- Primary (LGSE > FOCUS): **MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: +Random+LAPT > XLM-R (default) > +LGSE+LAPT > +FOCUS+LAPT > +LAPT

## Overall

Primary criterion (LGSE > FOCUS) met on **2 of 6** tasks with both systems complete: tigrinya/qa, tigrinya/tc

Separated at +/-1 sd on 0 of 6: none

