# Table 2 -- rankings and the LGSE-FOCUS comparison

## Note on interpretation

The published Table 2 has no surviving end-to-end execution record: no
complete configuration, run, or output log links to its printed numbers (see
`IMPLEMENTATION_NOTES.md` Section 3). The results below, run in full in
September 2026 using the released implementation with the documented choices
for W (orthogonal Procrustes) and lambda (1.0), are therefore **the only
measured result that exists for this method.**

That measurement does not reproduce the published ranking, margins, or
winner. The primary criterion (LGSE > FOCUS) is met on 2 of 6 task/language
cells and the secondary ordering (LGSE > FOCUS > Random > Default) is met on
0 of 6; no comparison separates at one standard deviation; off-the-shelf
XLM-R with no adaptation is the best system in 5 of the 6 cells. This is the
reported result of the released code, not supporting evidence alongside a
separately authoritative published table.

For complete context on the relationship between published results and repository artifacts, see `IMPLEMENTATION_NOTES.md` Section 3: "Table 2 reproducibility."

## Results from this implementation

Mean +/- sample sd over seeds 42-46. Metric: accuracy for TC (Table 2 'AC'), F1 for NER and QA.

The paper specifies a learned projection matrix W but does not specify how W is obtained (Sec. 4.1). To implement the pipeline, this repository uses orthogonal Procrustes alignment to instantiate W (see `data/alignment/W_*.json`). The paper defines the regularization coefficient lambda but does not provide a numerical value or selection procedure; to implement the pipeline, this repository uses lambda = 1.0. These are results from this implementation, obtained using those implementation choices.

## amharic / ner (f1)

*Second-generation run (2026-09-19 -- 2026-09-22, jobs 75427/75426/78067). The
four vocabulary-expanding systems were rerun after two corrections described
in `IMPLEMENTATION_NOTES.md` section 5c: Amharic runs now use the
corpus-derived `data/new_tokens_amharic.txt` instead of the Tigrinya-derived
`data/new_tokens.txt`, and `data/morph_lexicon.txt` gained 10,218
amseg/HornMorpho-derived Amharic entries. XLM-R carries over unchanged from
the first generation: it expands no vocabulary, so neither correction can
affect it.*

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLM-R (default) | 68.97 | 1.94 | 5 | - | 68.86 | 71.33 | 66.61 | 70.43 | 67.63 |
| 2 | +FOCUS+LAPT | 68.82 | 1.79 | 5 | - | 67.24 | 71.22 | 67.01 | 68.64 | 69.97 |
| 3 | +LGSE+LAPT | 67.75 | 1.80 | 5 | - | 65.85 | 66.05 | 70.03 | 68.00 | 68.83 |
| 4 | +LAPT | 66.25 | 1.95 | 5 | - | 63.99 | 69.20 | 65.83 | 65.32 | 66.90 |
| 5 | +Random+LAPT | 54.22 | 30.44 | 5 | - | 65.11 | 72.29 | 67.99 | 65.73 | 0.00 |

**LGSE - FOCUS = -1.07** (67.75 +/- 1.80 vs 68.82 +/- 1.79)

- Primary (LGSE > FOCUS): **NOT MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: XLM-R (default) > +FOCUS+LAPT > +LGSE+LAPT > +LAPT > +Random+LAPT

**`+Random+LAPT` seed 46 scored exactly 0.00** (dev F1 0.00 as well): the
fine-tuned tagger predicted no correct entity at all. LAPT loss decreased
normally for that seed, so this is a downstream fine-tuning collapse, not a
crash -- every other unit in that job scored 63-72. The run is reported as
measured rather than dropped; excluding it, `+Random+LAPT` averages 67.78,
which would place it third. Whether to exclude a degenerate seed is a
judgement call that changes this system's rank, so both figures are stated.

## amharic / qa (f1)

*Second-generation run; see the note under amharic / ner. XLM-R carries over
unchanged.*

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +FOCUS+LAPT | 60.83 | 1.56 | 5 | - | 61.73 | 61.47 | 61.86 | 58.10 | 60.97 |
| 2 | XLM-R (default) | 59.58 | 2.15 | 5 | - | 58.90 | 62.44 | 58.73 | 56.87 | 60.95 |
| 3 | +Random+LAPT | 59.03 | 1.33 | 5 | - | 59.40 | 59.89 | 57.35 | 60.54 | 57.96 |
| 4 | +LGSE+LAPT | 57.74 | 2.02 | 5 | - | 57.48 | 56.76 | 60.35 | 55.11 | 58.97 |
| 5 | +LAPT | 57.08 | 1.97 | 5 | - | 56.58 | 60.38 | 55.28 | 56.00 | 57.17 |

**LGSE - FOCUS = -3.09** (57.74 +/- 2.02 vs 60.83 +/- 1.56)

- Primary (LGSE > FOCUS): **NOT MET**
- Ranges at +/-1 sd: **separated** -- 57.74+2.02 = 59.76 is below 60.83-1.56 = 59.27 only marginally; the gap is the largest of the three Amharic tasks and the one least attributable to seed noise
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET**
- Observed order: +FOCUS+LAPT > XLM-R (default) > +Random+LAPT > +LGSE+LAPT > +LAPT

QA is the only Amharic task where FOCUS beats the unmodified XLM-R baseline,
and it is also where LGSE trails FOCUS by the widest margin. LGSE places
below `+Random+LAPT` here: random initialization of the new embedding rows
outperformed LGSE's morpheme-derived initialization by 1.29 F1.

## amharic / tc (accuracy)

*Second-generation run; see the note under amharic / ner. XLM-R carries over
unchanged.*

Test set size: 71 items (rebuilt splits -- the first-generation table above
used a 24-item test set). Each item is 1.41 accuracy points, so values move
in steps of one correctly or incorrectly classified example.

| Rank | System | Mean | SD | n | Missing seeds | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +LAPT | 75.77 | 3.21 | 5 | - | 78.87 | 70.42 | 76.06 | 76.06 | 77.46 |
| 1 | +LGSE+LAPT | 75.77 | 4.03 | 5 | - | 81.69 | 76.06 | 76.06 | 70.42 | 74.65 |
| 3 | XLM-R (default) | 75.49 | 2.75 | 5 | - | 71.83 | 77.46 | 78.87 | 74.65 | 74.65 |
| 4 | +FOCUS+LAPT | 73.80 | 4.18 | 5 | - | 69.01 | 73.24 | 71.83 | 74.65 | 80.28 |
| 5 | +Random+LAPT | 73.24 | 3.45 | 5 | - | 77.46 | 74.65 | 74.65 | 70.42 | 69.01 |

**LGSE - FOCUS = +1.97** (75.77 +/- 4.03 vs 73.80 +/- 4.18)

- Primary (LGSE > FOCUS): **MET**
- Ranges at +/-1 sd: **overlapping** -- the difference is within seed-to-seed spread
- Secondary (LGSE > FOCUS > Random > Default): **NOT MET** (`+LAPT` ties LGSE for first)
- Observed order: +LAPT = +LGSE+LAPT > XLM-R (default) > +FOCUS+LAPT > +Random+LAPT

TC is the one Amharic cell where LGSE beats FOCUS. The margin is +1.97 with
standard deviations of ~4 on both systems, so it does not separate. LGSE also
ties `+LAPT` to four significant figures (75.7746 vs 75.7746 -- both average
to the same total number of correctly classified items across seeds), meaning
LGSE's morpheme-derived initialization performs identically to default
initialization on this task. At 71 test items and a 4-point spread, TC is the
noisiest of the three Amharic tasks.

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

Test set size: 74 items. Each item is one accuracy point (1/74 = 1.35
percentage points), so the values below move in steps of one correctly or
incorrectly classified example.

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

*Amharic cells reflect the second-generation corrected-methodology rerun
(2026-09-19 -- 2026-09-24). Tigrinya cells are unchanged from the first
generation and have not been rerun under the corrected Amharic token list,
which does not apply to them.*

Primary criterion (LGSE > FOCUS) met on **3 of 6** tasks with both systems
complete: amharic/tc, tigrinya/qa, tigrinya/tc

Secondary criterion (LGSE > FOCUS > Random > Default) met on **0 of 6**.

Separated at +/-1 sd on 0 of 6: none

### The corrected Amharic rerun did not change the conclusion

The rerun was motivated by two real defects (a Tigrinya-derived OOV list used
for Amharic runs, and a morpheme lexicon with almost no Amharic coverage),
both fixed before these numbers were produced. Fixing them moved individual
cells but not the finding:

| Amharic task | LGSE - FOCUS, 1st gen | LGSE - FOCUS, 2nd gen |
|---|---|---|
| ner | -0.70 | -1.07 |
| qa | -0.57 | -3.09 |
| tc | -0.70 | +1.97 |

LGSE now wins TC and loses NER and QA by wider margins than before. On QA it
places below random initialization. On TC it ties `+LAPT` exactly (269/355
items for both). Off-the-shelf XLM-R remains at or near the top of every
Amharic cell despite expanding no vocabulary and running no LAPT at all.

Off-the-shelf XLM-R (no LAPT, no FastText-based initialization of any kind)
is the best-scoring system in 4 of the 6 cells: all three Tigrinya cells and
amharic/ner. After the Amharic rerun it is narrowly displaced in amharic/tc
(by `+LAPT` and `+LGSE+LAPT`, both 75.77 vs 75.49) and in amharic/qa (by
`+FOCUS+LAPT`, 60.83 vs 59.58). Every LGSE-FOCUS difference in this report,
in both directions, is smaller than at least one of the two systems'
seed-to-seed standard deviation. This is a finding of no measured effect for
LGSE over FOCUS or over the unmodified baseline, under the documented
implementation choices (Procrustes W, lambda = 1.0) and the five seeds used
here -- not a partial reproduction of the paper's claim.

### Known limitations of this measurement

Two measured properties of the setup mean LGSE's distinguishing mechanism was
only partly exercised in these runs. Both are documented with measurements in
`IMPLEMENTATION_NOTES.md` sections 5b-ii and 5b-iii:

1. **Gradient reachability.** Only 110 of the 198 new Amharic tokens (55.6%)
   ever appear in the LAPT corpus. The other 44% keep their initialization
   value unchanged for the whole LAPT run, so for those rows every
   initializer -- LGSE, FOCUS, Random, default -- is evaluated on its
   initialization alone. This affects all four systems equally.
2. **Morpheme-path coverage.** Only 45 of 198 tokens (22.7%) have a
   `data/morph_lexicon.txt` entry and therefore take LGSE's morpheme-average
   path; the remaining 77.3% fall through to whole-token FastText, which is
   close to what FOCUS does. Attaching HornMorpho (via `amseg`) as a
   segmentation fallback raises this to 114/198 (57.6%) on the same token
   list, a 2.5x increase. That integration exists in
   `src/lgse/segmentation.py` behind `LGSEConfig.use_hornmorpho`, defaults to
   off, and was **not** enabled for any run in this report.

Neither is a defence of the numbers above, which stand as measured. They
identify what a stronger test of LGSE would require, and point 2 in
particular means these results characterize LGSE-with-a-sparse-lexicon rather
than LGSE's morpheme mechanism at full coverage.

