# Reproduction paper

`main.tex` — the write-up of the September/October 2026 reproduction run.
`custom.bib` — bibliography.

## Building

This uses the ACL conference style. It needs two files from the official
template that are **not** committed here, because they are redistributed by
ACL rather than by this project:

- `acl.sty`
- `acl_natbib.bst`

Get them from
<https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj>
(or the `acl-org/acl-style-files` repository) and place them beside
`main.tex`. Then:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

`\usepackage[review]{acl}` produces the anonymised review version with line
numbers. Change to `\usepackage{acl}` for the camera-ready, which also
un-anonymises the author block.

## Where every number comes from

No figure in the paper is typed by hand. Each is computed from the per-seed
`experiment.json` records, and the mapping is:

| Paper element | Source |
|---|---|
| Table 1 (main matrix) | `results_gen2_snapshot_20260924/` — 150 runs, the frozen snapshot |
| Table 1, published rows | `report/TABLE2_REPORT.md`, transcribed from the LREC 2026 paper |
| Table 2 (controlled three-way, Amharic) | `results_hornmorpho_wam/{lgse_lapt,focus_lapt,xlmr}__{tc,ner,qa}__amharic__seed*` — NER uses seeds 42–51 for the two vocabulary-expanding systems |
| Table 3 (paired tests) | derived from the same records; paired `scipy.stats.ttest_rel` and `wilcoxon` on matched seeds where both systems converged |
| Table 4 (signal/noise) | derived from the frozen snapshot |
| §5.2 coverage, 16.7% / 50.0% | `IMPLEMENTATION_NOTES.md` §5b-ii |
| §5.3 split shift, +1.93..+8.21 | `IMPLEMENTATION_NOTES.md` §5a |
| §5.4 degenerate runs, 5/421 | scan of all `results*/*/experiment.json` |
| §7 arithmetic errors | published Table 2 Avg column vs `(ti+am)/2` |
| $W$ provenance (anchors, residuals) | `data/alignment/W_{am,ti}.json` |
| Split sizes | `data/*/manifest.json` |

To re-verify the tables against the result files:

```
python3 -m pytest tests/test_results_table_fidelity.py
```

## Scope

The paper reports the frozen snapshot as its primary matrix. The controlled
comparison in §6 covers **one cell** (Amharic TC); the remaining baselines on
the rebuilt split are not yet run, and the Limitations section says so. Do not
extend §6's conclusion to the other five cells without the corresponding runs.
