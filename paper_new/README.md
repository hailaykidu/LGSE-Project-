# Replacement paper

`main.tex` — "Can Five Seeds Tell Embedding Initializers Apart? A Controlled
Study of Vocabulary Extension for Amharic". Replaces the retracted LREC 2026
paper and cites it.

`custom.bib` — bibliography.

## Building

Needs two ACL-distributed files, not committed here:

- `acl.sty`
- `acl_natbib.bst`

from <https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj>.
Then:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`\usepackage[review]{acl}` gives the anonymised review version. For ARR
submission that is the correct setting; switch to `\usepackage{acl}` for
camera-ready.

## Every number comes from one script

```
python3 scripts/build_paper_tables.py --out tables
```

| Paper element | Source |
|---|---|
| Table 1 (three-way means) | `tables/tables.txt`, "TABLE 2 primary" |
| Table 2 (paired differences, CIs) | `tables/tables.txt`, "TABLE 3" |
| Table 3 (spread vs seed variation) | `tables/tables.txt`, "Seed variation" |
| Table 4 (the six failed runs) | `tables/tables.txt`, "FAILURE RULE" |
| §5 detectable differences | paired sd from `tables/per_seed.csv` |
| §6 split comparison | 71-item figures from the superseded batch, 235-item from this protocol; both stated as such |
| §7 per-class collapse | `tables/tables.txt`, "TABLE 4" |
| Coverage 50.0% / 16.7% | `tables/coverage.json` |
| Appendix A checksums, recipe | `configs/hornmorpho.yaml`, run records |

Verify the tables against the result files:

```
python3 -m pytest tests/
```

## Scope

Amharic, three systems (LGSE, FOCUS, XLM-R), three tasks, 60 runs, one
protocol. `+LAPT` and `+Random+LAPT` were not run under this protocol and
are **not** filled from the earlier batch — §6 shows why that would create an
ordering artifact of about 8 points. Do not add those rows without re-running
them on the rebuilt splits.

No ranking is claimed anywhere. Every paired interval contains zero; if a
revision adds a ranking it must also add the evidence for one.
