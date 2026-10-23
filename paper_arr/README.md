# paper_arr — ACL Rolling Review submission

*Below the Seed-Noise Floor: A Controlled Comparison of Embedding
Initializers for Amharic and Tigrinya Vocabulary Extension*

An anonymous ARR submission. Standalone: it reports the five-system
comparison on its own terms and does not narrate the history of the project.

## Building

`acl.sty` and `acl_natbib.bst` are ACL-distributed and are not kept here.
Fetch them from [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files),
place them beside `main.tex`, then:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`\usepackage[review]{acl}` gives the anonymised review version; switch to
`\usepackage{acl}` for camera-ready.

Note: the preamble includes `inconsolata`, which is standard in TeX Live and
on Overleaf but absent from the cluster used for development. A local build
there needs a stub providing `\ProvidesPackage{inconsolata}`.

## Page budget

Countable body (Introduction through Conclusion) is 8 pages, at the ACL long
limit. Limitations, Ethics, references and the two appendices follow and are
excluded from the count. Any addition therefore requires a cut; the §6 floor
table is the largest movable block.

## Where the numbers come from

Every figure in the paper is generated from the 165 released per-run records,
not transcribed:

```bash
python3 ../scripts/build_table2.py --out ../tables
```

Table 1's thirty cells and the floor table's seventy-two values were each
checked against that output programmatically. Degenerate-run exclusions are
the six Amharic runs identified from training-loss reduction alone
(`scripts/build_paper_tables.py`, `failure_analysis()`).

Two measurements in §8 are computed outside that script and are reproducible
from the saved LAPT checkpoints:

- new-token share of task tokens, from the tokenizers and the task splits
- initialization-to-post-LAPT cosine, by reconstructing the deterministic
  LGSE initialization from `data/alignment/W_ti.npy`, the morpheme lexicon
  and the FastText model, then comparing against the checkpoint

## Relationship to the other paper directories

- `paper_arr/` — this submission, current
- `paper_final/` — earlier full-disclosure write-up, superseded
- `paper_new/`, `paper/` — earlier drafts, superseded and containing figures
  since corrected; do not cite from them
