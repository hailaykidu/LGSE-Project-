# Implementation notes: what the paper specifies, and what it does not

This repository implements the LGSE method described in the paper. This
document records every aspect of the implementation that cannot be derived
directly from the paper, where a value, an artifact, or a procedure is
required for implementation but is not specified in the published method.

Each entry distinguishes between what the paper explicitly specifies and what
is implemented in this repository. Where the paper leaves an implementation
detail unspecified, the repository documents the corresponding requirement or
assumption without presenting it as part of the published method. The purpose
is to provide the information required to reproduce the implementation
faithfully and to identify which components originate from the paper and which
must be supplied externally.

> ## Fidelity prerequisites
>
> Sections 4.1 and 4.2 introduce two components of the method — the alignment
> matrix **W** and the regularization coefficient **λ**. This section records
> how each is specified in the LGSE implementation.
>
> | Component | What the paper specifies | What this implementation uses |
> |---|---|---|
> | Alignment matrix W | Section 4.1 specifies a learned projection matrix W used to align the FastText and pretrained model embedding spaces. It does not specify how W is obtained. | To implement the pipeline, we use orthogonal Procrustes alignment to instantiate W over shared FastText/XLM-R vocabulary anchors. |
> | Regularization coefficient λ | Section 4.2 defines λ in the regularization term `L_reg = λ‖e_new − μ‖²`. It does not provide a numerical value or a selection procedure. | To implement the pipeline, we use `reg_lambda = 1.0`. |
>
> **Projection matrix (W):** The paper specifies a learned projection matrix
> (W) but does not specify how (W) is obtained. To implement the pipeline, we
> use orthogonal Procrustes alignment to instantiate (W). The alignment
> metadata — anchor count, embedding dimensions, objective used, and random
> seed — is recorded alongside each matrix.
>
> **Regularization coefficient (λ):** The paper defines the regularization
> coefficient (λ) but does not provide a numerical value or selection
> procedure. To implement the pipeline, we use (λ = 1.0). The value is
> recorded in the experiment metadata written by each run.
>
> A third prerequisite concerns the input data rather than the configuration.
> Since Section 4.1 defines **W** as a square projection, the FastText vectors
> must have the same embedding dimension as the pretrained model (768 for
> `xlm-roberta-base`) (§1).
>
> ### Standing policy
>
> **No silent fallback may be added for an unspecified methodological
> component.** Where the paper does not specify something, this repository
> fails and says so; it does not infer, reconstruct, or default. That
> applies to future changes as much as to the current state.
> There is no third category, and no result should be presented as faithful without the artifacts above.
>
> | Invariant | Test |
> |---|---|
> | No entry point defaults W | `test_alignment_matrix_is_required_by_every_entry_point` |
> | No W is committed to the shipped config | `test_shipped_config_leaves_the_matrix_unset` |
> | λ has no default | `test_reg_lambda_is_mandatory` |
> | No dimension is silently reshaped | `test_no_silent_reshaping_of_mismatched_vectors` |
> | No unstated loss term trains W | `test_paper_objectives_give_w_no_gradient` |
> | Unfaithful runs are flagged in the table | `tests/test_results_table_fidelity.py` |
> | Baselines are *not* falsely flagged | `test_baselines_are_not_flagged_as_unfaithful` |

## 1. Projection W is square, and externally supplied

**Paper (Sec 4.1):** "To align the FastText embedding space with the
pretrained model embedding space, a learned linear projection
`W ∈ R^{d×d}` is applied, i.e. `e_aligned_t = W e_t`."

Two consequences, both implemented:

**W is square.** The paper aligns two spaces of the same dimension `d`; it
describes no dimensionality change. In practice this means FastText must be
trained at the model's embedding width (768 for xlm-roberta-base), not the
standard 300 — `configs/base.yaml` sets `fasttext_dim: 768` accordingly.

**The 300-dim CC vectors referenced under `data.fasttext` cannot be used
with a square W as specified in Sec 4.1. This is a data prerequisite, not
something the code can resolve**, so the implementation refuses rather than
adapting:

| Not done | Why |
|---|---|
| Rectangular `W ∈ R^(300×768)` | Not the published method; W is `d×d` |
| Truncating 768→300 or padding 300→768 | Silently discards or fabricates dimensions |
| PCA/SVD reduction of the model's space | Changes what the embeddings mean |
| Falling back to char n-grams on mismatch | Reports as LGSE while bypassing FastText entirely |

Each of these would produce numbers that look like results. None would be
the published method, and the substitution would not be visible in any
metric — which is why the code raises instead of choosing one.

Enforcement is layered so the failure comes as early as possible:

1. **`data/scripts/download_fasttext.py`** — checks at download time
   (`--expect-dim`, default 768), prints a prominent warning naming the fix,
   records `dimension_ok` in the manifest, and exits non-zero.
2. **`LGSEConfig.fasttext_path`** — reads the manifest's recorded dimension
   and raises before a multi-GB model is loaded.
3. **`build_projection` / `MorphemeEmbeddingBuilder`** — the authoritative
   check, via `check_dimensions`. Both share one implementation so they
   cannot disagree.

All three raise `IncompatibleFastTextDimension`, a distinct exception type,
with a message naming both dimensions, the reason, the required width, and
the `fasttext -dim 768` command that produces it.

**W has no default.** It is supplied by the author via
`lgse.alignment_matrix_path`; a run without one fails. See §1a-i for why no
default — including the identity — is substituted.

**W is the only supported projection.** `src/lgse/projection.py` defines
`AlignmentProjection` as a square `d×d` matrix, externally supplied and
frozen. There is no alternative projection path and no `projection:` config
key. Rectangular mappings between the two widths are not implemented, since
Sec 4.1 specifies `W ∈ R^{d×d}`.

### 1a. Under the paper's stated objectives, W receives no gradient

The paper calls W "learned" (Sec 4.1; and in the systems list, "aligned via
a learned projection layer"). But no objective it states is a function of W:

- **Initialization** (Sec 4.2) sets `e_new` to the average of projected
  morpheme embeddings. In code this value is written into the embedding
  matrix via `.data` (`src/lgse/initializer.py:60-63`), which severs the
  autograd graph. That in-place write is deliberate — replacing the
  `Parameter` would break weight tying with the MLM head.

- **The regularizer** (Sec 4.2) is `L_reg = λ‖e_new − μ‖²`, "where μ is the
  **initial** embedding vector". μ is a constant; the term measures drift
  from initialization and trains `e_new`, not W.

- **LAPT** (Sec 5) applies MLM with the encoder frozen, updating "only the
  new embeddings". It reads the embedding matrix, not W.

So W is initialized, its output is copied into the embeddings, and it is
never differentiated again. Verified empirically:

```
W is square      : torch.Size([768, 768])
identity init    : True
emb grad         : True
W grad under paper formulation: None
```

**Status: unspecified in paper; instantiated by this implementation.**

The paper specifies the existence and role of W, but does not specify its
construction. This implementation therefore chooses orthogonal Procrustes
alignment to instantiate W so that the pipeline can be executed.

### 1a-i. Implementation assumptions

W is therefore implemented as an **externally supplied alignment matrix**,
under three assumptions recorded here as assumptions, not findings:

| # | Assumption | Consequence |
|---|---|---|
| 1 | W is an externally supplied artifact | `alignment_matrix_path` loads a `d×d` matrix from `.pt`/`.npy`; **no default of any kind** |
| 2 | W is frozen | `requires_grad=False`, excluded from the optimizer |
| 3 | No objective trains W | No loss term is invented to give it a gradient |
| 4 | `reg_lambda` must be explicitly provided | No default; see §8a |
| 5 | W is not specified by the paper, so any W used here is **an implementation choice**, recorded as such | Runs fail rather than substitute one silently |

**There is no default W — not even the identity.**

The identity matrix is not a neutral choice. It assumes that the FastText and
model embedding spaces are already aligned, whereas Section 4.1 introduces W
specifically to learn this alignment. Replacing the unspecified transformation
with the identity therefore introduces an implementation decision that can
materially affect the results, rather than preserving fidelity to the
described method.

Every candidate default fails the same test:

| Candidate | Why it is not used |
|---|---|
| Identity | Asserts the two spaces are already aligned — the claim W exists to avoid |
| Random / seeded | An alignment strategy the paper does not describe |
| Fitted (Procrustes, CCA, …) | A method the paper does not describe, requiring anchor data it never mentions |
| A rectangular Johnson–Lindenstrauss map | Rectangular, where Sec 4.1 specifies `d×d` |

So the implementation refuses. `build_projection` raises
`MissingAlignmentMatrix` — a distinct type, because this marks a genuinely
unspecified part of the method rather than a misconfiguration to patch. The
message quotes Sec 4.1, states that the paper does not specify how W is
obtained, explains why neither the identity nor a random matrix is
substituted, and records that any W used here is an implementation choice
introduced to make the pipeline executable.

A *supplied* identity remains perfectly legitimate — it is then the author's
documented choice, recorded as such, not this project's silent one. The
trainer flags it in the log when it occurs.

**Why W is frozen rather than trainable-but-unused.** Placing W in the
optimizer would advertise a capability the run does not have: nothing in the
paper's stated objectives differentiates W, so a reader inspecting the
parameter groups would otherwise conclude W was being learned. W therefore
carries `requires_grad=False` with no switch to change it: `AlignmentProjection`
takes no `trainable` argument, and the optimizer contains the embedding
matrix alone.

**What is *not* done.** No loss term is invented to give W a gradient.
Candidates exist — a live regularizer anchor, a reconstruction loss, an
alignment loss against anchor translations — and any of them would make W
train and produce numbers. None is in the paper, so none is implemented, and
none is kept as a dormant alternative. Making the regularizer anchor a live
function of W would contradict Sec 4.2's statement that "μ is the initial
embedding vector," so `LGSERegularizer` takes a fixed anchor tensor and
nothing else.

### 1a-ii. How the status is surfaced

The gap is enforced before a run starts and reported wherever a result could
be read:

- **Before the run:** `run_experiment.py` refuses to start any
  FastText-using system (`lgse`, `focus`) when `lgse.alignment_matrix_path`
  is unset — checked before the backbone and the multi-GB FastText model are
  loaded. `build_projection` is the authoritative check.
- **Per run:** `[LGSELAPTrainer] W training status: author-required /
  unspecified in paper -- W is frozen`. A supplied identity is additionally
  flagged.
- **Per checkpoint:** `projection.pt` carries `source`, `trainable` and
  `training_status`; `projection_status.json` sits beside it.
- **Per result:** the run record's `projection` field records `source`,
  `author_supplied`, `training_status`, and `trained_during_this_run: false`.
- **In the generated table:** `scripts/aggregate_results.py` emits a notice
  naming any system whose runs lacked an author-supplied W, *and* an
  "Alignment matrix W" column per row — `author-supplied W`, `not faithful`,
  `MIXED` when seeds within one system disagree, or `n/a` for the baselines,
  which use no FastText and need no W. A reader gets the fidelity status
  from the table alone, without opening this file.
- **In tests:** `test_paper_objectives_give_w_no_gradient` fails if an
  unstated loss term is added; `test_alignment_matrix_is_required_by_every_entry_point`
  fails if any entry point introduces a default; `test_shipped_config_leaves_the_matrix_unset`
  fails if a W is ever committed to `configs/base.yaml`.

**To resolve this, the authors need to state which objective trains W.**
Until then, "learned" describes W's declared type in Sec 4.1, not its
observed behaviour under the published equations.

### 1b. W is part of the checkpoint

`save_pretrained()` covers only the model, so `LGSELAPTrainer.save()` writes
`projection.pt` alongside it and `load_projection()` restores it, refusing a
shape mismatch. This keeps a run self-describing: the exact alignment matrix
a result used travels with that result. Without it, a checkpoint made with
an author-supplied W would silently reload as the identity — a different run
from the one that produced the numbers. It also keeps the checkpoint correct
if W is ever trained under an author-supplied objective, where it could not
be recomputed at all.

## 2. Comparison baselines

Table 2 compares five systems, and the paper describes them in prose rather
than specifying their implementation.

`src/baselines/strategies.py` provides `default` (+LAPT),
`random` (+Random+LAPT) and `focus` (+FOCUS+LAPT), each exposing the same
interface as `LGSEInitializer` so the trainer swaps between them with no
other change.

FOCUS (Dobler & de Melo, 2023) is reimplemented here as a similarity-weighted
combination of pretrained embeddings using the FastText space as the
auxiliary signal — the same external signal LGSE uses, so the two differ
only in how they use it. **This is a reimplementation, not the authors'
code**, and has not been validated against their published results.

## 3. Table 2 reproducibility

Table 2 reports QA, NER and text classification. The evaluation harness is
in `src/evaluation/` (entity-level NER F1, SQuAD QA F1, mean/stdev over
seeds).

**Status:** The full Table 2 sweep has now been completed, in September 2026,
across five systems, three tasks, two languages, and five seeds. The results
are in `report/TABLE2_REPORT.md` and `report/per_seed_results.json`.

**Relationship to the published paper:**

There is no single surviving end-to-end execution record that links every
published Table 2 number to a complete configuration, run, and output.
Individual components of the original experiment survive in this repository
-- data, implementation code, configurations -- but no run log, checkpoint,
or environment record connects them to the printed numbers. The published
Table 2 values therefore have no reproducible provenance in this repository.

**The September 2026 sweep is the only measured result for this method that
exists.** It is not supporting evidence alongside a separately authoritative
published table; in the absence of any surviving record for the published
numbers, it is the reported result. It does not reproduce the paper's
ranking, margins, or winner:

- Primary criterion (LGSE > FOCUS): met on 2 of 6 task/language cells.
- Secondary criterion (LGSE > FOCUS > Random > Default): met on 0 of 6.
- Separated at +/-1 sd: 0 of 6 -- every LGSE-FOCUS difference is inside
  seed-to-seed spread.
- Off-the-shelf XLM-R with no adaptation is the best system in 5 of 6 cells.
- Tigrinya QA: 47.4 (LGSE) vs 49.2 (baseline) here, against 78.0 vs 61.3 in
  the paper. Tigrinya TC: 37.0 vs 37.0 here (see the item-count note in
  `report/TABLE2_REPORT.md` -- the Tigrinya TC test set is 74 items and the
  Amharic TC test set is 24, so these accuracies move in one-item steps),
  against 75.2 vs 63.2 in the paper.

This repository brings the surviving components together, makes the
currently executable implementation explicit, and reports what running it
produces, rather than asserting that it recovers the published numbers.

**Underspecified components and their resolution:**

The paper specifies a learned projection matrix W (Sec. 4.1) but does not
explicitly describe the procedure used to instantiate it, and it defines the
regularization coefficient λ (Sec. 4.2) without giving a numerical value.
Where the paper does not specify sufficient implementation detail to
reconstruct an implementation exactly, this repository documents the choices
used by its executable implementation. These are **repository implementation
choices**, not newly recovered historical values unless supported by
surviving primary evidence.

In particular:

| Component | Paper specifies | This repository implements |
|---|---|---|
| Alignment matrix W | Learned projection; no instantiation procedure given | Orthogonal Procrustes alignment over shared FastText/XLM-R vocabulary anchors |
| Regularization coefficient λ | Symbol defined in L_reg = λ‖e_new − μ‖²; no numerical value provided | λ = 1.0 |

These choices are documented in IMPLEMENTATION_NOTES.md (this file) and
implemented in the corresponding evaluation pipeline in `src/evaluation/`.

**Interpretation of repository results:**

The repository's results are the measured output of the released
implementation, run under the documented settings (W via orthogonal
Procrustes, λ=1.0). They are not a partial or noisy echo of an
already-established published result to be reconciled in the published
value's favor: no published value has a surviving record to defer to. Where
a repository result differs from a published Table 2 value -- and on every
task/language cell it does, often by tens of points -- that difference is not
resolved by treating the published number as authoritative. It is reported as
what it is: the released implementation, run in full, does not produce the
paper's ranking, margins, or winner.

The repository does not invent missing historical configuration or claim exact
historical reconstruction where the surviving record does not permit a published
number to be traced through one complete historical execution.

**Reproducibility claim:** The resulting Table 2 values constitute the
reproducible output of the released implementation under the explicitly
documented settings. We do not claim that these choices were numerically
specified in the paper itself; rather, we make the previously implicit or
underspecified implementation details explicit so that the experiment can be
deterministically reproduced from this repository's code and configuration.
This is the reported result for the released method, not a supporting
artifact subordinate to the published table.

Any future reproduction or replication should either:
1. Use these documented settings (W via Procrustes, λ=1.0) to obtain
   consistent results with this repository's implementation, or
2. Obtain author-supplied values from the original authors' materials and
   document any differences that result from alternative choices.

## 4. FastText model acquisition

FastText binaries are 2–3 GB and are not committed.
`data/scripts/download_fasttext.py` fetches them on demand:

| | Source | Dim | Vocab |
|---|---|---|---|
| Amharic | `cc.am.300.bin` (Grave et al., 2018) | 300 | — |
| Tigrinya | `Hailay/fasttext-tigrinya` | 300 | 156,687 |

The script loads each model and refuses to continue if it is empty; it never
substitutes random vectors, because a placeholder would silently reduce LGSE
to its own character-n-gram fallback while still reporting as LGSE.
Dimensions, vocabulary size and sha256 are recorded in
`data/fasttext_manifest.json`. The binaries themselves are gitignored.

## 5. TIGQA is abstractive, not extractive

TIGQA (Zenodo 11423987, CC-BY-4.0) is released as a `.docx` table — columns
R/no, Grade level, Topic, Context, Question, Answer — with several numbered
question-answer pairs packed into single cells. Parsing yields 107 context
rows and ~120 QA pairs.

Extractive QA requires each answer to be a span of its context, identified by
a character offset. **107 of the 120 TIGQA answers do not occur in their
context**, even after normalising whitespace: they are rewritten rather than
copied. Only 13 pairs are usable for extractive QA.

`data/scripts/prepare_qa.py` drops unmatched pairs and records the count in
the manifest rather than fabricating offsets. Consequences:

* An extractive QA model can be evaluated on 13 Tigrinya test items at most,
  which is far too few for a stable F1, let alone a standard deviation over
  five seeds.
* The paper reports F1 on "TIGQA train-dev-test splits". Those splits are not
  in the Zenodo release, and the release as published does not support the
  extractive setup that F1-over-spans implies.

A first SQuAD-format conversion of TIGQA (`TIGQA_squad_format.json`, version
TIGQA-1.0) addressed this by supplying 2,108 QA pairs over 433 contexts with
`answer_start` offsets. Of those 2,108, only 797 had a span that actually
resolved in its context (1,039 carried `answer_start == -1`, an abstractive
rewrite; 272 had no answer given). `prepare_qa.py --dataset tigqa_squad`
originally ran against this file and kept those 797 pairs, split 644/67/86
over 292/36/37 contexts — the numbers the committed Table 2 QA run used.

**Superseded 2026-09-18 by a second, later conversion.** A newer file,
`tigqa_extractive_squad.json` (version `TIGQA-extractive-1.0`, provided by
the user, saved to `data/qa/tigqa_squad/tigqa_extractive_squad.json`), covers
the same 107 TIGQA context rows but records *how* each answer's
`answer_start` was obtained, per-answer, via a `match_status` field:
`exact` (verbatim substring), `fuzzy` (nearest-substring alignment — not
necessarily the *correct* substring), `unmatched`/`no_answer_chunk` (no
usable span; `is_impossible: true`, original text kept in `raw_answer`).

Verified directly against the file (2,179 total QA pairs): 1,517 `exact`,
148 `fuzzy`, 498 `unmatched`, 16 `no_answer_chunk`. Every `exact` and every
`fuzzy` span was checked to resolve correctly at its `answer_start` offset
(0 mismatches in both groups) — but several `fuzzy` spans, despite
resolving, are verifiably *wrong* excerpts: e.g. one `fuzzy` answer text is
`ን ዳይ ኦክሳይድ`, a truncated fragment of the actually-annotated answer
`ካርዶንዳይኦክሳይድ` (`match_ratio` 0.727) — the alignment found the nearest
substring, not the right one. `fuzzy` is therefore excluded, not merely
deprioritized: `prepare_qa.py --dataset tigqa_extractive` keeps only
`match_status == "exact"`, giving **1,517 extractive pairs — roughly double
the 797 from the prior conversion — with zero risk of a corrupted span**.

`data/qa/tigqa_squad/{train,dev,test,manifest}.json` have been regenerated
from this source and now hold 1,517 pairs (1189/177/151 over 85/10/12
contexts, seed 42), replacing the prior 797-pair split in place. The old
`TIGQA_squad_format.json`-derived numbers above are kept in this note as
history, not as the current state — any Tigrinya QA result in
`report/TABLE2_REPORT.md` predates this replacement and has not yet been
rerun against it.

Two caveats remain for strict reproduction:

* These splits are derived here with seed 42, not taken from an official
  release. The paper refers to "TIGQA train-dev-test splits"; the split files
  in `hailaykidu/TigQA-Dataset` are not usable as published — `dev.json` and
  `test.json` contain malformed JSON, and in `train.json` all 37 answer
  offsets are relative to the source document rather than the merged
  paragraph, so none resolve.
* Tigrinya QA is now scored on 151 test items (was 86) against Amharic's 299
  (AmQA's official test split, `data/qa/amqa/manifest.json`).

## 5a. Text classification: the paper's 2,500-sample figure does not match either language's data as held here

The paper states (Sec 7): *"we introduce a new benchmark dataset comprising
2,500 human-annotated samples in Amharic and Tigrinya... split into 80% for
training, 10% for development, and 10% for testing."* This describes one
combined bilingual dataset of 2,500 samples. What this repository holds for
the two languages does not add up to that figure, and does not appear to be
that dataset:

| | Source | Documents | Split recorded |
|---|---|---|---|
| Tigrinya TC | `data-is-better-together/fineweb-c` (config `tir_Ethi`) | 745 | 597/74/74, stratified, seed 42 |
| Amharic TC | `data/Dataset_Educational_classifier_annotated.tsv`, self-annotated, independent of FineWeb-C | 700 (kept after cleaning; see below) | 558/71/71, stratified, seed 42 |

700 + 745 = 1,445, not 2,500, and the two halves still come from unrelated
sources — one is a FineWeb-C config, the other a separately annotated TSV
with no connection to FineWeb-C.

**Resolved 2026-09-24.** The Amharic annotation export was superseded by the
complete set, `data/merged_dataset_first_2500_rows.csv` (2,499 rows, sha256
`64dca1207da4907a…`), and the 2,500 figure refers to the **Amharic** dataset
alone, not a combined bilingual one. Verified rather than assumed:

* The new file shares **663 of the previous 692** Amharic texts (95.8%), with
  97.7% label agreement on the overlap (19 relabelled) — it is an expansion of
  the same annotation effort, not a different corpus.
* It shares **0 of 745** Tigrinya TC texts, and **0 of 2,329** documents carry
  three or more Tigrinya-specific orthographic markers (`ኸ`, `ዅ`, `ዂ`, the
  `ኣይ-` negation). It is monolingual Amharic.

After cleaning (2 spreadsheet-artifact rows — one `=AI(...)` formula and one
`#ERROR!` cell — and 157 exact `(text, label)` duplicates), 2,340 documents
remain, split **1,872 / 233 / 235** by the same `stratified_split()` used for
Tigrinya: stratified by label, duplicate texts grouped, seed 42. Both
languages' ratios are 0.80/0.10/0.10 and no text crosses splits.

Two properties of the expanded set affect how its accuracy should be read:

* **Label 2 is 65% of the corpus** (1,618/2,499). A majority-class baseline
  scores ~66% accuracy, so TC accuracy near that figure indicates little
  beyond class prior.
* **Label 5 has only 2 instances**, both retained in train by the split guard
  (a class with fewer than one example remaining after dev/test allocation is
  not split). Dev and test therefore contain no label-5 examples, and the
  top of the ordinal scale is untested.

The previous 558/71/71 split and every Table 2 TC number measured on it are
preserved in `results_gen2_snapshot_20260924/`. The new split has a test set
of 235 items against the previous 71, so TC results measured on the two are
not comparable and the TC column requires rerunning before it can be read
alongside the earlier one.

**Amharic TC source replaced 2026-09-18.** The Amharic file was replaced with
a larger raw annotation export, `data/Dataset_Educational_classifier_annotated.tsv`
(829 parseable data rows), superseding the earlier 236-document
`Educational classifier.csv`. Cleaning before splitting, applied in order:

* 1 row dropped: a leaked spreadsheet formula string
  (`=AI("Fill an appropriate value for this cell based on the table
  context")`) rated `6`, not real annotated content.
* 7 rows dropped: English meta-commentary strings (e.g. `"Factual news:
  joint agency team visits drought-affected zone — single event report."`)
  with no tab separating text from label — not well-formed data rows.
* 121 rows dropped as exact `(text, label)` duplicates. The raw file
  repeats two large blocks verbatim later in the file (a political essay
  beginning "የሰው ልጅ ኑሮ..." and a rural-energy-technology grant application
  form); these are source-file duplication, not independent documents.
* 700 rows kept. 8 short menu/navigation-label texts (e.g. `መደበኛ ተቋም`,
  `ማስታወቂያ`) recur with *different* human-assigned labels across
  occurrences — a genuine annotator inconsistency, not a formatting issue —
  and are kept as separate rows rather than resolved to one label, since
  there is no principled basis to prefer one annotation over another.

The split (`data/tc/amharic/{train,dev,test}.csv`) uses the identical
stratified-by-label, duplicate-texts-grouped, seed-42 method as
`data/scripts/prepare_tc.py`'s Tigrinya split, so the two languages'
train/dev/test files are constructed the same way. Full detail in
`data/tc/amharic/manifest.json`.

This was investigated once already, in-repo: an earlier commit
(`1643bc3`, 2026-09-07) recorded a 2,500 → 745 original-pool-to-processed
figure for *both* languages, on the assumption that Amharic used a FineWeb-C
`amh_Ethi` config mirroring Tigrinya's `tir_Ethi`. Five minutes later, commit
`a9318c7` corrected this after checking FineWeb-C's config list directly (127
configs total): no `amh_Ethi` config exists. The Amharic data is not a
FineWeb-C split at all, and the 2,500-sample original-pool figure was never
verified for either language independently of that mistaken assumption — it
appears once, in the paper, and nowhere in this repository's data pipeline
for either language.

As of 2026-09-18, Amharic TC now has a recorded, seed-42, stratified
train/dev/test split matching Tigrinya's construction (see above). This
supersedes the earlier state, in which no split was recorded at all. Any
Amharic TC results in `report/TABLE2_REPORT.md` measured before this date
were produced against the smaller, unsplit 236-document file and are not
comparable to a rerun against the new 700-document, 558/71/71 split without
rerunning the sweep.

**Status: unresolved (dataset-size gap), resolved (split provenance).** This
repository does not have, and does not claim to have, the paper's
2,500-sample joint Amharic+Tigrinya educational-quality dataset — 700 + 745 =
1,445 is still far short of 2,500. The Tigrinya TC pipeline uses a verified
FineWeb-C config; the Amharic TC pipeline uses a separate, self-annotated
TSV of a different size from a different source, now with a recorded split
built the same way as Tigrinya's. Both are documented as what they are, not
as the paper's dataset.

## 5b. LGSE's own fallback-chain usage is not logged

`LGSEInitializer.init_token_embedding` (`src/lgse/initializer.py`) resolves
each new token through a fallback chain: morpheme-average (FastText vectors
for the token's known morphemes, averaged) → whole-token FastText vector →
character n-gram encoding (`src/lgse/char_ngrams.py`, a fixed-seed but
otherwise content-free random vector per n-gram, scaled by 0.02). This chain
is correctly wired and reachable at every level (verified directly against
the source in a 2026-09-18 re-check; see git history around this note).

Unlike the FOCUS baseline, which logs `unique_rows`/`per_dim_std` for its
written initialization (`assert_focus_init_is_distinct`,
`src/baselines/focus_aux.py`), **LGSE's initializer records no equivalent
statistic.** There is no log output, for the already-committed Table 2 run or
any other, stating what fraction of new tokens for Amharic or Tigrinya
resolved via morpheme-average, whole-token FastText, or the random character
n-gram fallback.

This is not a correctness defect -- every fallback level is real and
reachable, and a token that falls through to character n-grams still gets an
embedding, per the paper's described fallback chain. It is a missing
diagnostic: if a large fraction of new tokens for a given language landed on
the content-free random fallback, that would be a legitimate, mundane
explanation for weak LGSE performance on that language, and there is
currently no way to check this from existing run artifacts.

**Measured directly, 2026-09-18** (offline, against the live FastText model
and the current `data/morph_lexicon.txt`, for `lgse_lapt`'s 198-token Amharic
vocabulary):

| tier | count | % |
|---|---|---|
| morpheme-average | 45 | 22.7% |
| whole-token FastText | 153 | 77.3% |
| character n-gram (random) fallback | 0 | 0.0% |

Zero tokens hit the content-free random fallback -- FastText's own subword
handling covers every token that has no lexicon entry, so the "weak
performance from random fallback" hypothesis above does not hold for the
current Amharic setup. But the split is still informative: fewer than a
quarter of new tokens get genuine morphological decomposition; the rest get
`lgse_lapt`'s embedding from whole-token FastText alone, which is close in
spirit to what `focus_lapt` already does from a different auxiliary space.
This means the two systems are less differentiated in practice than their
descriptions suggest, for this vocabulary. This logging is not wired into
the training code itself (still a gap for future runs); the numbers above
come from a one-off audit script, not from `LGSELAPTrainer`.

## 5b-ii. Roughly half of Amharic's new-vocabulary tokens never receive a
gradient during LAPT

A related, more consequential question: regardless of *how* a new token's
row is initialized, does LAPT ever actually move it? `train_epoch()`
(`src/lgse/lap_trainer.py`) only updates a row when the tokenizer emits that
token's ID somewhere in a training batch -- rows for tokens absent from the
LAPT corpus receive zero gradient for the entire run and stay exactly at
their initialization value.

Measured directly by tokenizing all 200,000 lines of `data/lapt/amharic.txt`
with `xlm-roberta-base` plus the 198 added Amharic tokens, and checking which
of the 198 new token IDs are ever produced:

* **101/198 (51.0%)** of the new tokens actually appear in the LAPT corpus
  and can receive gradient.
* **97/198 (49.0%)** never appear at all; their embedding rows are frozen at
  initialization for the whole of LAPT, no matter which system
  (`lapt`/`random_lapt`/`focus_lapt`/`lgse_lapt`) or initializer produced
  them.

This explains an otherwise-odd observation in the training logs: LGSE's
regularizer loss (`reg=` in `[LGSELAPTrainer] avg loss this epoch:` lines)
reports exactly `0.0000` on every epoch for the systems checked so far. The
regularizer measures drift from initialization (`L_reg = lambda *
||e_new - mu||^2`, paper Sec 4.2); with roughly half the rows never touched
by a gradient step, and the touched half moving only slightly against a
0.15-probability MLM objective, the reported value legitimately rounds to
`0.0000` at 4 decimal places -- this is not evidence the regularizer is
disconnected, and re-deriving it against `src/lgse/regularization.py`
confirms the loss computation itself is correct and wired into
`train_epoch()`'s backward pass as specified.

The same check against the *original*, pre-fix Tigrinya-shaped
`data/new_tokens.txt` (i.e. what Amharic runs used before section 5c's
fix) finds only **39/198 (19.7%)** of those tokens appear in the Amharic
LAPT corpus -- so the new-tokens-list fix in section 5c did produce a real,
measurable improvement in gradient coverage (19.7% -> 51.0%), independent of
any downstream score. It does not close the gap: roughly half the
vocabulary any vocab-expanding system is nominally adapting via LAPT is,
in the current setup, adapted in name only. The root cause is that the OOV
candidate list (`data/new_tokens_amharic.txt`) was built from the TC/NER
task text (see section 5c), a different corpus than
`data/lapt/amharic.txt`; a token can be frequent and highly fragmented in
one and entirely absent from the other. Building the new-tokens list from
the LAPT corpus itself, rather than from downstream task text, would be the
principled fix, but is not done here -- recorded as an open gap, not
resolved as part of this audit.

## 5b-iii. Small LAPT-corpus supplement from local Amharic sources (2026-09-18)

Per the user's suggestion, checked three other local projects
(`~/amseg`, `~/LLAMA3_M/HornMT.json`, `~/VEXMLM_Official`) for Amharic text
that could raise the 51.0% gradient-coverage figure above by supplementing
`data/lapt/amharic.txt` -- deliberately as a corpus fix (more of the
existing, already-justified 198 tokens get exercised), not as a way to
keep swapping inputs until a downstream score improves.

* `VEXMLM_Official` was already fully incorporated: `data/lapt/amharic.txt`
  *is* `VEXMLM_Official/datasets/raw/amharic/amharic.txt` (sha256-verified
  match, see section 5d). No additional signal there.
* `amseg`'s `geez_corpus.txt` / `mt_combined.txt` are classical/liturgical
  Ge'ez, a different language from Amharic despite the shared script --
  correctly excluded, not used.
* `amseg/data/evaluation/flores200/amharic_flores_{dev,devtest}.am` and
  `amseg/data/evaluation/amharic/test.am` are modern Amharic FLORES-200
  benchmark sentences (2,109 lines; FLORES-200 is a public Meta AI release).
  Measured contribution alone: 42/198 tokens covered, only 4 net-new beyond
  the existing corpus.
* `~/LLAMA3_M/HornMT.json` (2,030 JSON-lines records, `data.amh` field) is
  modern Amharic news-domain text resembling the public HornMT
  parallel corpus for Horn-of-Africa languages. **Its exact provenance
  inside this local repo is not independently verified here** -- no
  README or license file in `LLAMA3_M` mentions HornMT by name. Recorded as
  a caveat, not asserted as a specific license. Measured contribution
  alone: 58/198 tokens covered, 5 net-new beyond the existing corpus.

Combined (both sources, deduplicated against the existing 200,000-line
corpus -- only 1 line overlapped): **4,137 new unique lines appended** to
`data/lapt/amharic.txt`, taking it from 200,000 to 204,137 lines
(new sha256 `16a5041118febcade4a13efff9e91becf75de5433b0d262a894777ac923e0c64`;
original backed up before the append and verified against its known
sha256 `b5d5b1da90d3fff680d083dcab7ad8416037665e7762c99a2826b53ef8a849e1`
first). Re-measured against the actual final merged file (not just
predicted): gradient coverage for the 198 new Amharic tokens rises from
**101/198 (51.0%) to 110/198 (55.6%)** -- a real, verified, modest gain
(+9 tokens). The remaining 88 uncovered tokens (mostly proper nouns and
rare loanword spellings, per the OOV audit in section 5c) are not touched
by either source; closing the rest of the gap would need either a much
larger general-domain Amharic corpus or, more directly, building the
new-tokens list from the LAPT corpus itself rather than from downstream
task text -- still the open item recorded in section 5b-ii, not resolved
by this supplement.

No Table 2 result has been rerun against this supplemented corpus as of
this note; a future Amharic rerun should reflect it.

## 5c. `data/new_tokens.txt` is a single Tigrinya-derived list used for both languages, and the morpheme lexicon was correspondingly thin for Amharic

`data/new_tokens.txt` (198 unique tokens after `load_new_tokens()`'s
dedup/sort) is the sole new-vocabulary list `LGSEConfig.new_tokens_file`
defaults to, with no per-language override anywhere in
`scripts/run_table2.sh` or the `configs/` directory. Introduced once, in the
initial commit (`ec1c9a5`), and never split per language since.

Inspection of the list's content (e.g. `ትግራዋይ` "Tigrayan person",
`ኣይመፀን`, other words using the Tigrinya-specific ‘ኣ'-prefixed negation
pattern) shows it is Tigrinya vocabulary, not a bilingual or Amharic list.

**`MorphologicalSegmenter` has no language parameter and no per-language
lookup.** `from_file()` builds one shared `dict[word -> morphemes]` covering
both the Tigrinya and Amharic sections of `data/morph_lexicon.txt`;
`segment(token)` looks the token up in that single dict regardless of which
language's run is calling it. Checked directly (2026-09-18 diagnostic) against
the 198-token list: 179 (90.4%) match an entry that lives in the *Tigrinya*
section, 22 (11.1%) match an entry in the *Amharic* section (some tokens
match both), and only 1 (0.5%, `ኢልበወለድ`) matches neither.

**What this means for LGSE's measured Amharic results:** because the segmenter
is language-blind, an Amharic-language run ("language: am" in `LGSEConfig`)
initializing this same shared token list does **not** mostly fall through to
the character n-gram fallback as an earlier draft of this note stated —
instead, for ~90% of its new tokens, it receives a real morpheme
decomposition that was written into the lexicon as *Tigrinya* morphology,
applied to what the run treats as Amharic vocabulary. Whether this is
"wrong" or merely "makes performance depend on Tigrinya-Amharic morphological
overlap" depends on how similar the two languages' morphological systems are
for the specific words involved — that has not been checked here. Either
way, it is a genuine confound distinct from any W/λ parameter choice, and it
is now corrected as this note's finding rather than the earlier draft's
possibly wrong fallback-chain explanation.

**Amharic morpheme entries added, existing gap not closed (2026-09-18).** 224
new Amharic morpheme entries were added to `data/morph_lexicon.txt`'s Amharic
section, user-supplied and converted from a `WORD<TAB>MORPHEME1_MORPHEME2_..."`
format to the file's native space-separated format; one submitted entry
(`ኢትዮጵያዊ`) already existed with an identical decomposition and was not
duplicated. **These entries do not overlap at all with `data/new_tokens.txt`'s
198 tokens** — the submitted vocabulary is everyday/religious-register text
(e.g. ቤት "house", ኃጢአት "sin", ትንሳኤ "resurrection"), while `new_tokens.txt`
is dominated by loanwords, proper nouns and technical terms. Amharic-section
coverage of the 198-token list therefore remains 22/198 (11.1%), essentially
unchanged by this addition (net of the shared-dict effect described above,
where most of the list resolves via the Tigrinya section regardless).
Closing this gap in the sense of "Amharic runs decompose Amharic-appropriate
vocabulary using Amharic morphology" would require either a
language-aware lookup in `MorphologicalSegmenter` (a code change, not yet
made) or an Amharic-specific new-token list (a candidate 198-token list
derived from `data/tc/amharic` + `data/ner/amharic` word frequency and XLM-R subword
fragmentation exists in a scratch location as of this note, not yet
committed or wired in) with matching morpheme coverage, or morpheme entries
targeted at the existing Tigrinya-derived list's actual words — neither has
been done.

**10,218 amseg/HornMorpho-derived Amharic entries merged (2026-09-18).** A
separate local project, `~/teklehaymanot/amseg`, provides
`data/segmented/amharic_morpheme_segmented.json`: 10,293 Amharic words with
morphological analyses from HornMorpho (a dedicated, purpose-built Amharic/
Tigrinya analyzer), each recording `prefix`/`root`/`suffix`/`infix`/`clitic`
fields per `amseg/scripts/build_morpheme_json.py`.

Conversion note: amseg's `clitic` field merges proclitics (attach before the
root -- prepositions/relativizers like `የ`, `በ`, `ለ`, `ከ`, `እንደ`, `ስለ`,
`ወደ`, `በስተ`, `እስከ`) and enclitics (attach after -- accusative `ን`,
conjunction `ና`, definite-article allomorphs `ኡ`/`ው`/`ት`/`ዋ`/`አው`/`ኑ`/`ኢቱ`)
into one hyphen-joined string with no positional marker (e.g. `የውድድሩ`'s
clitic field is `የ-ኡ`, meaning proclitic `የ` before the root and enclitic
`ኡ` after it, not two morphemes in field order). Reconstructing the correct
`prefix + infix + root + suffix` order required classifying each clitic
piece against the two closed sets above; 23 of 10,293 words (0.22%)
contained a clitic piece matching neither set (e.g. bare `እ` in `እላፊ`,
`እስር`, `እረቡእ` -- plausibly a root-initial vowel HornMorpho split off, not a
genuine clitic) and were excluded rather than guessed at.

Of the remaining 10,270 converted words, 52 already existed in
`data/morph_lexicon.txt` (the original 20 + the 224 added above); 37 of
those 52 disagree with the existing decomposition (HornMorpho's
consonantal-root analysis vs. the existing entries' surface-form splits --
e.g. `ተወላጅ`: existing `ተ ወለጅ`, amseg `ውልድ` -- both linguistically
defensible under different conventions). Per an explicit choice made when
this was found, **existing entries were kept unchanged on conflict**; only
the 10,218 words not already present were merged in. `data/morph_lexicon.txt`
now holds 10,650 entries (up from 432).

**Effect on the coverage gap described above:** `data/new_tokens.txt`'s
198-token Tigrinya-shaped list was already covered at 197/198 (99.5%)
before this merge (via the pre-existing Tigrinya-section entries) and
remains 197/198 after -- unchanged, since this list's words were already
almost entirely resolved through the Tigrinya section (the shared-dict issue
above still applies regardless). The amseg merge does not materially change
this number. More directly relevant: the **Amharic-specific OOV candidate
list** built earlier in this
session (198 words drawn from `data/tc/amharic` + `data/ner/amharic`,
ranked by XLM-R subword fragmentation, not yet wired into
`data/new_tokens.txt`) went from 0/198 covered to **45/198 (22.7%) covered**
by this merge alone — the first concrete coverage improvement against
genuinely Amharic-appropriate vocabulary. The remaining 153/198 uncovered
OOV candidates are mostly proper nouns (place/person names such as
`ላምፔዱዛ`, `ፓንክኸርስት`) and rare loanword spellings HornMorpho's lexicon does
not carry — a data-availability limit, not a methodology gap.

This merge does not by itself change any Table 2 result: it only affects
`data/morph_lexicon.txt`, which is read at LAPT-initialization time, and no
sweep has been rerun against it. It is recorded here as what it is — a
real lexicon expansion with measured, partial effect on one specific,
previously-diagnosed coverage gap — not as a rerun or a new result.

## 5d. Rerunning Amharic under the corrected token list and lexicon

With `data/new_tokens_amharic.txt` wired into `src/training/run_experiment.py`
(language-conditional `new_tokens_file`, per §5c) and the expanded lexicon in
place, a full Amharic rerun (5 systems × 3 tasks × 5 seeds) was submitted via
`logs/rerun_amharic_tokenlist_{tc,ner,qa}.sbatch` on 2026-09-18.

**First attempt failed on missing data.** `data/lapt/amharic.txt` -- the
LAPT continued-pretraining corpus, distinct from the FastText training
corpus at `~/amseg/mt_finetune/data_am/all.am` -- was not present anywhere
on the filesystem, despite its sha256 (`b5d5b1da90d3fff680d083dcab7ad8416`
`037665e7762c99a2826b53ef8a849e1`) being recorded in every historical
Amharic result's `experiment.json`. All three jobs completed the `xlmr`
baseline (unaffected by the token-list/lexicon change, since it does no
vocabulary expansion) across all 5 seeds each, then crashed identically on
the first `lapt`-system seed with `FileNotFoundError`. A substitute corpus
was deliberately not used -- `all.am` is a different, differently-sourced
file with a different hash, and silently swapping it in would have been an
undocumented methodology change of exactly the kind this document exists to
prevent.

**Original file located and verified, not reconstructed.** A file at
`/homes/neumann/teklehaymanot/VEXMLM_Official/datasets/raw/amharic/amharic.txt`
(23,338,664 bytes, 200,000 lines) hashes to the exact recorded sha256 --
confirmed via `sha256sum`, not inferred from filename similarity. Two other
candidates found during the search (`MoVoC_Tok/01_collection/corpus_raw/`
and `02_cleaning/corpus_clean/amharic.txt`, both multi-gigabyte) did **not**
match and were not used. The verified file was copied to
`data/lapt/amharic.txt` and the three jobs resubmitted.

**Update, 2026-09-18 (later same day):** Tigrinya's equivalent corpus was
missing from the repository the same way Amharic's was. Restored by copying
`/homes/neumann/teklehaymanot/VEXMLM_Official/datasets/raw/tigrinya/tigrinya.txt`
to `data/lapt/tigrinya.txt` -- same directory pattern
(`datasets/raw/<language>/<language>.txt`) and same size (200,000 lines) as
the verified Amharic file, and the two files share a timestamp
(`Aug 5 14:00`), consistent with having been prepared in the same batch.
Unlike the Amharic case, **no historical `experiment.json` recorded this
file's sha256**, so this restoration is verified by directory-convention and
content-plausibility (line count, genuine flowing Tigrinya prose, not a
tagged/annotated file) rather than by hash match against a known-good value.
Recorded sha256 of the file now in place:
`e006c5ff19e3ec63f29b862871607d2be1cb94d3fe5db8447cd0f17898e0f6ec`. If this
turns out not to be the exact file the original Tigrinya Table 2 numbers
were produced from, that would only be discoverable if a hash is ever
recovered from elsewhere.

### 5d-i. Completion history of the corrected Amharic rerun

The rerun did not complete in one pass. Recording the job lineage because
the second-generation Amharic numbers in `report/TABLE2_REPORT.md` come from
three different SLURM jobs, and one of them failed part-way.

| Job | Scope | Outcome |
|---|---|---|
| 75113/75114/75115 | all 4 expanding systems × 3 tasks | Killed at ~14 h by a self-imposed `--time=24:00:00`; reached only `lapt` seeds 42-43. Resubmitted with a 7-day limit after confirming no cluster policy required 24 h. |
| 75427 (ner), 75428 (qa) | 4 systems × 5 seeds | `COMPLETED`, exit 0:0. All 20 units each. |
| 75426 (tc) | 4 systems × 5 seeds | `FAILED` at 17/20 (see below). Produced `lgse_lapt` seeds 42-43 before failing. |
| 78067 (tc) | `lgse_lapt` seeds 44-46 only | Seeds completed; job then exited non-zero on a post-run aggregation bug (see 5d-iii). |

`xlmr` was dropped from the resubmissions: it expands no vocabulary, so
neither §5c correction can affect it, and its first-generation results are
carried into the second-generation table unchanged.

### 5d-ii. A non-dict key in `data/fasttext_manifest.json` crashed job 75426

Job 75426 failed after `lgse_lapt` seed 43 with:

```
File "src/training/run_experiment.py", line 93, in provenance
    lang: {k: rec[k] for k in ("source", "dimension", "vocab_size", ...
TypeError: string indices must be integers, not 'str'
```

`provenance()` iterates **every** top-level key of the manifest and assumes
each value is a per-language record dict. A `"_comment"` key with a string
value had been added to the manifest earlier that day as documentation; the
dict comprehension then evaluated `"...some string..."["source"]`.

The failure is instructive beyond the immediate bug: it did not surface
until ~4 days into the run, because `provenance()` is only called when a run
finishes and writes its `experiment.json`. The seed's training and
evaluation had already completed successfully -- seed 44's `result.json` was
written and its score (`test AC 76.06`) printed -- and only the provenance
record was lost.

Fixed by removing the `_comment` key; per-language documentation fields such
as `repo_relative_path` live *inside* each language record, which
`provenance()` reads through a fixed key whitelist and therefore tolerates.
`logs/rerun3_amharic_lgse_tc.sbatch` additionally fails fast before any GPU
work if the manifest has a non-dict top-level key.

### 5d-iii. `aggregate_results.py` could not aggregate TC

Job 78067 produced all three of its seeds correctly but exited `1:0` on its
final `scripts/aggregate_results.py` call:

```
File "src/evaluation/metrics.py", line 107, in aggregate
    "mean": sum(values) / len(values),
TypeError: unsupported operand type(s) for +: 'int' and 'dict'
```

`aggregate()` averaged every key of a run's `test` dict across seeds. That
holds for NER and QA, whose `test` blocks contain only scalars
(`precision`/`recall`/`f1`), but TC also records a nested `per_class`
breakdown, and summing dicts raised. `aggregate()` now skips non-scalar
keys. This is a reporting-path bug only: no result value was affected, and
the Amharic table was cross-checked by reading `test.accuracy` / `test.f1`
directly from each `experiment.json` and confirming the aggregator's means
match to the digit (its standard deviations differ from the report's because
the aggregator uses population sd, as its docstring states, while
`TABLE2_REPORT.md` quotes sample sd).

### 5d-iv. Stale pre-fix records shared the results directory

Six `experiment.json` files dated 2026-08-09, from the original pre-fix run,
occupied exactly the six seed slots the second-generation run had not yet
reached (`lgse_lapt__tc__seed{44,45,46}`, `lgse_lapt__qa__seed{45,46}`,
`lgse_lapt__ner__seed46`). `aggregate_results.py` globs
`results/*/experiment.json` with no freshness check, so aggregating before
those slots were overwritten would have silently blended August pre-fix
numbers (60.0 / 72.0 / 64.0 for the TC seeds) into the corrected table as if
they were part of it. They were all overwritten by the completing jobs
before any table was produced, and every value in the second-generation
Amharic table was confirmed to come from a record dated 2026-09-18 or later
before being reported.

## 6. MasakhaNER source

The paper uses MasakhaNER (Adelani et al., 2021) for Amharic NER. The
HuggingFace mirrors — `masakhane/masakhaner`, `masakhane/masakhaner2`,
`Davlan/masakhanerV1` — are all script-based datasets, which current
`datasets` refuses to load ("Dataset scripts are no longer supported"), and
none carries data files for Amharic.

`data/scripts/prepare_ner.py --language amharic` therefore takes the CoNLL
files directly from the project's own repository,
`masakhane-io/masakhane-ner/data/amh/{train,dev,test}.txt`. These are the
official splits, used as released — no partition is derived. Counts:
1,750 / 250 / 500 sentences (25,819 / 3,749 / 7,449 tokens), tag set
PER/ORG/LOC/DATE, matching the Tigrinya label inventory.

## 7. Single seed

The release sets `seed=42` in one place with no CLI override, so
mean +/- standard deviation over multiple runs cannot be produced.

**Implemented here:** seed is a config field and CLI argument, and
`configs/base.yaml` sets the paper's five runs as seeds 42–46. The paper
states the experiments were "repeated five times with different random
seeds" but does not say which, so these are ours and are recorded in every
run record.

## 8. Table 1 hyperparameters — recovered and applied

Table 1 ("Hyperparameter settings used for further pretraining with
morpheme-aware tokenization and fine-tuning") has been recovered from the
paper and applied to `configs/base.yaml`, with each value annotated
`source: paper`:

| Hyperparameter | Value |
|---|---|
| Maximum sequence length | 256 |
| Batch size | 32 |
| Number of training epochs | 10 |
| Learning rate | 5 × 10⁻⁵ |
| Learning rate schedule | Constant |
| MLM probability | 0.15 |
| Weight decay | 0.01 |
| Optimizer | Adam |
| Adam ε | 1 × 10⁻⁸ |
| Adam β₁ | 0.9 |
| Adam β₂ | 0.999 |
| Mixed precision (fp16) | True |

Two notes on how the table was applied:

- The paper gives **one** table covering both further pretraining and
  fine-tuning, so the same values populate the `lapt:` and `finetune:`
  sections. Sec 5 states hyperparameters are "consistent" across Amharic and
  Tigrinya, so no per-language variation is introduced.

- Table 1 says "Adam" while also specifying weight decay 0.01. The config
  uses AdamW, since decoupled weight decay is what a nonzero `weight_decay`
  means in the HuggingFace/PyTorch stack this code targets. Recorded here
  because it is an interpretation, not a quotation.

The schedule is constant with no warmup stated, so `warmup_ratio` is 0.0.

**`source: unavailable`:** the regularization strength λ in
`L_reg = λ‖e_new − μ‖²`. The paper introduces λ but does not give its value,
and Table 1 does not list it. This is the only optimisation-relevant value
not from the paper.

### 8a. λ is a mandatory parameter

Because there is no published value, **`reg_lambda` has no default**.
`LGSEConfig` raises `MissingRequiredParameter` when it is not supplied, and
`run_experiment.py` refuses to start if `lgse.reg_lambda` is absent from the
run config.

This is deliberate friction. A silent default would bury an experimenter's
choice in a dataclass field, and every result would then carry a value that
*looks* like it came from the paper. Requiring it means whoever runs an
experiment states λ, and the value is recorded in the run record alongside
`reg_lambda_source: "unavailable -- not stated in the paper"`.

The paper defines λ symbolically but does not provide its numerical value or
selection procedure. This implementation therefore uses λ = 1.0 so that the
pipeline can be executed.

`configs/base.yaml` ships `reg_lambda: 1.0`, marked `source: unavailable`.
**That value is an implementation choice, not a value specified by the
paper**, and λ is a plausible candidate for sensitivity analysis: it sets the
balance between preserving the lexically grounded initialization and adapting
to the target language, which is the trade-off the method turns on. Deleting
the key from a config makes runs fail rather than fall back.
