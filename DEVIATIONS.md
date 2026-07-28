# Deviations from the paper and the original release

This branch implements the LGSE method as described in the paper. Where the
implementation differs from the published release
(`published-release-77987c5`), or where the paper underdetermines a choice,
it is recorded here. Nothing in this list is a claim about which is correct
-- only about what differs.

> ## Fidelity prerequisites
>
> Two artifacts cannot be derived from the paper, and the implementation
> **fails rather than substituting a value** for either:
>
> | Required | Why it cannot be defaulted |
> |---|---|
> | `lgse.alignment_matrix_path` (W) | Sec 4.1 introduces W but never says how it is obtained (§1a) |
> | `lgse.reg_lambda` (λ) | Sec 4.2 introduces λ but never assigns it (§8a) |
>
> **Any result produced without an author-provided W is not faithful to the
> published method.** It may still be a useful experiment, but it is an
> experiment under a documented substitution, not a reproduction — and the
> substitution is the experimenter's, recorded in the run record.
>
> A third prerequisite is data, not configuration: FastText vectors at the
> model's embedding width (768), since W is square (§1).
>
> ### Standing policy
>
> **No silent fallback may be added for an unspecified methodological
> component.** Where the paper does not specify something, this repository
> fails and says so; it does not infer, reconstruct, or default. That
> applies to future changes as much as to the current state.
>
> A run either satisfies all three prerequisites, or it reports itself as a
> partial reproduction / implementation validation. There is no third
> category, and no result should be presented as faithful without the
> artifacts above.
>
> This is enforced, not merely stated:
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
with the paper's W. This is a data prerequisite, not something the code can
resolve**, so the implementation refuses rather than adapting:

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

**W defaults to the identity.** Alignment between same-dimension spaces
starts from "no change", so the initial embeddings are exactly the FastText
morpheme averages Sec 4.1 defines. An arbitrary W would distort those
vectors before training saw them, defeating the point of grounding the
initialization lexically. (The original release's scaled-Gaussian init was
tied to its rectangular map and does not carry over.) An author-supplied
matrix replaces the identity — see §1a-i.

**Original release:** `lgse/morpheme_embeddings.py:24-31` builds a fixed,
seeded Johnson-Lindenstrauss projection with `np.random.default_rng`. It is
never a `torch.nn.Parameter`, never passed to an optimizer, and never
updated. The code comment states the rationale: bridging 300-dim FastText to
768-dim XLM-R "without requiring extra training data to fit a learned
projection."

**This branch implements the paper's W as the only supported projection.**
`src/lgse/projection.py` defines `AlignmentProjection` as a square `d×d`
matrix, externally supplied and frozen. There is no fixed-projection path
and no `projection:` config key. The release's rectangular
Johnson–Lindenstrauss map is removed, not retained as a fallback.

### 1a. Under the paper's stated objectives, W receives no gradient

**This is an open discrepancy in the paper, recorded rather than resolved.**

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

**Status: author-required / unspecified in paper.**

### 1a-i. Implementation assumptions

W is therefore implemented as an **externally supplied alignment matrix**,
under three assumptions recorded here as assumptions, not findings:

| # | Assumption | Consequence |
|---|---|---|
| 1 | W is an externally supplied artifact | `alignment_matrix_path` loads a `d×d` matrix from `.pt`/`.npy`; **no default of any kind** |
| 2 | W is frozen | `requires_grad=False`, excluded from the optimizer |
| 3 | No objective trains W | No loss term is invented to give it a gradient |
| 4 | `reg_lambda` must be explicitly provided | No default; see §8a |
| 5 | A result without an author-provided W is **not faithful** to the published method | Runs fail rather than proceed |

**There is no default W — not even the identity.**

An earlier revision of this branch defaulted to the identity, on the
reasoning that it "adds nothing". That reasoning was wrong. The identity is
not a neutral choice: it asserts that the FastText and model embedding
spaces are *already aligned*, which is precisely the claim Sec 4.1
introduces W to avoid having to make. Substituting it would replace a
missing specification with an implementation decision that materially
affects results, while the run still looked faithful.

Every candidate default fails the same test:

| Candidate | Why it is not used |
|---|---|
| Identity | Asserts the two spaces are already aligned — the claim W exists to avoid |
| Random / seeded | An alignment strategy the paper does not describe |
| Fitted (Procrustes, CCA, …) | A method the paper does not describe, requiring anchor data it never mentions |
| Release's Johnson–Lindenstrauss map | Rectangular, and tied to the release's 300→768 setup |

So the implementation refuses. `build_projection` raises
`MissingAlignmentMatrix` — a distinct type, because this marks a genuinely
unspecified part of the method rather than a misconfiguration to patch. The
message quotes Sec 4.1, states that the paper never says how W is obtained,
explains why neither the identity nor a random matrix is substituted, and
records that any result produced without an author-provided W is not
faithful to the published method.

A *supplied* identity remains perfectly legitimate — it is then the author's
documented choice, recorded as such, not this project's silent one. The
trainer flags it in the log when it occurs.

**Why W is frozen rather than trainable-but-unused.** An earlier revision
placed W in the optimizer on the reasoning that "any objective which is a
function of it would train it". But nothing differentiates W, so that
optimizer entry advertised a capability the run did not have — a reader
inspecting the parameter groups would conclude W was being learned.
`AlignmentProjection` still accepts `trainable=True` for a future run under
an author-supplied objective; nothing in this repository sets it, and
setting it alone does not create a gradient path.

**What is *not* done.** No loss term is invented to give W a gradient.
Candidates exist — a live regularizer anchor, a reconstruction loss, an
alignment loss against anchor translations — and any of them would make W
train and produce numbers. None is in the paper, so none is implemented.
An earlier revision of this branch did make the regularizer anchor a live
function of W; that was reverted on reading Sec 4.2, since it contradicts
"μ is the initial embedding vector". The capability remains in
`LGSERegularizer` (`anchor_is_live`), unused by any configured run.

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
- **In tests:** `test_paper_objectives_give_w_no_gradient` fails if a future
  change adds an unstated loss term;
  `test_alignment_matrix_is_required_by_every_entry_point` fails if any
  entry point reintroduces a default;
  `test_shipped_config_leaves_the_matrix_unset` fails if a W is ever
  committed to `configs/base.yaml`.

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

## 2. Baselines absent from the release

Table 2 compares five systems. The published release implements only the
LGSE path; the string "FOCUS" does not appear anywhere in it, and there is no
random-initialization or no-expansion baseline.

**This branch:** `src/baselines/strategies.py` adds `default` (+LAPT),
`random` (+Random+LAPT) and `focus` (+FOCUS+LAPT), each exposing the same
interface as `LGSEInitializer` so the trainer swaps between them with no
other change.

FOCUS (Dobler & de Melo, 2023) is reimplemented here as a similarity-weighted
combination of pretrained embeddings using the FastText space as the
auxiliary signal -- the same external signal LGSE uses, so the two differ
only in how they use it. **This is a reimplementation, not the authors'
code**, and has not been validated against their published results.

## 3. Evaluation absent from the release

The published release contains no evaluation code: no occurrence of
`evaluate`, `f1`, or `accuracy` in any Python file. Table 2 reports QA, NER
and text classification.

**This branch:** pending. The datasets, splits, metrics and hyperparameters
are to be taken from the paper's experimental section; they are not
guessable from the release, and nothing here claims to reproduce Table 2
until they are wired in and run.

## 4. Data placeholders

`data/fasttext_Amharic.bin` and `data/fasttext_Tigriyna.bin` in the release
are 208-byte text files containing download URLs, not FastText binaries.
Loading either raises `ValueError: ... has wrong file format!`
(`lgse/lap_trainer.py:60`). The training corpus in
`scripts/run_lgse_lap.py` is two hardcoded sentences, self-documented as a
smoke test.

**This branch:** the placeholder files are removed and
`data/scripts/download_fasttext.py` fetches the real models on demand:

| | Source | Dim | Vocab |
|---|---|---|---|
| Amharic | `cc.am.300.bin` (Grave et al., 2018) | 300 | -- |
| Tigrinya | `Hailay/fasttext-tigrinya` | 300 | 156,687 |

The script loads each model and refuses to continue if it is empty; it never
substitutes random vectors, because a placeholder would silently reduce LGSE
to its own character-n-gram fallback while still reporting as LGSE.
Dimensions, vocabulary size and sha256 are recorded in
`data/fasttext_manifest.json`. The binaries themselves are gitignored.

## 5. TIGQA is abstractive, not extractive

TIGQA (Zenodo 11423987, CC-BY-4.0) is released as a `.docx` table -- columns
R/no, Grade level, Topic, Context, Question, Answer -- with several numbered
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

**Resolved.** A SQuAD-format conversion of TIGQA (`TIGQA_squad_format.json`,
version TIGQA-1.0) supplies 2,108 QA pairs over 433 contexts with
`answer_start` offsets -- an order of magnitude more than the .docx yields,
and enough for span-extraction F1.

Of those 2,108:

| | count | usable for extractive F1 |
|---|---|---|
| valid span (offset resolves) | 797 | yes |
| `answer_start == -1` | 1,039 | no -- abstractive rewrite |
| no answer given | 272 | no |

`prepare_qa.py --dataset tigqa_squad` keeps the 797 extractive pairs and
records the other two counts in the manifest. Splits: 644/67/86 QA pairs over
292/36/37 contexts.

The `answer_start == -1` marker is the conversion's own signal that an answer
could not be located in its context; verified independently here, none of the
1,039 occurs verbatim in its context. They are not recoverable by string
matching, and fabricating offsets for them would corrupt the metric.

Two caveats remain for strict reproduction:

* These splits are derived here with seed 42, not taken from an official
  release. The paper refers to "TIGQA train-dev-test splits"; the split files
  in `hailaykidu/TigQA-Dataset` are not usable as published -- `dev.json` and
  `test.json` contain malformed JSON, and in `train.json` all 37 answer
  offsets are relative to the source document rather than the merged
  paragraph, so none resolve.
* Tigrinya QA is scored on 86 test items against Amharic's 285.

## 6. MasakhaNER source

The paper uses MasakhaNER (Adelani et al., 2021) for Amharic NER. The
HuggingFace mirrors -- `masakhane/masakhaner`, `masakhane/masakhaner2`,
`Davlan/masakhanerV1` -- are all script-based datasets, which current
`datasets` refuses to load ("Dataset scripts are no longer supported"), and
none carries data files for Amharic.

`data/scripts/prepare_ner.py --language amharic` therefore takes the CoNLL
files directly from the project's own repository,
`masakhane-io/masakhane-ner/data/amh/{train,dev,test}.txt`. These are the
official splits, used as released -- no partition is derived. Counts:
1,750 / 250 / 500 sentences (25,819 / 3,749 / 7,449 tokens), tag set
PER/ORG/LOC/DATE, matching the Tigrinya label inventory.

## 7. Single seed

The release sets `seed=42` in one place with no CLI override, so
mean +/- standard deviation over multiple runs cannot be produced.

**This branch:** seed is a config field and CLI argument, and
`configs/base.yaml` sets the paper's five runs as seeds 42-46. The paper
states the experiments were "repeated five times with different random
seeds" but does not say which, so these are ours and are recorded in every
run record.

## 8. Table 1 hyperparameters — recovered and applied

Table 1 ("Hyperparameter settings used for further pretraining with
morpheme-aware tokenization and fine-tuning") has been recovered from the
paper and applied to `configs/base.yaml`:

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

The values previously marked `source: unavailable` are now `source: paper`.
Two notes on how the table was applied:

- The paper gives **one** table covering both further pretraining and
  fine-tuning, so the same values populate the `lapt:` and `finetune:`
  sections. Sec 5 states hyperparameters are "consistent" across Amharic and
  Tigrinya, so no per-language variation is introduced.

- Table 1 says "Adam" while also specifying weight decay 0.01. The config
  uses AdamW, since decoupled weight decay is what a nonzero `weight_decay`
  means in the HuggingFace/PyTorch stack the release targets. Recorded here
  because it is an interpretation, not a quotation.

The schedule is constant with no warmup stated, so `warmup_ratio` is 0.0.

**Still `source: unavailable`:** the regularization strength λ in
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

`configs/base.yaml` ships `reg_lambda: 1.0`, carried over from the original
release and marked `source: unavailable`. **That value is the release's, not
the paper's**, and λ is a plausible candidate for sensitivity analysis: it
sets the balance between preserving the lexically grounded initialization
and adapting to the target language, which is the trade-off the method turns
on. Deleting the key from a config makes runs fail rather than fall back.
