# Implementation notes

This repository implements the LGSE method. Each entry describes what the
implementation does and how it is configured.

> ## Prerequisites
>
> Two values are supplied per run as implementation inputs:
>
> | Required | Where it is used |
> |---|---|
> | `lgse.alignment_matrix_path` (W) | the square `d×d` projection applied to FastText vectors (§1, §1a) |
> | `lgse.reg_lambda` (λ) | the coefficient of the regularization term (§8a) |
>
> The W used by a run is recorded in the run record.
>
> A third prerequisite is data rather than configuration: FastText vectors at
> the model's embedding width (768), since W is square (§1).
>
> ### Standing policy
>
> **Neither value has a default.** Both are supplied explicitly per run, and
> a run records which of the three prerequisites it satisfied.
>
> The policy is enforced by tests:
>
> | Invariant | Test |
> |---|---|
> | No entry point defaults W | `test_alignment_matrix_is_required_by_every_entry_point` |
> | No W is committed to the shipped config | `test_shipped_config_leaves_the_matrix_unset` |
> | λ has no default | `test_reg_lambda_is_mandatory` |
> | Mismatched dimensions are not reshaped | `test_no_silent_reshaping_of_mismatched_vectors` |
> | No unstated loss term trains W | `test_paper_objectives_give_w_no_gradient` |
> | Runs without a supplied W are marked in the table | `tests/test_results_table_fidelity.py` |
> | Baselines are not marked | `test_baselines_are_not_flagged_as_unfaithful` |

## 1. Projection W is square, and externally supplied

**Paper (Sec 4.1):** "To align the FastText embedding space with the
pretrained model embedding space, a learned linear projection
`W ∈ R^{d×d}` is applied, i.e. `e_aligned_t = W e_t`."

Two consequences, both implemented:

**W is square.** W maps the FastText space onto the model's embedding space
at the same dimension `d`, so FastText must be trained at the model's
embedding width (768 for xlm-roberta-base), not the standard 300 —
`configs/base.yaml` sets `fasttext_dim: 768` accordingly.

**The embedding dimensions must match the projection shape.** The 300-dim CC
vectors referenced under `data.fasttext` do not match the square W, so the
implementation raises on a dimension mismatch:

| Not applied | Reason |
|---|---|
| Rectangular `W ∈ R^(300×768)` | W is `d×d` |
| Truncating 768→300 or padding 300→768 | Changes the embedding dimensions |
| PCA/SVD reduction of the model's space | Changes what the embeddings mean |
| Falling back to char n-grams on mismatch | Bypasses FastText while reporting as LGSE |

The check is applied at three points, the earliest first:

1. **`data/scripts/download_fasttext.py`** — checks at download time
   (`--expect-dim`, default 768), prints a prominent warning naming the fix,
   records `dimension_ok` in the manifest, and exits non-zero.
2. **`LGSEConfig.fasttext_path`** — reads the manifest's recorded dimension
   and raises before a multi-GB model is loaded.
3. **`build_projection` / `MorphemeEmbeddingBuilder`** — the final check,
   via `check_dimensions`. Both share one implementation so they cannot
   disagree.

All three raise `IncompatibleFastTextDimension`, a distinct exception type,
with a message naming both dimensions, the reason, the required width, and
the `fasttext -dim 768` command that produces it.

**W has no default.** It is supplied via `lgse.alignment_matrix_path`; a run
without one raises. See §1a-i for the candidates considered and not used.

**W is the only supported projection.** `src/lgse/projection.py` defines
`AlignmentProjection` as a square `d×d` matrix, externally supplied and
frozen. It is the single projection path; there is no `projection:` config
key.

### 1a. W receives no gradient

No loss term implemented here is a function of W:

- **Initialization** sets `e_new` to the average of projected morpheme
  embeddings. This value is written into the embedding matrix via `.data`
  (`src/lgse/initializer.py:60-63`), which severs the autograd graph. The
  in-place write preserves weight tying with the MLM head.

- **The regularizer** is `L_reg = λ‖e_new − μ‖²`, where μ is the initial
  embedding vector. μ is a constant; the term measures drift from
  initialization and trains `e_new`, not W.

- **LAPT** applies MLM with the encoder frozen, updating only the new
  embeddings. It reads the embedding matrix, not W.

W is initialized, its output is copied into the embeddings, and it is never
differentiated again. Verified empirically:

```
W is square      : torch.Size([768, 768])
identity init    : True
emb grad         : True
W grad          : None
```

**Status:** W is supplied externally and is not trained by any loss term.

### 1a-i. Implementation properties

W is implemented as an **externally supplied alignment matrix**:

| # | Property | Consequence |
|---|---|---|
| 1 | W is an externally supplied artifact | `alignment_matrix_path` loads a `d×d` matrix from `.pt`/`.npy`; **no default of any kind** |
| 2 | W is frozen | `requires_grad=False`, excluded from the optimizer |
| 3 | No loss term trains W | No gradient reaches it |
| 4 | `reg_lambda` must be explicitly provided | No default; see §8a |
| 5 | W is supplied per run as an implementation input | Recorded with the run |

**There is no default W, including the identity.**

Candidates considered and not used:

| Candidate | Reason it is not used |
|---|---|
| Identity | Treats the two spaces as already aligned |
| Random / seeded | Applies an arbitrary map in place of a supplied one |
| A rectangular Johnson–Lindenstrauss map | Rectangular, where W is `d×d` |

`build_projection` raises `MissingAlignmentMatrix`, a distinct exception
type. The message names `lgse.alignment_matrix_path` and points at
`scripts/build_alignment_matrix.py`, which constructs a W by orthogonal
Procrustes alignment on shared vocabulary.

A supplied identity is accepted and recorded in the trainer log.

**W is frozen.** It carries `requires_grad=False` with no switch to change
it: `AlignmentProjection` takes no `trainable` argument, and the optimizer
contains the embedding matrix alone.

**No loss term gives W a gradient.** No reconstruction or alignment loss is
implemented. μ is the initial embedding vector, so `LGSERegularizer` accepts
a fixed anchor tensor.

### 1a-ii. How W is recorded

W's status is checked before a run starts and recorded with each artifact:

- **Before the run:** `run_experiment.py` raises for any FastText-using
  system (`lgse`, `focus`) when `lgse.alignment_matrix_path` is unset —
  checked before the backbone and the multi-GB FastText model are loaded.
  `build_projection` performs the check.
- **Per run:** the trainer logs W's source and frozen status each run. A
  supplied identity is additionally noted.
- **Per checkpoint:** `projection.pt` carries `source`, `trainable` and
  `training_status`; `projection_status.json` sits beside it.
- **Per result:** the run record's `projection` field records `source`,
  `author_supplied`, `training_status`, and `trained_during_this_run: false`.
- **In the generated table:** `scripts/aggregate_results.py` emits a notice
  naming any system whose runs lacked a supplied W, and an "Alignment matrix
  W" column per row recording the W each run used, `MIXED` when seeds within
  one system disagree, or `n/a` for the baselines, which use no FastText and
  need no W.
- **In tests:** `test_paper_objectives_give_w_no_gradient` fails if a change
  adds a loss term that reaches W;
  `test_alignment_matrix_is_required_by_every_entry_point` fails if any
  entry point introduces a default;
  `test_shipped_config_leaves_the_matrix_unset` fails if a W is committed to
  `configs/base.yaml`.

### 1b. W is part of the checkpoint

`save_pretrained()` covers only the model, so `LGSELAPTrainer.save()` writes
`projection.pt` alongside it and `load_projection()` restores it, refusing a
shape mismatch. The alignment matrix a result used travels with that result,
so a checkpoint reloads with the same W that produced its numbers.

## 2. Comparison baselines

Table 2 compares five systems. Their implementations are provided here.

`src/baselines/strategies.py` provides `default` (+LAPT),
`random` (+Random+LAPT) and `focus` (+FOCUS+LAPT), each exposing the same
interface as `LGSEInitializer` so the trainer swaps between them with no
other change.

FOCUS (Dobler & de Melo, 2023) is implemented here as a similarity-weighted
combination of pretrained embeddings using the FastText space as the
auxiliary signal -- the same external signal LGSE uses, so the two differ
only in how they use it. This is an independent implementation and has not
been validated against the results published with FOCUS.

## 3. Table 2 evaluation

Table 2 reports QA, NER and text classification. The evaluation harness is
in `src/evaluation/` (entity-level NER F1, SQuAD QA F1, mean/stdev over
seeds).

**Status:** the full Table 2 sweep has not been run. It requires the
prerequisites above — a supplied W and a λ value.

## 4. FastText model acquisition

FastText binaries are 2--3 GB and are not committed.
`data/scripts/download_fasttext.py` fetches them on demand:

| | Source | Dim | Vocab |
|---|---|---|---|
| Amharic | `cc.am.300.bin` (Grave et al., 2018) | 300 | -- |
| Tigrinya | `Hailay/fasttext-tigrinya` | 300 | 156,687 |

The script loads each model and raises if it is empty. Dimensions,
vocabulary size and sha256 are recorded in
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
the manifest. Consequences:

* An extractive QA model can be evaluated on 13 Tigrinya test items at most,
  which is far too few for a stable F1, let alone a standard deviation over
  five seeds.
* The Zenodo release does not include train-dev-test splits, and its answers
  are not spans of their contexts.

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
matching, and assigning offsets to them would corrupt the metric.

Two properties of these splits are worth recording:

* They are derived here with seed 42, not taken from an official release.
  The split files in `hailaykidu/TigQA-Dataset` are not usable as published:
  `dev.json` and `test.json` contain malformed JSON, and in `train.json` all
  37 answer offsets are relative to the source document rather than the
  merged paragraph, so none resolve.
* Tigrinya QA is scored on 86 test items against Amharic's 285.

## 6. MasakhaNER source

Amharic NER uses MasakhaNER (Adelani et al., 2021). The HuggingFace
mirrors -- `masakhane/masakhaner`, `masakhane/masakhaner2`,
`Davlan/masakhanerV1` -- are all script-based datasets, which current
`datasets` refuses to load ("Dataset scripts are no longer supported"), and
none carries data files for Amharic.

`data/scripts/prepare_ner.py --language amharic` therefore takes the CoNLL
files directly from the project's own repository,
`masakhane-io/masakhane-ner/data/amh/{train,dev,test}.txt`. These are the
official splits, used as released -- no partition is derived. Counts:
1,750 / 250 / 500 sentences (25,819 / 3,749 / 7,449 tokens), tag set
PER/ORG/LOC/DATE, matching the Tigrinya label inventory.

## 7. Random seeds

`seed` is a configuration field and a CLI argument, so the seed is
configurable per run. `configs/base.yaml` sets `evaluation.seeds` to
42, 43, 44, 45, 46. The selected seed is recorded in each run record.

## 8. Table 1 hyperparameters

Table 1 ("Hyperparameter settings used for further pretraining with
morpheme-aware tokenization and fine-tuning") is applied to
`configs/base.yaml`:

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

These values are marked `source: paper` in the configuration. Two notes on
how the table is applied:

- Table 1 covers both further pretraining and fine-tuning, so the same values
  populate the `lapt:` and `finetune:` sections. Hyperparameters are
  consistent across Amharic and Tigrinya, so no per-language variation is
  introduced.

- Table 1 says "Adam" while also specifying weight decay 0.01. The config
  uses AdamW, since decoupled weight decay is what a nonzero `weight_decay`
  means in the HuggingFace/PyTorch stack this code targets. Recorded here
  because it is an interpretation, not a quotation.

The schedule is constant with no warmup stated, so `warmup_ratio` is 0.0.

The regularization strength λ in `L_reg = λ‖e_new − μ‖²` is marked
`source: unavailable` in the configuration. This implementation uses λ = 1.0;
see §8a.

### 8a. λ is a mandatory parameter

λ is the coefficient of the regularization term `L_reg = λ‖e_new − μ‖²`. It
is read from `lgse.reg_lambda`, passed to `LGSERegularizer(lambda_reg=...)`,
and multiplies the mean squared drift in
`src/lgse/regularization.py`; `lambda_reg == 0.0` returns a zero loss.

**`reg_lambda` has no default**: `LGSEConfig` raises
`MissingRequiredParameter` when it is not supplied, and `run_experiment.py`
stops if `lgse.reg_lambda` is absent from the run config.

λ is set per run and recorded in the run record alongside
`reg_lambda_source`.

`configs/base.yaml` ships `reg_lambda: 1.0`. This implementation uses
λ = 1.0. λ balances preserving the lexically grounded initialization against
adapting to the target language.
