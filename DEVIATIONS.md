# Deviations from the paper and the original release

This branch implements the LGSE method as described in the paper. Where the
implementation differs from the published release
(`published-release-77987c5`), or where the paper underdetermines a choice,
it is recorded here. Nothing in this list is a claim about which is correct
-- only about what differs.

## 1. Projection W is learned, and square

**Paper (Sec 4.1):** "To align the FastText embedding space with the
pretrained model embedding space, a learned linear projection
`W ∈ R^{d×d}` is applied, i.e. `e_aligned_t = W e_t`."

Two consequences, both implemented:

**W is square.** The paper aligns two spaces of the same dimension `d`; it
describes no dimensionality change. `build_projection` enforces this and
raises on a mismatch rather than substituting a rectangular map, which would
be a different method than the published one. In practice this means
FastText must be trained at the model's embedding width (768 for
xlm-roberta-base), not the standard 300 — `configs/base.yaml` sets
`fasttext_dim: 768` accordingly. **The 300-dim CC vectors referenced under
`data.fasttext` cannot be used with the paper's W without retraining or
reducing at dimension 768; this is a data prerequisite, not something the
code can resolve.**

**W is identity-initialized.** Alignment between same-dimension spaces
starts from "no change", so the initial embeddings are exactly the FastText
morpheme averages Sec 4.1 defines. A random init would scramble those
vectors before training saw them, defeating the point of grounding the
initialization lexically. (The original release's scaled-Gaussian init was
tied to its rectangular map and does not carry over.)

**Original release:** `lgse/morpheme_embeddings.py:24-31` builds a fixed,
seeded Johnson-Lindenstrauss projection with `np.random.default_rng`. It is
never a `torch.nn.Parameter`, never passed to an optimizer, and never
updated. The code comment states the rationale: bridging 300-dim FastText to
768-dim XLM-R "without requiring extra training data to fit a learned
projection."

**This branch implements the paper's W as the only supported projection.**
`src/lgse/projection.py` defines `LearnedProjection` as a square `d×d`
trainable map. There is no fixed-projection path and no `projection:` config
key. The release's rectangular Johnson–Lindenstrauss map is removed, not
retained as a fallback.

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

**What this implementation does.** W is registered with the optimizer, so
any objective that is a function of it would train it — but none is, so W
remains at its identity initialization throughout LAPT. This is faithful to
the paper's equations. Constructing a gradient path for W would require
inventing a loss term the paper does not state, which would be implementing
a different method.

`LGSELAPTrainer.projection_receives_gradient()` reports the actual situation
after each epoch, so the gap is visible per run rather than only in this
document. `test_paper_objectives_give_w_no_gradient` asserts it, and is
written to fail loudly if a future change silently adds such a term.

**To resolve this, the paper's authors would need to state which objective
trains W.** Until then, "learned" describes W's declared type, not its
observed behaviour under the published equations.

An earlier revision of this branch made the regularizer anchor a live
function of W, which does give W a gradient. That was reverted on reading
Sec 4.2: it contradicts "μ is the initial embedding vector". The capability
remains in `LGSERegularizer` (`anchor_is_live`) for deliberate
departures-from-paper experiments, but is not used by any configured run.

### 1b. W is part of the checkpoint

W is trained, so it cannot be recovered from its seed once optimized.
`save_pretrained()` covers only the model, so `LGSELAPTrainer.save()` writes
`projection.pt` alongside it and `load_projection()` restores it, refusing a
shape mismatch. Without this, a resumed run would silently restart from a
fresh initialization and discard the training that produced W.

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
and Table 1 does not list it. `reg_lambda: 1.0` is carried over from the
release. This is now the only optimisation-relevant value not from the paper.
