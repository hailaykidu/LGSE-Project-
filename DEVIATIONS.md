# Deviations from the paper and the original release

This branch implements the LGSE method as described in the paper. Where the
implementation differs from the published release
(`published-release-77987c5`), or where the paper underdetermines a choice,
it is recorded here. Nothing in this list is a claim about which is correct
-- only about what differs.

## 1. Projection W is learned

**Paper:** W is learned, trained jointly with the new token embeddings.

**Original release:** `lgse/morpheme_embeddings.py:24-31` builds a fixed,
seeded Johnson-Lindenstrauss projection with `np.random.default_rng`. It is
never a `torch.nn.Parameter`, never passed to an optimizer, and never
updated. The code comment states the rationale: bridging 300-dim FastText to
768-dim XLM-R "without requiring extra training data to fit a learned
projection."

**This branch implements the paper's learned W as the only supported
projection.** `src/lgse/projection.py` defines `LearnedProjection` (230,400
trainable parameters for 300->768). There is no fixed-projection path and no
`projection:` config key to select one. When FastText and the model share a
width, `build_projection` returns `None` and the vectors are used directly --
there is nothing to map between. A dimension mismatch with no W raises rather
than substituting an untrained map, so a run cannot silently fall back to the
released behaviour while still reporting as LGSE.

### 1a. Making "learned" true required a change to the regularizer

Putting W in the optimizer is necessary but **not sufficient**, and the
insufficiency is invisible in the reported numbers.

The initializer writes W's output into the embedding matrix through `.data`
(`src/lgse/initializer.py:60-63`), which severs the autograd graph. That
in-place write is deliberate: replacing the `Parameter` would break weight
tying with the MLM head. But if the regularizer's anchor is then a detached
constant, **no LAPT loss term is a function of W at all**. W sits in the
optimizer, reports as trainable, receives `grad = None` on every step, and
never moves -- an end state identical to a frozen random map, reached by a
different route.

This branch therefore recomputes the regularizer anchor through W on every
step (`src/lgse/regularization.py`, `anchor_is_live`). The penalty
`lambda*||E_new - W(f)||^2` is live on both sides, so it pulls the new
embeddings toward their lexically grounded targets and simultaneously adapts
W toward the representations the LM is learning -- "trained jointly with the
new embeddings" in fact, not merely in the optimizer's parameter list.

The baselines have no projection; their anchor remains a fixed tensor and
their behaviour is unchanged.

`tests/test_learned_projection_pipeline.py` asserts both directions:
`test_w_is_updated_by_a_lapt_step` fails if W stops moving, and
`test_detached_anchor_leaves_w_without_gradient` pins the reason the live
anchor is required.

**Scope note.** The paper specifies that W is learned jointly with the new
embeddings; it does not spell out which loss term carries W's gradient.
Routing it through the existing regularization term is the smallest
formulation consistent with that description -- it introduces no new loss
term and no new hyperparameter. It is an implementation choice the paper
does not dictate, recorded here as such.

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

## 8. Table 1 hyperparameters not recovered

The paper states that "complete training configurations and hyperparameter
settings are presented in Table 1". Table 1 was not recoverable from the
paper text available to this reproduction, nor from the released artifacts:
the release has no config file, no logs and no checkpoints from which
optimiser settings could be read back.

Ten values in `configs/base.yaml` are therefore marked `source: unavailable`
-- LAPT learning rate, batch size, epochs and MLM probability, and the six
downstream fine-tuning settings. The values in place are conventional
defaults for XLM-R, **not the paper's**.

`scripts/aggregate_results.py` reads that marker and prefixes any generated
table with a "Not a replication" notice naming the count, so a produced
number can never be mistaken for a replication of the published Table 2.
Replacing those ten values with Table 1's is the single remaining
requirement for a faithful comparison.
