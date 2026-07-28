# Deviations from the paper and the original release

Every difference between this reproduction branch, the published release
(`published-release-77987c5`), and the paper is recorded here. Nothing in this
list is a claim about which is correct -- only about what differs.

## 1. Projection W: learned (paper) vs fixed random (release)

**Paper:** W is learned, trained jointly with the new token embeddings.

**Original release:** `lgse/morpheme_embeddings.py:24-31` builds a fixed,
seeded Johnson-Lindenstrauss projection with `np.random.default_rng`. It is
never a `torch.nn.Parameter`, never passed to an optimizer, and never
updated. The code comment states the rationale: bridging 300-dim FastText to
768-dim XLM-R "without requiring extra training data to fit a learned
projection."

**This branch:** `src/lgse/projection.py` provides both.
`LearnedProjection` implements the paper (230,400 trainable parameters for
300->768) and is the default; `RandomProjection` reproduces the release
behaviour (0 trainable parameters). Both start from the same scaled-Gaussian
distribution, so a difference in results is attributable to W being learned
rather than to a different initialization. The choice is recorded in each run
config.

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

**This branch:** documented download and preparation scripts under
`data/scripts/`, with checksums. No large binaries are committed.

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

## 6. Single seed

The release sets `seed=42` in one place with no CLI override, so
mean +/- standard deviation over multiple runs cannot be produced.

**This branch:** seed is a config field and CLI argument.
