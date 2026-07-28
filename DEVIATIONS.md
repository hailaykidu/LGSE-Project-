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
