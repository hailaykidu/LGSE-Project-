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

## 5. Single seed

The release sets `seed=42` in one place with no CLI override, so
mean +/- standard deviation over multiple runs cannot be produced.

**This branch:** seed is a config field and CLI argument.
