# Pipeline validation

Evidence that the complete LGSE path runs end to end. None of the stages
below are exercised together by unit tests, which cover these components in
isolation — this run is retained as the check that they compose correctly.

> **This validation run is not evidence of a faithful reproduction under the
> paper's method.** Under the current implementation, W has no default and
> must be an author-supplied artifact set via `lgse.alignment_matrix_path`;
> the command below requires this to be configured in the run config, or the
> run refuses to start (see `IMPLEMENTATION_NOTES.md` §1a-i).
>
> The run is retained because the stages it confirms — tokenizer expansion,
> initialization, regularization, backbone freezing, checkpointing,
> downstream fine-tuning — are unaffected by where W comes from. It is
> **not** evidence of a faithful reproduction, and its scores were never
> results (see *Scores* below).

## Run

    python src/training/run_experiment.py \
        --system lgse_lapt --task ner --language tigrinya --seed 42 \
        --data-dir <tigrinya NER split> --corpus <LAPT corpus>
    # requires lgse.alignment_matrix_path in the config

Environment: python 3.13, torch 2.5.1+cu118, transformers 4.51.3,
fasttext 0.9.3, CPU.

## Stages confirmed

| Stage | Evidence from the run log |
|---|---|
| FastText loading | resolved via `data/fasttext_manifest.json` |
| Morphological lexicon | `Loaded morphological lexicon: 210 words` |
| Vocabulary expansion | `Added 198/198 new tokens to the tokenizer` |
| Alignment matrix W | `[LGSELAPTrainer] alignment matrix: W 768x768 from <supplied path> (frozen)` |
| W training status | `[LGSELAPTrainer] W training status: author-required / unspecified in paper -- W is frozen` |
| Morpheme composition | init vectors written for all 198 tokens |
| LGSE regularization | `avg loss this epoch: 9.8771 (mlm=9.8710 reg=0.0000)` |
| Frozen backbone | only the embedding matrix and W receive gradients |
| Checkpoint saved | `Saved LGSE-specialized model to .../lapt` |
| Downstream fine-tune | loaded the LAPT checkpoint, trained, evaluated |
| Record written | `experiment.json` with system, seed, projection, scores |

The `xlmr` baseline was validated separately and correctly skips LAPT,
returning the unmodified backbone.

`reg=0.0000` on the last batch is expected here: with `reg_lambda=1.0` and a
single LAPT epoch over a 60-line smoke corpus, the new embeddings have barely
moved from their initialization, which is exactly what the regularizer
penalises drift from.

## Scores

F1 is 0.00 in this validation. That is expected and is **not** a result: the
run uses a 60-line LAPT corpus and ~20 NER sentences for one epoch, purely to
exercise the code path. No conclusion about LGSE follows from it. Real numbers
require the full corpora and the hyperparameters the paper places in Table 1
(see `IMPLEMENTATION_NOTES.md` section 8).

## Implementation behaviors confirmed by this run

None of the following would be visible to unit tests that exercise these
components separately:

1. **W is not included in the optimizer.** No objective stated in the paper
   is a function of W, so an optimizer entry for it would advertise a
   capability the run does not have. W is implemented as an externally
   supplied, frozen alignment matrix — see `IMPLEMENTATION_NOTES.md` §1a.

2. **Morpheme embeddings are averaged in the tensor's own framework, not
   with NumPy.** `word_from_morphemes` (`src/lgse/morpheme_embeddings.py`)
   averages using the tensor framework directly, since a grad-tracking
   tensor cannot be passed to `np.mean`: doing so raises `RuntimeError:
   Can't call numpy() on Tensor that requires grad`.

3. **W has no gradient path under the paper's stated objectives.** The
   initializer writes W's output into the embedding matrix via `.data`,
   which severs the autograd graph, and the regularizer anchors to a
   constant μ. No LAPT loss term is a function of W, so it receives
   `grad = None` every step and never moves. This follows directly from the
   paper's own equations (arXiv:2603.22629), which call W "learned" while
   specifying no objective that depends on it; the implementation reports
   W's gradient status each epoch rather than introducing an unstated
   training signal. See `IMPLEMENTATION_NOTES.md` §1a.

4. **W is serialized alongside the model checkpoint.** `projection.pt`
   (written by `LGSELAPTrainer.save`, restored by `load_projection`) travels
   with the model and tokenizer, so a resumed run restores the exact W a
   result used rather than reinitializing. This matters for any run that
   does train W, and costs nothing for runs that do not.

`tests/test_learned_projection_pipeline.py` covers all four; 28 tests pass.
`test_paper_objectives_give_w_no_gradient` is the load-bearing one: it
asserts the documented gap and fails loudly if a future change adds a loss
term the paper does not describe.

## Provenance recorded per run

Every `experiment.json` carries what is needed to re-run it:

| Field | Purpose |
|---|---|
| `commit`, `branch`, `dirty` | which code produced the result |
| `config_file`, `config_sha256` | which configuration, verifiable |
| `dataset_manifest` | dataset source URL, raw sha256, split seed, counts |
| `lapt_corpus`, `lapt_corpus_sha256` | which adaptation corpus |
| `fasttext` | model source, dimension, vocab size, sha256 per language |
| `python`, `packages` | torch / transformers / fasttext / numpy versions |
| `unavailable_hyperparameters` | count still marked `source: unavailable` |

`dirty` is recorded rather than assumed false: a run made with uncommitted
changes cannot be recovered from its commit hash alone, and a table built
from such runs says so.

`scripts/aggregate_results.py` surfaces the commit and config hash behind any
table it generates, and warns when runs span more than one commit or when any
was made with a dirty tree.

## Dataset manifest checks

A missing dataset manifest is never silent. Three cases, all verified:

| Case | Behaviour |
|---|---|
| `--require-manifest` and no manifest | the run refuses to start, naming the directory and the script that writes one |
| no manifest, flag not passed | the run proceeds; the table renders **manifest unavailable -- splits not traceable** and a warning naming the affected language/task |
| some runs documented, others not | the table shows the source and split seed, followed by **N of M runs had no manifest** |

Use `--require-manifest` for official experiment runs so no reported figure
can rest on splits that cannot be traced to a source, checksum and seed.
