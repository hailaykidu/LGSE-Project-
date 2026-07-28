# Pipeline validation

Evidence that the complete LGSE path runs end to end, recorded because two
integration failures previously reached it unnoticed.

## Run

    python src/training/run_experiment.py \
        --system lgse_lapt --task ner --language tigrinya --seed 42 \
        --data-dir <tigrinya NER split> --corpus <LAPT corpus>

Environment: python 3.13, torch 2.5.1+cu118, transformers 4.51.3,
fasttext 0.9.3, CPU.

## Stages confirmed

| Stage | Evidence from the run log |
|---|---|
| FastText loading | resolved via `data/fasttext_manifest.json` |
| Morphological lexicon | `Loaded morphological lexicon: 210 words` |
| Vocabulary expansion | `Added 198/198 new tokens to the tokenizer` |
| Projection W | `[LGSELAPTrainer] learned projection W: 768 x 768 (589824 trainable parameters, identity-initialized)` |
| W gradient status | `[LGSELAPTrainer] projection W received gradient: False` — expected; see DEVIATIONS.md §1a |
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
(see `DEVIATIONS.md` section 8).

## Failures this validation caught

All were invisible to unit tests, which exercised these components
separately:

1. **The learned projection was never optimized.** Its parameters were not in
   the AdamW parameter group, so W was initialized and then frozen, behaving
   identically to a fixed map while reporting as learned. Fixed in
   `src/lgse/lap_trainer.py`.

2. **The learned projection crashed on real vectors.**
   `word_from_morphemes` averaged with `np.mean`, which cannot consume a
   grad-tracking tensor: `RuntimeError: Can't call numpy() on Tensor that
   requires grad`. Fixed in `src/lgse/morpheme_embeddings.py` by averaging in
   the tensor's own framework.

3. **W has no gradient path even once it is in the optimizer.** The
   initializer writes W's output into the embedding via `.data`, severing the
   graph; the regularizer anchors to a constant μ. No LAPT loss term is a
   function of W, so it receives `grad = None` every step and never moves.

   Reading the paper (arXiv:2603.22629) established this is **not a defect in
   this implementation** — it follows from the paper's own equations, which
   call W "learned" while specifying no objective that depends on it. The
   implementation now follows the paper and reports W's gradient status each
   epoch rather than inventing a training signal. See `DEVIATIONS.md` §1a.

4. **A trained W was discarded at checkpoint time.** `save()` wrote only the
   model and tokenizer, so a resumed run silently restarted from a fresh
   initialization. Fixed by serializing `projection.pt`
   (`LGSELAPTrainer.save` / `load_projection`). This matters for any run that
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
