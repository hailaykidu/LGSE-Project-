# Pipeline validation

A record of the complete LGSE path running end to end: tokenizer expansion,
initialization, regularization, backbone freezing, checkpointing, and
downstream fine-tuning. Its scores are not results (see *Scores* below).

The run recorded here did not use a supplied W. W has no default in this
implementation, so the command below requires `lgse.alignment_matrix_path`
to be set.

## Run

    python src/training/run_experiment.py \
        --system lgse_lapt --task ner --language tigrinya --seed 42 \
        --data-dir <tigrinya NER split> --corpus <LAPT corpus>
    # requires lgse.alignment_matrix_path to be set in the config

Environment: python 3.13, torch 2.5.1+cu118, transformers 4.51.3,
fasttext 0.9.3, CPU.

## Stages confirmed

| Stage | Evidence from the run log |
|---|---|
| FastText loading | resolved via `data/fasttext_manifest.json` |
| Morphological lexicon | `Loaded morphological lexicon: 210 words` |
| Vocabulary expansion | `Added 198/198 new tokens to the tokenizer` |
| Alignment matrix W | `[LGSELAPTrainer] alignment matrix: W 768x768 from <supplied path> (frozen)` |
| W training status | the trainer logs W as externally supplied and frozen |
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

F1 is 0.00 in this validation. The run uses a 60-line LAPT corpus and ~20 NER
sentences for one epoch to exercise the code path, so the score is expected
and is not a result. Meaningful numbers require the full corpora and the
hyperparameters the paper places in Table 1 (see `IMPLEMENTATION_NOTES.md`
section 8).

## Integration properties this validation covers

These properties involve more than one component:

1. **W is not in the optimizer.** W is an externally supplied, frozen
   alignment matrix — see `IMPLEMENTATION_NOTES.md` §1a.

2. **The projection runs on real vectors.** `word_from_morphemes` averages in
   the tensor's own framework rather than with `np.mean`, which cannot consume
   a grad-tracking tensor (`RuntimeError: Can't call numpy() on Tensor that
   requires grad`). See `src/lgse/morpheme_embeddings.py`.

3. **W has no gradient path even once it is in the optimizer.** The
   initializer writes W's output into the embedding via `.data`, severing the
   graph; the regularizer anchors to a constant μ. No LAPT loss term is a
   function of W, so it receives `grad = None` every step and never moves.

   The implementation reports W's gradient status each epoch. See
   `IMPLEMENTATION_NOTES.md` §1a.

4. **W is preserved at checkpoint time.** `save()` serializes `projection.pt`
   alongside the model and tokenizer (`LGSELAPTrainer.save` /
   `load_projection`), so a resumed run does not restart from a fresh
   initialization.

`tests/test_learned_projection_pipeline.py` covers all four; 28 tests pass.
`test_paper_objectives_give_w_no_gradient` asserts W's gradient status.

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

`dirty` records whether the working tree had uncommitted changes; the
generated table reports it.

`scripts/aggregate_results.py` surfaces the commit and config hash behind any
table it generates, and warns when runs span more than one commit or when any
was made with a dirty tree.

## Dataset manifest checks

A missing dataset manifest is never silent. Three cases, all verified:

| Case | Behaviour |
|---|---|
| `--require-manifest` and no manifest | the run raises, naming the directory and the script that writes one |
| no manifest, flag not passed | the run proceeds; the table renders **manifest unavailable -- splits not traceable** and a warning naming the affected language/task |
| some runs documented, others not | the table shows the source and split seed, followed by **N of M runs had no manifest** |

`--require-manifest` requires every run to carry a manifest recording its
source, checksum and split seed.
