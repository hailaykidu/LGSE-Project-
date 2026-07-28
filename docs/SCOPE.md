# Scope of this branch

The `reproduction` branch adds reproduction and engineering scaffolding. It
does not change the paper's methodology, datasets, experimental definitions
or claims. This file records where that boundary sits, with the evidence for
each statement.

## Original release preserved

| Ref | Commit | Moved? |
|---|---|---|
| `original-release-e33c97f` | `e33c97f` | no |
| `published-release-77987c5` | `77987c5` | no |
| local `main` | `e33c97f` | no |
| `origin/main` | `77987c5` | no |

All work is on `reproduction`. Neither tag nor either main branch has been
modified, and nothing has been pushed.

## Relationship to the published release

This branch implements the method the paper describes. Where the release
and the paper disagree, **the paper governs** -- the release's behaviour is
not preserved for its own sake.

Byte-identical to the published release, relocated only:

    segmentation.py  char_ngrams.py  token_selection.py  initializer.py

Changed, deliberately:

**`projection.py`** implements the paper's learned W as the only supported
projection. The release's fixed Johnson-Lindenstrauss map is not retained,
not selectable, and has no config key. See DEVIATIONS.md §1.

**`regularization.py`** implements the paper's `L_reg = λ‖e_new − μ‖²` with
μ a constant (Sec 4.2: "μ is the initial embedding vector"). It additionally
accepts a live anchor, unused by any configured run and retained only for
deliberate departures-from-paper experiments. See DEVIATIONS.md §1a.

**`morpheme_embeddings.py`** applies W to FastText vectors and averages in
the tensor's own framework so W stays on the autograd graph. A dimension
mismatch with no W raises rather than silently substituting an untrained map.

**`lap_trainer.py`** reads `system`/`initializer` from the config and
dispatches accordingly, builds W, registers it with the optimizer, and
serializes it with the checkpoint.

## What is added, and where it lives

| Addition | Location | Touches the method? |
|---|---|---|
| Projection W | `src/lgse/projection.py` | **Yes -- deliberately.** Implements the paper's square `d×d` W as the only projection; the release's rectangular random map is removed. Under the paper's own objectives W receives no gradient — documented, not worked around. See DEVIATIONS.md §1, §1a |
| Table 2 baselines | `src/baselines/` | New systems the paper compares against; LGSE itself untouched |
| Evaluation harness | `src/evaluation/` | Implements the metrics the paper reports; no metric redefined |
| Dataset preparation | `data/scripts/` | Fetches the datasets the paper names; splits follow the paper's stated policy |
| Provenance, manifests, warnings | `run_experiment.py`, `aggregate_results.py` | Reporting only -- no effect on architecture, objective, data or metrics |

## What is deliberately not done

* No preprocessing rule the paper does not describe. The SQuAD answer
  normalizer is not applied, because it strips English articles and would
  alter Ge'ez-script scoring.
* No metric redefined. NER is entity-level F1 (CoNLL); QA is SQuAD token-F1.
* No dataset substituted. Where a required artifact is missing it is recorded
  as missing rather than replaced -- see `DEVIATIONS.md`.
* No synthetic data. `download_fasttext.py` fails rather than generating
  placeholder vectors, which would silently reduce LGSE to its own
  character-n-gram fallback.
* No hyperparameter presented as the paper's. Ten remain marked
  `source: unavailable`, and any generated table carries a "Not a
  replication" notice while they do.

## Reading the results

The safeguards are transparency measures. They do not change what is
computed. A number produced here is a *reproduction under documented
substitutions* until Table 1's values are supplied; at that point the
notice clears automatically and the same pipeline produces a comparable
figure with no code change.
