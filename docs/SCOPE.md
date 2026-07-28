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

## The paper's method files are unchanged

Byte-identical to the published release, relocated only:

    segmentation.py  char_ngrams.py  regularization.py
    token_selection.py  initializer.py

Two files changed, both additively:

**`morpheme_embeddings.py`** takes an optional `projection` argument. With
`projection=None` -- the default -- it reproduces the release's fixed
Johnson-Lindenstrauss map exactly; verified numerically against an
independent recomputation of the release's own formula. The learned
projection is opt-in via config.

**`lap_trainer.py`** reads `system`/`initializer`/`projection` from the
config and dispatches accordingly. With the default config it follows the
release's LGSE path.

## What is added, and where it lives

| Addition | Location | Touches the method? |
|---|---|---|
| Learned projection W | `src/lgse/projection.py` | Implements the paper's stated W; the release's random map remains available and is the fallback |
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
