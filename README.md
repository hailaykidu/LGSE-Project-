# LGSE: Lexically Grounded Subword Embedding Initialization

### Official implementation — LREC 2026

This is the official repository for **LGSE: Lexically Grounded Subword
Embedding Initialization for Low-Resource Language Adaptation**
(Teklehaymanot, Fazlija & Nejdl, LREC 2026).
Paper: https://aclanthology.org/2026.lrec-1.812/

It implements the method described in the paper for **Amharic** and
**Tigrinya**, two morphologically rich Ethio-Semitic languages.

---

## Overview

Adapting pretrained multilingual language models to low-resource,
morphologically rich languages is limited by how new vocabulary is
initialized. Standard vocabulary expansion relies on arbitrary subword
units, which fragment morphological structure and degrade semantic
alignment.

LGSE initializes new token embeddings from linguistic structure rather than
from random vectors:

1. **Morphological decomposition** — words are segmented into meaningful
   morphemes using a supervised lexicon.
2. **Morpheme-averaged embeddings** — each new token's embedding is the
   average of its morphemes' FastText representations, aligned into the
   model's embedding space by a linear projection `W ∈ R^(d×d)` (Sec 4.1).
3. **Character n-gram fallback** — tokens with no usable segmentation fall
   back to character n-gram representations.
4. **Regularized LAPT** — during language-adaptive pretraining the encoder
   is frozen and only the new embeddings are updated, with

   ```
   L_total = L_MLM + λ · ‖e_new − μ‖²
   ```

   penalizing drift from the initialized values (Sec 4.2).

### Pipeline

| Stage | Module |
|---|---|
| Token selection | `src/lgse/token_selection.py` |
| Morphological segmentation | `src/lgse/segmentation.py` |
| Morpheme embeddings + projection W | `src/lgse/morpheme_embeddings.py`, `src/lgse/projection.py` |
| Character n-gram fallback | `src/lgse/char_ngrams.py` |
| Embedding initialization | `src/lgse/initializer.py` |
| Regularization | `src/lgse/regularization.py` |
| Language-adaptive pretraining | `src/lgse/lap_trainer.py` |
| Downstream evaluation | `src/evaluation/` |

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `transformers`, `fasttext`, `numpy`,
`sentencepiece`.

---

## Usage

### Requirements

Three artifacts are supplied per run:

| Requirement | Where | Published specification |
|---|---|---|
| Alignment matrix **W** | `lgse.alignment_matrix_path` | Sec 4.1 defines W; its construction is not specified |
| Regularization strength **λ** | `lgse.reg_lambda` | Sec 4.2 defines λ; no value is given |
| FastText at the model's width | `data/fasttext_manifest.json` | W is square (`d×d`), so FastText must match the model's embedding width |

A run missing any of these stops with an explanatory error. See
[`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md) §1, §1a and §8a.

### FastText models

FastText binaries are 2–3 GB and are not committed. Fetch them with:

```bash
python data/scripts/download_fasttext.py --language both
```

- **Amharic** — FastText Common Crawl vectors (Grave et al., 2018)
- **Tigrinya** — https://huggingface.co/Hailay/fasttext-tigrinya

Both are released at 300 dimensions. Because W is square, they must be
trained at the model's embedding width (768 for `xlm-roberta-base`) before
use:

```bash
fasttext skipgram -input <corpus> -output <model> -dim 768
```

Mismatched vectors are not truncated, padded, or rectangularly projected to
fit; the dimension is checked at download, at config resolution, and at
construction.

### Language-adaptive pretraining

```bash
python scripts/run_lgse_lap.py \
    --language ti \
    --corpus_file /path/to/tigrinya_corpus.txt
```

`--language` is `am` or `ti`. Model, lexicon, FastText and W paths come from
`configs/base.yaml` via `src/lgse/config.py`.

### Downstream evaluation

```bash
python src/training/run_experiment.py \
    --system lgse_lapt --task ner --language tigrinya --seed 42 \
    --data-dir <split> --corpus <LAPT corpus>

python scripts/aggregate_results.py        # build the results table
```

Systems: `xlmr`, `lapt`, `random_lapt`, `focus_lapt`, `lgse_lapt`. The
baselines use no FastText and need no alignment matrix.

---

## Repository structure

```
LGSE-Project/
├── configs/          base.yaml, systems.yaml
├── data/
│   ├── morph_lexicon.txt, new_tokens.txt
│   └── scripts/      download_fasttext.py, prepare_ner.py, prepare_qa.py
├── src/
│   ├── lgse/         the method (segmentation, projection, initializer,
│   │                 regularization, lap_trainer, ...)
│   ├── baselines/    comparison systems
│   ├── evaluation/   NER / QA metrics and runners
│   └── training/     run_experiment.py
├── scripts/          run_lgse_lap.py, run_table2.sh, aggregate_results.py
├── tests/
└── IMPLEMENTATION_NOTES.md
```

---

## Reproducibility

Results are produced by running the pipeline; none are committed to this
repository. `scripts/aggregate_results.py` builds a table from your own runs
in `results/`, recording for each figure the commit, configuration hash,
dataset manifest, seeds and environment that produced it, and which alignment
matrix the run used.

The paper's reported numbers are not copied into this repository; the tables
here are built from runs performed in this repository.

[`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md) records each point where
an implementation choice is supplied alongside the published specification,
and [`docs/VALIDATION.md`](docs/VALIDATION.md) records end-to-end pipeline
validation.

---

## Supported languages

Amharic and Tigrinya, with the method designed to extend to other
Ge'ez-script and Semitic languages where morphological segmentation
resources exist.

---

## License

- Code follows the license of the original LGSE / FOCUS implementation.
- FastText models follow their respective licenses.
- Morphological lexicons are released for research use.

---

## Citation

```bibtex
@@inproceedings{teklehaymanot-etal-2026-lgse,
  title = {LGSE: Lexically Grounded Subword Embedding Initialization for Low-Resource Language Adaptation},
  author = {Teklehaymanot, Hailay Kidu and Fazlija, Dren and Nejdl, Wolfgang},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {10341--10352},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/5i7acx7v348w},
   }
```
