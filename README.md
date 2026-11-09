# LGSE: Lexically Grounded Subword Embedding Initialization

Implementation of **LGSE: Lexically Grounded Subword Embedding
Initialization for Low-Resource Language Adaptation** (Teklehaymanot,
Fazlija & Nejdl, LREC 2026), for Amharic and Tigrinya.
Paper: https://aclanthology.org/2026.lrec-1.812/

The repository also contains a controlled comparison of LGSE against four
other initialization strategies: 165 runs, with every run record, the
training logs the exclusion rule reads, and the scripts that build every
table. Those measurements do **not** reproduce the ranking reported in the
LREC paper; see [Reproducing the tables](#reproducing-the-tables).

## Method

When the vocabulary of a pretrained encoder (here `xlm-roberta-base`) is
extended, each new token needs an embedding. LGSE builds it from the
token's morphemes:

1. segment the token with a morpheme lexicon (optionally HornMorpho);
2. average the morphemes' FastText vectors and map the result into the
   model's space with a square matrix `W`;
3. fall back to character n-grams when no segmentation exists;
4. during language-adaptive pretraining (LAPT), keep the new rows near
   their initial values with `L = L_MLM + λ·‖e_new − μ‖²`.

| Stage | Module |
|---|---|
| Token selection | `src/lgse/token_selection.py` |
| Segmentation | `src/lgse/segmentation.py` |
| Morpheme embeddings, projection `W` | `src/lgse/morpheme_embeddings.py`, `src/lgse/projection.py` |
| Character n-gram fallback | `src/lgse/char_ngrams.py` |
| Initialization | `src/lgse/initializer.py` |
| Regularization | `src/lgse/regularization.py` |
| LAPT | `src/lgse/lap_trainer.py` |
| Baselines (FOCUS, random) | `src/baselines/` |
| Task runners (TC, NER, QA) | `src/evaluation/` |

The paper defines `W` and `λ` but gives neither a construction nor a value.
This implementation uses orthogonal Procrustes on shared FastText/XLM-R
anchors for `W` (`scripts/build_alignment_matrix.py`; matrices and sidecars
in `data/alignment/`) and `λ = 1.0`. A run missing either stops with an
error rather than falling back to a default.

## Installation

```bash
pip install -r requirements.txt
python data/scripts/download_fasttext.py --language both
```

FastText models are 2–3 GB and are not committed. Because `W` is square,
the vectors must have the model's width (768); `data/scripts/train_fasttext_768.py`
trains them, and `data/fasttext_manifest.json` records the command and
checksum of each.

## Running

One run (LAPT followed by fine-tuning and evaluation):

```bash
python src/training/run_experiment.py \
    --system lgse_lapt --task ner --language amharic --seed 42 \
    --data-dir data/ner/amharic --corpus <LAPT corpus> \
    --config configs/hornmorpho.yaml
```

`--system` is one of `xlmr`, `lapt`, `random_lapt`, `focus_lapt`,
`lgse_lapt`; `--task` is `tc`, `ner` or `qa`. Use `configs/tigrinya.yaml`
for Tigrinya. `scripts/run_table2.sh` runs all systems and seeds for one
task and language. Each run writes an `experiment.json` with the commit,
configuration checksum, alignment matrix, corpus checksum and test metrics.

## Reproducing the tables

```bash
python3 scripts/build_table2.py --out tables   # every table and statistic
python3 scripts/build_table2.py --audit        # which runs are used, and why
python3 scripts/loss_rule.py                   # recompute the excluded runs from logs/
python3 scripts/token_share.py                 # share of test tokens on new rows
python3 -m pytest tests/ -q
```

No GPU is needed; the first command takes about ten seconds. It reads the
165 run records in `results_hornmorpho_wam/`, `results_hornmorpho/` (Amharic
`lapt` and `random_lapt` only) and `results_tigrinya_gen2/`, and the
exclusions in `tables/loss_rule.json`.

Nine of the 165 runs failed to fit their training data. They are identified
from training loss alone (`scripts/loss_rule.py`), never from test scores,
and `build_table2.py` prints each affected mean both with and without them.

The provenance of every data split is in its `manifest.json`.

## Data

Every split has a `manifest.json` with its source, checksum, cleaning counts
and split seed.

| Task | Language | Source | Split |
|---|---|---|---|
| NER | Amharic | MasakhaNER (Adelani et al., 2021) | official |
| NER | Tigrinya | Tigrinya NER (Yohannes & Amagasa, 2022) | 80/10/10, seed 42 |
| QA | Amharic | AmQA (Taffa et al., 2024) | as released |
| QA | Tigrinya | extractive conversion of TIGQA (Teklehaymanot et al., 2024) | seed 42 |
| TC | Amharic | annotated for this work | 1872/233/235 |
| TC | Tigrinya | FineWeb-C, `tir_Ethi` | 597/74/74 |

- MasakhaNER does not cover Tigrinya. The two NER datasets use different tag
  sets (Tigrinya adds `MISC`) and are not compared with each other.
- The TIGQA release reports about 2.68K questions; the converted file used
  here has 2,179, of which the 1,517 with an exactly matching answer span are
  kept. The difference is not reconciled, so Tigrinya QA results here are not
  comparable to published TIGQA numbers.
- The Amharic TC corpus was labelled by eleven annotators (one author and
  ten native-speaker volunteers) with a six-class rubric, one label per
  document, followed by a full adjudication pass. Labels were merged without
  annotator identifiers, so no inter-annotator agreement can be computed.

## Layout

```
configs/                base, Amharic (hornmorpho), Tigrinya, systems
data/                   splits and manifests, alignment matrices, token lists,
                        morpheme lexicon; data/scripts/ prepares each split
src/lgse/               the method
src/baselines/          FOCUS and random initialization
src/evaluation/         TC, NER and QA runners
src/training/           run_experiment.py
scripts/                build_table2.py, loss_rule.py, token_share.py,
                        build_alignment_matrix.py, run_table2.sh,
                        run_lgse_lap.py, aggregate_results.py
results_*/              one experiment.json per run
logs/                   the fine-tuning logs loss_rule.py reads
tables/                 generated tables and exclusions
tests/
```

## License

- Code follows the license of the original LGSE / FOCUS implementation.
- FastText models follow their respective licenses.
- Morphological lexicons are released for research use.

## Citation

```bibtex
@inproceedings{teklehaymanot-etal-2026-lgse,
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
