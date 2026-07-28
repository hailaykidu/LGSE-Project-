# LGSE: Linguistically-Guided Subword Embeddings for Amharic & Tigrinya

### Official Code Release — Accepted at LREC 2026

This repository accompanies our paper accepted for publication at **LREC 2026**.

It provides a complete implementation of **LGSE (Linguistically-Guided Subword Embedding Initialization)** adapted for **Amharic** and **Tigrinya**, two morphologically rich Ethio-Semitic languages.

LGSE improves multilingual pretrained language models (e.g., Hugging Face Transformers such as XLM-RoBERTa) by initializing newly introduced vocabulary tokens using:

* Morpheme-aware decomposition
* FastText subword representations
* Character n-gram fallback embeddings
* Regularized Language-Adaptive Pretraining (LAP)

---

## ⚠️ Prerequisites you must supply

Three artifacts cannot be derived from the paper. **The implementation fails
rather than substituting a value for any of them** — explicit incompleteness
is preferred to a silent choice the authors never described.

| Required | Where | Why it has no default |
|---|---|---|
| **Alignment matrix W** | `lgse.alignment_matrix_path` | Sec 4.1 introduces W but never says how it is obtained |
| **Regularization strength λ** | `lgse.reg_lambda` | Sec 4.2 introduces λ but never assigns it |
| **FastText at dim 768** | `data/fasttext_manifest.json` | W is square, so FastText must match the model's width |

> **Any result produced without an author-provided W is not faithful to the
> published method.** It may still be a useful experiment, but it is one run
> under a documented substitution — and the substitution is yours, recorded
> in the run record.

A run either satisfies all three prerequisites, or it reports itself as a
**partial reproduction / implementation validation**. The generated results
table states this per row, so fidelity is readable from the table itself:

| System | F1 | seeds | Alignment matrix W |
|---|---|---|---|
| XLM-R | … | 5 | n/a — no alignment matrix required |
| +FOCUS+LAPT | … | 5 | author-supplied W |
| +LGSE+LAPT | … | 5 | **not faithful — no author-supplied W** |

The baselines (`xlmr`, `lapt`, `random_lapt`) use no FastText, need no W,
and run normally — the prerequisite applies only to the LGSE/FOCUS systems
that consume FastText.

See `IMPLEMENTATION_NOTES.md` §1, §1a and §8a.

---

## 📌 Motivation

Adapting pretrained multilingual language models to **low-resource, morphologically rich languages** remains challenging.
Standard vocabulary expansion methods rely on arbitrary subword units, which fragment morphological structure and degrade semantic alignment.

LGSE addresses this by:

1. Decomposing words into linguistically meaningful morphemes.
2. Constructing semantically coherent embeddings via morpheme representation averaging.
3. Applying embedding regularization during LAP to preserve alignment with the original embedding space.

---

## 🚀 Project Structure

```
LGSE-Project/
│
├── data/
│   ├── morph_lexicon.txt
│   ├── new_tokens.txt
│   ├── fasttext_Amharic.bin       # https://fasttext.cc/docs/en/crawl-vectors.html /download  FastText Models
│   └── fasttext_Tigriyna.bin      # https://huggingface.co/Hailay/fasttext-tigrinya/ download FastText Models
│
├── lgse/
│   ├── config.py
│   ├── lap_trainer.py
│   ├── segmentation.py
│   ├── morpheme_embeddings.py
│   ├── initializer.py
│   ├── regularization.py
│   ├── char_ngrams.py
│   └── token_selection.py
│
├── lgse_tokenizers/                # unrelated helper utilities, not used by
│   ├── __init__.py                 # the LAP pipeline; see Implementation
│   ├── spm_utils.py                # Notes below for why it isn't named
│   └── vocab_expansion.py          # `tokenizers/`
│
└── scripts/
    ├── run_lgse_lap.py
    └── analyze_tokens.py
```

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies

* transformers
* torch
* fasttext
* numpy
* sentencepiece

---

## 🌍 FastText Models

LGSE uses FastText embeddings for morpheme and fallback representations.

### ⚠️ Required embedding dimension

> **FastText vectors must have the same dimension as the model's embedding
> space — 768 for `xlm-roberta-base`, the model used in the paper.**
>
> LGSE aligns the two spaces with a **square** learned projection
> `W ∈ R^(d×d)` (paper Sec 4.1). W performs an alignment, not a change of
> dimensionality, so both spaces must already be of width `d`.
>
> **The standard 300-dimensional FastText vectors cannot be used with a
> 768-dim model under the published method.** LGSE will refuse to start
> rather than reshape them.

Train FastText at the required width:

```bash
fasttext skipgram -input <corpus> -output fasttext_amharic_768 -dim 768
```

The dimension is checked in three places, earliest first, so a mismatch
never reaches training:

| Where | When |
|---|---|
| `data/scripts/download_fasttext.py` | at download — warns and exits non-zero |
| `LGSEConfig.fasttext_path` | at config resolution — before the model is loaded |
| `build_projection` / `MorphemeEmbeddingBuilder` | at construction — authoritative |

**Mismatched vectors are never truncated, zero-padded, or rectangularly
projected to fit.** Any of those would change the method into one the paper
does not describe, and would do so invisibly in the reported numbers. See
`IMPLEMENTATION_NOTES.md` §1.

### FastText Embeddings

The LGSE framework initializes token embeddings using pretrained FastText
models to provide morphology-aware semantic representations.

- **Tigrinya:** Pretrained FastText model released with this project:
  https://huggingface.co/Hailay/fasttext-tigrinya
- **Amharic:** Official FastText Common Crawl vectors:
  https://fasttext.cc/docs/en/crawl-vectors.html

Both sources above are **300-dimensional** and therefore require retraining
at dimension 768 before use with XLM-R — see the requirement above.

These pretrained models are used during vocabulary initialization to obtain
word-level embeddings. For out-of-vocabulary items, LGSE automatically falls
back to FastText's subword (character n-gram) representations, ensuring that
embeddings can still be generated for unseen words without requiring explicit
vocabulary entries.

### 🔹 Amharic FastText

Download from Facebook FastText:
https://fasttext.cc/

```
cc.am.300.bin        # 300-dim: NOT usable with 768-dim XLM-R as-is
```

Rename to:

```
fasttext_amharic.bin
```

---

### 🔹 Tigrinya FastText

Download from Hugging Face:
https://huggingface.co/Hailay/fasttext-tigrinya

Rename to:

```
fasttext_tigrinya.bin
```

Place both models inside:

```
data/
```

---

## 🔧 Running LGSE Language-Adaptive Pretraining (LAP)

```bash
python scripts/run_lgse_lap.py --language ti --corpus_file /path/to/tigrinya_corpus.txt
```

`--language` is `am` or `ti` (selects which FastText model + config
defaults to use). `--corpus_file` is a plain-text file, one sentence per
line; without it, the script runs on a 2-sentence placeholder just to
smoke-test that the pipeline runs end to end -- nowhere near enough data
for a real experiment. Model, lexicon, and FastText paths come from
`lgse.config.LGSEConfig` rather than separate CLI flags (see
`lgse/config.py`); pass a custom `LGSEConfig` in Python if you need to
override them for your setup.

---

## ✅ Implementation Notes

## Implementation Notes

This implementation provides a complete end-to-end realization of
the LGSE framework described in the paper. It integrates morpheme-based
embedding initialization, FastText representations, character n-gram
fallback embeddings, embedding regularization, and frozen-backbone
language-adaptive pretraining.

The implementation has been validated with XLM-RoBERTa and Tigrinya, Amharic 
FastText embeddings. During training, pretrained vocabulary embeddings
remain fixed while newly introduced vocabulary embeddings are optimized
through the LGSE-guided objective.

To align the FastText embedding space with the pretrained model's, LGSE
applies a linear projection **W ∈ R^(d×d)** (paper Sec 4.1). W is square, so
FastText must be trained at the model's embedding width (768 for XLM-R base)
— `build_projection` refuses a rectangular map rather than silently changing
the method.

### ⚠️ W must be supplied — there is no default

> The paper introduces W (Sec 4.1) but **never states how it is obtained**:
> no initialization, no fitting procedure, and no training objective
> anywhere in the paper is a function of W. `L_reg` anchors to a constant μ
> (Sec 4.2), and LAPT's MLM loss reads the embedding matrix, not W.
>
> W is therefore an **author-supplied artifact**. LGSE runs **fail** unless
> `lgse.alignment_matrix_path` points at a `.pt`/`.npy` file holding a `d×d`
> matrix.
>
> **No default is substituted — not even the identity.** The identity is not
> neutral: it asserts the FastText and model embedding spaces are already
> aligned, which is exactly the claim W exists to avoid making. A random or
> fitted W would be an alignment strategy the paper does not describe.
> Either would materially affect results while the run still looked
> faithful.
>
> **Any result produced without an author-provided W is not faithful to the
> published method.**
>
> W is frozen in all cases — no gradient path is manufactured for it.
> Resolving this requires the authors to state both how W is obtained and
> which objective, if any, trains it. See `IMPLEMENTATION_NOTES.md` §1a.

```yaml
lgse:
  alignment_matrix_path: path/to/W.pt   # required; d x d
```

W is saved with the checkpoint, and its source, `author_supplied` flag, and
training status are recorded in every run record. A *supplied* identity is
fine — that is the author's documented choice, and the trainer flags it.

### `reg_lambda` is mandatory

`lgse.reg_lambda` — λ in `L_reg = λ‖e_new − μ‖²` (Sec 4.2) — **has no
default and must be set explicitly.** The paper introduces λ but never
states its value, so any setting is the experimenter's choice; a silent
default would make every result look as if it followed a published one. Runs
fail with an explanatory error if it is absent. `configs/base.yaml` ships
`1.0`, carried over from the original release and marked
`source: unavailable`.

---

## 🧠 LGSE Pipeline Overview

1. **Token Selection**
   Identify new vocabulary items for expansion.

2. **Morphological Decomposition**
   Use Amharic + Tigrinya lexicon for segmentation.

3. **Embedding Initialization**

   * Morpheme averaging (FastText or pretrained subwords)
   * Character n-gram fallback

4. **Regularized LAP**

Loss formulation:

```
L_total = L_MLM + lambda * ||E_new − E_init||^2
```

5. **Evaluation**

   * Question Answering
   * Named Entity Recognition
   * Text Classification

---

## 📊 Experimental Findings

LGSE consistently:

* Outperforms random initialization
* Outperforms subword averaging baselines
* Preserves embedding space alignment
* Improves downstream performance in low-resource settings

Best improvements observed in:

* Morphologically productive suffixes
* Derivational morphology
* Negation constructions

---

## 📚 Supported Languages

* Amharic
* Tigrinya

Designed for extension to:

* Oromo
* Geez-script languages
* Other Semitic languages

---

## 🏛 Conference

This repository accompanies our paper accepted at:

**LREC 2026**

---

## 📄 License

* Code follows the license of the original LGSE / FOCUS implementation.
* FastText models follow their respective licenses.
* Morphological lexicons are released for research use.

---

## 🤝 Citation

If you use this code or lexicon, please cite our LREC 2026 paper.

 Teklehaymanot, H., Fazlija, D., & Nejdl, W. (2026). LGSE: Lexically Grounded Subword Embedding Initialization for Low-Resource Language Adaptation. arXiv preprint arXiv:2603.22629.

---

## 🔮 Future Extensions

* Hybrid neural + rule-based morphological analyzer
* LGSE + LoRA integration
* Joint tokenization learning
* Automatic morpheme discovery
* Open benchmark for Ethio-Semitic NLP

---

## 👩‍🔬 Authors
Teklehaymanot, H., Fazlija, D., & Nejdl, W. (2026). LGSE: Lexically Grounded Subword Embedding Initialization for Low-Resource Language Adaptation. arXiv preprint arXiv:2603.22629.

---

## ⭐ Acknowledgements

We thank the open-source NLP community and contributors to multilingual pretrained models and low-resource language research.

---

## 📬 Contact

For questions, collaborations, or contributions, please open an issue or contact the authors.

---
