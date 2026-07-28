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

### FastText Embeddings

The LGSE framework initializes token embeddings using pretrained FastText
models to provide morphology-aware semantic representations.

- **Tigrinya:** Pretrained FastText model released with this project:
  https://huggingface.co/Hailay/fasttext-tigrinya
- **Amharic:** Official FastText Common Crawl vectors:
  https://fasttext.cc/docs/en/crawl-vectors.html

These pretrained models are used during vocabulary initialization to obtain
word-level embeddings. For out-of-vocabulary items, LGSE automatically falls
back to FastText's subword (character n-gram) representations, ensuring that
embeddings can still be generated for unseen words without requiring explicit
vocabulary entries.

### 🔹 Amharic FastText

Download from Facebook FastText:
https://fasttext.cc/

Use:

```
cc.am.300.bin
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
applies a **learned linear projection W ∈ R^(d×d)** (paper Sec 4.1). W is
square, so FastText must be trained at the model's embedding width (768 for
XLM-R base) — `build_projection` refuses a rectangular map rather than
silently changing the method. W is identity-initialized, registered with the
optimizer, and saved with the checkpoint.

> **Note.** Under the paper's stated objectives W receives no gradient during
> LAPT: `L_reg` anchors to a constant μ (Sec 4.2) and the MLM loss reads the
> embedding matrix, not W. This implementation follows the paper and reports
> W's gradient status each epoch rather than adding an unstated loss term.
> See `DEVIATIONS.md` §1a.

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
