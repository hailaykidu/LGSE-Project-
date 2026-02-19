# LGSE: Linguistically-Guided Subword Embeddings for Amharic & Tigrinya

### Official Code Release — Accepted at LREC-COLING 2026

This repository accompanies our paper accepted for publication at **LREC-COLING 2026**.

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
│   ├── fasttext_amharic.bin
│   └── fasttext_tigrinya.bin
│
├── lgse/
│   ├── config.py
│   ├── lap_trainer.py
│   ├── morph_lexicon.py
│   ├── morpheme_embeddings.py
│   ├── initializer.py
│   ├── regularization.py
│   ├── char_ngrams.py
│   └── token_selection.py
│
├── tokenizers/
│   ├── __init__.py
│   ├── spm_utils.py
│   └── vocab_expansion.py
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

Example:

```bash
python scripts/run_lgse_lap.py \
  --base_model xlm-roberta-base \
  --language tigrinya \
  --morph_lexicon data/morph_lexicon.txt \
  --new_tokens data/new_tokens.txt \
  --fasttext_model data/fasttext_tigrinya.bin
```

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

**LREC-COLING 2026**

---

## 📄 License

* Code follows the license of the original LGSE / FOCUS implementation.
* FastText models follow their respective licenses.
* Morphological lexicons are released for research use.

---

## 🤝 Citation

If you use this code or lexicon, please cite our LREC-COLING 2026 paper.

(Citation details will be updated after publication.)

---

## 🔮 Future Extensions

* Hybrid neural + rule-based morphological analyzer
* LGSE + LoRA integration
* Joint tokenization learning
* Automatic morpheme discovery
* Open benchmark for Ethio-Semitic NLP

---

## 👩‍🔬 Authors

This work was conducted as part of ongoing research on multilingual NLP and low-resource languages at the L3S research center in Germany.

---

## ⭐ Acknowledgements

We thank the open-source NLP community and contributors to multilingual pretrained models and low-resource language research.

---

## 📬 Contact

For questions, collaborations, or contributions, please open an issue or contact the authors.

---
