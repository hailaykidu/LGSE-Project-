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
│   ├── fasttext_Amharic.bin       # placeholder until replaced -- see FastText Models
│   └── fasttext_Tigriyna.bin      # placeholder until replaced -- see FastText Models
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

> **The `.bin` files currently committed in `data/` are placeholders, not
> real FastText models** (208 bytes each -- a real FastText model is
> hundreds of MB to several GB). Replace them with the real downloads
> below before running anything for a real experiment; until then, every
> new token falls through to the character n-gram fallback tier (still
> produces valid embeddings, just loses the FastText/morpheme grounding).

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

An earlier draft of this code defined every class described above
(morpheme embeddings, character n-gram fallback, regularization, frozen-
backbone training) but never actually connected them: `LGSELAPTrainer` was
imported by the entry-point script but never defined anywhere, the
character n-gram fallback was unreachable dead code, `LGSERegularizer` was
never called, and the "training loop" was a literal placeholder
(`loss = torch.zeros(1)`) optimizing the whole model rather than just the
new embeddings. All of that has since been fixed, covered by 17 real tests
(`pytest tests/ -v`, replacing four `assert True` placeholders), and
verified end-to-end against real `xlm-roberta-base` and a real Tigrinya
FastText model: loss is finite and decreases, old-vocabulary embeddings
stay frozen throughout training, and new-token embeddings only move when
their tokens actually appear in the training text.

One additional fix worth knowing about if you extend this code: FastText
vectors are 300-dim by default, which won't match most target models'
embedding width (768 for `xlm-roberta-base`). `MorphemeEmbeddingBuilder`
now applies a fixed, seeded random projection to bridge the two spaces
automatically when the widths differ -- this was a real crash on the
first end-to-end run with a real FastText model, not a theoretical
concern.

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
