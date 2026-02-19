# LGSE: Linguistically‑Guided Subword Embeddings for Amharic & Tigrinya

This repository provides an implementation of **LGSE (Linguistically‑Guided Subword Embeddings)** adapted for **Amharic** and **Tigrinya**, two morphologically rich Semitic languages.  
LGSE improves multilingual pretrained models (e.g., XLM‑R) by initializing new vocabulary items using **morpheme‑aware embeddings** and **FastText subword vectors**.

This project includes:

- A combined **Amharic + Tigrinya morphological lexicon**
- A **new token list** for vocabulary expansion
- **FastText embeddings** for both languages
- A complete **LGSE Language‑Adaptive Pretraining (LAP)** pipeline

---

## 🚀 1. Project Structure

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
│   ├── init.py
│   ├── spm_utils.py
│   └── vocab_expansion.py
│
└── scripts/
├── run_lgse_lap.py
└── analyze_tokens.py

---

## 📦 2. Installation
Install dependencies:

```bash
pip install -r requirements.txt
Key libraries:

transformers
fasttext
torch
numpy
3. FastText Models
Amharic FastText
Download from Facebook FastText:
https://fasttext.cc/ Use:cc.am.300.bin → fasttext_amharic.bin
Tigrinya FastText
Download from HuggingFace:
https://huggingface.co/Hailay/fasttext-tigrinya/blob/main/fasttext_tigrinya.bin

Notes
LGSE significantly improves representation quality for morphologically rich languages.
Works best with large lexicons (Amharic: 2000 entries, Tigrinya: 2000+).
FastText models are essential for fallback embeddings.

License
This project follows the license of the original LGSE/FOCUS implementation.
FastText models follow their respective licenses.