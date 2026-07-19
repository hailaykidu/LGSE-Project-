# LGSE: Lexically Grounded Subword Embedding Initialization for Amharic & Tigrinya

This repository implements **LGSE (Lexically Grounded Subword Embedding
Initialization)** for **Amharic** and **Tigrinya**, two morphologically rich
Semitic languages. LGSE improves vocabulary expansion for multilingual
pretrained models (e.g. XLM-R) by initializing new vocabulary items with
**morpheme-aware embeddings** and **FastText subword vectors**, instead of
random vectors or arbitrary subword-derived initialization.

## Method (and how the code implements each part)

1. **New tokens get morpheme-averaged embeddings, not random vectors.**
   `MorphologicalSegmenter` (`lgse/segmentation.py`) looks up a word's
   morpheme decomposition from `data/morph_lexicon.txt`; `MorphemeEmbeddingBuilder`
   (`lgse/morpheme_embeddings.py`) averages FastText vectors over those
   morphemes.
2. **When a token can't be segmented, fall back to whole-token FastText,
   then character n-grams** (`lgse/char_ngrams.py`) — never a random
   vector. `LGSEInitializer` (`lgse/initializer.py`) is what chains these
   three steps together in order.
3. **A regularization term anchors new embeddings to their init values**
   during training (`LGSERegularizer`, `lgse/regularization.py`),
   preventing them from drifting arbitrarily far from the
   linguistically-informed starting point.
4. **Only the new-token embeddings are updated during LAP** — the rest of
   the model (and the old-vocabulary embedding rows) stays frozen, to
   isolate the effect of the initialization strategy. `LGSELAPTrainer`
   (`lgse/lap_trainer.py`) freezes every parameter except the embedding
   matrix, then zeros the gradient on old-vocabulary rows every step so
   only new-token rows actually move.

**Note on a prior version of this code:** an earlier draft defined all of
the classes above but never actually connected them — `LGSELAPTrainer`
didn't exist (the entry point script imported a class with no
definition anywhere), the two fallback/regularization mechanisms were
either unreachable or never called, and the training loop was a literal
placeholder (`loss = torch.zeros(1); loss.backward()`) with no real model,
corpus, or MLM objective involved. Everything above was fixed, wired
together, covered by real tests (`tests/`, previously all `assert True`
placeholders), and verified end-to-end against a real model
(`xlm-roberta-base`) and a real FastText model before being written up here.

`lgse_tokenizers/` (SentencePiece / vocab-expansion helpers) is unrelated
to the LAP pipeline above and isn't used by it; it's named `lgse_tokenizers`
rather than `tokenizers` because a bare `tokenizers/` directory here would
shadow the real PyPI `tokenizers` package that `transformers` depends on.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/run_lgse_lap.py --language ti --corpus_file /path/to/tigrinya_corpus.txt
```

`--corpus_file` should be a plain-text file, one sentence per line. Without
it, the script runs on a 2-sentence placeholder just to smoke-test that the
pipeline runs end to end — nowhere near enough data for a real experiment.
A real, already-cleaned Amharic corpus is available in this user's
`GeezTokenizer` project at
`../../GeezTokenizer/02_cleaning/corpus_clean/amharic.txt` (12.19M lines) if
you want real-scale Amharic LAP data without re-collecting anything.

## FastText Models

**The `.bin` files currently in `data/` are placeholders, not real FastText
models** (208 bytes each — a real FastText model is hundreds of MB to
several GB). Replace them before running anything for real:

- **Amharic**: download from [fasttext.cc](https://fasttext.cc/) —
  `cc.am.300.bin` → `data/fasttext_Amharic.bin`
- **Tigrinya**: [huggingface.co/Hailay/fasttext-tigrinya](https://huggingface.co/Hailay/fasttext-tigrinya/blob/main/fasttext_tigrinya.bin),
  or reuse the real 2.78GB one already on this machine at
  `../MPETokenization/Paralleldata/Focus/fasttext_tigrinya.bin` (point
  `fasttext_tigrinya_path` in your config at it directly rather than
  copying 2.78GB).

FastText vectors are 300-dim by default, which will not match most target
models' embedding width (768 for `xlm-roberta-base`, for instance) --
`MorphemeEmbeddingBuilder` applies a fixed, seeded random projection to
bridge the two spaces automatically when the widths differ (see
`lgse/morpheme_embeddings.py`); this was a real crash the first time this
code was actually run end to end with a real FastText model, not a
theoretical concern.

## Testing

```bash
pytest tests/ -v
```

17 tests, covering the real lexicon-parsing format (verified against
`data/morph_lexicon.txt`'s actual mixed format, not an assumed one), the
morpheme/whole-token/char-n-gram fallback chain, the regularizer's math, and
that writing new-token embeddings doesn't break weight tying with a tied MLM
output head.

## Notes

- Works best with large lexicons (Amharic: 2000 entries, Tigrinya: 2000+).
  The current `data/morph_lexicon.txt` has 210 usable entries after parsing
  (226 raw lines minus comments/headers/section markers) -- enough to
  exercise the pipeline, but smaller than what's recommended for a real
  experiment.
- FastText models are essential for the morpheme/whole-token fallback tiers;
  without them (or with the placeholder files still in place), every new
  token falls through to the character n-gram encoder, which still produces
  valid embeddings but loses the "lexically grounded" part of the method.

## License

This project follows the license of the original LGSE/FOCUS implementation.
FastText models follow their respective licenses.
