import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lgse.morpheme_embeddings import MorphemeEmbeddingBuilder
from lgse.segmentation import MorphologicalSegmenter


class FakeFastText:
    """Minimal stand-in for a real FastText model: fixed vectors for known
    strings, raises for anything else (mirrors how a real model would only
    have coverage for strings it was trained on)."""

    def __init__(self, vectors: dict, dim: int = 4):
        self.vectors = vectors
        self.dim = dim

    def get_dimension(self):
        return self.dim

    def get_word_vector(self, text):
        if text in self.vectors:
            return np.array(self.vectors[text], dtype=np.float32)
        raise KeyError(text)


def test_word_from_morphemes_averages_known_vectors():
    ft = FakeFastText({"ሰላም": [1.0, 0.0, 0.0, 0.0], "ኣዊ": [0.0, 1.0, 0.0, 0.0]})
    builder = MorphemeEmbeddingBuilder(fasttext_model=ft, segmenter=None, embedding_dim=4)
    v = builder.word_from_morphemes(["ሰላም", "ኣዊ"])
    np.testing.assert_allclose(v, [0.5, 0.5, 0.0, 0.0])


def test_word_from_morphemes_returns_none_when_no_coverage():
    ft = FakeFastText({})
    builder = MorphemeEmbeddingBuilder(fasttext_model=ft, segmenter=None, embedding_dim=4)
    assert builder.word_from_morphemes(["unknown1", "unknown2"]) is None


def test_word_from_morphemes_never_returns_a_random_vector():
    # A previous version silently substituted a random vector when nothing
    # matched; the fix returns None so the caller (LGSEInitializer) can
    # fall through to the character n-gram encoder instead, matching the
    # paper's described fallback chain.
    ft = FakeFastText({})
    builder = MorphemeEmbeddingBuilder(fasttext_model=ft, segmenter=None, embedding_dim=4)
    for _ in range(5):
        assert builder.word_from_morphemes(["nope"]) is None


def test_build_embedding_for_token_uses_segmenter_then_falls_back_to_whole_token():
    seg = MorphologicalSegmenter({"ሰላማዊ": ["ሰላም", "ኣዊ"]})
    ft = FakeFastText({
        "ሰላም": [1.0, 0.0, 0.0, 0.0],
        "ኣዊ": [0.0, 1.0, 0.0, 0.0],
        "ኮምፒተር": [0.0, 0.0, 1.0, 0.0],  # whole-token vector for an unsegmentable word
    })
    builder = MorphemeEmbeddingBuilder(fasttext_model=ft, segmenter=seg, embedding_dim=4)

    # Known word: averages its morpheme vectors.
    v1 = builder.build_embedding_for_token("ሰላማዊ")
    np.testing.assert_allclose(v1, [0.5, 0.5, 0.0, 0.0])

    # Unsegmentable word, but FastText has a whole-token vector: use that.
    v2 = builder.build_embedding_for_token("ኮምፒተር")
    np.testing.assert_allclose(v2, [0.0, 0.0, 1.0, 0.0])

    # Unsegmentable word with zero FastText coverage: None, not a random
    # vector -- caller must supply the character n-gram fallback.
    v3 = builder.build_embedding_for_token("completely_unknown")
    assert v3 is None
