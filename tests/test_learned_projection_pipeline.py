"""
Regression tests for the FastText + learned projection + initializer path.

Two integration failures reached the full pipeline unnoticed because every
existing test exercised these components separately:

  1. The learned projection's parameters were not passed to the optimizer,
     so W was initialized and then frozen -- `projection: learned` behaved
     identically to `random` while reporting as learned.

  2. `word_from_morphemes` averaged with `np.mean`, which cannot consume a
     grad-tracking tensor: "Can't call numpy() on Tensor that requires
     grad". The paper's method crashed as soon as it met real FastText
     vectors.

Both are cheap to assert and neither is visible unless FastText, the
projection and the initializer are wired together, so they are tested
together here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lgse.initializer import LGSEInitializer          # noqa: E402
from lgse.morpheme_embeddings import MorphemeEmbeddingBuilder  # noqa: E402
from lgse.projection import build_projection          # noqa: E402


class StubFastText:
    """Deterministic stand-in with FastText's interface and dimension."""

    def __init__(self, dim=300):
        self.dim = dim

    def get_dimension(self):
        return self.dim

    def get_word_vector(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return rng.normal(size=self.dim).astype(np.float32)


class StubSegmenter:
    def segment(self, word):
        return [word[:2], word[2:]] if len(word) > 2 else [word]


class StubCharEncoder:
    def __init__(self, dim):
        self.dim = dim

    def encode(self, token):
        return torch.zeros(self.dim)


def test_learned_projection_keeps_gradients():
    """Morpheme averaging must not drop W from the autograd graph."""
    projection = build_projection("learned", 300, 768, seed=42)
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=768, projection=projection)

    vec = builder.build_embedding_for_token("ኣይመፀን")

    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (768,)
    assert vec.requires_grad, "learned W must stay on the autograd graph"


def test_random_projection_is_not_trainable():
    """The released fixed projection must remain frozen."""
    projection = build_projection("random", 300, 768, seed=42)
    assert not projection.is_learned
    assert sum(p.numel() for p in projection.parameters()
               if p.requires_grad) == 0

    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=768, projection=projection)
    vec = builder.build_embedding_for_token("ኣይመፀን")
    assert not getattr(vec, "requires_grad", False)


def test_learned_projection_has_trainable_parameters():
    projection = build_projection("learned", 300, 768, seed=42)
    assert projection.is_learned
    trainable = sum(p.numel() for p in projection.parameters()
                    if p.requires_grad)
    assert trainable == 300 * 768


def test_gradient_reaches_projection_weights():
    """A loss on the initialized embedding must produce a gradient on W.

    This is the assertion that would have caught the optimizer bug: if W is
    detached anywhere in the chain, .grad stays None.
    """
    projection = build_projection("learned", 300, 768, seed=42)
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=768, projection=projection)

    vec = builder.build_embedding_for_token("ኣይመፀን")
    vec.sum().backward()

    weight = projection.linear.weight
    assert weight.grad is not None, "no gradient reached W"
    assert torch.any(weight.grad != 0), "gradient reached W but is all zero"


def test_initializer_writes_rows_without_breaking_weight_tying():
    """Init vectors are written in place, so a tied output head survives."""
    embedding = torch.nn.Embedding(120, 768)
    original = embedding.weight
    projection = build_projection("learned", 300, 768, seed=42)
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=768, projection=projection)
    initializer = LGSEInitializer(
        embedding_layer=embedding, morph_builder=builder,
        char_encoder=StubCharEncoder(768))

    token_to_id = {"ኣይመፀን": 100, "ሰላማዊ": 101}
    init_matrix = initializer.write_embeddings_for_new_tokens(token_to_id)

    assert init_matrix.shape == (2, 768)
    assert embedding.weight is original, "Parameter was replaced, breaking tying"
    for token, idx in token_to_id.items():
        assert torch.allclose(embedding.weight.data[idx],
                              init_matrix[list(token_to_id).index(token)])


@pytest.mark.parametrize("kind", ["learned", "random"])
def test_projection_is_deterministic_under_seed(kind):
    a = build_projection(kind, 300, 768, seed=7)
    b = build_projection(kind, 300, 768, seed=7)
    x = torch.randn(4, 300)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))
