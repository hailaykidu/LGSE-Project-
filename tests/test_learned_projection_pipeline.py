"""
Regression tests for the FastText + learned projection + initializer path.

LGSE's projection W is learned, and "learned" is a claim with three
separable failure modes -- each of which leaves a run that still trains,
still reports, and still looks correct, while W is in fact not being
learned. All three have occurred here:

  1. W's parameters were not passed to the optimizer, so it was initialized
     and then frozen -- indistinguishable from a fixed random map.

  2. `word_from_morphemes` averaged with `np.mean`, which cannot consume a
     grad-tracking tensor: "Can't call numpy() on Tensor that requires
     grad". The method crashed on real FastText vectors.

  3. `save()` wrote only the model and tokenizer, so a trained W was
     discarded at checkpoint time and a restored run silently began from a
     fresh initialization.

None of these are visible unless FastText, the projection and the
initializer are wired together, so they are tested together here.
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
    projection = build_projection(300, 768, seed=42)
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=768, projection=projection)

    vec = builder.build_embedding_for_token("ኣይመፀን")

    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (768,)
    assert vec.requires_grad, "learned W must stay on the autograd graph"


def test_no_silent_fallback_when_projection_is_missing():
    """A dimension mismatch with no W must raise, not substitute a map.

    Quietly bridging 300->768 with an untrained projection would leave the
    run reporting as LGSE while the mapping never learns anything -- the
    exact failure this refactor removes.
    """
    with pytest.raises(ValueError, match="no learned projection"):
        MorphemeEmbeddingBuilder(
            fasttext_model=StubFastText(), segmenter=StubSegmenter(),
            embedding_dim=768, projection=None)


def test_matching_dims_need_no_projection():
    """When FastText and the model share a width, W is unnecessary."""
    assert build_projection(768, 768, seed=42) is None
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(dim=768), segmenter=StubSegmenter(),
        embedding_dim=768, projection=None)
    assert builder.build_embedding_for_token("ኣይመፀን").shape == (768,)


def test_learned_projection_has_trainable_parameters():
    projection = build_projection(300, 768, seed=42)
    assert projection.is_learned
    trainable = sum(p.numel() for p in projection.parameters()
                    if p.requires_grad)
    assert trainable == 300 * 768


def test_gradient_reaches_projection_weights():
    """A loss on the initialized embedding must produce a gradient on W.

    This is the assertion that would have caught the optimizer bug: if W is
    detached anywhere in the chain, .grad stays None.
    """
    projection = build_projection(300, 768, seed=42)
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
    projection = build_projection(300, 768, seed=42)
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


def test_projection_is_deterministic_under_seed():
    a = build_projection(300, 768, seed=7)
    b = build_projection(300, 768, seed=7)
    x = torch.randn(4, 300)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


def test_w_is_updated_by_a_lapt_step():
    """The load-bearing test: W must actually move during training.

    Every other assertion here passes even when W is frozen in practice.
    The initializer writes W's output into the embedding via `.data`, which
    severs the autograd graph, so if the regularizer's anchor is a detached
    constant then no LAPT loss term is a function of W: it sits in the
    optimizer, reports as trainable, and never moves -- the same end state
    as the fixed random map, reached by a different route.

    Recomputing the anchor through W keeps it on the graph. This test fails
    if that ever regresses.
    """
    from lgse.regularization import LGSERegularizer

    projection = build_projection(300, 768, seed=42)
    embedding = torch.nn.Embedding(120, 768)
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=768, projection=projection)
    initializer = LGSEInitializer(
        embedding_layer=embedding, morph_builder=builder,
        char_encoder=StubCharEncoder(768))

    token_to_id = {"ኣይመፀን": 100, "ሰላማዊ": 101}
    initializer.write_embeddings_for_new_tokens(token_to_id)

    def live_anchor():
        return torch.stack(
            [initializer.init_token_embedding(t) for t in token_to_id], dim=0)

    regularizer = LGSERegularizer(init_embeddings=live_anchor,
                                  token_ids=token_to_id, lambda_reg=1.0)
    assert regularizer.anchor_is_live

    optimizer = torch.optim.AdamW(
        [embedding.weight] + list(projection.parameters()), lr=1e-2)
    before = projection.linear.weight.detach().clone()

    for _ in range(3):
        mlm = embedding(torch.tensor([100, 101])).pow(2).mean()
        loss = mlm + regularizer.loss(embedding)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert projection.linear.weight.grad is not None, "W got no gradient"
    assert not torch.allclose(before, projection.linear.weight), \
        "W did not move across a LAPT step -- it is frozen in practice"


def test_detached_anchor_leaves_w_without_gradient():
    """Documents precisely why the live anchor is required.

    This is the pre-refactor behaviour, asserted so the reason for the
    change stays visible rather than becoming folklore.
    """
    from lgse.regularization import LGSERegularizer

    projection = build_projection(300, 768, seed=42)
    embedding = torch.nn.Embedding(120, 768)
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=768, projection=projection)
    initializer = LGSEInitializer(
        embedding_layer=embedding, morph_builder=builder,
        char_encoder=StubCharEncoder(768))

    token_to_id = {"ኣይመፀን": 100}
    init_matrix = initializer.write_embeddings_for_new_tokens(token_to_id)

    regularizer = LGSERegularizer(init_embeddings=init_matrix,
                                  token_ids=token_to_id, lambda_reg=1.0)
    assert not regularizer.anchor_is_live

    loss = embedding(torch.tensor([100])).sum() + regularizer.loss(embedding)
    loss.backward()

    assert projection.linear.weight.grad is None, \
        "a detached anchor should give W no gradient path"


def test_trained_projection_survives_a_checkpoint_round_trip(tmp_path):
    """A trained W must be restored, not silently re-initialized.

    `save_pretrained()` covers the model only, so W needs explicit
    serialization; without it a resumed run starts from a fresh Gaussian
    and the training that produced W is thrown away.
    """
    from lgse.lap_trainer import LGSELAPTrainer

    projection = build_projection(300, 768, seed=42)
    with torch.no_grad():                      # stand in for training
        projection.linear.weight.add_(1.0)
    trained = projection.linear.weight.detach().clone()

    torch.save({"state_dict": projection.state_dict(),
                "source_dim": 300, "target_dim": 768},
               tmp_path / LGSELAPTrainer.PROJECTION_FILE)

    restored = build_projection(300, 768, seed=42)
    assert not torch.allclose(restored.linear.weight, trained), \
        "fresh projection already equals the trained one; test is vacuous"

    trainer = LGSELAPTrainer.__new__(LGSELAPTrainer)   # no model download
    trainer.projection = restored
    trainer.device = torch.device("cpu")
    trainer.load_projection(str(tmp_path))

    assert torch.allclose(restored.linear.weight, trained)


def test_restoring_a_mismatched_projection_is_refused(tmp_path):
    """Loading a differently-shaped W would mean loading another run's map."""
    from lgse.lap_trainer import LGSELAPTrainer

    other = build_projection(300, 512, seed=42)
    torch.save({"state_dict": other.state_dict(),
                "source_dim": 300, "target_dim": 512},
               tmp_path / LGSELAPTrainer.PROJECTION_FILE)

    trainer = LGSELAPTrainer.__new__(LGSELAPTrainer)
    trainer.projection = build_projection(300, 768, seed=42)
    trainer.device = torch.device("cpu")

    with pytest.raises(ValueError, match="shape mismatch"):
        trainer.load_projection(str(tmp_path))
