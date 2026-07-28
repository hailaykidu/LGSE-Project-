"""
Regression tests for the FastText + projection W + initializer path.

Paper Sec 4.1 defines W as a *square* learned linear map, R^{d x d}, aligning
the FastText space to the model's. Sec 4.2 defines the regularizer as
L_reg = lambda * ||e_new - mu||^2 "where mu is the initial embedding vector"
-- a constant.

Those two facts together produce the situation these tests pin down: W is a
trainable parameter registered with the optimizer, but under the paper's
stated objectives **nothing is a function of it** once initialization is
done. The initializer writes W's output into the embedding via `.data`
(severing the graph, deliberately, so weight tying with the MLM head
survives), and L_reg anchors to a constant. So W receives no gradient during
LAPT.

That is a gap in the paper, and these tests assert it rather than papering
over it -- inventing a loss term to make W train would be implementing a
different method than the published one. See DEVIATIONS.md section 1a.

Two real bugs previously reached the full pipeline unnoticed because every
unit test exercised these components separately:

  1. `word_from_morphemes` averaged with `np.mean`, which cannot consume a
     grad-tracking tensor: "Can't call numpy() on Tensor that requires
     grad". The method crashed on real FastText vectors.

  2. `save()` wrote only the model and tokenizer, so a trained W was
     discarded and a resumed run silently began from a fresh initialization.
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

# The paper's W is square, so FastText and the model share dimension d.
DIM = 768


class StubFastText:
    """Deterministic stand-in with FastText's interface and dimension."""

    def __init__(self, dim=DIM):
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


def _builder(projection):
    return MorphemeEmbeddingBuilder(
        fasttext_model=StubFastText(), segmenter=StubSegmenter(),
        embedding_dim=DIM, projection=projection)


# --- shape: W is square (Sec 4.1) ------------------------------------

def test_projection_is_square():
    projection = build_projection(DIM, DIM)
    assert projection.linear.weight.shape == (DIM, DIM)


def test_rectangular_projection_is_refused():
    """A 300->768 map would be a different method than the published one."""
    from lgse.projection import IncompatibleFastTextDimension

    with pytest.raises(IncompatibleFastTextDimension):
        build_projection(300, DIM)


def test_dimension_error_is_actionable():
    """The message must name both dims, the requirement, and the fix.

    A user hitting this has a large, slow-to-replace data artifact; the
    error is the only place they learn that 300-dim CC vectors cannot be
    used with the published method at all.
    """
    from lgse.projection import IncompatibleFastTextDimension

    with pytest.raises(IncompatibleFastTextDimension) as exc:
        build_projection(300, DIM)
    msg = str(exc.value)

    assert "300" in msg and str(DIM) in msg          # both dimensions
    assert "square" in msg                            # why
    assert "Sec 4.1" in msg                           # where in the paper
    assert f"-dim {DIM}" in msg                       # how to fix it
    assert "DEVIATIONS.md" in msg                     # where to read more


def test_mismatched_fasttext_is_refused_at_builder_construction():
    """The builder must refuse too, not just build_projection.

    A caller who constructs the builder without a projection would
    otherwise get vectors of the wrong width flowing into the initializer.
    """
    from lgse.projection import IncompatibleFastTextDimension

    with pytest.raises(IncompatibleFastTextDimension):
        MorphemeEmbeddingBuilder(
            fasttext_model=StubFastText(dim=300), segmenter=StubSegmenter(),
            embedding_dim=DIM, projection=None)


def test_no_silent_reshaping_of_mismatched_vectors():
    """Nothing anywhere may pad, truncate or rectangularly project.

    Asserted as a property of the whole entry surface rather than of one
    function, since any single lenient path would reintroduce the silent
    method change.
    """
    from lgse.projection import IncompatibleFastTextDimension, check_dimensions

    for ft_dim in (100, 300, 512, 1024):        # smaller and larger than DIM
        if ft_dim == DIM:
            continue
        with pytest.raises(IncompatibleFastTextDimension):
            check_dimensions(ft_dim, DIM)
        with pytest.raises(IncompatibleFastTextDimension):
            build_projection(ft_dim, DIM)


def test_projection_is_identity_initialized():
    """Alignment starts from 'no change', so the initial embeddings are
    exactly the FastText morpheme averages Sec 4.1 defines."""
    projection = build_projection(DIM, DIM)
    assert torch.allclose(projection.linear.weight, torch.eye(DIM))

    x = torch.randn(4, DIM)
    with torch.no_grad():
        assert torch.allclose(projection(x), x, atol=1e-5)


def test_projection_has_trainable_parameters():
    projection = build_projection(DIM, DIM)
    assert projection.is_learned
    trainable = sum(p.numel() for p in projection.parameters()
                    if p.requires_grad)
    assert trainable == DIM * DIM


# --- autograd: W stays on the graph through morpheme averaging -------

def test_projection_keeps_gradients_through_morpheme_averaging():
    """Regression: `np.mean` used to break this path outright."""
    projection = build_projection(DIM, DIM)
    vec = _builder(projection).build_embedding_for_token("ኣይመፀን")

    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (DIM,)
    assert vec.requires_grad, "W must stay on the autograd graph"


def test_gradient_reaches_projection_weights_from_a_live_loss():
    """W is differentiable: a loss computed *through* it does reach it.

    This isolates W's mechanics from the separate question of whether the
    paper's objectives actually produce such a loss (they do not -- see
    test_paper_objectives_give_w_no_gradient).
    """
    projection = build_projection(DIM, DIM)
    vec = _builder(projection).build_embedding_for_token("ኣይመፀን")
    vec.sum().backward()

    weight = projection.linear.weight
    assert weight.grad is not None, "no gradient reached W"
    assert torch.any(weight.grad != 0), "gradient reached W but is all zero"


# --- the paper's objectives leave W without a gradient ---------------

def test_paper_objectives_give_w_no_gradient():
    """The documented gap, asserted so it cannot regress into a false claim.

    Sec 4.2's L_reg anchors to a constant mu, and the MLM loss reads the
    embedding matrix that initialization wrote into via `.data`. Neither is
    a function of W, so W stays at its identity initialization for the whole
    of LAPT even though the paper calls it "learned".
    """
    from lgse.regularization import LGSERegularizer

    projection = build_projection(DIM, DIM)
    embedding = torch.nn.Embedding(120, DIM)
    initializer = LGSEInitializer(
        embedding_layer=embedding, morph_builder=_builder(projection),
        char_encoder=StubCharEncoder(DIM))

    token_to_id = {"ኣይመፀን": 100}
    init_matrix = initializer.write_embeddings_for_new_tokens(token_to_id)

    regularizer = LGSERegularizer(init_embeddings=init_matrix,
                                  token_ids=token_to_id, lambda_reg=1.0)
    assert not regularizer.anchor_is_live, "paper's mu is a constant"

    loss = embedding(torch.tensor([100])).sum() + regularizer.loss(embedding)
    loss.backward()

    assert projection.linear.weight.grad is None, (
        "W has a gradient under the paper's objectives -- if this starts "
        "passing, the implementation has added a loss term the paper does "
        "not describe, and DEVIATIONS.md section 1a must be revisited")


# --- initialization writes rows in place -----------------------------

def test_initializer_writes_rows_without_breaking_weight_tying():
    """Init vectors are written in place, so a tied output head survives."""
    embedding = torch.nn.Embedding(120, DIM)
    original = embedding.weight
    projection = build_projection(DIM, DIM)
    initializer = LGSEInitializer(
        embedding_layer=embedding, morph_builder=_builder(projection),
        char_encoder=StubCharEncoder(DIM))

    token_to_id = {"ኣይመፀን": 100, "ሰላማዊ": 101}
    init_matrix = initializer.write_embeddings_for_new_tokens(token_to_id)

    assert init_matrix.shape == (2, DIM)
    assert embedding.weight is original, "Parameter was replaced, breaking tying"
    for token, idx in token_to_id.items():
        assert torch.allclose(embedding.weight.data[idx],
                              init_matrix[list(token_to_id).index(token)])


def test_projection_is_deterministic():
    a = build_projection(DIM, DIM, seed=7)
    b = build_projection(DIM, DIM, seed=7)
    x = torch.randn(4, DIM)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


# --- checkpointing ---------------------------------------------------

def test_trained_projection_survives_a_checkpoint_round_trip(tmp_path):
    """A trained W must be restored, not silently re-initialized.

    `save_pretrained()` covers the model only, so W needs explicit
    serialization; without it a resumed run starts from a fresh identity
    and any training that produced W is thrown away.
    """
    from lgse.lap_trainer import LGSELAPTrainer

    projection = build_projection(DIM, DIM)
    with torch.no_grad():                      # stand in for training
        projection.linear.weight.add_(1.0)
    trained = projection.linear.weight.detach().clone()

    torch.save({"state_dict": projection.state_dict(),
                "source_dim": DIM, "target_dim": DIM},
               tmp_path / LGSELAPTrainer.PROJECTION_FILE)

    restored = build_projection(DIM, DIM)
    assert not torch.allclose(restored.linear.weight, trained), \
        "fresh projection already equals the trained one; test is vacuous"

    trainer = LGSELAPTrainer.__new__(LGSELAPTrainer)   # no model download
    trainer.projection = restored
    trainer.device = torch.device("cpu")
    trainer.load_projection(str(tmp_path))

    assert torch.allclose(restored.linear.weight, trained)


def test_manifest_rejects_wrong_dimension_before_loading_the_model(tmp_path):
    """Config resolution must fail on a bad dimension recorded in the
    manifest, so the run stops before loading a multi-GB model."""
    import json

    from lgse.config import LGSEConfig
    from lgse.projection import IncompatibleFastTextDimension

    manifest = tmp_path / "fasttext_manifest.json"
    manifest.write_text(json.dumps({
        "amharic": {"path": "/nonexistent/cc.am.300.bin", "dimension": 300,
                    "vocab_size": 100, "sha256": "x"}}))

    config = LGSEConfig(language="am", model_name="xlm-roberta-base",
                        fasttext_manifest=str(manifest))

    with pytest.raises(IncompatibleFastTextDimension, match="300"):
        _ = config.fasttext_path


def test_manifest_accepts_the_required_dimension(tmp_path):
    import json

    from lgse.config import LGSEConfig

    manifest = tmp_path / "fasttext_manifest.json"
    manifest.write_text(json.dumps({
        "amharic": {"path": "/models/am.768.bin", "dimension": DIM,
                    "vocab_size": 100, "sha256": "x"}}))

    config = LGSEConfig(language="am", model_name="xlm-roberta-base",
                        fasttext_manifest=str(manifest))
    assert config.fasttext_path == "/models/am.768.bin"


def test_restoring_a_mismatched_projection_is_refused(tmp_path):
    """Loading a differently-shaped W would mean loading another run's map."""
    from lgse.lap_trainer import LGSELAPTrainer

    other = build_projection(512, 512)
    torch.save({"state_dict": other.state_dict(),
                "source_dim": 512, "target_dim": 512},
               tmp_path / LGSELAPTrainer.PROJECTION_FILE)

    trainer = LGSELAPTrainer.__new__(LGSELAPTrainer)
    trainer.projection = build_projection(DIM, DIM)
    trainer.device = torch.device("cpu")

    with pytest.raises(ValueError, match="shape mismatch"):
        trainer.load_projection(str(tmp_path))
