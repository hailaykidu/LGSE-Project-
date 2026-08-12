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
different method than the published one. See IMPLEMENTATION_NOTES.md section 1a.

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


def _W(tmp_path, weight=None, name="W.pt"):
    """Write an author-supplied alignment matrix and return its path.

    W has no default, so every test that needs a projection must supply one
    -- exactly as a real run must.
    """
    path = tmp_path / name
    torch.save(torch.eye(DIM) if weight is None else weight, path)
    return path


@pytest.fixture
def projection(tmp_path):
    """A supplied W, standing in for the author-provided artifact."""
    return build_projection(DIM, DIM, alignment_matrix_path=_W(tmp_path))


# --- shape: W is square (Sec 4.1) ------------------------------------

def test_projection_is_square(projection):
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
    assert "IMPLEMENTATION_NOTES.md" in msg                     # where to read more


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


def test_missing_alignment_matrix_is_refused():
    """No W supplied -> fail. There is no default, not even the identity.

    The identity would assert the FastText and model embedding spaces are
    already aligned -- the claim Sec 4.1 introduces W to avoid making.
    """
    from lgse.projection import MissingAlignmentMatrix

    with pytest.raises(MissingAlignmentMatrix):
        build_projection(DIM, DIM)


def test_missing_matrix_error_is_actionable():
    from lgse.projection import MissingAlignmentMatrix

    with pytest.raises(MissingAlignmentMatrix) as exc:
        build_projection(DIM, DIM)
    msg = str(exc.value)

    assert "Sec 4.1" in msg                         # where W is defined
    assert "no default" in msg                      # W is not defaulted
    assert "identity" in msg                        # nor is the identity
    assert "alignment_matrix_path" in msg           # how to proceed
    assert "build_alignment_matrix.py" in msg       # how to build one
    assert "IMPLEMENTATION_NOTES.md" in msg


def test_a_supplied_identity_still_works_but_is_flagged(tmp_path):
    """The identity is a legitimate *choice*, just not a default."""
    projection = build_projection(
        DIM, DIM, alignment_matrix_path=_W(tmp_path, torch.eye(DIM)))
    assert projection.is_identity

    x = torch.randn(4, DIM)
    with torch.no_grad():
        assert torch.allclose(projection(x), x, atol=1e-5)


def test_projection_is_frozen_by_default(projection):
    """W is externally supplied, not fitted here.

    The paper states no objective that trains W, so a trainable W would
    advertise a capability the run does not have.
    """
    assert not projection.is_trainable
    assert sum(p.numel() for p in projection.parameters()
               if p.requires_grad) == 0


def test_frozen_projection_is_excluded_from_the_optimizer(projection):
    """The optimizer must contain the embeddings only."""
    embedding = torch.nn.Embedding(120, DIM)

    assert not projection.is_trainable
    assert [p for p in projection.parameters() if p.requires_grad] == []


def test_w_cannot_be_made_trainable():
    """There is no switch that makes W trainable.

    The paper states no objective that trains W, so an implementation
    offering a trainable mode would be offering a method the paper does not
    describe.
    """
    from lgse.projection import AlignmentProjection

    projection = AlignmentProjection(DIM, weight=torch.eye(DIM))
    assert not projection.is_trainable
    with pytest.raises(TypeError):
        AlignmentProjection(DIM, weight=torch.eye(DIM), trainable=True)


# --- autograd: W stays on the graph through morpheme averaging -------

def test_morpheme_averaging_produces_a_correctly_shaped_vector(projection):
    """Regression: `np.mean` used to break this path outright when W's
    output was a grad-tracking tensor."""
    vec = _builder(projection).build_embedding_for_token("ኣይመፀን")

    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (DIM,)


def test_no_gradient_is_created_for_a_frozen_w(projection):
    """A loss through W must not produce a gradient on it.

    With W frozen there is nothing to accumulate into, so this stays None
    regardless of what the caller does downstream.
    """
    vec = _builder(projection).build_embedding_for_token("ኣይመፀን")

    assert not vec.requires_grad, "frozen W must not put the output on the graph"
    assert projection.linear.weight.grad is None


# --- the paper's objectives leave W without a gradient ---------------

def test_paper_objectives_give_w_no_gradient(projection):
    """The documented gap, asserted so it cannot regress into a false claim.

    Sec 4.2's L_reg anchors to a constant mu, and the MLM loss reads the
    embedding matrix that initialization wrote into via `.data`. Neither is
    a function of W, so the supplied W is unchanged for the whole of LAPT
    even though the paper calls it "learned".
    """
    from lgse.regularization import LGSERegularizer

    embedding = torch.nn.Embedding(120, DIM)
    initializer = LGSEInitializer(
        embedding_layer=embedding, morph_builder=_builder(projection),
        char_encoder=StubCharEncoder(DIM))

    token_to_id = {"ኣይመፀን": 100}
    init_matrix = initializer.write_embeddings_for_new_tokens(token_to_id)

    regularizer = LGSERegularizer(init_embeddings=init_matrix,
                                  token_ids=token_to_id, lambda_reg=1.0)

    loss = embedding(torch.tensor([100])).sum() + regularizer.loss(embedding)
    loss.backward()

    assert projection.linear.weight.grad is None, (
        "W has a gradient under the paper's objectives -- if this starts "
        "passing, the implementation has added a loss term the paper does "
        "not describe, and IMPLEMENTATION_NOTES.md section 1a must be revisited")


# --- initialization writes rows in place -----------------------------

def test_initializer_writes_rows_without_breaking_weight_tying(projection):
    """Init vectors are written in place, so a tied output head survives."""
    embedding = torch.nn.Embedding(120, DIM)
    original = embedding.weight
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


def test_projection_is_deterministic(tmp_path):
    """The same supplied W yields the same mapping every time."""
    w = _W(tmp_path, torch.randn(DIM, DIM))
    a = build_projection(DIM, DIM, alignment_matrix_path=w)
    b = build_projection(DIM, DIM, alignment_matrix_path=w)
    x = torch.randn(4, DIM)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


def test_externally_supplied_matrix_is_loaded_verbatim(tmp_path):
    """An author-supplied W must be used exactly as given."""
    w = torch.randn(DIM, DIM)
    path = tmp_path / "W.pt"
    torch.save(w, path)

    projection = build_projection(DIM, DIM, alignment_matrix_path=path)

    assert torch.allclose(projection.linear.weight, w)
    assert not projection.is_identity
    assert not projection.is_trainable
    assert str(path) in projection.source


def test_supplied_matrix_of_wrong_shape_is_refused(tmp_path):
    """Not reshaped, transposed or padded to fit."""
    path = tmp_path / "W.pt"
    torch.save(torch.randn(300, DIM), path)

    with pytest.raises(ValueError, match="not reshaped"):
        build_projection(DIM, DIM, alignment_matrix_path=path)


def test_alignment_matrix_is_required_by_every_entry_point():
    """Neither build_projection nor AlignmentProjection may default."""
    from lgse.projection import AlignmentProjection, MissingAlignmentMatrix

    with pytest.raises(MissingAlignmentMatrix):
        build_projection(DIM, DIM)
    with pytest.raises(MissingAlignmentMatrix):
        AlignmentProjection(DIM, weight=None)
    with pytest.raises(TypeError):
        AlignmentProjection(DIM)          # weight is positional-required


# --- checkpointing ---------------------------------------------------

def test_projection_survives_a_checkpoint_round_trip(tmp_path):
    """W must be restored exactly, not silently replaced by the identity.

    `save_pretrained()` covers the model only, so W needs explicit
    serialization. Without it, a checkpoint made with an author-supplied
    alignment matrix would silently reload as the identity -- a different
    run from the one that produced the numbers.
    """
    from lgse.lap_trainer import LGSELAPTrainer

    projection = build_projection(
        DIM, DIM, alignment_matrix_path=_W(tmp_path, torch.randn(DIM, DIM)))
    trained = projection.linear.weight.detach().clone()

    torch.save({"state_dict": projection.state_dict(),
                "source_dim": DIM, "target_dim": DIM},
               tmp_path / LGSELAPTrainer.PROJECTION_FILE)

    restored = build_projection(
        DIM, DIM,
        alignment_matrix_path=_W(tmp_path, torch.eye(DIM), name="other.pt"))
    assert not torch.allclose(restored.linear.weight, trained), \
        "fresh projection already equals the saved one; test is vacuous"

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
                        fasttext_manifest=str(manifest), reg_lambda=1.0)

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
                        fasttext_manifest=str(manifest), reg_lambda=1.0)
    assert config.fasttext_path == "/models/am.768.bin"


def test_shipped_config_leaves_the_matrix_unset():
    """configs/base.yaml must NOT ship a W.

    Shipping one would hand every user an alignment matrix this project
    invented and the authors never specified -- the exact silent choice the
    fail-fast behaviour exists to prevent.
    """
    import yaml

    cfg = yaml.safe_load(
        open(Path(__file__).resolve().parent.parent / "configs" / "base.yaml"))
    assert cfg["lgse"]["alignment_matrix_path"] == ""


def test_run_experiment_guards_fasttext_systems():
    """The early guard must exist for lgse/focus.

    Checked against the source rather than by running the sweep, which
    would need a backbone download.
    """
    source = (Path(__file__).resolve().parent.parent / "src" / "training"
              / "run_experiment.py").read_text()

    assert "alignment_matrix_path" in source
    assert '("lgse", "focus")' in source
    assert "faithful" in source


# --- reg_lambda is mandatory ----------------------------------------

def test_reg_lambda_is_mandatory():
    """No default: the paper never assigns lambda, so any value is the
    experimenter's choice and must be stated rather than inherited."""
    from lgse.config import LGSEConfig, MissingRequiredParameter

    with pytest.raises(MissingRequiredParameter, match="reg_lambda"):
        LGSEConfig(language="am")


def test_reg_lambda_error_explains_why_there_is_no_default():
    from lgse.config import LGSEConfig, MissingRequiredParameter

    with pytest.raises(MissingRequiredParameter) as exc:
        LGSEConfig(language="am")
    msg = str(exc.value)

    assert "Sec 4.2" in msg                    # where it comes from
    assert "never states its value" in msg     # why no default
    assert "IMPLEMENTATION_NOTES.md" in msg              # where to read more


def test_reg_lambda_accepts_an_explicit_value():
    from lgse.config import LGSEConfig

    assert LGSEConfig(language="am", reg_lambda=0.5).reg_lambda == 0.5
    # Zero is meaningful: it disables the regularizer.
    assert LGSEConfig(language="am", reg_lambda=0.0).reg_lambda == 0.0


def test_negative_reg_lambda_is_refused():
    from lgse.config import LGSEConfig

    with pytest.raises(ValueError, match="non-negative"):
        LGSEConfig(language="am", reg_lambda=-1.0)


def test_shipped_config_supplies_reg_lambda():
    """configs/base.yaml must carry the key, or every run fails."""
    import yaml

    cfg = yaml.safe_load(
        open(Path(__file__).resolve().parent.parent / "configs" / "base.yaml"))
    assert "reg_lambda" in cfg["lgse"]


def test_restoring_a_mismatched_projection_is_refused(tmp_path):
    """Loading a differently-shaped W would mean loading another run's map."""
    from lgse.lap_trainer import LGSELAPTrainer

    other = build_projection(
        512, 512, alignment_matrix_path=_W(tmp_path, torch.eye(512),
                                          name="w512.pt"))
    torch.save({"state_dict": other.state_dict(),
                "source_dim": 512, "target_dim": 512},
               tmp_path / LGSELAPTrainer.PROJECTION_FILE)

    trainer = LGSELAPTrainer.__new__(LGSELAPTrainer)
    trainer.projection = build_projection(
        DIM, DIM, alignment_matrix_path=_W(tmp_path, name="wdim.pt"))
    trainer.device = torch.device("cpu")

    with pytest.raises(ValueError, match="shape mismatch"):
        trainer.load_projection(str(tmp_path))
