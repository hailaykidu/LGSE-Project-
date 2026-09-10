"""Focused tests for the alignment matrix W and the regularization weight lambda.

Four groups for W -- dimensions, gradient flow, optimizer inclusion, and
save/load -- and one for lambda's effect on the loss and its gradient.

These tests verify that W is implemented as a fixed projection: it is
initialized from the supplied matrix, excluded from optimization, and does
not receive parameter updates during training.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lgse.projection import (  # noqa: E402
    AlignmentProjection,
    IncompatibleFastTextDimension,
    MissingAlignmentMatrix,
    build_projection,
    check_dimensions,
    load_alignment_matrix,
)
from lgse.regularization import LGSERegularizer  # noqa: E402

DIM = 16


def a_matrix(dim: int = DIM) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(dim, dim)


# ---------------------------------------------------------------- dimensions

def test_w_is_square_at_the_model_width():
    p = AlignmentProjection(DIM, weight=a_matrix(), source="test")
    assert tuple(p.linear.weight.shape) == (DIM, DIM)
    assert p.dim == p.source_dim == p.target_dim == DIM


def test_w_maps_the_space_onto_itself():
    p = AlignmentProjection(DIM, weight=a_matrix(), source="test")
    out = p(torch.randn(4, DIM))
    assert out.shape == (4, DIM)


def test_non_square_w_is_rejected():
    with pytest.raises(ValueError, match="must be square"):
        AlignmentProjection(DIM, weight=torch.randn(DIM, DIM + 1), source="test")


def test_mismatched_fasttext_width_raises():
    """300-dim vectors against a 768-dim model: W is square, so this raises."""
    with pytest.raises(IncompatibleFastTextDimension) as e:
        check_dimensions(300, 768, source="xlm-roberta-base")
    assert "768" in str(e.value) and "300" in str(e.value)


def test_matching_width_passes():
    assert check_dimensions(768, 768) is None


# ------------------------------------------------------------- gradient flow

def test_w_carries_no_gradient():
    """W is frozen: requires_grad is False and no gradient accumulates."""
    p = AlignmentProjection(DIM, weight=a_matrix(), source="test")
    assert p.linear.weight.requires_grad is False
    assert p.is_trainable is False

    out = p(torch.randn(3, DIM))
    assert out.requires_grad is False
    assert p.linear.weight.grad is None


def test_the_initializer_write_severs_the_graph():
    """Writing W's output via .data detaches it, as the initializer does."""
    p = AlignmentProjection(DIM, weight=a_matrix(), source="test")
    projected = p(torch.randn(3, DIM))

    emb = torch.nn.Embedding(8, DIM)
    with torch.no_grad():
        emb.weight.data[0:3] = projected

    assert emb.weight.grad_fn is None
    emb.weight.sum().backward()
    assert p.linear.weight.grad is None


# --------------------------------------------------------- optimizer updates

def test_w_is_excluded_from_the_optimizer():
    p = AlignmentProjection(DIM, weight=a_matrix(), source="test")
    trainable = [q for q in p.parameters() if q.requires_grad]
    assert trainable == []


def test_w_does_not_move_when_the_embedding_trains():
    """One optimizer step on the embedding leaves W bit-for-bit unchanged."""
    p = AlignmentProjection(DIM, weight=a_matrix(), source="test")
    before = p.linear.weight.detach().clone()

    emb = torch.nn.Embedding(8, DIM)
    opt = torch.optim.SGD([q for q in emb.parameters()], lr=0.1)

    reg = LGSERegularizer(
        init_embeddings=torch.zeros(3, DIM),
        token_ids={"a": 0, "b": 1, "c": 2},
        lambda_reg=1.0,
    )
    loss = reg.loss(emb)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert not torch.equal(emb.weight[0:3].detach(), torch.zeros(3, DIM))
    assert torch.equal(p.linear.weight.detach(), before)


# --------------------------------------------------------------- save / load

@pytest.mark.parametrize("suffix", [".npy", ".pt"])
def test_w_round_trips_through_disk(tmp_path, suffix):
    w = a_matrix()
    path = tmp_path / f"W_test{suffix}"
    if suffix == ".npy":
        np.save(path, w.numpy())
    else:
        torch.save(w, path)

    loaded = load_alignment_matrix(path, DIM)
    assert tuple(loaded.shape) == (DIM, DIM)
    assert loaded.dtype is torch.float32
    assert torch.allclose(loaded, w, atol=1e-6)


def test_w_loads_from_a_state_dict_style_file(tmp_path):
    w = a_matrix()
    path = tmp_path / "W_dict.pt"
    torch.save({"weight": w}, path)
    assert torch.allclose(load_alignment_matrix(path, DIM), w, atol=1e-6)


def test_a_mismatched_file_is_not_reshaped(tmp_path):
    path = tmp_path / "W_wrong.npy"
    np.save(path, np.zeros((DIM + 4, DIM + 4), dtype=np.float32))
    with pytest.raises(ValueError, match="not reshaped"):
        load_alignment_matrix(path, DIM)


def test_build_projection_round_trips_a_saved_matrix(tmp_path):
    w = a_matrix()
    path = tmp_path / "W.npy"
    np.save(path, w.numpy())

    p = build_projection(DIM, DIM, alignment_matrix_path=path)
    assert torch.allclose(p.linear.weight.detach(), w, atol=1e-6)
    assert p.is_trainable is False


def test_build_projection_requires_a_matrix():
    with pytest.raises(MissingAlignmentMatrix):
        build_projection(DIM, DIM, alignment_matrix_path=None)


# -------------------------------------------------------------------- lambda

def test_lambda_scales_the_loss():
    emb = torch.nn.Embedding(8, DIM)
    torch.nn.init.ones_(emb.weight)
    ids = {"a": 0, "b": 1, "c": 2}
    anchor = torch.zeros(3, DIM)

    one = LGSERegularizer(anchor, ids, lambda_reg=1.0).loss(emb)
    two = LGSERegularizer(anchor, ids, lambda_reg=2.0).loss(emb)
    assert two.item() == pytest.approx(2.0 * one.item(), rel=1e-6)


def test_lambda_scales_the_gradient():
    ids = {"a": 0, "b": 1, "c": 2}
    anchor = torch.zeros(3, DIM)

    grads = []
    for lam in (1.0, 2.0):
        emb = torch.nn.Embedding(8, DIM)
        torch.nn.init.ones_(emb.weight)
        LGSERegularizer(anchor, ids, lambda_reg=lam).loss(emb).backward()
        grads.append(emb.weight.grad[0:3].abs().sum().item())

    assert grads[1] == pytest.approx(2.0 * grads[0], rel=1e-6)


def test_lambda_zero_contributes_nothing():
    emb = torch.nn.Embedding(8, DIM)
    torch.nn.init.ones_(emb.weight)
    loss = LGSERegularizer(torch.zeros(3, DIM), {"a": 0, "b": 1, "c": 2},
                           lambda_reg=0.0).loss(emb)
    assert loss.item() == 0.0


def test_the_regularizer_measures_drift_from_the_anchor():
    """mu is constant: no drift means no penalty, drift means a positive one."""
    ids = {"a": 0, "b": 1, "c": 2}
    emb = torch.nn.Embedding(8, DIM)
    torch.nn.init.ones_(emb.weight)
    anchor = emb.weight[0:3].detach().clone()

    assert LGSERegularizer(anchor, ids, lambda_reg=1.0).loss(emb).item() == 0.0

    with torch.no_grad():
        emb.weight[0:3] += 1.0
    assert LGSERegularizer(anchor, ids, lambda_reg=1.0).loss(emb).item() > 0.0


def test_the_configured_lambda_reaches_the_loss():
    """configs/base.yaml's reg_lambda is the coefficient actually applied."""
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
        .read_text(encoding="utf-8"))
    lam = cfg["lgse"]["reg_lambda"]

    emb = torch.nn.Embedding(8, DIM)
    torch.nn.init.ones_(emb.weight)
    ids = {"a": 0, "b": 1, "c": 2}
    anchor = torch.zeros(3, DIM)

    configured = LGSERegularizer(anchor, ids, lambda_reg=lam).loss(emb)
    unit = LGSERegularizer(anchor, ids, lambda_reg=1.0).loss(emb)
    assert configured.item() == pytest.approx(lam * unit.item(), rel=1e-6)
