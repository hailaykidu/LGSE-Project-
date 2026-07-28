"""
projection.py -- the learned projection W of the LGSE method.

Paper Sec 4.1:

    "To align the FastText embedding space with the pretrained model
     embedding space, a learned linear projection W in R^{d x d} is
     applied, i.e. e_aligned_t = W e_t."

Two things follow directly from that line, and both are load-bearing:

  * **W is square** (d x d). The paper aligns two spaces of the same
    dimension d; it does not describe a rectangular map that changes
    dimensionality. Running with 300-dim FastText against 768-dim XLM-R
    therefore requires FastText vectors trained at d=768, not a 300->768
    rectangular W. `build_projection` enforces the square shape and reports
    the mismatch rather than silently reshaping the method.

  * **W is learned.** It is a trainable parameter, saved with the
    checkpoint and restored on resume.

What the paper does *not* do is give W a gradient path through the
regularization term: L_reg anchors to a constant mu (Sec 4.2), so it
trains the embeddings, not W. See DEVIATIONS.md section 1a for where W's
gradient comes from and what remains underdetermined.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LearnedProjection(nn.Module):
    """Trainable square W (d x d), aligning FastText space to the model's.

    Initialized to the identity: alignment between two spaces of the same
    dimension starts from "no change", so the initial embeddings are exactly
    the FastText morpheme averages the paper's Sec 4.1 defines, and W then
    learns the alignment away from there. A random initialization would
    instead scramble those vectors before training ever sees them, which
    would defeat the point of grounding the initialization lexically.
    """

    def __init__(self, dim: int, seed: int = 42, bias: bool = False):
        super().__init__()
        self.dim = dim
        # Retained for checkpoint compatibility and shape assertions.
        self.source_dim = dim
        self.target_dim = dim
        self.linear = nn.Linear(dim, dim, bias=bias)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))
            if bias:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def is_learned(self) -> bool:
        return True


class IncompatibleFastTextDimension(ValueError):
    """Raised when FastText's width does not match the embedding space.

    A distinct type so callers can catch this specific, actionable data
    problem rather than pattern-matching on a generic ValueError.
    """


def check_dimensions(fasttext_dim: int, embedding_dim: int,
                     source: str = "") -> None:
    """Verify FastText matches the model's embedding width, or raise.

    LGSE's W is square (paper Sec 4.1), so these two must be equal. When
    they are not, the only correct fix is different FastText vectors --
    reshaping, truncating, padding or rectangular-projecting the ones in
    hand would change the method into one the paper does not describe, and
    would do so invisibly in the reported numbers.
    """
    if fasttext_dim == embedding_dim:
        return

    where = f"\n  Model:      {source}" if source else ""
    raise IncompatibleFastTextDimension(
        f"Incompatible FastText embedding dimension.\n"
        f"\n"
        f"  FastText:   {fasttext_dim}-dim{where}\n"
        f"  Model:      {embedding_dim}-dim embedding space\n"
        f"\n"
        f"LGSE's projection W is square (W in R^(d x d), paper Sec 4.1): it "
        f"aligns the FastText space with the model's embedding space, both "
        f"of dimension d. It does not change dimensionality.\n"
        f"\n"
        f"Required: FastText vectors trained at dimension {embedding_dim}.\n"
        f"\n"
        f"This is a data prerequisite, not a configuration problem. The "
        f"standard 300-dim FastText CC vectors (Grave et al., 2018) cannot "
        f"be used with a {embedding_dim}-dim model under the published "
        f"method. Options:\n"
        f"  - train FastText at dimension {embedding_dim} on the target "
        f"language corpus (`fasttext ... -dim {embedding_dim}`);\n"
        f"  - use a model whose embedding width is {fasttext_dim}.\n"
        f"\n"
        f"Reshaping, truncating, zero-padding or rectangularly projecting "
        f"the {fasttext_dim}-dim vectors would silently change the method "
        f"and is deliberately not implemented. See DEVIATIONS.md section 1.")


def build_projection(source_dim: int, target_dim: int,
                     seed: int = 42) -> nn.Module:
    """Return the learned square projection W (paper Sec 4.1).

    Raises `IncompatibleFastTextDimension` on a dimension mismatch: the
    paper's W is square, so unequal dimensions mean the FastText model does
    not match the target embedding width. That is a data problem to fix by
    training or fetching FastText at dimension d, not something to paper
    over with a rectangular map.
    """
    check_dimensions(source_dim, target_dim)
    return LearnedProjection(source_dim, seed=seed)
