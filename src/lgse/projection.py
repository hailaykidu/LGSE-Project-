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


def build_projection(source_dim: int, target_dim: int,
                     seed: int = 42) -> nn.Module:
    """Return the learned square projection W (paper Sec 4.1).

    Raises on a dimension mismatch. The paper's W is square, so unequal
    dimensions mean the FastText model does not match the target embedding
    width -- a data problem to fix by training or fetching FastText at
    dimension d, not something to paper over with a rectangular map. A
    rectangular W would be a different method than the one published.
    """
    if source_dim != target_dim:
        raise ValueError(
            f"LGSE's projection W is square (d x d, paper Sec 4.1), but "
            f"FastText is {source_dim}-dim and the embedding space is "
            f"{target_dim}-dim. Use FastText vectors trained at "
            f"dimension {target_dim}; a {source_dim}->{target_dim} "
            f"rectangular map is not the published method.")
    return LearnedProjection(source_dim, seed=seed)
