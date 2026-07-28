"""
projection.py -- the learned projection W of the LGSE method.

FastText vectors are 300-dimensional; the target model's embedding space is
wider (768 for xlm-roberta-base). W bridges the two.

W is a *learned* parameter. It is initialized once and then optimized jointly
with the new token embeddings during LAPT, so the mapping adapts to the
geometry of the target embedding space instead of merely preserving relative
distances between FastText vectors. This is the method LGSE describes, and it
is the only projection this package supports.

Three properties are load-bearing, and each is covered by a regression test in
tests/test_learned_projection_pipeline.py:

  1. W carries trainable parameters (`requires_grad`).
  2. Those parameters reach the optimizer, and gradients flow back through
     the morpheme-averaging path into W.
  3. W is serialized with the checkpoint, so a restored run continues with
     the trained mapping rather than a fresh initialization.

A projection that fails any one of these is indistinguishable, at the level
of reported numbers, from a fixed random map -- which is why they are
asserted rather than assumed.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LearnedProjection(nn.Module):
    """Trainable W: FastText space -> the model's embedding space.

    Initialized as a scaled Gaussian (std 1/sqrt(source_dim)), which keeps
    the initial mapping approximately norm-preserving so training starts
    from a well-conditioned point rather than one that inflates or crushes
    the projected vectors.
    """

    def __init__(self, source_dim: int, target_dim: int, seed: int = 42,
                 bias: bool = False):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.linear = nn.Linear(source_dim, target_dim, bias=bias)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.randn(target_dim, source_dim, generator=generator)
                / np.sqrt(source_dim)
            )
            if bias:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def is_learned(self) -> bool:
        return True


def build_projection(source_dim: int, target_dim: int,
                     seed: int = 42) -> Optional[nn.Module]:
    """Return the learned projection W, or None when the dims already match.

    None is not a silent fallback to identity-by-accident: when FastText and
    the model share a width there is nothing for W to map between, and the
    vectors are used directly.
    """
    if source_dim == target_dim:
        return None
    return LearnedProjection(source_dim, target_dim, seed=seed)
