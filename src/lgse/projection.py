"""
projection.py -- projection W from FastText space to the model's embedding space.

FastText vectors are 300-dimensional; the target model's embedding space is
wider (768 for xlm-roberta-base). Something must map between them.

The paper specifies a **learned** projection: W is a trainable parameter
optimized jointly with the new token embeddings during LAPT, so the mapping
adapts to the target embedding space rather than merely preserving relative
distances.

The released implementation instead used a fixed, seeded Johnson-Lindenstrauss
random projection (lgse/morpheme_embeddings.py in the original release). That
is a defensible way to bridge two spaces without extra training data, but it
is not what the paper describes: a random W is never updated, so it cannot
learn anything about the target space. Both are provided here --
`LearnedProjection` implements the paper, `RandomProjection` reproduces the
original release -- and the choice is recorded in the run config so any
result states which was used.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LearnedProjection(nn.Module):
    """Trainable W: FastText space -> model embedding space (paper).

    Initialized as a scaled Gaussian, the same distribution the fixed
    random projection uses, so training starts from an equivalent mapping
    and any difference in results comes from W being learned rather than
    from a different starting point.
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


class RandomProjection(nn.Module):
    """Fixed Johnson-Lindenstrauss projection (original release).

    Not trainable: `requires_grad` is False on the buffer, and it is never
    handed to an optimizer. Kept so the released behaviour stays
    reproducible and comparable against the paper's learned W.
    """

    def __init__(self, source_dim: int, target_dim: int, seed: int = 42):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        rng = np.random.default_rng(seed)
        w = rng.normal(scale=1.0 / np.sqrt(source_dim),
                       size=(source_dim, target_dim)).astype(np.float32)
        self.register_buffer("weight", torch.from_numpy(w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

    @property
    def is_learned(self) -> bool:
        return False


def build_projection(kind: str, source_dim: int, target_dim: int,
                     seed: int = 42) -> Optional[nn.Module]:
    """Return the configured projection, or None when dims already match."""
    if source_dim == target_dim:
        return None
    if kind == "learned":
        return LearnedProjection(source_dim, target_dim, seed=seed)
    if kind == "random":
        return RandomProjection(source_dim, target_dim, seed=seed)
    raise ValueError(f"unknown projection {kind!r}; expected 'learned' or 'random'")
