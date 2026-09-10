"""
projection.py -- the alignment matrix W of the LGSE method.

W is a square d x d linear map applied to FastText vectors before they are
written into the model's embedding space: e_aligned_t = W e_t (Sec 4.1).

Two properties follow, and both are enforced here:

  * **W is square.** It maps the FastText space onto the model's embedding
    space at the same dimension d, so FastText vectors must be trained at the
    model's embedding width (768 for xlm-roberta-base). `check_dimensions`
    verifies this and raises on a mismatch.

  * **W is fixed during training.** It is supplied per run, held with
    `requires_grad=False`, and excluded from the optimizer. Initialization
    writes W's output into the embedding through `.data`, so no gradient
    path reaches W. `scripts/build_alignment_matrix.py` constructs W before
    training; see IMPLEMENTATION_NOTES.md section 1a.

W has no default: `build_projection` requires an explicit matrix and raises
`MissingAlignmentMatrix` without one. The identity is not applied as a
fallback, since it would treat the FastText and model embedding spaces as
already aligned.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class MissingAlignmentMatrix(ValueError):
    """Raised when no alignment matrix W was supplied.

    A distinct type so callers can catch a missing W specifically.
    """


class AlignmentProjection(nn.Module):
    """Square alignment matrix W (d x d), supplied per run.

    `weight` is required: there is no default, and the identity is not
    applied as a fallback.

    W is fixed: `requires_grad=False`, excluded from the optimizer. See
    IMPLEMENTATION_NOTES.md section 1a.
    """

    def __init__(self, dim: int, weight: torch.Tensor,
                 source: str = "unspecified"):
        super().__init__()
        self.dim = dim
        # Retained for checkpoint compatibility and shape assertions.
        self.source_dim = dim
        self.target_dim = dim
        self.source = source

        if weight is None:
            raise MissingAlignmentMatrix(
                "AlignmentProjection requires an explicit weight matrix; "
                "there is no default. See build_projection() for the "
                "user-facing error and IMPLEMENTATION_NOTES.md section 1a.")
        if tuple(weight.shape) != (dim, dim):
            raise ValueError(
                f"alignment matrix W must be square {dim}x{dim}, got "
                f"{tuple(weight.shape)} from {source}")
        w = weight.to(dtype=torch.float32)

        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(w)
        self.linear.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def is_trainable(self) -> bool:
        return self.linear.weight.requires_grad

    @property
    def is_identity(self) -> bool:
        with torch.no_grad():
            return bool(torch.allclose(self.linear.weight,
                                       torch.eye(self.dim,
                                                 device=self.linear.weight.device)))

    def describe(self) -> str:
        state = "trainable" if self.is_trainable else "frozen"
        return (f"W {self.dim}x{self.dim} from {self.source} ({state}"
                f"{', identity' if self.is_identity else ''})")


def load_alignment_matrix(path, dim: int) -> torch.Tensor:
    """Read W from a .pt or .npy file.

    The file must contain exactly a d x d matrix; a mismatched file is not
    reshaped, transposed or padded to fit.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"alignment matrix not found: {path}")

    if path.suffix == ".npy":
        w = torch.from_numpy(np.load(path))
    elif path.suffix in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            if "weight" not in obj:
                raise ValueError(
                    f"{path} is a dict without a 'weight' key; expected a "
                    f"bare {dim}x{dim} tensor or {{'weight': tensor}}")
            obj = obj["weight"]
        w = obj
    else:
        raise ValueError(
            f"unsupported alignment matrix format {path.suffix!r}; "
            f"expected .npy, .pt or .pth")

    if tuple(w.shape) != (dim, dim):
        raise ValueError(
            f"alignment matrix in {path} has shape {tuple(w.shape)}, "
            f"expected ({dim}, {dim}). It is not reshaped to fit.")
    return w.to(dtype=torch.float32)


class IncompatibleFastTextDimension(ValueError):
    """Raised when FastText's width does not match the embedding space.

    A distinct type so callers can catch this specific, actionable data
    problem rather than pattern-matching on a generic ValueError.
    """


def check_dimensions(fasttext_dim: int, embedding_dim: int,
                     source: str = "") -> None:
    """Verify FastText matches the model's embedding width, or raise.

    W is square, so the two dimensions must be equal. A mismatch is resolved
    by supplying FastText vectors at the model's width; the vectors in hand
    are not reshaped, truncated or padded.
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
        f"W is square (W in R^(d x d), Sec 4.1): it aligns the FastText "
        f"space with the model's embedding space, both of dimension d, and "
        f"does not change dimensionality.\n"
        f"\n"
        f"Required: FastText vectors trained at dimension {embedding_dim}.\n"
        f"\n"
        f"This is a data prerequisite. The standard 300-dim FastText CC "
        f"vectors (Grave et al., 2018) do not match a {embedding_dim}-dim "
        f"model. Options:\n"
        f"  - train FastText at dimension {embedding_dim} on the target "
        f"language corpus (`fasttext ... -dim {embedding_dim}`);\n"
        f"  - use a model whose embedding width is {fasttext_dim}.\n"
        f"\n"
        f"The {fasttext_dim}-dim vectors are not reshaped, truncated, "
        f"zero-padded or rectangularly projected to fit. See "
        f"IMPLEMENTATION_NOTES.md section 1.")


def build_projection(source_dim: int, target_dim: int,
                     alignment_matrix_path=None) -> nn.Module:
    """Return the square alignment matrix W.

    `alignment_matrix_path` is required: without it this raises
    `MissingAlignmentMatrix`. The returned W is fixed.

    Raises `IncompatibleFastTextDimension` when the FastText width does not
    match the target embedding width; the fix is FastText vectors trained at
    dimension d.
    """
    check_dimensions(source_dim, target_dim)

    if not alignment_matrix_path:
        raise MissingAlignmentMatrix(
            f"No alignment matrix W was supplied.\n"
            f"\n"
            f"W is the square projection applied to FastText vectors before\n"
            f"they are written into the embedding space (Sec 4.1):\n"
            f"    e_aligned_t = W e_t\n"
            f"\n"
            f"W is supplied per run. There is no default, and the identity\n"
            f"is not applied as a fallback.\n"
            f"\n"
            f"To proceed, set `lgse.alignment_matrix_path` to a .pt/.npy\n"
            f"file holding a {source_dim}x{source_dim} matrix. Build one\n"
            f"with:\n"
            f"    python scripts/build_alignment_matrix.py --language <am|ti>\n"
            f"\n"
            f"See IMPLEMENTATION_NOTES.md section 1a.")

    weight = load_alignment_matrix(alignment_matrix_path, source_dim)
    return AlignmentProjection(source_dim, weight=weight,
                               source=str(alignment_matrix_path))
