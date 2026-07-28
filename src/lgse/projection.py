"""
projection.py -- the alignment matrix W of the LGSE method.

Paper Sec 4.1:

    "To align the FastText embedding space with the pretrained model
     embedding space, a learned linear projection W in R^{d x d} is
     applied, i.e. e_aligned_t = W e_t."

Two things follow directly, and both are load-bearing:

  * **W is square** (d x d). The paper aligns two spaces of the same
    dimension d; it does not describe a rectangular map that changes
    dimensionality. Running with 300-dim FastText against 768-dim XLM-R
    therefore requires FastText vectors trained at d=768. `check_dimensions`
    enforces this and reports the mismatch rather than reshaping the method.

  * **W is described as "learned", but the paper specifies no objective
    that trains it.** Sec 4.2's L_reg anchors to a constant mu; Sec 5's
    LAPT applies MLM to the embedding matrix with the encoder frozen.
    Neither is a function of W. Initialization writes W's output into the
    embedding through `.data`, which severs the autograd graph, and nothing
    downstream depends on W again.

STATUS: author-required / unspecified in paper
------------------------------------------------------------------
W is therefore treated here as an **externally supplied alignment matrix**:
a given of the run, loaded from disk, with **no default of any kind**. A run
without an author-provided W fails rather than starting.

That includes the identity. The identity is not a neutral fallback -- it
asserts that the FastText and model embedding spaces are already aligned,
which is precisely the claim Sec 4.1 introduces W to avoid making. Choosing
it silently would substitute an implementation decision for a specification
the paper does not give, and would materially affect results while looking
like a faithful run.

W is frozen -- excluded from the optimizer and carrying
`requires_grad=False`.

No gradient path is manufactured for W. Constructing one would require
inventing a loss term the paper does not state, which would be implementing
a different method while reporting it as LGSE. `AlignmentProjection`
accepts `trainable=True` for a future run under an author-supplied
objective, but nothing in this repository sets it, and setting it does not
by itself create a gradient path.

Resolving this requires the authors to state which objective trains W.
See DEVIATIONS.md section 1a.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class MissingAlignmentMatrix(ValueError):
    """Raised when no author-supplied W is available.

    A distinct type because this is not a misconfiguration to patch with a
    default -- it marks a genuinely unspecified part of the method.
    """


class AlignmentProjection(nn.Module):
    """Square alignment matrix W (d x d), externally supplied.

    W is a *given* of the run, never something this pipeline fits or
    invents. `weight` is required: there is no default, not even the
    identity. The identity would be an implementation choice the paper does
    not describe, and one that materially affects results -- it asserts that
    the FastText and model embedding spaces are already aligned, which is
    exactly the claim Sec 4.1 introduces W to avoid having to make.

    W is **frozen**: `requires_grad=False`, excluded from the optimizer.
    This is not an oversight but the documented consequence of the paper
    specifying no objective that trains it (see DEVIATIONS.md section 1a).
    Marking it trainable while nothing differentiates it would report a
    capability the run does not have.

    `trainable=True` exists only for a future run under an author-supplied
    objective. Nothing in this repository sets it, and setting it alone does
    not create a gradient path -- it only makes W eligible for one.
    """

    def __init__(self, dim: int, weight: torch.Tensor,
                 trainable: bool = False, source: str = "unspecified"):
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
                "user-facing error and DEVIATIONS.md section 1a.")
        if tuple(weight.shape) != (dim, dim):
            raise ValueError(
                f"alignment matrix W must be square {dim}x{dim}, got "
                f"{tuple(weight.shape)} from {source}")
        w = weight.to(dtype=torch.float32)

        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(w)
        self.linear.weight.requires_grad = bool(trainable)

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
    """Read an author-supplied W from a .pt or .npy file.

    The file must contain exactly a d x d matrix. Nothing is reshaped,
    transposed or padded to make a mismatched file fit -- see the note on
    dimensions in check_dimensions.
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
                     alignment_matrix_path=None,
                     trainable: bool = False) -> nn.Module:
    """Return the square alignment matrix W (paper Sec 4.1).

    `alignment_matrix_path` supplies an author-provided W; without one, W is
    the identity. W is frozen unless `trainable=True`, which no configured
    run sets -- see this module's docstring for why.

    Raises `IncompatibleFastTextDimension` on a dimension mismatch: W is
    square, so unequal dimensions mean the FastText model does not match the
    target embedding width. That is a data problem to fix by training
    FastText at dimension d, not something to paper over with a rectangular
    map.
    """
    check_dimensions(source_dim, target_dim)

    if not alignment_matrix_path:
        raise MissingAlignmentMatrix(
            f"No alignment matrix W was supplied.\n"
            f"\n"
            f"Paper Sec 4.1 introduces W:\n"
            f"    \"a learned linear projection W in R^(d x d) is applied,\n"
            f"     i.e. e_aligned_t = W e_t\"\n"
            f"but never states how W is obtained -- no initialization, no\n"
            f"fitting procedure, and no training objective anywhere in the\n"
            f"paper is a function of W.\n"
            f"\n"
            f"W is therefore an author-supplied artifact. It cannot be\n"
            f"derived from the paper, and this implementation will not\n"
            f"choose one on the authors' behalf:\n"
            f"\n"
            f"  - The identity would assert that the FastText and model\n"
            f"    embedding spaces are already aligned -- exactly the claim\n"
            f"    W exists to avoid making.\n"
            f"  - A random or fitted W would be an alignment strategy the\n"
            f"    paper does not describe.\n"
            f"\n"
            f"Either would materially affect results while looking like a\n"
            f"faithful run.\n"
            f"\n"
            f"To proceed, set `lgse.alignment_matrix_path` to a .pt/.npy\n"
            f"file holding a {source_dim}x{source_dim} matrix.\n"
            f"\n"
            f"Any result produced without an author-provided W is NOT\n"
            f"faithful to the published method. See DEVIATIONS.md 1a.")

    weight = load_alignment_matrix(alignment_matrix_path, source_dim)
    return AlignmentProjection(source_dim, weight=weight,
                               trainable=trainable,
                               source=str(alignment_matrix_path))
