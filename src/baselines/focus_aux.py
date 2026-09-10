"""focus_aux.py -- auxiliary-space wiring for the FOCUS baseline.

FOCUS (Dobler & de Melo, 2023) initializes a new token as a
similarity-weighted combination of pretrained embedding rows. The
similarities are measured in an *auxiliary* space that represents old and
new tokens alike; here that space is the FastText model already loaded for
LGSE, so both methods see the same external signal.

This module builds the auxiliary table over the pretrained vocabulary and
the per-token lookup for new vocabulary, and asserts loudly that both are
present. The arm previously ran with `aux_vectors=None` and no lookup, in
which case `FocusInit` falls back to the mean pretrained embedding for every
new token -- indistinguishable from `+LAPT`. See
`report/LGSE_FORENSIC_IMPLEMENTATION_AUDIT.md`.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

#: SentencePiece word-boundary marker used by XLM-R's tokenizer.
MARKER = "▁"


def surface_form(token: str) -> str:
    """Token as it appears in FastText: boundary marker stripped."""
    return token[1:] if token.startswith(MARKER) else token


def build_focus_aux_table(
        tokenizer,
        fasttext_model,
        old_vocab_size: int,
        device=None,
) -> Tuple[torch.Tensor, List[int]]:
    """Auxiliary vectors for the *pretrained* vocabulary.

    Returns `(aux_vectors, aux_index)` where `aux_vectors` is an
    (n_aux, d_aux) L2-normalized tensor and `aux_index[i]` is the
    embedding-matrix row that vector describes. Only pretrained rows
    (id < old_vocab_size) are candidates: FOCUS combines embeddings that
    were actually trained, never other new rows.

    Anchor selection matches `scripts/build_alignment_matrix.py`: surface
    form of at least two characters, present in the FastText vocabulary.
    """
    if fasttext_model is None:
        raise ValueError(
            "FOCUS requires a FastText model for its auxiliary space, but "
            "None was supplied. Without it FocusInit returns the mean "
            "pretrained embedding for every new token and the arm reduces "
            "to +LAPT.")

    ft_vocab = set(fasttext_model.get_words())
    vectors: List[np.ndarray] = []
    index: List[int] = []
    for token, token_id in tokenizer.get_vocab().items():
        if token_id >= old_vocab_size:
            continue
        surface = surface_form(token)
        if len(surface) < 2 or surface not in ft_vocab:
            continue
        vectors.append(fasttext_model.get_word_vector(surface))
        index.append(token_id)

    if not vectors:
        raise ValueError(
            "FOCUS auxiliary table is empty: no pretrained token has a "
            "FastText vector. Check that the FastText model matches the "
            "language of the backbone vocabulary.")

    aux = torch.as_tensor(np.asarray(vectors), dtype=torch.float32)
    aux = aux / (aux.norm(dim=1, keepdim=True) + 1e-8)
    # Match the embedding matrix's device: FocusInit multiplies the softmax
    # weights derived from these vectors against pretrained rows read
    # straight off the embedding layer, so a CPU aux table against a CUDA
    # embedding raises at the combination step.
    if device is not None:
        aux = aux.to(device)
    return aux, index


def make_focus_aux_lookup(fasttext_model, device=None):
    """Lookup mapping a *new* token to its auxiliary vector.

    FastText is subword-based, so it returns a vector for unseen strings
    too; `None` is reserved for the degenerate empty-surface case, where
    FocusInit correctly falls back to the mean.

    The vector is returned on `device` so it can be multiplied against the
    auxiliary table without a cross-device error.
    """
    if fasttext_model is None:
        raise ValueError("FOCUS auxiliary lookup requires a FastText model.")

    def lookup(token: str) -> Optional[torch.Tensor]:
        surface = surface_form(token)
        if not surface:
            return None
        vec = torch.as_tensor(
            np.asarray(fasttext_model.get_word_vector(surface)),
            dtype=torch.float32)
        return vec.to(device) if device is not None else vec

    return lookup


def assert_focus_is_wired(initializer, fasttext_model) -> None:
    """Fail loudly if the corrected FOCUS path is not actually active.

    Guards exactly the conditions that made the original defect silent:
    either branch of `FocusInit.init_token_embedding`'s fallback test being
    true means every new token collapses to the mean pretrained embedding.
    """
    if getattr(initializer, "aux_vectors", None) is None:
        raise AssertionError(
            "FOCUS: aux_vectors is None -- every new token would be "
            "initialized to the mean pretrained embedding.")
    if getattr(initializer, "aux_index", None) is None:
        raise AssertionError("FOCUS: aux_index is None.")
    if getattr(initializer, "_aux_lookup", None) is None:
        raise AssertionError(
            "FOCUS: set_aux_lookup() was never called -- every new token "
            "would be initialized to the mean pretrained embedding.")

    n_vec = initializer.aux_vectors.shape[0]
    n_idx = len(initializer.aux_index)
    if n_vec != n_idx:
        raise AssertionError(
            f"FOCUS: aux_vectors has {n_vec} rows but aux_index has "
            f"{n_idx} entries; they must correspond one-to-one.")

    old = initializer.old_vocab_size
    if old is not None and max(initializer.aux_index) >= old:
        raise AssertionError(
            "FOCUS: aux_index points at a new-vocabulary row; the auxiliary "
            "table must only reference pretrained embeddings.")


def assert_focus_init_is_distinct(init_matrix: torch.Tensor,
                                  tolerance: float = 1e-6) -> Dict[str, float]:
    """Fail if the written FOCUS rows collapsed to a single vector.

    This is the post-hoc counterpart to `assert_focus_is_wired`: it checks
    the outcome rather than the plumbing, so a future regression that
    bypasses the wiring check is still caught.
    """
    if init_matrix.ndim != 2 or init_matrix.shape[0] < 2:
        return {}

    rows = init_matrix.detach().to(torch.float32).cpu()
    unique_rows = torch.unique(
        (rows / tolerance).round(), dim=0).shape[0]
    per_dim_std = float(rows.std(dim=0).mean())

    if unique_rows <= 1:
        raise AssertionError(
            "FOCUS: all new-token rows are identical -- the mean-vector "
            "fallback is active despite the wiring check passing.")
    if per_dim_std <= 0.0:
        raise AssertionError(
            "FOCUS: new-token rows have zero variance across tokens.")

    return {
        "unique_rows": float(unique_rows),
        "n_rows": float(rows.shape[0]),
        "per_dim_std": per_dim_std,
        "rms_norm": float(rows.norm(dim=1).pow(2).mean().sqrt()),
    }
