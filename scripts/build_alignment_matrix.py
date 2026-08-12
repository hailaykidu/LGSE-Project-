"""
build_alignment_matrix.py -- W by orthogonal Procrustes on shared vocabulary.

W is the square d x d projection of Sec 4.1, applied to FastText vectors
before they are written into the model's embedding space.

This script constructs W by orthogonal Procrustes alignment. W is computed
here, before any training, and used as a precomputed matrix, matching
`AlignmentProjection`, which holds it with `requires_grad=False`.

METHOD (an implementation choice; see IMPLEMENTATION_NOTES.md section 1a)
------------------------------------------------------------------------
Orthogonal Procrustes on anchor tokens present in both spaces:

    W* = argmin_{W: W^T W = I}  || W X - Y ||_F
       = U V^T   where   U S V^T = SVD(Y X^T)

  X   FastText vectors for the anchors      (d x n)
  Y   XLM-R input embeddings for the same   (d x n)

The orthogonality constraint makes this a rotation of the FastText space onto
XLM-R's, without rescaling, so distances among FastText neighbours are
preserved. An unconstrained least-squares solution fits the anchors more
closely but may rescale or collapse directions.

ANCHORS
-------
Tokens that XLM-R already represents and FastText also covers: the tokens for
which both spaces hold an independent representation. New tokens are excluded
by construction, since their XLM-R rows are what LGSE produces.

The XLM-R side is read with the sentencepiece marker stripped, since FastText
is trained on whitespace-tokenized text.

Usage:
    python scripts/build_alignment_matrix.py --language ti
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 42
MARKER = "▁"  # sentencepiece word-start marker


def procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return the orthogonal W minimizing ||W X - Y||_F.

    X and Y are (d x n): one column per anchor token. The solution is
    U V^T from the SVD of Y X^T.
    """
    if X.shape != Y.shape:
        raise ValueError(f"anchor matrices must match, got {X.shape} and {Y.shape}")
    U, _S, Vt = np.linalg.svd(Y @ X.T)
    return U @ Vt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=("am", "ti"), required=True)
    p.add_argument("--fasttext", type=Path, default=None,
                   help="FastText .bin at the model's embedding width")
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--out-dir", type=Path, default=Path("data/alignment"))
    p.add_argument("--min-anchors", type=int, default=1000,
                   help="minimum shared vocabulary required to write a matrix")
    args = p.parse_args()

    import fasttext
    import torch
    from transformers import AutoModel, AutoTokenizer

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ft_path = args.fasttext or (
        Path.home() / f".cache/lgse/fasttext768/fasttext_{args.language}_768.bin")
    if not ft_path.exists():
        raise SystemExit(
            f"FastText model not found: {ft_path}\n"
            "  Train one at the model's embedding width, or pass --fasttext.")

    print(f"loading FastText {ft_path}")
    ft = fasttext.load_model(str(ft_path))
    ft_dim = ft.get_dimension()
    ft_vocab = set(ft.get_words())

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    xlmr_dim = emb.shape[1]

    if ft_dim != xlmr_dim:
        raise SystemExit(
            f"Dimension mismatch: FastText {ft_dim}, {args.model} {xlmr_dim}.\n"
            "  W is square; neither space is padded or truncated to fit.\n"
            "  Train FastText at the model's embedding width.")

    # Anchors: tokens both spaces represent.
    xs, ys, anchors = [], [], []
    for token, index in tokenizer.get_vocab().items():
        surface = token[1:] if token.startswith(MARKER) else token
        if len(surface) < 2 or surface not in ft_vocab:
            continue
        xs.append(ft.get_word_vector(surface))
        ys.append(emb[index])
        anchors.append(surface)

    n = len(anchors)
    print(f"anchor tokens: {n:,}")
    if n < args.min_anchors:
        raise SystemExit(
            f"Too few anchors: {n} < {args.min_anchors} shared tokens.")

    X = np.asarray(xs, dtype=np.float64).T          # d x n
    Y = np.asarray(ys, dtype=np.float64).T          # d x n

    W = procrustes(X, Y)

    orthogonality_error = float(np.abs(W @ W.T - np.eye(xlmr_dim)).max())
    residual = float(np.linalg.norm(W @ X - Y, "fro"))
    baseline = float(np.linalg.norm(X - Y, "fro"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"W_{args.language}"
    np.save(f"{stem}.npy", W.astype(np.float32))

    record = {
        "language": {"am": "amharic", "ti": "tigrinya"}[args.language],
        "path": str(Path(f"{stem}.npy").resolve()),
        "anchor_tokens": n,
        "fasttext_dimension": ft_dim,
        "xlmr_dimension": xlmr_dim,
        "shape": list(W.shape),
        "objective": (
            "orthogonal Procrustes: W* = argmin_{W^T W = I} ||W X - Y||_F, "
            "solved as U V^T from SVD(Y X^T)"),
        "seed": SEED,
        "fasttext_model": str(ft_path),
        "fasttext_vocab_size": len(ft_vocab),
        "backbone": args.model,
        "residual_frobenius": round(residual, 4),
        "unaligned_frobenius": round(baseline, 4),
        "orthogonality_max_abs_error": orthogonality_error,
        "source": (
            "implementation choice: orthogonal Procrustes alignment, "
            "precomputed before training and used frozen"),
    }
    (args.out_dir / f"W_{args.language}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"  FastText dim {ft_dim}  {args.model} dim {xlmr_dim}  W {W.shape}")
    print(f"  orthogonality max|WW^T - I| = {orthogonality_error:.2e}")
    print(f"  ||WX - Y||_F {residual:.2f}  (unaligned {baseline:.2f})")
    print(f"  seed {SEED}")
    print(f"  wrote {stem}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
