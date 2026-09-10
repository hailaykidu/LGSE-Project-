"""
build_alignment_matrix.py -- W by orthogonal Procrustes on shared vocabulary.

The paper specifies a learned projection matrix W in R^{d x d} (Sec 4.1) but
does not specify how W is obtained. No loss stated in the paper is a function
of W: Sec 4.2's L_reg anchors to a constant mu, and Sec 5's LAPT applies MLM
to the embedding matrix with the encoder frozen. Initialization writes W's
output into the embedding through `.data`, which severs the autograd graph.
There is therefore no gradient path by which LAPT could fit W.

To implement the pipeline, this repository uses orthogonal Procrustes
alignment to instantiate W. W is computed *here*, before any training, and
used as a precomputed alignment matrix -- consistent with
`AlignmentProjection`, which holds it frozen with `requires_grad=False`.

METHOD (used to implement the pipeline; not specified by the paper)
-------------------------------------------------------------------
Orthogonal Procrustes on anchor tokens present in both spaces:

    W* = argmin_{W: W^T W = I}  || W X - Y ||_F
       = U V^T   where   U S V^T = SVD(Y X^T)

  X   FastText vectors for the anchors      (d x n)
  Y   XLM-R input embeddings for the same   (d x n)

Orthogonality is what makes this defensible as an *alignment* rather than a
fitted regression: it rotates the FastText space onto XLM-R's without
rescaling or distorting relative geometry, so distances among FastText
neighbours survive the map. An unconstrained least-squares solution would fit
the anchors better while being free to collapse directions, which is a
different claim about the two spaces.

ANCHORS
-------
Tokens that XLM-R already represents and FastText also covers. These are
exactly the tokens for which both spaces hold an independent opinion, so they
are the only evidence available about how the spaces relate. New tokens are
excluded by construction -- their XLM-R rows are what LGSE exists to produce,
so using them would be circular.

The XLM-R side is read with the sentencepiece marker stripped, since FastText
is trained on whitespace-tokenized text and would otherwise miss nearly every
subword.

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=("am", "ti"), required=True)
    p.add_argument("--fasttext", type=Path, default=None,
                   help="768-dim .bin; defaults to the cache path")
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--out-dir", type=Path, default=Path("data/alignment"))
    p.add_argument("--min-anchors", type=int, default=1000,
                   help="refuse to write a matrix fitted on fewer anchors")
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
            f"MISSING 768-dim FASTTEXT: {ft_path}\n"
            "  Train it first: data/scripts/train_fasttext_768.py")

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
            f"DIMENSION MISMATCH: FastText {ft_dim} vs XLM-R {xlmr_dim}.\n"
            "  Sec 4.1's W is square; neither space is padded or truncated "
            "to fit. Train FastText at the model's width.")

    # Anchors: tokens both spaces represent independently.
    rows, xs, ys, anchors = [], [], [], []
    for token, index in tokenizer.get_vocab().items():
        surface = token[1:] if token.startswith(MARKER) else token
        if len(surface) < 2 or surface not in ft_vocab:
            continue
        xs.append(ft.get_word_vector(surface))
        ys.append(emb[index])
        anchors.append(surface)
        rows.append(index)

    n = len(anchors)
    print(f"anchor tokens: {n:,}")
    if n < args.min_anchors:
        raise SystemExit(
            f"TOO FEW ANCHORS: {n} < {args.min_anchors}. A rotation fitted on "
            "this little shared vocabulary would not be an alignment of the "
            "two spaces.")

    X = np.asarray(xs, dtype=np.float64).T          # d x n
    Y = np.asarray(ys, dtype=np.float64).T          # d x n

    # W* = U V^T from SVD(Y X^T): the orthogonal map closest to taking X to Y.
    U, S, Vt = np.linalg.svd(Y @ X.T)
    W = U @ Vt

    orthogonality_error = float(
        np.abs(W @ W.T - np.eye(xlmr_dim)).max())
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
        "status": (
            "The paper specifies a learned projection matrix W but does not "
            "specify how W is obtained. To implement the pipeline, this "
            "repository uses orthogonal Procrustes alignment to instantiate "
            "W. This matrix is precomputed here and used frozen."),
    }
    (args.out_dir / f"W_{args.language}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"  FastText dim {ft_dim}  XLM-R dim {xlmr_dim}  W {W.shape}")
    print(f"  orthogonality max|WW^T - I| = {orthogonality_error:.2e}")
    print(f"  ||WX - Y||_F {residual:.2f}  (unaligned {baseline:.2f})")
    print(f"  seed {SEED}")
    print(f"  wrote {stem}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
