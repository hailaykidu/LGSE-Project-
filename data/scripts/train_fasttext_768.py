"""
train_fasttext_768.py -- FastText vectors at the model's embedding width.

Sec 4.1 defines W as a square d x d map, so the FastText space must match
XLM-R's 768 dimensions. The released vectors for both languages are 300-dim
(`data/fasttext_manifest.json`), which `check_dimensions` rejects rather
than reshaping -- padding 300 to 768 would invent 468 dimensions of signal
and a rectangular map would be a different method from the paper's.

This trains replacements at dim=768 with the command the README documents:

    fasttext skipgram -input <corpus> -output <model> -dim 768

Corpora are the monolingual sides of the same NLLB bitext used elsewhere in
this work, which keeps the vector vocabulary in the same domain as the LAPT
and downstream data. The corpus path is recorded in the manifest so a later
reader can tell which text produced which vectors.

Usage:
    python data/scripts/train_fasttext_768.py --language ti --corpus PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fasttext

DIM = 768


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=("am", "ti"), required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out-dir", type=Path,
                   default=Path.home() / ".cache/lgse/fasttext768")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"MISSING CORPUS: {args.corpus}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"fasttext_{args.language}_{DIM}"

    print(f"training {args.language} skipgram dim={DIM} on {args.corpus}")
    model = fasttext.train_unsupervised(
        str(args.corpus),
        model="skipgram",
        dim=DIM,
        epoch=args.epochs,
        minCount=args.min_count,
        thread=args.threads,
    )
    model.save_model(f"{stem}.bin")

    path = Path(f"{stem}.bin")
    record = {
        "language": {"am": "amharic", "ti": "tigrinya"}[args.language],
        "path": str(path),
        "dimension": model.get_dimension(),
        "vocab_size": len(model.get_words()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "source": (f"trained here: fasttext skipgram dim={DIM} "
                   f"epoch={args.epochs} minCount={args.min_count}"),
        "corpus": str(args.corpus),
        "corpus_sha256": sha256(args.corpus),
        # Recorded because the 300-dim released vectors remain on disk and a
        # reader must be able to tell which set produced a given result.
        "replaces": "300-dim vectors in data/fasttext_manifest.json",
    }
    (args.out_dir / f"manifest_{args.language}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert model.get_dimension() == DIM, model.get_dimension()
    print(f"  dim {model.get_dimension()}  vocab {len(model.get_words()):,}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
