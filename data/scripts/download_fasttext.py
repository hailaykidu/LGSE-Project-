"""
download_fasttext.py -- fetch the pretrained FastText models.

The released repository ships `data/fasttext_Amharic.bin` and
`data/fasttext_Tigriyna.bin` as 208-byte text files containing download
URLs, not FastText binaries; loading either raises
`ValueError: ... has wrong file format!`. The real models are ~2.6 GB and
~1.1 GB and are not committed here.

  Amharic   cc.am.300.bin from the FastText CC vectors (Grave et al., 2018)
  Tigrinya  Hailay/fasttext-tigrinya on the HuggingFace Hub

Both are 300-dimensional, which is why a projection into the model's
embedding space is needed at all (see src/lgse/projection.py).

Nothing synthetic is generated: if a model cannot be downloaded the script
fails rather than substituting random vectors, because a placeholder would
silently turn LGSE into its own character-n-gram fallback.
"""

import argparse
import hashlib
import json
from pathlib import Path

AMHARIC_URL = ("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/"
               "cc.am.300.bin.gz")
TIGRINYA_REPO = "Hailay/fasttext-tigrinya"
TIGRINYA_FILE = "fasttext_tigrinya.bin"
CACHE = Path.home() / ".cache" / "lgse"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_amharic() -> Path:
    import gzip
    import shutil
    import urllib.request

    CACHE.mkdir(parents=True, exist_ok=True)
    final = CACHE / "cc.am.300.bin"
    if final.exists():
        return final
    archive = CACHE / "cc.am.300.bin.gz"
    if not archive.exists():
        print(f"downloading {AMHARIC_URL} (~2.6 GB)")
        urllib.request.urlretrieve(AMHARIC_URL, archive)
    print("decompressing")
    with gzip.open(archive, "rb") as src, open(final, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return final


def fetch_tigrinya() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(TIGRINYA_REPO, TIGRINYA_FILE))


def verify(path: Path, language: str) -> dict:
    """Load the model and confirm it behaves like a FastText model."""
    import fasttext

    model = fasttext.load_model(str(path))
    dim, vocab = model.get_dimension(), len(model.get_words())
    if dim <= 0 or vocab <= 0:
        raise SystemExit(f"{path} loaded but is empty (dim={dim}, vocab={vocab})")
    return {"language": language, "path": str(path), "dimension": dim,
            "vocab_size": vocab, "sha256": sha256(path),
            "size_bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser(description="Download FastText models")
    p.add_argument("--language", choices=("amharic", "tigrinya", "both"),
                   default="both")
    p.add_argument("--manifest", type=Path,
                   default=Path("data/fasttext_manifest.json"))
    args = p.parse_args()

    records = {}
    if args.language in ("tigrinya", "both"):
        records["tigrinya"] = verify(fetch_tigrinya(), "tigrinya")
        records["tigrinya"]["source"] = f"https://huggingface.co/{TIGRINYA_REPO}"
    if args.language in ("amharic", "both"):
        records["amharic"] = verify(fetch_amharic(), "amharic")
        records["amharic"]["source"] = AMHARIC_URL

    for lang, r in records.items():
        print(f"  {lang:9} dim={r['dimension']} vocab={r['vocab_size']:,} "
              f"{r['size_bytes'] / 1e9:.2f} GB")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.manifest.exists():
        existing = json.load(open(args.manifest, encoding="utf-8"))
    existing.update(records)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
