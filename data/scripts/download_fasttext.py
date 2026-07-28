"""
download_fasttext.py -- fetch the pretrained FastText models.

The released repository ships `data/fasttext_Amharic.bin` and
`data/fasttext_Tigriyna.bin` as 208-byte text files containing download
URLs, not FastText binaries; loading either raises
`ValueError: ... has wrong file format!`. The real models are ~2.6 GB and
~1.1 GB and are not committed here.

  Amharic   cc.am.300.bin from the FastText CC vectors (Grave et al., 2018)
  Tigrinya  Hailay/fasttext-tigrinya on the HuggingFace Hub

DIMENSION REQUIREMENT
---------------------
Both of the above are **300-dimensional**, and LGSE's projection W is
**square** (W in R^(d x d), paper Sec 4.1): it aligns the FastText space
with the model's embedding space, both of dimension d, and does not change
dimensionality.

So these downloads are usable only with a model whose embedding width is
300. They are **not** usable with xlm-roberta-base (768-dim), which is the
model the paper uses. Running LGSE with 768-dim XLM-R requires FastText
vectors trained at dimension 768:

    fasttext skipgram -input <corpus> -output <model> -dim 768

This script therefore warns loudly whenever it records a model whose
dimension is not the expected one, so the mismatch surfaces at download
time rather than hours into a run. `--expect-dim` sets the target width.

Nothing synthetic is generated, and nothing is reshaped: if a model cannot
be downloaded the script fails rather than substituting random vectors, and
mismatched vectors are never truncated or padded to fit. Either would
silently change the method -- see DEVIATIONS.md section 1.
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

# The paper's model is xlm-roberta-base, whose embedding space is 768-dim.
# Because W is square, FastText must match it.
DEFAULT_EXPECTED_DIM = 768


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


def verify(path: Path, language: str, expected_dim: int) -> dict:
    """Load the model, confirm it is a FastText model, and check its width.

    A dimension mismatch is recorded in the manifest and warned about here
    rather than raising: downloading is still useful (the file is large and
    slow to fetch), but the run that consumes it will refuse to start, so
    the warning must be impossible to miss.
    """
    import fasttext

    model = fasttext.load_model(str(path))
    dim, vocab = model.get_dimension(), len(model.get_words())
    if dim <= 0 or vocab <= 0:
        raise SystemExit(f"{path} loaded but is empty (dim={dim}, vocab={vocab})")

    record = {"language": language, "path": str(path), "dimension": dim,
              "vocab_size": vocab, "sha256": sha256(path),
              "size_bytes": path.stat().st_size,
              "expected_dimension": expected_dim,
              "dimension_ok": dim == expected_dim}

    if dim != expected_dim:
        print(
            f"\n"
            f"  !!  WARNING: {language} FastText is {dim}-dim, but "
            f"{expected_dim}-dim is required.\n"
            f"  !!\n"
            f"  !!  LGSE's projection W is square (paper Sec 4.1), so "
            f"FastText must match\n"
            f"  !!  the model's embedding width. LGSE will REFUSE TO RUN "
            f"with this model.\n"
            f"  !!\n"
            f"  !!  These vectors are not reshaped or truncated to fit -- "
            f"that would\n"
            f"  !!  silently change the method. Train FastText at "
            f"dimension {expected_dim}:\n"
            f"  !!\n"
            f"  !!    fasttext skipgram -input <corpus> -output <model> "
            f"-dim {expected_dim}\n"
            f"  !!\n"
            f"  !!  See DEVIATIONS.md section 1.\n")

    return record


def main():
    p = argparse.ArgumentParser(description="Download FastText models")
    p.add_argument("--language", choices=("amharic", "tigrinya", "both"),
                   default="both")
    p.add_argument("--manifest", type=Path,
                   default=Path("data/fasttext_manifest.json"))
    p.add_argument("--expect-dim", type=int, default=DEFAULT_EXPECTED_DIM,
                   help="required FastText width; must equal the model's "
                        "embedding dimension because LGSE's W is square "
                        f"(default {DEFAULT_EXPECTED_DIM}, xlm-roberta-base)")
    args = p.parse_args()

    records = {}
    if args.language in ("tigrinya", "both"):
        records["tigrinya"] = verify(fetch_tigrinya(), "tigrinya",
                                     args.expect_dim)
        records["tigrinya"]["source"] = f"https://huggingface.co/{TIGRINYA_REPO}"
    if args.language in ("amharic", "both"):
        records["amharic"] = verify(fetch_amharic(), "amharic",
                                    args.expect_dim)
        records["amharic"]["source"] = AMHARIC_URL

    for lang, r in records.items():
        status = "OK" if r["dimension_ok"] else \
            f"UNUSABLE (need dim {r['expected_dimension']})"
        print(f"  {lang:9} dim={r['dimension']} vocab={r['vocab_size']:,} "
              f"{r['size_bytes'] / 1e9:.2f} GB  [{status}]")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.manifest.exists():
        existing = json.load(open(args.manifest, encoding="utf-8"))
    existing.update(records)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"manifest -> {args.manifest}")

    unusable = [l for l, r in records.items() if not r["dimension_ok"]]
    if unusable:
        raise SystemExit(
            f"\n{len(unusable)} model(s) have the wrong dimension: "
            f"{', '.join(unusable)}.\n"
            f"They are downloaded and recorded, but LGSE will refuse to run "
            f"with them.\nSee the dimension requirement in this script's "
            f"docstring and DEVIATIONS.md section 1.")


if __name__ == "__main__":
    main()
