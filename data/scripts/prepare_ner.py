"""
prepare_ner.py -- build NER train/dev/test splits (paper Sec 6.3).

Amharic  MasakhaNER (Adelani et al., 2021), which ships official splits.
Tigrinya Yohannes and Amagasa (2022). No official split is provided, so the
         paper creates one by randomly splitting 80/10/10. That partition is
         reproduced here with an explicit seed so it is reproducible; the
         paper does not state its seed, so ours is recorded rather than
         claimed to match.

Downloads nothing large into the repository: the raw corpora are fetched to
a cache directory and only the derived splits are written.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

TIGRINYA_URL = ("https://raw.githubusercontent.com/mehari-eng/"
                "Tigrinya-NER/main/Tigrinya-NER-Dataset.txt")

# MasakhaNER ships official train/dev/test splits, so no partition is
# derived for Amharic -- the released files are used as-is. The HuggingFace
# mirrors are script-based datasets, which current `datasets` refuses to
# load, so the CoNLL files are taken from the project's own repository.
MASAKHANER_BASE = ("https://raw.githubusercontent.com/masakhane-io/"
                   "masakhane-ner/main/data/amh")

# A handful of tags in the Tigrinya release are typos for a valid tag
# ('B-DATe' for 'B-DATE') or malformed entirely ('IO', 'BO', 'B-LO').
# Mapping the typo and dropping the rest keeps the label set well formed.
TAG_FIXES = {"B-DATe": "B-DATE"}
VALID_PREFIXES = ("B-", "I-")


def normalize_tag(tag: str) -> str:
    tag = TAG_FIXES.get(tag, tag)
    if tag == "O" or tag.startswith(VALID_PREFIXES):
        return tag
    return "O"          # malformed -> outside


def read_conll(path: Path):
    """CoNLL: one `token tag` per line, blank line between sentences."""
    sentences, current = [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.replace("\r", "").rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                current.append((parts[0], normalize_tag(parts[-1])))
    if current:
        sentences.append(current)
    return sentences


def split_80_10_10(sentences, seed: int):
    """The paper's random 80/10/10 partition, with an explicit seed."""
    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_train, n_dev = int(0.8 * n), int(0.1 * n)
    parts = {
        "train": idx[:n_train],
        "dev": idx[n_train:n_train + n_dev],
        "test": idx[n_train + n_dev:],
    }
    return {k: [sentences[i] for i in v] for k, v in parts.items()}


def write_split(sentences, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for token, tag in sent:
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Prepare NER splits")
    p.add_argument("--language", choices=("amharic", "tigrinya"), required=True)
    p.add_argument("--raw", type=Path, default=None,
                   help="local raw file; downloaded if omitted (Tigrinya)")
    p.add_argument("--out-dir", type=Path, default=Path("data/ner"))
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the 80/10/10 split (Tigrinya only)")
    args = p.parse_args()

    if args.language == "amharic":
        import urllib.request
        out = args.out_dir / "amharic"
        out.mkdir(parents=True, exist_ok=True)
        cache = Path.home() / ".cache" / "lgse"
        cache.mkdir(parents=True, exist_ok=True)

        manifest = {
            "source": MASAKHANER_BASE,
            "citation": "Adelani et al. (2021), MasakhaNER",
            "split": "official MasakhaNER splits, used as released",
            "files": {},
        }
        for name in ("train", "dev", "test"):
            local = cache / f"masakhaner_amh_{name}.txt"
            if not local.exists():
                url = f"{MASAKHANER_BASE}/{name}.txt"
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, local)
            sentences = read_conll(local)
            write_split(sentences, out / f"{name}.conll")
            manifest["files"][name] = {
                "sha256": sha256(local),
                "sentences": len(sentences),
                "tokens": sum(len(s) for s in sentences),
            }
            print(f"  {name:5} {len(sentences):5} sentences")
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"manifest -> {out / 'manifest.json'}")
        return

    raw = args.raw
    if raw is None:
        import urllib.request
        cache = Path.home() / ".cache" / "lgse"
        cache.mkdir(parents=True, exist_ok=True)
        raw = cache / "Tigrinya-NER-Dataset.txt"
        if not raw.exists():
            print(f"downloading {TIGRINYA_URL}")
            urllib.request.urlretrieve(TIGRINYA_URL, raw)

    sentences = read_conll(raw)
    splits = split_80_10_10(sentences, args.seed)

    out = args.out_dir / "tigrinya"
    manifest = {
        "source": TIGRINYA_URL,
        "citation": "Yohannes and Amagasa (2022)",
        "raw_sha256": sha256(raw),
        "split": "random 80/10/10 (paper Sec 6.3); seed recorded below",
        "seed": args.seed,
        "sentences": len(sentences),
        "counts": {k: len(v) for k, v in splits.items()},
    }
    for name, sents in splits.items():
        write_split(sents, out / f"{name}.conll")
        print(f"  {name:5} {len(sents):5} sentences")
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
