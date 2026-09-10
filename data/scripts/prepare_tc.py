"""
prepare_tc.py -- build the Tigrinya text-classification split from FineWeb-C.

Source: `data-is-better-together/fineweb-c`, config `tir_Ethi` -- 745
community-annotated Tigrinya documents rated for educational quality. This is
the annotation effort behind the Tigrinya TC column of Table 2; the Amharic
column uses `data/Educational classifier.csv`.

Label mapping. FineWeb-C rates on a named scale; the Amharic file uses the
numeric 1-6 scale of the paper's Sec 7. The two are the same rubric under
different notation, so the named labels are mapped onto the numbers rather
than either file being rewritten:

    None                      -> 1   (no educational value)
    Minimal                   -> 2
    Basic                     -> 3
    Good                      -> 4
    Excellent                 -> 5
    ❗ Problematic Content ❗   -> 6

Multi-annotator resolution. Every document carries labels from 2-4
annotators. A single label per document is required for classification, and
the choice of how to resolve disagreement is a real decision, so it is stated
rather than buried:

  * **majority** (default) -- the most frequent label; ties broken toward the
    lower rating, which is the conservative reading of educational value.
  * **first** -- the first annotator's label, for comparison.
  * **strict** -- only documents where all annotators agree, which trades
    quantity for label quality.

The problematic-content flag is *not* folded into the rating. A document can
be both problematic and educationally basic, and FineWeb-C records those
separately; collapsing them would lose that distinction.

Usage:
    python data/scripts/prepare_tc.py --out data/tc/tigrinya
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

DATASET = "data-is-better-together/fineweb-c"
CONFIG = "tir_Ethi"
ROWS_URL = "https://datasets-server.huggingface.co/rows"

#: FineWeb-C's named scale onto the paper's numeric 1-6 scale.
LABEL_MAP = {
    "None": "1",
    "Minimal": "2",
    "Basic": "3",
    "Good": "4",
    "Excellent": "5",
    "❗ Problematic Content ❗": "6",
}


def fetch_rows(limit: int = 1000) -> list[dict]:
    """Page through the datasets-server rows endpoint."""
    rows: list[dict] = []
    offset = 0
    while offset < limit:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=120) as response:
            batch = json.load(response).get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
    return rows


def resolve_label(labels: list[str], strategy: str) -> str | None:
    """One rating per document from several annotators' labels."""
    mapped = [LABEL_MAP[l] for l in labels if l in LABEL_MAP]
    if not mapped:
        return None
    if strategy == "first":
        return mapped[0]
    if strategy == "strict":
        return mapped[0] if len(set(mapped)) == 1 else None
    counts = Counter(mapped)
    top = max(counts.values())
    # Ties resolve toward the lower rating: the conservative reading of
    # educational value when annotators disagree.
    return min(label for label, n in counts.items() if n == top)


def stratified_split(rows, seed: int, dev_frac=0.1, test_frac=0.1):
    """80/10/10 per Sec 7, stratified, with duplicate texts grouped.

    Mirrors `src/evaluation/run_tc.py` so a split written here and one derived
    in-process agree in construction.
    """
    import random

    grouped: dict[str, list] = {}
    for text, label in rows:
        grouped.setdefault(text, []).append((text, label))

    by_label: dict[str, list] = {}
    for group in grouped.values():
        by_label.setdefault(group[0][1], []).append(group)

    rng = random.Random(seed)
    train, dev, test = [], [], []
    for _, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_dev, n_test = int(round(n * dev_frac)), int(round(n * test_frac))
        if n - n_dev - n_test < 1:
            n_dev = n_test = 0
        dev += [r for g in items[:n_dev] for r in g]
        test += [r for g in items[n_dev : n_dev + n_test] for r in g]
        train += [r for g in items[n_dev + n_test :] for r in g]

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return {"train": train, "dev": dev, "test": test}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/tc/tigrinya"))
    p.add_argument("--cache", type=Path, default=None,
                   help="pre-downloaded rows JSON, to avoid refetching")
    p.add_argument("--strategy", choices=("majority", "first", "strict"),
                   default="majority")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-chars", type=int, default=1,
                   help="drop documents shorter than this")
    args = p.parse_args()

    if args.cache and args.cache.exists():
        raw = json.loads(args.cache.read_text(encoding="utf-8"))
    else:
        raw = fetch_rows()
    print(f"source rows: {len(raw)}")

    rows, dropped_unlabelled, dropped_short = [], 0, 0
    for item in raw:
        label = resolve_label(item.get("educational_value_labels") or [],
                              args.strategy)
        text = (item.get("text") or "").strip()
        if label is None:
            dropped_unlabelled += 1
            continue
        if len(text) < args.min_chars:
            dropped_short += 1
            continue
        rows.append((text, label))

    print(f"labelled: {len(rows)}  "
          f"dropped: {dropped_unlabelled} unresolved, {dropped_short} empty")
    print(f"distribution: {dict(sorted(Counter(l for _, l in rows).items()))}")

    splits = stratified_split(rows, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with open(args.out / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Text", "Label"])
            writer.writerows(items)
        print(f"  {name}.csv: {len(items)} rows")

    manifest = {
        "source": f"{DATASET} (config {CONFIG})",
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "documents": len(raw),
        "labelled": len(rows),
        "label_scale": "paper Sec 7, 1-6",
        "label_map": LABEL_MAP,
        "annotator_resolution": args.strategy,
        "tie_break": "lower rating",
        "problematic_flag_folded_into_rating": False,
        "split": "stratified 80/10/10, duplicate texts grouped",
        "seed": args.seed,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "label_distribution": {
            k: dict(sorted(Counter(l for _, l in v).items()))
            for k, v in splits.items()
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
