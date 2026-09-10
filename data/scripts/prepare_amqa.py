"""
prepare_amqa.py -- Amharic QA splits from the AmQA release.

Source: https://github.com/semantic-systems/amharic-qa (MIT). The release
ships official `train_data.json` / `dev_data.json` / `test_data.json` in
SQuAD v2.0 layout, so no partition is derived here -- the splits are used
as released, matching how MasakhaNER is handled for Amharic NER.

Two departures from a byte-copy, both mechanical and both counted in the
manifest rather than applied silently:

**One malformed article.** In `dev_data.json` a single article stores its
paragraph as a dict where every other article uses a list of dicts. The
content is intact; only the container differs. It is wrapped in a list so
the file parses uniformly. Dropping the article would discard usable
questions over a JSON typo.

**Off-by-one answer offsets.** Twenty answers across train and dev have an
`answer_start` one character early, pointing at the space before the
answer: the span at the recorded offset is `' ቤጂንግ እና በሀቤይ'` where the
answer text is `'ቤጂንግ እና በሀቤይ '`. Span-extraction training reads the
offset, not the text, so an uncorrected record teaches a boundary shifted
by one. Each is repaired only when the answer text occurs verbatim
elsewhere in the context and the correction is unambiguous; anything that
does not resolve cleanly is dropped and counted.

No answers are absent and none carry `answer_start == -1`, so unlike TIGQA
there is no abstractive or unanswerable subset to separate.

Usage:
    python data/scripts/prepare_amqa.py --out data/qa/amqa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/semantic-systems/amharic-qa"
FILES = {"train": "train_data.json", "dev": "dev_data.json",
         "test": "test_data.json"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paragraphs_of(article: dict) -> list[dict]:
    """Paragraph list for an article, tolerating the dict-wrapped case."""
    paragraphs = article.get("paragraphs", [article])
    if isinstance(paragraphs, dict):
        return [paragraphs]
    return paragraphs


def fix_answer(context: str, answer: dict, stats: dict) -> dict | None:
    """Repair an off-by-one offset, or drop the answer if it cannot be.

    Returns the answer with a verified offset, or None when the text does
    not occur in the context at all -- a record whose answer is unfindable
    cannot be scored by span extraction and would train against a wrong
    span if kept.
    """
    text = answer.get("text", "")
    start = answer.get("answer_start", -1)
    if not text:
        stats["dropped_empty_text"] += 1
        return None

    if 0 <= start and context[start:start + len(text)] == text:
        return answer

    found = context.find(text)
    if found < 0:
        stats["dropped_text_not_in_context"] += 1
        return None

    # Unambiguous only if the text occurs exactly once; otherwise the
    # original offset identified a specific occurrence and guessing which
    # would silently relabel the example.
    if context.count(text) > 1:
        nearest = min(
            (i for i in range(len(context)) if context.startswith(text, i)),
            key=lambda i: abs(i - start))
        if abs(nearest - start) > 2:
            stats["dropped_ambiguous"] += 1
            return None
        found = nearest

    stats["offset_corrected"] += 1
    return {**answer, "answer_start": found}


def convert(raw: dict, stats: dict) -> tuple[dict, int]:
    articles, kept = [], 0
    for article in raw.get("data", []):
        out_paragraphs = []
        for para in paragraphs_of(article):
            context = para["context"]
            qas = []
            for qa in para.get("qas", []):
                answers = []
                for answer in qa.get("answers") or []:
                    fixed = fix_answer(context, answer, stats)
                    if fixed is not None:
                        answers.append(fixed)
                if not answers:
                    stats["dropped_questions"] += 1
                    continue
                qas.append({**qa, "answers": answers})
                kept += 1
            if qas:
                out_paragraphs.append({**para, "context": context, "qas": qas})
        if out_paragraphs:
            articles.append({"paragraphs": out_paragraphs})
    return {"version": raw.get("version", "2.0"), "data": articles}, kept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/qa/amqa"))
    p.add_argument("--source", type=Path, default=None,
                   help="local clone; cloned to a temp dir if omitted")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        src = args.source
        if src is None:
            print(f"cloning {REPO}")
            subprocess.run(["git", "clone", "--depth", "1", REPO, tmp],
                           check=True, capture_output=True)
            src = Path(tmp)

        args.out.mkdir(parents=True, exist_ok=True)
        stats = {k: 0 for k in ("offset_corrected", "dropped_empty_text",
                                "dropped_text_not_in_context",
                                "dropped_ambiguous", "dropped_questions")}
        manifest = {
            "source": REPO,
            "license": "MIT",
            "citation": "AmQA, semantic-systems/amharic-qa",
            "split": "official train/dev/test, used as released",
            "format": "SQuAD v2.0",
            "files": {},
        }

        for name, filename in FILES.items():
            raw_bytes = (src / filename).read_bytes()
            raw = json.loads(raw_bytes)
            before = sum(len(para.get("qas", []))
                         for article in raw.get("data", [])
                         for para in paragraphs_of(article))
            converted, kept = convert(raw, stats)
            (args.out / f"{name}.json").write_text(
                json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            manifest["files"][name] = {
                "raw_file": filename,
                "raw_sha256": sha256_bytes(raw_bytes),
                "qas_in_release": before,
                "qas_written": kept,
            }
            print(f"  {name:5} {kept:5} / {before} qas")

        manifest["repairs"] = {
            "dict_wrapped_paragraphs": (
                "one dev article stores paragraphs as a dict; wrapped in a "
                "list, content unchanged"),
            "offset_corrected": stats["offset_corrected"],
            "dropped_empty_text": stats["dropped_empty_text"],
            "dropped_text_not_in_context": stats["dropped_text_not_in_context"],
            "dropped_ambiguous": stats["dropped_ambiguous"],
            "dropped_questions": stats["dropped_questions"],
        }
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    print(f"  offsets corrected: {stats['offset_corrected']}")
    dropped = sum(v for k, v in stats.items() if k.startswith("dropped_")
                  and k != "dropped_questions")
    print(f"  answers dropped:   {dropped}")
    print(f"manifest -> {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
