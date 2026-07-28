"""
prepare_qa.py -- build QA train/dev/test splits (paper Sec 6.3).

Amharic  AmQA (Taffa et al., 2024), released as SQuAD 2.0 JSON with
         answer_start offsets. Ships without splits, so an 80/10/10
         partition is derived here with an explicit seed.
Tigrinya TIGQA (Teklehaymanot et al., 2024), Zenodo record 11423987,
         CC-BY-4.0. Released as a .docx table, not a dataset: columns are
         R/no, Grade level, Topic, Context, Question, Answer, with several
         numbered question-answer pairs packed into single cells.

TIGQA answers are free text with no character offsets. Extractive QA needs
`answer_start`, so this script locates each answer inside its context by
exact string match and **drops** pairs where no match exists rather than
guessing an offset. The number dropped is recorded in the manifest: it is a
measure of how much of the released dataset is usable for extractive QA,
not a detail to hide.
"""

import argparse
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TIGQA_URL = ("https://zenodo.org/records/11423987/files/"
             "TIGQA%20Tigrinya%20Question%20Answering%20dataset.docx?download=1")
AMQA_URL = ("https://raw.githubusercontent.com/semantic-systems/"
            "amharic-qa/refs/heads/main/AmQA_Dataset.json")

NUMBERED = re.compile(r"(?m)(?=\d+\s*[.)])")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_numbering(s: str) -> str:
    return re.sub(r"^\d+\s*[.)]\s*", "", s).strip()


def read_tigqa(path: Path):
    """(context, question, answer) triples from the TIGQA .docx table."""
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    table = next(root.iter(W + "tbl"))
    rows = [[
        "".join(t.text or "" for t in tc.iter(W + "t")).strip()
        for tc in tr.findall(W + "tc")
    ] for tr in table.findall(W + "tr")]

    triples = []
    for row in rows[1:]:
        if len(row) < 6:
            continue
        context, q_cell, a_cell = row[3], row[4], row[5]
        questions = [strip_numbering(x) for x in NUMBERED.split(q_cell) if x.strip()]
        answers = [strip_numbering(x) for x in NUMBERED.split(a_cell) if x.strip()]
        for q, a in zip(questions, answers):
            if q and a and context:
                triples.append((context, q, a))
    return triples


def to_squad(triples, title="TIGQA"):
    """SQuAD-style records, keeping only answers found verbatim in context."""
    data, kept, dropped = [], 0, 0
    for i, (context, question, answer) in enumerate(triples):
        start = context.find(answer)
        if start < 0:
            dropped += 1
            continue
        kept += 1
        data.append({
            "title": title,
            "context": context,
            "qas": [{
                "id": f"tigqa-{i}",
                "question": question,
                "answers": [{"text": answer, "answer_start": start}],
                "is_impossible": False,
            }],
        })
    return data, kept, dropped


def read_amqa(path: Path):
    raw = json.load(open(path, encoding="utf-8"))["data"]
    out = []
    for article in raw:
        for para in article.get("paragraphs", []):
            if not isinstance(para, dict) or not para.get("context"):
                continue          # 3 paragraphs in the release are bare strings
            out.append({"title": article.get("title", "AmQA"),
                        "context": para["context"],
                        "qas": para.get("qas", [])})
    return out


def split_80_10_10(items, seed: int):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_train, n_dev = int(0.8 * n), int(0.1 * n)
    return {
        "train": [items[i] for i in idx[:n_train]],
        "dev": [items[i] for i in idx[n_train:n_train + n_dev]],
        "test": [items[i] for i in idx[n_train + n_dev:]],
    }


def main():
    p = argparse.ArgumentParser(description="Prepare QA splits")
    p.add_argument("--dataset", choices=("amqa", "tigqa", "tigqa_squad"),
                   required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/qa"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    manifest = {"raw_sha256": sha256(args.raw), "seed": args.seed,
                "split": "random 80/10/10 with the seed recorded here"}

    if args.dataset == "tigqa_squad":
        # TIGQA already converted to SQuAD form, with answer_start offsets.
        # Answers the conversion could not locate in their context carry
        # answer_start == -1; those are abstractive rewrites and are kept
        # aside rather than dropped silently, so the extractive subset is
        # explicit.
        raw = json.load(open(args.raw, encoding="utf-8"))
        items, extractive, abstractive, unanswerable = [], 0, 0, 0
        for article in raw["data"]:
            for para in article["paragraphs"]:
                ctx = para["context"]
                keep = []
                for qa in para["qas"]:
                    answers = qa.get("answers") or []
                    if not answers:
                        unanswerable += 1
                        continue
                    valid = [x for x in answers
                             if x.get("answer_start", -1) >= 0
                             and ctx[x["answer_start"]:
                                     x["answer_start"] + len(x["text"])] == x["text"]]
                    if valid:
                        extractive += 1
                        keep.append({**qa, "answers": valid})
                    else:
                        abstractive += 1
                if keep:
                    items.append({"title": article.get("title", "TIGQA"),
                                  "context": ctx, "qas": keep})
        manifest.update({
            "source": str(args.raw),
            "citation": "Teklehaymanot et al. (2024), TIGQA",
            "version": raw.get("version"),
            "extractive_qa_pairs": extractive,
            "abstractive_dropped": abstractive,
            "unanswerable_dropped": unanswerable,
            "note": ("answer_start == -1 marks answers not present verbatim in "
                     "the context; those are abstractive and cannot be scored "
                     "by span-extraction F1"),
        })
    elif args.dataset == "tigqa":
        triples = read_tigqa(args.raw)
        items, kept, dropped = to_squad(triples)
        manifest.update({
            "source": TIGQA_URL,
            "citation": "Teklehaymanot et al. (2024), Zenodo 11423987, CC-BY-4.0",
            "raw_qa_pairs": len(triples),
            "kept_extractive": kept,
            "dropped_no_span": dropped,
            "note": ("answers not found verbatim in their context were dropped: "
                     "extractive QA needs a character offset and the release "
                     "provides none"),
        })
    else:
        items = read_amqa(args.raw)
        manifest.update({
            "source": AMQA_URL,
            "citation": "Taffa et al. (2024)",
            "paragraphs": len(items),
            "qa_pairs": sum(len(x["qas"]) for x in items),
        })

    splits = split_80_10_10(items, args.seed)
    out = args.out_dir / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "data": part}, f,
                      ensure_ascii=False, indent=1)
        n_qas = sum(len(x["qas"]) for x in part)
        print(f"  {name:5} {len(part):5} contexts  {n_qas:5} QA pairs")
    manifest["counts"] = {k: len(v) for k, v in splits.items()}
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
