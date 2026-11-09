#!/usr/bin/env python3
"""Share of test-split tokens that resolve to a newly added embedding row.

An initializer can only act through the rows it sets. This counts, per
language and task, how many tokens of the test split the extended tokenizer
maps to an added row (id >= the base vocabulary size), and how many test
examples contain at least one such token. Reported in the paper, section
"Why the Metric Cannot Move".

The extended tokenizer is rebuilt by adding the language's token list
(data/new_tokens_amharic.txt, data/new_tokens.txt) to the XLM-R base
tokenizer, exactly as the pipeline does. Where saved checkpoints are present
(the working repository), the script also checks that every checkpoint
carries the same added-token set as the rebuilt tokenizer. The pipeline
assigns the new ids in a different order, which does not affect which tokens
are split off, and so does not affect any count here.

Items counted: TC, each document; NER, each sentence (words joined by
spaces); QA, each distinct context once and each question once.

Usage:
    python3 scripts/token_share.py
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPLITS = {
    ("amharic", "tc"): "data/tc/amharic/test.csv",
    ("amharic", "ner"): "data/ner/amharic/test.conll",
    ("amharic", "qa"): "data/qa/amqa/test.json",
    ("tigrinya", "tc"): "data/tc/tigrinya/test.csv",
    ("tigrinya", "ner"): "data/ner/tigrinya/test.conll",
    ("tigrinya", "qa"): "data/qa/tigqa_squad/test.json",
}
TOKEN_LIST = {"amharic": "data/new_tokens_amharic.txt",
              "tigrinya": "data/new_tokens.txt"}
CHECKPOINT_DIR = {"amharic": "results_hornmorpho_wam",
                  "tigrinya": "results_tigrinya_gen2"}


def examples(lang: str, task: str) -> list[str]:
    path = ROOT / SPLITS[(lang, task)]
    if task == "tc":
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        col = next(c for c in rows[0] if c.strip().lower() == "text")
        return [r[col] for r in rows]
    if task == "ner":
        out, words = [], []
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if not parts:
                if words:
                    out.append(" ".join(words)); words = []
                continue
            words.append(parts[0])
        if words:
            out.append(" ".join(words))
        return out
    # QA: each distinct context once, and each question once, as separate
    # items. A context is shared by many questions; counting it per question
    # would multiply the same text by the number of questions asked about it.
    data = json.load(open(path, encoding="utf-8"))["data"]
    contexts, questions = [], []
    for art in data:
        for p in art.get("paragraphs") or [art]:
            contexts.append(p["context"])
            questions.extend(qa["question"] for qa in p["qas"])
    return list(dict.fromkeys(contexts)) + questions


def extended_tokenizer(lang: str):
    """XLM-R base plus the language's added tokens; checked against checkpoints."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    n_base = len(tok)
    words = list(dict.fromkeys(
        l.strip() for l in open(ROOT / TOKEN_LIST[lang], encoding="utf-8")
        if l.strip()))
    tok.add_tokens(words)
    rebuilt = {c for c, i in tok.get_added_vocab().items() if i >= n_base}

    hits = sorted(glob.glob(str(ROOT / CHECKPOINT_DIR[lang]
                                / f"*lapt__*__{lang}__seed*" / "lapt"
                                / "tokenizer.json")))
    for h in hits:
        saved = {a["content"] for a in
                 json.load(open(h, encoding="utf-8"))["added_tokens"]
                 if not a["special"] and a["id"] >= n_base}
        if saved != rebuilt:
            raise SystemExit(f"{lang}: {h} adds a different token set")
    note = (f"matches all {len(hits)} saved checkpoints" if hits
            else "no checkpoints present to cross-check")
    return tok, f"{len(words)} listed, {len(rebuilt)} new rows; {note}"


def main() -> int:
    import os
    # the base tokenizer is expected in the local cache; without this a node
    # with no network spends minutes on Hub timeouts
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained("xlm-roberta-base")
    n_base = len(base)          # len() on a fast tokenizer rebuilds the vocab
    print(f"base vocabulary: {n_base}")
    print(f"\n  {'lang':4s}{'task':5s}{'tokens':>9s}{'new':>6s}{'%':>7s}"
          f"{'examples':>10s}{'with new':>10s}")
    for lang in ("amharic", "tigrinya"):
        tok, note = extended_tokenizer(lang)
        print(f"  {lang}: {note}")
        for task in ("tc", "ner", "qa"):
            ex = examples(lang, task)
            total = new = with_new = 0
            for text in ex:
                ids = tok(text, add_special_tokens=False)["input_ids"]
                k = sum(1 for i in ids if i >= n_base)
                total += len(ids); new += k; with_new += k > 0
            print(f"  {lang[:2]:4s}{task:5s}{total:9d}{new:6d}"
                  f"{100 * new / total:7.2f}{len(ex):10d}"
                  f"{100 * with_new / len(ex):9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
