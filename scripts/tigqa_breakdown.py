#!/usr/bin/env python3
"""Why TIGQA pairs are not kept as extractive QA.

The conversion holds 2,179 question-answer pairs; 1,517 have an answer that
occurs verbatim in its passage and are used. For the others with answer text,
this finds the passage span of the answer's length most similar to it
(difflib ratio) and classifies the pair:

    >= 0.85     same answer, different surface form (spelling variant,
                inflectional affix, punctuation)
    0.60-0.85   partial overlap (paraphrase, extra or missing words)
    < 0.60      abstractive (the answer is not in the passage)

Usage: python3 scripts/tigqa_breakdown.py
"""
from __future__ import annotations

import collections
import difflib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/qa/tigqa_squad/tigqa_extractive_squad.json"


def norm(s: str) -> str:
    return " ".join(str(s).split())


def best_ratio(answer: str, context: str) -> float:
    n, best = len(answer), 0.0
    for i in range(max(1, len(context) - n + 1)):
        r = difflib.SequenceMatcher(None, answer, context[i:i + n],
                                    autojunk=False).ratio()
        best = max(best, r)
    return best


def main() -> int:
    data = json.load(open(SOURCE, encoding="utf-8"))["data"]
    status = collections.Counter()
    pairs, questions, kinds = 0, set(), collections.Counter()
    for art in data:
        for p in art.get("paragraphs") or [art]:
            ctx = norm(p["context"])
            for q in p["qas"]:
                pairs += 1
                questions.add((norm(q["question"]), ctx))
                st = q.get("match_status") or q.get("status")
                status[st] += 1
                answer = norm(q.get("raw_answer")
                              or (q["answers"][0]["text"] if q.get("answers") else ""))
                if st in ("fuzzy", "unmatched") and answer:
                    r = best_ratio(answer, ctx)
                    kinds["surface form" if r >= 0.85 else
                          "partial overlap" if r >= 0.60 else "abstractive"] += 1
    print(f"pairs {pairs}, distinct questions {len(questions)}, "
          f"passages {len(data)}")
    print("match status:", dict(status))
    print("not kept, with answer text:", sum(kinds.values()))
    for k in ("surface form", "partial overlap", "abstractive"):
        print(f"  {k:16s}{kinds[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
