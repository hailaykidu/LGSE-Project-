"""
metrics.py -- F1 for NER and QA (paper Sec 6.3).

The paper reports F1 for NER, QA and text classification, averaged over five
random seeds with the standard deviation.

NER uses entity-level F1 (CoNLL convention): a prediction counts only when
both the entity's span and its type match the gold annotation exactly.
Token-level accuracy is not reported -- it is dominated by the O class.

QA uses SQuAD F1: token overlap between the predicted answer string and the
gold answer, maximised over the available gold answers.
"""

from collections import Counter
from typing import Dict, List, Sequence


def extract_entities(tags: Sequence[str]) -> set:
    """(type, start, end) spans from a BIO tag sequence.

    A B- tag opens a span; an I- tag of the same type continues it; anything
    else closes it. I- tags that open a span without a preceding B- are
    treated as B-, which is the lenient CoNLL reading and avoids silently
    discarding entities in datasets that use IOB1.
    """
    spans, start, etype = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-"):
            if etype is not None:
                spans.add((etype, start, i))
            start, etype = i, tag[2:]
        elif tag.startswith("I-"):
            if etype != tag[2:]:
                if etype is not None:
                    spans.add((etype, start, i))
                start, etype = i, tag[2:]
        else:
            if etype is not None:
                spans.add((etype, start, i))
            start, etype = None, None
    return spans


def ner_f1(predictions: List[Sequence[str]],
           references: List[Sequence[str]]) -> Dict[str, float]:
    """Micro-averaged entity-level precision, recall and F1."""
    tp = fp = fn = 0
    for pred, gold in zip(predictions, references):
        p, g = extract_entities(pred), extract_entities(gold)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": 100 * precision, "recall": 100 * recall, "f1": 100 * f1}


def _normalize(text: str) -> List[str]:
    """Whitespace tokenization. Deliberately no lowercasing or punctuation
    stripping: the SQuAD normalizer is English-specific (it strips articles
    'a', 'an', 'the'), and applying it to Ge'ez script would be meaningless.
    """
    return text.split()


def qa_f1_single(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = _normalize(prediction), _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1(predictions: Dict[str, str],
          references: Dict[str, List[str]]) -> Dict[str, float]:
    """SQuAD F1 and exact match, keyed by question id."""
    f1_total = em_total = 0.0
    for qid, golds in references.items():
        pred = predictions.get(qid, "")
        f1_total += max(qa_f1_single(pred, g) for g in golds)
        em_total += max(float(pred.strip() == g.strip()) for g in golds)
    n = max(len(references), 1)
    return {"f1": 100 * f1_total / n, "exact_match": 100 * em_total / n}


def aggregate(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation across seeds, as Table 2 reports.

    Uses the population standard deviation (ddof=0), which is what numpy's
    default gives; with five runs the choice shifts the reported spread
    slightly, so it is stated rather than left implicit.
    """
    import statistics

    keys = runs[0].keys() if runs else []
    out = {}
    for k in keys:
        values = [r[k] for r in runs]
        out[k] = {
            "mean": sum(values) / len(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "runs": values,
        }
    return out
