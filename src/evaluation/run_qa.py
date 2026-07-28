"""
run_qa.py -- fine-tune and evaluate a model on extractive QA (paper Sec 6.3).

Span extraction over SQuAD-format data: AmQA for Amharic, TIGQA for
Tigrinya. Model selection is on dev, the reported score is on test. One seed
per invocation; scripts/run_table2.sh sweeps the five seeds.
"""

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_squad(path: Path):
    """Flatten SQuAD JSON to (id, question, context, answers) records."""
    raw = json.load(open(path, encoding="utf-8"))
    data = raw["data"] if isinstance(raw, dict) else raw
    out = []
    for article in data:
        paragraphs = article.get("paragraphs", [article])
        for para in paragraphs:
            context = para.get("context")
            if not context:
                continue
            for qa in para.get("qas", []):
                answers = qa.get("answers") or []
                if not answers:
                    continue
                out.append({
                    "id": str(qa.get("id")),
                    "question": qa["question"],
                    "context": context,
                    "answers": answers,
                })
    return out


def prepare_features(examples, tokenizer, max_length, stride, training):
    """Tokenize with overflow, mapping answers to token-level start/end.

    Long contexts are split into overlapping windows; for training, a window
    that does not contain the answer is labelled with the CLS position,
    which is the standard SQuAD treatment for unanswerable windows.
    """
    features = []
    for ex in examples:
        enc = tokenizer(
            ex["question"], ex["context"],
            truncation="only_second", max_length=max_length,
            stride=stride, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding=False,
        )
        for i in range(len(enc["input_ids"])):
            offsets = enc["offset_mapping"][i]
            sequence_ids = enc.sequence_ids(i)
            item = {"input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i]}
            if training:
                answer = ex["answers"][0]
                start_char = answer["answer_start"]
                end_char = start_char + len(answer["text"])
                # locate the context window
                ctx_start = sequence_ids.index(1)
                ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
                if not (offsets[ctx_start][0] <= start_char
                        and offsets[ctx_end][1] >= end_char):
                    item["start_positions"] = 0
                    item["end_positions"] = 0
                else:
                    s = ctx_start
                    while s <= ctx_end and offsets[s][0] <= start_char:
                        s += 1
                    e = ctx_end
                    while e >= ctx_start and offsets[e][1] >= end_char:
                        e -= 1
                    item["start_positions"] = s - 1
                    item["end_positions"] = e + 1
            else:
                item["example_id"] = ex["id"]
                item["offset_mapping"] = [
                    o if sequence_ids[k] == 1 else None
                    for k, o in enumerate(offsets)
                ]
            features.append(item)
    return features


def postprocess(examples, features, starts, ends, n_best=20, max_answer_len=64):
    """Best-scoring valid span per example."""
    by_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        by_example[feat["example_id"]].append(i)

    predictions = {}
    for ex in examples:
        best_text, best_score = "", -1e9
        for i in by_example[ex["id"]]:
            offsets = features[i]["offset_mapping"]
            s_logits, e_logits = starts[i], ends[i]
            s_idx = np.argsort(s_logits)[-n_best:][::-1]
            e_idx = np.argsort(e_logits)[-n_best:][::-1]
            for s in s_idx:
                for e in e_idx:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_len:
                        continue
                    score = s_logits[s] + e_logits[e]
                    if score > best_score:
                        best_score = score
                        best_text = ex["context"][offsets[s][0]:offsets[e][1]]
        predictions[ex["id"]] = best_text
    return predictions


def main():
    p = argparse.ArgumentParser(description="QA fine-tuning and evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.metrics import qa_f1

    from torch.utils.data import Dataset
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                              Trainer, TrainingArguments)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    splits = {n: read_squad(args.data_dir / f"{n}.json")
              for n in ("train", "dev", "test")}

    class QADataset(Dataset):
        def __init__(self, feats, keep_meta=False):
            self.feats = feats
            self.keep_meta = keep_meta

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, i):
            f = dict(self.feats[i])
            f.pop("example_id", None)
            f.pop("offset_mapping", None)
            return f

    train_feats = prepare_features(splits["train"], tokenizer,
                                   args.max_length, args.stride, True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed, data_seed=args.seed,
        save_strategy="no", logging_steps=50, report_to=[],
    )
    def collate(batch):
        """Pad to the longest item in the batch.

        DefaultDataCollator assumes every feature is already the same length,
        which is false here: features are produced without padding so that
        offset_mapping stays aligned with the untruncated context.
        """
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        out = {"input_ids": [], "attention_mask": []}
        has_pos = "start_positions" in batch[0]
        if has_pos:
            out["start_positions"], out["end_positions"] = [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * n)
            out["attention_mask"].append(b["attention_mask"] + [0] * n)
            if has_pos:
                out["start_positions"].append(b["start_positions"])
                out["end_positions"].append(b["end_positions"])
        return {k: torch.tensor(v) for k, v in out.items()}

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=QADataset(train_feats),
                      data_collator=collate)
    trainer.train()

    def score(examples):
        feats = prepare_features(examples, tokenizer, args.max_length,
                                 args.stride, False)
        out = trainer.predict(QADataset(feats))
        starts, ends = out.predictions[0], out.predictions[1]
        preds = postprocess(examples, feats, starts, ends)
        refs = {ex["id"]: [a["text"] for a in ex["answers"]] for ex in examples}
        return qa_f1(preds, refs)

    result = {"task": "qa", "model": args.model, "seed": args.seed,
              "dev": score(splits["dev"]), "test": score(splits["test"])}
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"seed {args.seed}  dev F1 {result['dev']['f1']:.2f}  "
          f"test F1 {result['test']['f1']:.2f}")


if __name__ == "__main__":
    main()
