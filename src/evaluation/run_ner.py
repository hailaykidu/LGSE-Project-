"""
run_ner.py -- fine-tune and evaluate a model on NER (paper Sec 6.3).

Token classification over BIO tags. Model selection is on dev F1 and the
reported score is on test, as the paper specifies. One seed per invocation;
`scripts/run_table2.sh` sweeps the five seeds and aggregates.
"""

import argparse
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


def read_conll(path: Path):
    sentences, current = [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.replace("\r", "").rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 2:
                current.append((parts[0], parts[-1]))
    if current:
        sentences.append(current)
    return sentences


def align_labels(tokenizer, words, tags, label2id, max_length):
    """Label the first sub-token of each word; mask the rest with -100.

    Sub-word continuations are masked rather than repeating the tag, so a
    long word cannot dominate the loss simply by producing more pieces --
    which matters here, since the whole point of the vocabulary expansion is
    to change how these languages fragment.
    """
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=max_length)
    labels, previous = [], None
    for word_id in enc.word_ids():
        if word_id is None:
            labels.append(-100)
        elif word_id != previous:
            labels.append(label2id[tags[word_id]])
        else:
            labels.append(-100)
        previous = word_id
    enc["labels"] = labels
    return enc


def main():
    p = argparse.ArgumentParser(description="NER fine-tuning and evaluation")
    p.add_argument("--model", required=True, help="model dir or hub name")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.metrics import ner_f1

    from torch.utils.data import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments)

    set_seed(args.seed)

    splits = {name: read_conll(args.data_dir / f"{name}.conll")
              for name in ("train", "dev", "test")}
    labels = sorted({t for s in splits["train"] for _, t in s}
                    | {t for s in splits["dev"] for _, t in s}
                    | {t for s in splits["test"] for _, t in s})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class NERDataset(Dataset):
        def __init__(self, sentences):
            self.items = [
                align_labels(tokenizer, [w for w, _ in s], [t for _, t in s],
                             label2id, args.max_length)
                for s in sentences if s
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed, data_seed=args.seed,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=50, report_to=[], save_total_limit=1,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=NERDataset(splits["train"]),
        eval_dataset=NERDataset(splits["dev"]),
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()

    def score(sentences):
        ds = NERDataset(sentences)
        logits = trainer.predict(ds).predictions
        preds, golds = [], []
        for row, item in zip(logits, ds.items):
            ids = row.argmax(-1)
            keep = [(i, l) for i, l in zip(ids, item["labels"]) if l != -100]
            preds.append([id2label[int(i)] for i, _ in keep])
            golds.append([id2label[int(l)] for _, l in keep])
        return ner_f1(preds, golds)

    result = {
        "task": "ner", "model": args.model, "seed": args.seed,
        "dev": score(splits["dev"]), "test": score(splits["test"]),
        "labels": labels,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"seed {args.seed}  dev F1 {result['dev']['f1']:.2f}  "
          f"test F1 {result['test']['f1']:.2f}")


if __name__ == "__main__":
    main()
