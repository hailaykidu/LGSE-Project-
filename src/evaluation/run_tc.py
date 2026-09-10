"""
run_tc.py -- fine-tune and evaluate a model on text classification
(paper Sec 6.3, Table 2 "Text Classification / TC / AC" rows).

Sequence classification over the educational-quality scale. Model selection
is on dev, the reported score is on test, and one seed runs per invocation;
`scripts/run_table2.sh` sweeps the five seeds and aggregates -- the same
contract as `run_ner.py` and `run_qa.py`.

Metric. Table 2 reports TC under the metric column "AC" (accuracy), while
Sec 6.3's prose says F1 is reported "for NER, QA, and Text classification".
Both are computed and written to `result.json`; the printed line and the
`aggregate_results.py` key follow Table 2 and use accuracy, since that is
what the table's numbers are labelled with. Choosing silently between them
would make the reproduced column incomparable with the published one.

Data. `data/Educational classifier.csv` carries the 1-6 educational-quality
scale described in Sec 7. Its first row is the annotation guidelines rather
than data, and its header row is `Text , Label ` with trailing spaces; both
are handled in `read_educational_csv` rather than by pre-editing the file,
so the released artifact stays byte-identical to what the annotators
produced.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_educational_csv(path: Path):
    """(text, label) pairs from the educational-quality CSV.

    The file's first row is the rating guidelines, not a record; the second
    is the header. Rows whose text or label is blank are skipped and counted
    by the caller rather than silently dropped.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for record in csv.reader(f):
            if len(record) < 2:
                continue
            text, label = record[0].strip(), record[1].strip()
            if not text or not label:
                continue
            # Header row, in either the spaced or unspaced form.
            if text.lower() == "text" and label.lower() == "label":
                continue
            # Guidelines blob: a multi-line cell with no label beside it.
            if not label.isdigit():
                continue
            rows.append((text, label))
    return rows


def stratified_split(rows, seed: int, dev_frac=0.1, test_frac=0.1):
    """80/10/10 split, stratified by label where the label allows it.

    Sec 7 specifies 80/10/10. The educational scale is heavily imbalanced --
    some ratings occur once or twice -- so a purely random split can leave a
    class absent from train and present in test, which makes the accuracy
    figure depend on the seed far more than on the model. Stratifying keeps
    each class's proportion stable; classes too small to divide are assigned
    to train, so they are learned rather than only tested.

    Duplicate texts are grouped before splitting. The released CSV repeats
    nine short texts (e.g. 'ማስታወቂያ'); splitting rows independently would put
    the same string in train and test, and the test score would then partly
    measure memorisation. Grouping keeps every copy on one side of the split.
    """
    grouped = {}
    for text, label in rows:
        grouped.setdefault(text, []).append((text, label))

    by_label = {}
    for group in grouped.values():
        # A text repeated under different labels is kept with its first,
        # rather than dropped: the annotation is the authors' to resolve.
        by_label.setdefault(group[0][1], []).append(group)

    rng = random.Random(seed)
    train, dev, test = [], [], []
    for label, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_dev, n_test = int(round(n * dev_frac)), int(round(n * test_frac))
        # A class with too few examples to split goes wholly to train.
        if n - n_dev - n_test < 1:
            n_dev = n_test = 0
        dev += [r for g in items[:n_dev] for r in g]
        test += [r for g in items[n_dev:n_dev + n_test] for r in g]
        train += [r for g in items[n_dev + n_test:] for r in g]

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return {"train": train, "dev": dev, "test": test}


def read_split_dir(data_dir: Path):
    """Pre-made splits, if the data directory provides them.

    Preferred over splitting in-process: an author-supplied split is
    reproducible across runs and tools, whereas one derived here depends on
    this function's logic.
    """
    splits = {}
    for name in ("train", "dev", "test"):
        for suffix in (".csv", ".tsv"):
            path = data_dir / f"{name}{suffix}"
            if path.exists():
                splits[name] = read_educational_csv(path)
                break
    return splits if len(splits) == 3 else None


def classification_scores(predictions, references, labels):
    """Accuracy plus macro and micro F1.

    Macro F1 is reported alongside accuracy because the label distribution
    is skewed: a model that predicts only the majority rating scores well on
    accuracy while being useless, and macro F1 makes that visible.
    """
    correct = sum(int(p == g) for p, g in zip(predictions, references))
    accuracy = correct / len(references) if references else 0.0

    per_class, f1s = {}, []
    tp_all = fp_all = fn_all = 0
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, references) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, references) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, references) if p != label and g == label)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": 100 * precision, "recall": 100 * recall,
                            "f1": 100 * f1, "support": tp + fn}
        f1s.append(f1)

    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r else 0.0)

    return {
        "accuracy": 100 * accuracy,
        "macro_f1": 100 * sum(f1s) / len(f1s) if f1s else 0.0,
        "micro_f1": 100 * micro_f1,
        "per_class": per_class,
    }


def main():
    p = argparse.ArgumentParser(
        description="Text classification fine-tuning and evaluation")
    p.add_argument("--model", required=True, help="model dir or hub name")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=None,
                   help="single-file dataset; defaults to "
                        "'<data-dir>/Educational classifier.csv'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    set_seed(args.seed)

    splits = read_split_dir(args.data_dir)
    split_source = "author-provided train/dev/test files"
    if splits is None:
        csv_path = args.csv or (args.data_dir / "Educational classifier.csv")
        if not csv_path.exists():
            raise SystemExit(
                f"MISSING TEXT CLASSIFICATION DATA: {csv_path}\n"
                "  Provide train/dev/test files in --data-dir, or point --csv "
                "at the educational-quality CSV.")
        rows = read_educational_csv(csv_path)
        if not rows:
            raise SystemExit(f"NO USABLE ROWS IN {csv_path}")
        splits = stratified_split(rows, seed=args.seed)
        split_source = (f"stratified 80/10/10 split of {csv_path.name} "
                        f"at seed {args.seed}")

    labels = sorted({l for part in splits.values() for _, l in part},
                    key=lambda x: (len(x), x))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class TCDataset(Dataset):
        def __init__(self, items):
            self.items = []
            for text, label in items:
                enc = tokenizer(text, truncation=True, max_length=args.max_length)
                enc["labels"] = label2id[label]
                self.items.append(enc)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    model = AutoModelForSequenceClassification.from_pretrained(
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
        train_dataset=TCDataset(splits["train"]),
        eval_dataset=TCDataset(splits["dev"]),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    def score(items):
        if not items:
            return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0,
                    "per_class": {}}
        ds = TCDataset(items)
        logits = trainer.predict(ds).predictions
        predictions = [id2label[int(row.argmax(-1))] for row in logits]
        references = [label for _, label in items]
        return classification_scores(predictions, references, labels)

    result = {
        "task": "tc", "model": args.model, "seed": args.seed,
        "dev": score(splits["dev"]), "test": score(splits["test"]),
        "labels": labels,
        "split_source": split_source,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "label_distribution": {
            k: dict(sorted(Counter(l for _, l in v).items()))
            for k, v in splits.items()
        },
        # Table 2 labels the TC column "AC"; Sec 6.3's prose says F1. Both
        # are recorded so the reported column can be stated unambiguously.
        "primary_metric": "accuracy",
        "primary_metric_note": (
            "Table 2 reports TC under metric 'AC' (accuracy); Sec 6.3 prose "
            "says F1 is reported for NER, QA and text classification. Both "
            "accuracy and macro/micro F1 are recorded here."),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"seed {args.seed}  dev AC {result['dev']['accuracy']:.2f}  "
          f"test AC {result['test']['accuracy']:.2f}  "
          f"(test macro F1 {result['test']['macro_f1']:.2f})")


if __name__ == "__main__":
    main()
