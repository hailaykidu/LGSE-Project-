"""
run_experiment.py -- one Table 2 cell, end to end.

A cell is (system, task, language, seed). This runs the whole pipeline for
one cell and writes a single result record:

    1. LAPT      expand the vocabulary, initialize the new embeddings with
                 the system's strategy, adapt with a frozen backbone
                 (skipped for the XLM-R baseline, which does neither)
    2. Fine-tune the adapted model on the downstream task
    3. Evaluate  F1 on dev for model selection, F1 on test to report

Systems differ only in step 1. Steps 2 and 3 are identical across them, so
any difference in the reported F1 is attributable to initialization.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_configs(base: Path, systems: Path):
    return (yaml.safe_load(open(base, encoding="utf-8")),
            yaml.safe_load(open(systems, encoding="utf-8"))["systems"])


def experiment_name(system: str, task: str, language: str, seed: int) -> str:
    return f"{system}__{task}__{language}__seed{seed}"


def run_lapt(cfg, system_cfg, language: str, seed: int, corpus: Path,
             out_dir: Path) -> str:
    """Stage 1. Returns the model path the downstream stage should load."""
    from lgse.config import LGSEConfig
    from lgse.lap_trainer import LGSELAPTrainer
    from torch.utils.data import Dataset

    base_model = cfg["model"]["name"]
    if not system_cfg["lapt"]:
        return base_model          # XLM-R baseline: unmodified backbone

    class LineDataset(Dataset):
        def __init__(self, path, tokenizer, max_length):
            self.lines = [l.strip() for l in
                          open(path, encoding="utf-8") if l.strip()]
            self.tokenizer, self.max_length = tokenizer, max_length

        def __len__(self):
            return len(self.lines)

        def __getitem__(self, i):
            enc = self.tokenizer(self.lines[i], truncation=True,
                                 max_length=self.max_length)
            return {"input_ids": enc["input_ids"]}

    ft_key = ("fasttext_amharic_path" if language == "amharic"
              else "fasttext_tigrinya_path")
    lgse_cfg = LGSEConfig(
        model_name=base_model,
        language="am" if language == "amharic" else "ti",
        system=system_cfg.get("name", "lgse_lapt"),
        expand_vocab=system_cfg["expand_vocab"],
        initializer=system_cfg["initializer"] or "default",
        projection=cfg["lgse"]["projection"],
        ngram_min=cfg["lgse"]["ngram_min"],
        ngram_max=cfg["lgse"]["ngram_max"],
        reg_lambda=cfg["lgse"]["reg_lambda"],
        seed=seed,
        learning_rate=cfg["lapt"]["learning_rate"],
        batch_size=cfg["lapt"]["batch_size"],
        mlm_probability=cfg["lapt"]["mlm_probability"],
        output_dir=str(out_dir / "lapt"),
    )
    setattr(lgse_cfg, ft_key, getattr(lgse_cfg, ft_key))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = LineDataset(corpus, tokenizer, cfg["lapt"]["max_length"])

    trainer = LGSELAPTrainer(lgse_cfg, dataset)
    for _ in range(cfg["lapt"]["epochs"]):
        trainer.train_epoch()
    trainer.save(str(out_dir / "lapt"))
    return str(out_dir / "lapt")


def run_downstream(task: str, model_path: str, data_dir: Path, seed: int,
                   cfg, out_dir: Path):
    """Stage 2+3, delegated to the task runner so both share one code path."""
    script = ROOT / "src" / "evaluation" / (
        "run_ner.py" if task == "ner" else "run_qa.py")
    ft = cfg["finetune"]
    cmd = [sys.executable, str(script),
           "--model", model_path,
           "--data-dir", str(data_dir),
           "--seed", str(seed),
           "--learning-rate", str(ft["learning_rate"]),
           "--batch-size", str(ft["batch_size"]),
           "--epochs", str(ft["epochs"]),
           "--output", str(out_dir / task)]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return json.load(open(out_dir / task / "result.json", encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description="Run one Table 2 cell")
    p.add_argument("--system", required=True)
    p.add_argument("--task", required=True, choices=("ner", "qa"))
    p.add_argument("--language", required=True, choices=("amharic", "tigrinya"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=None,
                   help="LAPT corpus; required unless the system skips LAPT")
    p.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    p.add_argument("--systems", type=Path, default=ROOT / "configs/systems.yaml")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    cfg, systems = load_configs(args.config, args.systems)
    if args.system not in systems:
        raise SystemExit(f"unknown system {args.system!r}; "
                         f"expected one of {sorted(systems)}")
    system_cfg = dict(systems[args.system], name=args.system)

    name = experiment_name(args.system, args.task, args.language, args.seed)
    out_dir = args.results_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {name} ===")

    started = time.time()
    model_path = run_lapt(cfg, system_cfg, args.language, args.seed,
                          args.corpus, out_dir)
    result = run_downstream(args.task, model_path, args.data_dir,
                            args.seed, cfg, out_dir)

    record = {
        "experiment": name,
        "system": args.system,
        "system_description": system_cfg["description"],
        "task": args.task,
        "language": args.language,
        "seed": args.seed,
        "projection": cfg["lgse"]["projection"],
        "model_path": model_path,
        "dev": result["dev"],
        "test": result["test"],
        "elapsed_seconds": round(time.time() - started, 1),
        "config": cfg,
    }
    with open(out_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"{name}: test F1 {result['test']['f1']:.2f}")


if __name__ == "__main__":
    main()
