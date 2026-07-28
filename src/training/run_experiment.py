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
import hashlib
import json
import platform
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


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(config_path: Path, data_dir: Path, corpus: Path) -> dict:
    """Everything needed to re-run this exact experiment later.

    A result is only reproducible if the code, the configuration and the
    data it used can all be identified afterwards. `dirty` matters: a run
    made with uncommitted changes cannot be recovered from the commit hash
    alone, so it is recorded rather than assumed clean.
    """
    # A missing manifest must be visible, not silent: a hand-assembled data
    # directory would otherwise render as fully documented while nothing
    # records where its splits came from.
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        dataset = json.load(open(manifest, encoding="utf-8"))
        dataset_available = True
    else:
        dataset = {"status": "unavailable",
                   "reason": f"no manifest.json in {data_dir}",
                   "note": ("splits cannot be traced to a source; prepare data "
                            "with data/scripts/prepare_{ner,qa}.py, which "
                            "writes a manifest recording source URL, raw "
                            "checksum and split seed")}
        dataset_available = False

    versions = {}
    for module in ("torch", "transformers", "fasttext", "numpy"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = "unavailable"

    fasttext_manifest = ROOT / "data" / "fasttext_manifest.json"
    embeddings = {}
    if fasttext_manifest.exists():
        embeddings = {
            lang: {k: rec[k] for k in ("source", "dimension", "vocab_size",
                                       "sha256") if k in rec}
            for lang, rec in json.load(
                open(fasttext_manifest, encoding="utf-8")).items()
        }

    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "config_file": str(config_path),
        "config_sha256": _sha256(config_path) if config_path.exists() else None,
        "data_dir": str(data_dir),
        "dataset_manifest": dataset,
        "dataset_manifest_available": dataset_available,
        "lapt_corpus": str(corpus) if corpus else None,
        "lapt_corpus_sha256": _sha256(corpus) if corpus and corpus.exists() else None,
        "fasttext": embeddings,
        "python": platform.python_version(),
        "packages": versions,
    }


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

    # reg_lambda is mandatory: the paper never assigns lambda, so there is
    # no published value to fall back on. A bare KeyError here would not say
    # that, and a default would present our choice as the paper's.
    if "reg_lambda" not in cfg["lgse"]:
        raise SystemExit(
            f"`lgse.reg_lambda` is missing from the run config.\n"
            f"\n"
            f"It is lambda in L_reg = lambda * ||e_new - mu||^2 (paper Sec "
            f"4.2). The paper introduces lambda but never states its value, "
            f"so there is no default to apply -- set it explicitly under "
            f"`lgse:` in your config. See DEVIATIONS.md section 8.")

    lgse_cfg = LGSEConfig(
        model_name=base_model,
        language="am" if language == "amharic" else "ti",
        system=system_cfg.get("name", "lgse_lapt"),
        expand_vocab=system_cfg["expand_vocab"],
        initializer=system_cfg["initializer"] or "default",
        ngram_min=cfg["lgse"]["ngram_min"],
        ngram_max=cfg["lgse"]["ngram_max"],
        reg_lambda=cfg["lgse"]["reg_lambda"],
        alignment_matrix_path=cfg["lgse"].get("alignment_matrix_path", ""),
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

    # Record W's provenance and training status beside the checkpoint, so a
    # result carries the alignment matrix it used and the fact that W was
    # not trained under any objective the paper states.
    import json as _json
    with open(out_dir / "lapt" / "projection_status.json", "w",
              encoding="utf-8") as f:
        _json.dump(trainer.projection_status(), f, indent=2)

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
    p.add_argument("--require-manifest", action="store_true",
                   help="refuse to run without a dataset manifest; use for "
                        "official experiment runs so no reported figure can "
                        "rest on untraceable splits")
    args = p.parse_args()

    if args.require_manifest and not (args.data_dir / "manifest.json").exists():
        raise SystemExit(
            f"--require-manifest: no manifest.json in {args.data_dir}.\n"
            "Prepare the data with data/scripts/prepare_ner.py or "
            "prepare_qa.py, which record the source, checksum and split seed.")

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

    config_text = args.config.read_text(encoding="utf-8")
    record = {
        "experiment": name,
        "system": args.system,
        "system_description": system_cfg["description"],
        "task": args.task,
        "language": args.language,
        "seed": args.seed,
        # W's status travels with the result. "learned" would be a claim the
        # run cannot support: no objective the paper states trains W.
        "projection": {
            "kind": "externally supplied alignment matrix (paper Sec 4.1)",
            "source": cfg["lgse"].get("alignment_matrix_path") or "identity",
            "training_status": "author-required / unspecified in paper",
            "trained_during_this_run": False,
        },
        "reg_lambda": cfg["lgse"]["reg_lambda"],
        "reg_lambda_source": "unavailable -- not stated in the paper",
        "model_path": model_path,
        "dev": result["dev"],
        "test": result["test"],
        "elapsed_seconds": round(time.time() - started, 1),
        "config": cfg,
        "provenance": provenance(args.config, args.data_dir, args.corpus),
        # The paper places several hyperparameters in Table 1, which could not
        # be recovered. Any run made while some remain marked carries the
        # count, so a record can never be mistaken for a faithful replication.
        "unavailable_hyperparameters": config_text.count("source: unavailable"),
    }
    with open(out_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"{name}: test F1 {result['test']['f1']:.2f}")


if __name__ == "__main__":
    main()
