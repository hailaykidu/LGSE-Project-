"""The failure rule, recomputed from the training logs, must reproduce the
committed selection in tables/loss_rule.json, and every run scoring exactly
zero must be among the runs it excludes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _failed(path: Path) -> dict[str, set]:
    d = json.load(open(path, encoding="utf-8"))
    return {lang: {(f["task"], f["system"], f["seed"]) for f in v["failed"]}
            for lang, v in d["languages"].items()}


def test_rule_reproduces_committed_selection(tmp_path):
    out = tmp_path / "loss_rule.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/loss_rule.py"),
                        "--write", str(out)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "matched to a log: 165" in r.stdout
    assert _failed(out) == _failed(ROOT / "tables/loss_rule.json")
    fresh = json.load(open(out, encoding="utf-8"))["languages"]
    assert sum(len(v["failed"]) for v in fresh.values()) == 9


def test_every_zero_score_run_is_excluded():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_table2 as bt
    valid, invalid = bt.load_runs()
    assert not invalid
    excluded = bt.load_exclusions(ROOT / "tables/loss_rule.json")
    zeros = {(e["language"], e["task"], e["system"], e["seed"])
             for e in valid if bt.score(e) == 0.0}
    assert zeros and zeros <= excluded
