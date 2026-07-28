"""
The generated results table must state its own fidelity.

A reader should be able to look at Table 2 and see whether each system's
numbers came from a faithful run, without opening DEVIATIONS.md. Two
mechanisms carry that, and both are asserted here:

  * a notice above the table naming any system whose runs lacked an
    author-supplied alignment matrix W;
  * a per-row "Alignment matrix W" column, so a reader knows which of the
    listed systems the notice actually refers to.

The distinction that matters most: the baselines (xlmr, lapt, random_lapt)
use no FastText and therefore need no W. They must read "n/a", never
"not faithful" -- flagging them would be a false warning that trains readers
to ignore the real one.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "aggregate_results.py"


def _record(results_dir, name, system, seed, projection):
    (results_dir / name).mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": name, "system": system, "task": "ner",
        "language": "tigrinya", "seed": seed, "reg_lambda": 1.0,
        "dev": {"f1": 0.5}, "test": {"f1": 0.5},
    }
    if projection is not None:
        record["projection"] = projection
    with open(results_dir / name / "experiment.json", "w") as f:
        json.dump(record, f)


def _supplied(yes):
    return {"kind": "externally supplied alignment matrix (paper Sec 4.1)",
            "author_supplied": yes,
            "source": "/models/W.pt" if yes else None,
            "training_status": "author-required / unspecified in paper",
            "trained_during_this_run": False}


def _table(results_dir):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--results-dir", str(results_dir)],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_baselines_are_not_flagged_as_unfaithful(tmp_path):
    """Systems that use no FastText need no W and must read 'n/a'.

    A false warning here would be worse than none: it teaches readers to
    discount the notice that does matter.
    """
    for name, system in [("a", "xlmr"), ("b", "lapt"), ("c", "random_lapt")]:
        _record(tmp_path, name, system, 42, None)

    table = _table(tmp_path)

    assert "no alignment matrix required" in table
    assert "not faithful" not in table
    assert "Not faithful to the published method" not in table


def test_run_without_author_supplied_w_is_flagged(tmp_path):
    _record(tmp_path, "a", "lgse_lapt", 42, _supplied(False))

    table = _table(tmp_path)

    assert "Not faithful to the published method" in table   # the notice
    assert "lgse_lapt" in table                              # names the system
    assert "**not faithful -- no author-supplied W**" in table  # the row
    assert "DEVIATIONS.md" in table


def test_run_with_author_supplied_w_is_not_flagged(tmp_path):
    _record(tmp_path, "a", "lgse_lapt", 42, _supplied(True))

    table = _table(tmp_path)

    assert "author-supplied W" in table
    assert "Not faithful to the published method" not in table


def test_mixed_seeds_are_reported_as_mixed(tmp_path):
    """One system, two seeds, only one with a W.

    Averaging over them would hide the unfaithful run inside a mean, so the
    row must say so rather than reporting the majority.
    """
    _record(tmp_path, "a", "lgse_lapt", 42, _supplied(True))
    _record(tmp_path, "b", "lgse_lapt", 43, _supplied(False))

    table = _table(tmp_path)

    assert "MIXED" in table
    assert "Not faithful to the published method" in table


def test_full_sweep_distinguishes_every_case(tmp_path):
    """The realistic case: baselines, a faithful LGSE run, an unfaithful one."""
    _record(tmp_path, "a", "xlmr", 42, None)
    _record(tmp_path, "b", "lapt", 42, None)
    _record(tmp_path, "c", "random_lapt", 42, None)
    _record(tmp_path, "d", "focus_lapt", 42, _supplied(True))
    _record(tmp_path, "e", "lgse_lapt", 42, _supplied(False))

    table = _table(tmp_path)
    rows = {line.split("|")[1].strip(): line.split("|")[4].strip()
            for line in table.splitlines()
            if line.startswith("| ") and "±" in line}

    assert rows["XLM-R"].startswith("n/a")
    assert rows["+LAPT"].startswith("n/a")
    assert rows["+Random+LAPT"].startswith("n/a")
    assert rows["+FOCUS+LAPT"] == "author-supplied W"
    assert "not faithful" in rows["+LGSE+LAPT"]


def test_fidelity_column_is_present(tmp_path):
    """The column itself must exist -- the notice alone is not enough,
    since a reader cannot tell which rows it refers to."""
    _record(tmp_path, "a", "lgse_lapt", 42, _supplied(True))

    assert "| Alignment matrix W |" in _table(tmp_path)
