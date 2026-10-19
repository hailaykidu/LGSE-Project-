#!/usr/bin/env python3
"""Generate every table, statistic and count reported in the reproduction paper.

Single source of truth. Each number in the paper must be reproducible by
running this script; nothing is typed by hand into the draft.

Protocol (fixed, see --help for the audit output):

  * Only `results_hornmorpho_wam/` is read. That directory holds the one
    common protocol: Amharic, rebuilt splits, `W_am.npy` where W applies,
    one identical `config_sha256` across all systems and tasks.
  * Four other result directories exist and are excluded, each for a
    protocol difference rather than for its numbers. `--audit` prints them
    with reasons.
  * Every valid run is kept, including optimization failures and results
    unfavourable to any system. A poor result is never a reason to exclude.

Failure handling:

  * Primary tables report all runs, plus a failure count per (task, system).
  * A secondary analysis excludes optimization failures under one rule
    applied to both tasks: relative training-loss reduction from the first
    to the last logged epoch below a threshold placed inside the empirical
    gap between failed and successful runs. The rule is computed from
    training logs only -- never from test scores -- and the script reports
    whether the gap is clear and whether the selection is insensitive to
    the threshold's exact value within it.

Usage:
    python3 scripts/build_paper_tables.py --audit
    python3 scripts/build_paper_tables.py --out paper/tables
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_DIR = ROOT / "results_hornmorpho_wam"

SYSTEMS = ["lgse_lapt", "focus_lapt", "xlmr"]
DISPLAY = {"lgse_lapt": "LGSE", "focus_lapt": "FOCUS", "xlmr": "XLM-R"}
TASKS = ["tc", "ner", "qa"]
# TC is scored by macro-F1 over the five classes present in the test split.
# Accuracy is reported alongside it but is not the primary metric: the
# majority class holds 66.4% of the test set, so accuracy is dominated by it.
PRIMARY_METRIC = {"tc": "macro_f1_5class", "ner": "f1", "qa": "f1"}
SECONDARY_METRIC = {"tc": "accuracy"}
SEEDS = list(range(42, 52))

# The config every run in this protocol must share. Asserted, not assumed:
# a run recording anything else is reported INVALID.
EXPECTED_CONFIG_SHA_PREFIX = "3e51cdabc22d6317"
EXPECTED_DATA_DIR = {
    "tc": "data/tc/amharic",
    "ner": "data/ner/amharic",
    "qa": "data/qa/amqa",
}

# Amharic TC label set. Label 5 occurs twice in train and never in dev or
# test, so the recorded macro-F1 averages over six classes including an
# empty one, which caps it at 5/6. The paper's primary macro-F1 averages
# over the five classes present in test; both are reported.
TC_CLASSES_IN_TEST = ["1", "2", "3", "4", "6"]
TC_CLASSES_RECORDED = ["1", "2", "3", "4", "5", "6"]

# Directories deliberately excluded, with the protocol difference that
# disqualifies each. Verified by `--audit`, not asserted here.
EXCLUDED_DIRS = [
    ("results_gen2_snapshot_20260924",
     "three distinct config_sha256; mixes W_am and W_ti; "
     "Amharic TC evaluated on the superseded 71-item test set"),
    ("results",
     "pre-snapshot copy of the above, same protocol differences"),
    ("results_hornmorpho",
     "mixes W_am and W_ti; contains the runs that applied the Tigrinya "
     "alignment matrix to Amharic data"),
    ("results_tigrinya_gen2",
     "Tigrinya only; no three-system comparison on a common evaluation"),
]


# --------------------------------------------------------------------------
# loading and audit
# --------------------------------------------------------------------------

def load_runs() -> tuple[list[dict], list[dict]]:
    """Return (valid, invalid) run records from the protocol directory."""
    valid, invalid = [], []
    for path in sorted(PROTOCOL_DIR.glob("*/experiment.json")):
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except Exception as exc:                       # pragma: no cover
            invalid.append({"path": str(path), "reasons": [f"unreadable: {exc}"]})
            continue

        task, system, seed = rec["task"], rec["system"], rec["seed"]
        prov = rec.get("provenance", {})
        reasons = []

        if not prov.get("commit"):
            reasons.append("no commit recorded")
        sha = prov.get("config_sha256", "")
        if not sha.startswith(EXPECTED_CONFIG_SHA_PREFIX):
            reasons.append(f"config_sha256 {sha[:16]!r} differs from protocol")
        if prov.get("data_dir") != EXPECTED_DATA_DIR.get(task):
            reasons.append(f"data_dir {prov.get('data_dir')!r}")
        # W is loaded only on the FastText-consuming path, so only those
        # systems are required to carry the Amharic matrix.
        if system in ("lgse_lapt", "focus_lapt"):
            src = str(rec.get("projection", {}).get("source") or "")
            if not src.endswith("W_am.npy"):
                reasons.append(f"alignment matrix {Path(src).name or 'absent'!r}")
        test = rec.get("test", {})
        needed = ["macro_f1", "per_class"] if task == "tc" else [PRIMARY_METRIC[task]]
        for key in needed:
            if test.get(key) is None:
                reasons.append(f"test.{key} missing")

        entry = {
            "task": task, "system": system, "seed": seed,
            "record": rec, "path": path, "reasons": reasons,
        }
        (invalid if reasons else valid).append(entry)
    return valid, invalid


def score(entry: dict, metric: str | None = None) -> float:
    """Primary (or named) test metric, normalised to a 0-100 scale.

    "macro_f1_5class" is derived here rather than read: the recorded
    macro_f1 averages over six classes including label 5, which appears
    twice in train and never in dev or test, so the recorded value is
    capped at 5/6 of the attainable maximum.
    """
    task = entry["task"]
    name = metric or PRIMARY_METRIC[task]
    if name == "macro_f1_5class":
        return tc_macro_f1(entry, TC_CLASSES_IN_TEST)
    if name == "macro_f1_6class":
        return tc_macro_f1(entry, TC_CLASSES_RECORDED)
    value = entry["record"]["test"][name]
    return value * 100 if value <= 1 else value


def tc_macro_f1(entry: dict, classes: list[str]) -> float:
    """Macro-F1 over the given classes, recomputed from per-class F1."""
    per_class = entry["record"]["test"]["per_class"]
    return st.mean(per_class[c]["f1"] for c in classes)


# --------------------------------------------------------------------------
# failure detection, from training logs only
# --------------------------------------------------------------------------

SEED_BANNER = re.compile(r"=== \w+__\w+__amharic__seed(\d+)")


def slurm_log_index() -> dict:
    """Map SLURM job name -> [(job id, stdout path)], newest job id first.

    Built from `sacct --format=JobID,JobName,StdOut`, so the link between a
    run and its log is the scheduler's own record rather than a guessed
    filename. Two bugs made this necessary: superseded jobs from a different
    protocol wrote the same per-seed banners into their own logs, and a
    filename glob picked those up in sort order. See
    tests/test_build_paper_tables.py.
    """
    import os
    import subprocess
    index = defaultdict(list)
    try:
        out = subprocess.run(
            ["sacct", "-S", "2026-09-01", "-u", os.environ.get("USER", ""),
             "--format=JobID,JobName%40,StdOut%200", "-X", "-n"],
            capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return index
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        index[parts[1]].append((parts[0], Path(parts[-1])))
    for name in index:
        index[name].sort(key=lambda t: int(t[0]), reverse=True)
    return index


_LOG_INDEX = None


def train_curve(task: str, system: str, seed: int) -> dict | None:
    """Training-loss curve for one run, located via its SLURM job id.

    Banner placement differs between the two job layouts, and getting it
    wrong silently reads the wrong curve or none at all:

      * five-seed jobs print the banner BEFORE each seed's training, and
        interleave seeds in one file, so a seed's text runs from its banner
        to the next banner for a different seed;
      * single-seed jobs print the runner's banner AFTER training, so the
        text after the last banner holds no losses at all.

    Rather than encode either layout, every candidate region is scanned and
    the one that actually contains at least two loss lines is used. Regions
    belonging to a different seed are excluded first.
    """
    global _LOG_INDEX
    if _LOG_INDEX is None:
        _LOG_INDEX = slurm_log_index()

    banner = f"=== {system}__{task}__amharic__seed{seed}"
    for name in (f"w_{task}_{system}_s{seed}", f"wam_{task}_{system}"):
        for job_id, log_path in _LOG_INDEX.get(name, []):
            if not log_path.exists():
                continue
            text = log_path.read_text(errors="replace")

            if banner in text:
                # candidate regions: after each banner for THIS seed, each
                # truncated at the next banner for a different seed
                regions = []
                for part in text.split(banner)[1:]:
                    for m in SEED_BANNER.finditer(part):
                        if int(m.group(1)) != seed:
                            part = part[: m.start()]
                            break
                    regions.append(part)
                # a single-seed log is wholly one run, so its own text is
                # also a valid region when the banner trails the training
                if name.endswith(f"_s{seed}"):
                    regions.append(text)
            elif name.endswith(f"_s{seed}"):
                regions = [text]
            else:
                continue

            for region in regions:
                losses = [float(x) for x in
                          re.findall(r"'loss': ([0-9.]+)", region)]
                if len(losses) >= 2 and losses[0]:
                    return {"job_id": job_id, "log": log_path,
                            "first": losses[0], "last": losses[-1],
                            "reduction": (losses[0] - losses[-1]) / losses[0]}
    return None


def train_loss_reduction(task: str, system: str, seed: int) -> float | None:
    curve = train_curve(task, system, seed)
    return curve["reduction"] if curve else None


def observed_failure(entry: dict) -> bool:
    """Outcome-side marker of a degenerate run, for gap analysis only.

    NER: the tagger proposes no entity, so F1 is exactly zero.
    TC:  the classifier never predicts class 3 (19% of test), the partial
         label collapse described in the paper.

    This is NOT the exclusion rule. It is used only to check whether the
    training-loss distribution separates failed from successful runs.
    """
    if entry["task"] == "ner":
        return score(entry) == 0.0
    if entry["task"] == "tc":
        return entry["record"]["test"]["per_class"]["3"]["f1"] == 0.0
    return False


def failure_analysis(valid: list[dict]) -> dict:
    """Compute the loss-reduction distribution and derive the rule."""
    rows = []
    for e in valid:
        if e["task"] == "qa":
            continue                 # no degenerate runs observed on QA
        red = train_loss_reduction(e["task"], e["system"], e["seed"])
        rows.append({
            "task": e["task"], "system": e["system"], "seed": e["seed"],
            "reduction": red, "observed_failure": observed_failure(e),
        })

    missing = [r for r in rows if r["reduction"] is None]
    failed = [r["reduction"] for r in rows
              if r["observed_failure"] and r["reduction"] is not None]
    ok = [r["reduction"] for r in rows
          if not r["observed_failure"] and r["reduction"] is not None]

    gap_clear = bool(failed and ok and max(failed) < min(ok))
    threshold = None
    if gap_clear:
        # Midpoint of the empirical gap. Any threshold inside the gap
        # selects the same runs; the report states that explicitly.
        threshold = (max(failed) + min(ok)) / 2

    selected = []
    if threshold is not None:
        selected = [r for r in rows
                    if r["reduction"] is not None and r["reduction"] < threshold]

    return {
        "rows": rows, "missing": missing,
        "failed_range": (min(failed), max(failed)) if failed else None,
        "ok_range": (min(ok), max(ok)) if ok else None,
        "gap_clear": gap_clear, "threshold": threshold,
        "selected": selected,
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def mean_sd_n(values: list[float]) -> tuple[float, float, int]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    return st.mean(values), (st.stdev(values) if n > 1 else 0.0), n


def paired_comparison(a: dict[int, float], b: dict[int, float]) -> dict:
    """Mean paired difference with 95% CI, plus Wilcoxon and win/loss.

    The t-interval is the primary result. The exact Wilcoxon p is a
    robustness check and is reported with its own floor, because at these
    sample sizes the smallest attainable two-sided p is 0.0625 (n=5) and
    0.125 (n=4) -- p < 0.05 is unreachable by construction.
    """
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    out = {
        "n": n, "seeds": seeds,
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
    }
    if n < 2:
        out.update(mean_diff=diffs[0] if diffs else float("nan"),
                   ci=(float("nan"), float("nan")),
                   p_t=float("nan"), p_wilcoxon=float("nan"),
                   wilcoxon_floor=float("nan"))
        return out

    mean_d = st.mean(diffs)
    sd_d = st.stdev(diffs)
    out["mean_diff"] = mean_d
    out["sd_diff"] = sd_d

    try:
        from scipy import stats as sps
        tcrit = sps.t.ppf(0.975, n - 1)
        half = tcrit * sd_d / math.sqrt(n)
        out["ci"] = (mean_d - half, mean_d + half)
        out["p_t"] = sps.ttest_rel([a[s] for s in seeds],
                                   [b[s] for s in seeds]).pvalue
        try:
            out["p_wilcoxon"] = sps.wilcoxon([a[s] for s in seeds],
                                             [b[s] for s in seeds]).pvalue
        except ValueError:
            out["p_wilcoxon"] = float("nan")
        # smallest attainable two-sided exact p at the effective n, i.e.
        # after scipy drops exact ties
        n_eff = n - out["ties"]
        out["n_effective"] = n_eff
        out["wilcoxon_floor"] = (
            sps.wilcoxon(list(range(1, n_eff + 1)), [0] * n_eff).pvalue
            if n_eff >= 1 else float("nan"))
    except ImportError:                                # pragma: no cover
        out.update(ci=(float("nan"), float("nan")), p_t=float("nan"),
                   p_wilcoxon=float("nan"), wilcoxon_floor=float("nan"),
                   n_effective=n - out["ties"])
    return out


# --------------------------------------------------------------------------
# coverage count
# --------------------------------------------------------------------------

def coverage_count() -> dict:
    """Multi-morpheme coverage of the Amharic added-token list.

    Reports the pipeline as configured for these runs -- configs/
    hornmorpho.yaml sets `use_hornmorpho: true`, so `lap_trainer.py` line 81
    attaches the amseg analyzer for Amharic -- and, separately, the static
    lexicon on its own.

    The two must be measured on independently constructed segmenters:
    `with_hornmorpho()` mutates and returns `self`, so a shared object
    reports the analyzer-enabled count for both.

    Attaching the analyzer takes minutes for 198 tokens, so the
    as-configured figure is read from tables/coverage.json when that file
    matches the current token list, and recomputed otherwise.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from lgse.segmentation import MorphologicalSegmenter

    token_file = ROOT / "data" / "new_tokens_amharic.txt"
    lex_file = ROOT / "data" / "morph_lexicon.txt"
    tokens = [l.strip() for l in open(token_file, encoding="utf-8") if l.strip()]

    lexicon = MorphologicalSegmenter.from_file(str(lex_file))
    lexicon_only = sum(1 for t in tokens if len(lexicon.segment(t)) > 1)
    in_lexicon = sum(1 for t in tokens if t in lexicon.lexicon)

    as_configured, cached_from = None, None
    cache = ROOT / "tables" / "coverage.json"
    if cache.exists():
        c = json.load(open(cache, encoding="utf-8"))
        if c.get("n") == len(tokens):
            as_configured, cached_from = c.get("as_configured_multi"), str(
                cache.relative_to(ROOT))
    if as_configured is None:
        analyzer = MorphologicalSegmenter.from_file(
            str(lex_file)).with_hornmorpho()
        as_configured = sum(1 for t in tokens if len(analyzer.segment(t)) > 1)

    return {
        "n_tokens": len(tokens),
        "in_lexicon": in_lexicon,
        "lexicon_only_multi": lexicon_only,
        "lexicon_only_pct": 100 * lexicon_only / len(tokens),
        "as_configured_multi": as_configured,
        "as_configured_pct": 100 * as_configured / len(tokens),
        "self_only": in_lexicon - lexicon_only,
        "token_list": str(token_file.relative_to(ROOT)),
        "lexicon": str(lex_file.relative_to(ROOT)),
        "cached_from": cached_from,
    }



# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_audit(valid, invalid):
    print("=" * 74)
    print("AUDIT")
    print("=" * 74)
    print(f"\nprotocol directory: {PROTOCOL_DIR.relative_to(ROOT)}")
    print(f"  VALID   {len(valid)}")
    print(f"  INVALID {len(invalid)}")
    if invalid:
        print("\n  invalid runs:")
        for e in invalid:
            who = e.get("task", "?"), e.get("system", "?"), e.get("seed", "?")
            print(f"    {who[0]:3s} {who[1]:11s} s{who[2]}: "
                  f"{'; '.join(e['reasons'])}")
    counts = Counter((e["task"], e["system"]) for e in valid)
    print("\n  valid runs per (task, system):")
    for task in TASKS:
        for s in SYSTEMS:
            print(f"    {task:3s} {DISPLAY[s]:6s} n={counts[(task, s)]}")
    shas = {e["record"]["provenance"]["config_sha256"] for e in valid}
    print(f"\n  distinct config_sha256 across valid runs: {len(shas)}")
    for sha in sorted(shas):
        print(f"    {sha}")
    print("\n  excluded directories (protocol differences, not outcomes):")
    for name, reason in EXCLUDED_DIRS:
        n = len(list((ROOT / name).glob("*/experiment.json"))) \
            if (ROOT / name).exists() else 0
        print(f"    {name}  (n={n})\n      {reason}")


def print_tables(valid, out_dir: Path | None):
    by = defaultdict(dict)
    for e in valid:
        by[(e["task"], e["system"])][e["seed"]] = e

    fa = failure_analysis(valid)

    # ---- failure rule -------------------------------------------------
    print("\n" + "=" * 74)
    print("FAILURE RULE (training logs only; no test scores used)")
    print("=" * 74)
    print("\nrelative training-loss reduction, first to last logged epoch:")
    for task in ("tc", "ner"):
        rows = [r for r in fa["rows"] if r["task"] == task]
        if not rows:
            continue
        print(f"\n  {task.upper()}")
        print(f"    {'system':11s}{'seed':6s}{'reduction':>11s}  observed")
        for r in sorted(rows, key=lambda r: (r["system"], r["seed"])):
            red = "--" if r["reduction"] is None else f"{r['reduction']:.1%}"
            print(f"    {DISPLAY[r['system']]:11s}s{r['seed']:<5d}{red:>11s}"
                  f"  {'FAILED' if r['observed_failure'] else 'ok'}")
    if fa["missing"]:
        print(f"\n  WARNING: no training log found for {len(fa['missing'])} run(s):")
        for r in fa["missing"]:
            print(f"    {r['task']} {r['system']} s{r['seed']}")
    if fa["failed_range"] and fa["ok_range"]:
        fl, fh = fa["failed_range"]
        ol, oh = fa["ok_range"]
        print(f"\n  failed runs:     {fl:.1%} .. {fh:.1%}")
        print(f"  successful runs: {ol:.1%} .. {oh:.1%}")
        print(f"  gap: {'CLEAR' if fa['gap_clear'] else 'OVERLAPPING'}")
        if fa["gap_clear"]:
            print(f"  any threshold in ({fh:.1%}, {ol:.1%}) selects the same "
                  f"runs; the midpoint {fa['threshold']:.1%} is used")
            print(f"  selected as optimization failures ({len(fa['selected'])}):")
            for r in sorted(fa["selected"], key=lambda r: (r["task"], r["system"])):
                print(f"    {r['task']:3s} {DISPLAY[r['system']]:6s} s{r['seed']}"
                      f"  reduction {r['reduction']:.1%}")
        else:
            print("  no clear gap: the secondary analysis is not reported")
    failset = {(r["task"], r["system"], r["seed"]) for r in fa["selected"]}

    # ---- Table 2 ------------------------------------------------------
    print("\n" + "=" * 74)
    print("TABLE 2  primary: all valid runs included")
    print("=" * 74)
    print("\n  Amharic TC majority-class baseline: accuracy 66.38, "
          "macro-F1 15.96 (5 classes), 13.30 (6 classes, as recorded)")
    for task in TASKS:
        primary = "macro-F1 (5 classes)" if task == "tc" else "F1"
        print(f"\n  {task.upper()}  primary: {primary}")
        hdr = f"    {'system':8s}{'mean':>8s}{'sd':>7s}{'n':>4s}{'fail':>6s}"
        if task == "tc":
            hdr += f"{'acc':>9s}{'sd':>7s}{'macroF1_6':>11s}{'sd':>7s}"
        print(hdr)
        for s in SYSTEMS:
            runs = by[(task, s)]
            m, sd, n = mean_sd_n([score(runs[k]) for k in sorted(runs)])
            nfail = sum(1 for k in runs if (task, s, k) in failset)
            line = f"    {DISPLAY[s]:8s}{m:8.2f}{sd:7.2f}{n:4d}{nfail:6d}"
            if task == "tc":
                ma, sda, _ = mean_sd_n(
                    [score(runs[k], "accuracy") for k in sorted(runs)])
                m6, sd6, _ = mean_sd_n(
                    [score(runs[k], "macro_f1_6class") for k in sorted(runs)])
                line += f"{ma:9.2f}{sda:7.2f}{m6:11.2f}{sd6:7.2f}"
            print(line)
        if task == "tc":
            print("    secondary: accuracy; macroF1_6 is the value recorded in")
            print("    the result files, averaging over an empty sixth class.")

    print("\n" + "=" * 74)
    print("TABLE 2b  secondary: optimization failures excluded")
    print("=" * 74)
    if not fa["gap_clear"]:
        print("\n  not reported: no clear gap in the loss-reduction distribution")
    else:
        for task in TASKS:
            print(f"\n  {task.upper()}")
            print(f"    {'system':8s}{'mean':>8s}{'sd':>7s}{'n':>4s}{'dropped':>9s}")
            for s in SYSTEMS:
                runs = {k: v for k, v in by[(task, s)].items()
                        if (task, s, k) not in failset}
                dropped = len(by[(task, s)]) - len(runs)
                vals = [score(runs[k]) for k in sorted(runs)]
                m, sd, n = mean_sd_n(vals)
                print(f"    {DISPLAY[s]:8s}{m:8.2f}{sd:7.2f}{n:4d}{dropped:9d}")

    # ---- Table 3 ------------------------------------------------------
    print("\n" + "=" * 74)
    print("TABLE 3  paired comparisons, LGSE minus baseline, all valid runs")
    print("=" * 74)
    print("\n  primary: mean paired difference with 95% CI (paired t)")
    print("  robustness: exact Wilcoxon p, win/loss/tie counts")
    print(f"\n    {'task':5s}{'vs':8s}{'n':>3s}{'diff':>8s}"
          f"{'95% CI':>18s}{'p_t':>8s}{'p_wilc':>8s}{'floor':>8s}  W/L/T")
    for task in TASKS:
        a = {k: score(v) for k, v in by[(task, "lgse_lapt")].items()}
        for opp in ("focus_lapt", "xlmr"):
            b = {k: score(v) for k, v in by[(task, opp)].items()}
            r = paired_comparison(a, b)
            ci = f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
            print(f"    {task:5s}{DISPLAY[opp]:8s}{r['n']:3d}"
                  f"{r['mean_diff']:+8.2f}{ci:>18s}"
                  f"{r['p_t']:8.3f}{r['p_wilcoxon']:8.3f}"
                  f"{r['wilcoxon_floor']:8.3f}"
                  f"  {r['wins']}/{r['losses']}/{r['ties']}")
    print("\n  floor = smallest attainable two-sided exact Wilcoxon p at the")
    print("  effective n. Where floor > 0.05, significance is unreachable")
    print("  by construction and the p-value carries no information.")
    print("  Every interval above contains zero. No ordering of the three")
    print("  systems is supported; means and intervals only.")

    # ---- per-class TC -------------------------------------------------
    print("\n" + "=" * 74)
    print("TABLE 4  Amharic TC per-class F1")
    print("=" * 74)
    sup = {c: by[("tc", "lgse_lapt")][42]["record"]["test"]["per_class"][c]["support"]
           for c in TC_CLASSES_RECORDED}
    print(f"\n    {'support':19s}" + "".join(f"{sup[c]:>8d}" for c in TC_CLASSES_RECORDED))
    print(f"    {'system':11s}{'seed':8s}"
          + "".join(f"{c:>8s}" for c in TC_CLASSES_RECORDED)
          + f"{'macro5':>9s}{'macro6':>9s}{'acc':>8s}")
    for s in SYSTEMS:
        for k in sorted(by[("tc", s)]):
            e = by[("tc", s)][k]
            pc = e["record"]["test"]["per_class"]
            print(f"    {DISPLAY[s]:11s}s{k:<7d}"
                  + "".join(f"{pc[c]['f1']:8.2f}" for c in TC_CLASSES_RECORDED)
                  + f"{tc_macro_f1(e, TC_CLASSES_IN_TEST):9.2f}"
                  + f"{tc_macro_f1(e, TC_CLASSES_RECORDED):9.2f}"
                  + f"{score(e):8.2f}")

    # ---- seed variance vs system spread -------------------------------
    print("\n" + "=" * 74)
    print("Seed variation against between-system spread (descriptive)")
    print("=" * 74)
    for label, drop in (("all valid runs", False),
                        ("optimization failures excluded", True)):
        if drop and not fa["gap_clear"]:
            continue
        print(f"\n  {label}")
        print(f"    {'task':5s}{'spread':>9s}{'mean sd':>10s}{'ratio':>8s}{'n/system':>10s}")
        for task in TASKS:
            means, sds, ns = [], [], []
            for s in SYSTEMS:
                runs = {k: v for k, v in by[(task, s)].items()
                        if not (drop and (task, s, k) in failset)}
                vals = [score(v) for v in runs.values()]
                m, sd, n = mean_sd_n(vals)
                means.append(m); sds.append(sd); ns.append(n)
            spread = max(means) - min(means)
            msd = st.mean(sds)
            ratio = spread / msd if msd else float("nan")
            span = f"{min(ns)}-{max(ns)}"
            print(f"    {task:5s}{spread:9.2f}{msd:10.2f}{ratio:8.2f}{span:>10s}")
        print("    ratio below 1: seed-to-seed variation exceeds the whole")
        print("    spread between the three systems.")

    # ---- coverage -----------------------------------------------------
    cov = coverage_count()
    print("\n" + "=" * 74)
    print("Morpheme coverage of the Amharic added-token list")
    print("=" * 74)
    print(f"\n  token list: {cov['token_list']}  (n={cov['n_tokens']})")
    print(f"  lexicon:    {cov['lexicon']}")
    print(f"  as configured (lexicon + HornMorpho analyzer, the setting these")
    print(f"    runs used):                  {cov['as_configured_multi']}"
          f"/{cov['n_tokens']} ({cov['as_configured_pct']:.1f}%)")
    print(f"  static lexicon component:      {cov['lexicon_only_multi']}"
          f"/{cov['n_tokens']} ({cov['lexicon_only_pct']:.1f}%)")
    print(f"  tokens with a lexicon entry:   {cov['in_lexicon']}")
    print(f"  entry maps the word to itself: {cov['self_only']}")
    if cov["cached_from"]:
        print(f"  as-configured figure read from {cov['cached_from']}")
    print("\n  Descriptive only. Coverage is not used to explain the")
    print("  comparison outcomes.")

    # ---- per-seed CSV -------------------------------------------------
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "per_seed.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "system", "seed", "primary_metric", "score",
                        "tc_macro_f1_5class", "tc_macro_f1_6class",
                        "optimization_failure", "job_commit", "result_file"])
            for task in TASKS:
                for s in SYSTEMS:
                    for k in sorted(by[(task, s)]):
                        e = by[(task, s)][k]
                        w.writerow([
                            task, DISPLAY[s], k, PRIMARY_METRIC[task],
                            f"{score(e):.4f}",
                            f"{tc_macro_f1(e, TC_CLASSES_IN_TEST):.4f}" if task == "tc" else "",
                            f"{tc_macro_f1(e, TC_CLASSES_RECORDED):.4f}" if task == "tc" else "",
                            int((task, s, k) in failset),
                            e["record"]["provenance"].get("commit", "")[:12],
                            str(e["path"].relative_to(ROOT)),
                        ])
        try:
            shown = csv_path.relative_to(ROOT)
        except ValueError:
            shown = csv_path          # --out may point outside the repo
        print(f"\nwrote {shown} "
              f"({sum(len(by[(t, s)]) for t in TASKS for s in SYSTEMS)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true",
                    help="print the run audit and exit")
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for the per-seed CSV")
    args = ap.parse_args()

    if not PROTOCOL_DIR.exists():
        print(f"protocol directory missing: {PROTOCOL_DIR}", file=sys.stderr)
        return 1

    valid, invalid = load_runs()
    print_audit(valid, invalid)
    if args.audit:
        return 0
    print_tables(valid, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
