#!/usr/bin/env python3
"""Generate the complete Table 2: five systems, three tasks, two languages.

Single source of truth for every number in the replacement paper's main
table. Nothing is typed by hand; running this reproduces the tables.

WHERE THE RUNS COME FROM, AND WHY
---------------------------------
Three directories supply the 150 runs, each for a stated reason. The split
is not arbitrary and is asserted at load time rather than assumed:

  results_hornmorpho_wam/   Amharic XLM-R, FOCUS, LGSE.
                            configs/hornmorpho.yaml, W_am.npy, 235-item TC.

  results_hornmorpho/       Amharic +LAPT and +Random+LAPT only.
                            Its config names W_ti.npy, which would normally
                            disqualify an Amharic run -- and does disqualify
                            that directory's LGSE and FOCUS runs
                            (IMPLEMENTATION_NOTES.md 5d-v). It does not
                            disqualify these two systems, because neither
                            consumes FastText and W is read only on the
                            FastText path (src/lgse/lap_trainer.py:78). That
                            is verified, not argued: all ten +LAPT and
                            +Random+LAPT NER runs are bit-identical to the
                            W_am snapshot, delta 0.000000 on every seed,
                            including one seed that recorded W_am in one
                            directory and W_ti in the other. The two configs
                            differ only in the alignment_matrix_path line.

  results_tigrinya_gen2/    all five Tigrinya systems.
                            configs/tigrinya.yaml, W_ti.npy, 151-item QA.

Four other result directories exist and are excluded for protocol
differences, never for their outcomes; --audit prints them with reasons.

WITHIN vs ACROSS LANGUAGE
-------------------------
Every hyperparameter is identical between the two language configs (LAPT and
fine-tuning settings, lambda, n-gram range, FastText dimension), so the
within-language comparisons this script reports are sound. What differs
between languages is the data each must use, plus one asymmetry worth
disclosing: configs/hornmorpho.yaml enables the amseg analyzer, which exists
for Amharic only, so LGSE's morpheme coverage is higher on Amharic than on
Tigrinya.

Across languages the TC columns are not the same quantity: the Amharic test
split has a 66.38% majority class and the Tigrinya one 32.43%. Accuracy on
the two is differently scaled and must not be pooled or averaged. This is
why macro-F1 over the classes present in test is the primary TC metric here,
with accuracy secondary.

Usage:
    python3 scripts/build_table2.py --audit
    python3 scripts/build_table2.py --out tables
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LANGUAGES = ["amharic", "tigrinya"]
SYSTEMS = ["xlmr", "lapt", "random_lapt", "focus_lapt", "lgse_lapt"]
BASELINES = ["lapt", "random_lapt", "focus_lapt", "xlmr"]   # LGSE's comparators
TASKS = ["tc", "ner", "qa"]
SEEDS = list(range(42, 52))

DISPLAY = {"xlmr": "XLM-R", "lapt": "+LAPT", "random_lapt": "+Random+LAPT",
           "focus_lapt": "+FOCUS+LAPT", "lgse_lapt": "+LGSE+LAPT"}

# (language, system) -> directory, with the config checksum each must carry.
SOURCE = {
    ("amharic", "xlmr"):        ("results_hornmorpho_wam", "3e51cdabc22d6317"),
    ("amharic", "focus_lapt"):  ("results_hornmorpho_wam", "3e51cdabc22d6317"),
    ("amharic", "lgse_lapt"):   ("results_hornmorpho_wam", "3e51cdabc22d6317"),
    ("amharic", "lapt"):        ("results_hornmorpho", "1fb98f621d25"),
    ("amharic", "random_lapt"): ("results_hornmorpho", "1fb98f621d25"),
    ("tigrinya", "xlmr"):        ("results_tigrinya_gen2", "0fe240d4f75000af"),
    ("tigrinya", "lapt"):        ("results_tigrinya_gen2", "0fe240d4f75000af"),
    ("tigrinya", "random_lapt"): ("results_tigrinya_gen2", "0fe240d4f75000af"),
    ("tigrinya", "focus_lapt"):  ("results_tigrinya_gen2", "0fe240d4f75000af"),
    ("tigrinya", "lgse_lapt"):   ("results_tigrinya_gen2", "0fe240d4f75000af"),
}

# W is required to match the language, but only for the systems that read it.
NEEDS_W = {"lgse_lapt", "focus_lapt"}
EXPECTED_W = {"amharic": "W_am.npy", "tigrinya": "W_ti.npy"}

EXPECTED_DATA_DIR = {
    ("amharic", "tc"): "data/tc/amharic",
    ("amharic", "ner"): "data/ner/amharic",
    ("amharic", "qa"): "data/qa/amqa",
    ("tigrinya", "tc"): "data/tc/tigrinya",
    ("tigrinya", "ner"): "data/ner/tigrinya",
    ("tigrinya", "qa"): "data/qa/tigqa_squad",
}

# Both TC tasks use six label values, but which of them occur in the test
# split differs by language, so the set cannot be hardcoded. Amharic test
# holds classes 1,2,3,4,6 -- label 5 occurs in train and in neither dev nor
# test -- while Tigrinya test holds all six, class 5 with a single instance.
# The recorded macro_f1 averages over all six regardless, so on Amharic it is
# capped at five sixths of its maximum. The primary metric averages over the
# classes actually present, read per run from the per-class supports.
TC_CLASSES_RECORDED = ["1", "2", "3", "4", "5", "6"]


def tc_classes_in_test(entry: dict) -> list[str]:
    """Label values with at least one test instance, for this run."""
    pc = entry["record"]["test"]["per_class"]
    return [c for c in TC_CLASSES_RECORDED
            if c in pc and pc[c]["support"] > 0]

# Majority-class baselines, computed from the test splits (see --audit).
MAJORITY = {"amharic": {"accuracy": 66.38, "macro_f1_5": 15.96},
            "tigrinya": {"accuracy": 32.43, "macro_f1_5": None}}

EXCLUDED_DIRS = [
    ("results_gen2_snapshot_20260924",
     "the September sweep. Withdrawn: all fifteen Amharic LGSE runs loaded "
     "W_ti.npy on Amharic data, and its Tigrinya QA used the superseded "
     "86-item split. See report/TABLE2_REPORT.md."),
    ("results",
     "same sweep, and its Amharic records were overwritten by a later run; "
     "only the committed report/per_seed_results.json preserves them"),
    ("results_hornmorpho (LGSE and FOCUS runs)",
     "those two systems consume FastText and so loaded the W_ti.npy this "
     "directory's config names; its +LAPT and +Random+LAPT runs are used"),
    ("results_b",
     "eight directories, no experiment.json; an abandoned partial run"),
]


# --------------------------------------------------------------------------
# loading and audit
# --------------------------------------------------------------------------

def load_runs() -> tuple[list[dict], list[dict]]:
    """Return (valid, invalid) records across both languages."""
    valid, invalid = [], []
    for (lang, system), (dirname, sha_prefix) in SOURCE.items():
        base = ROOT / dirname
        for task in TASKS:
            for seed in SEEDS:
                path = base / f"{system}__{task}__{lang}__seed{seed}" / "experiment.json"
                if not path.exists():
                    continue
                try:
                    rec = json.load(open(path, encoding="utf-8"))
                except Exception as exc:               # pragma: no cover
                    invalid.append({"path": path, "language": lang,
                                    "system": system, "task": task,
                                    "seed": seed,
                                    "reasons": [f"unreadable: {exc}"]})
                    continue
                prov = rec.get("provenance", {})
                reasons = []
                if not prov.get("commit"):
                    reasons.append("no commit recorded")
                if not prov.get("config_sha256", "").startswith(sha_prefix):
                    reasons.append(
                        f"config_sha256 {prov.get('config_sha256','')[:16]!r} "
                        f"!= {sha_prefix!r}")
                if prov.get("data_dir") != EXPECTED_DATA_DIR[(lang, task)]:
                    reasons.append(f"data_dir {prov.get('data_dir')!r}")
                if system in NEEDS_W:
                    src = str(rec.get("projection", {}).get("source") or "")
                    if not src.endswith(EXPECTED_W[lang]):
                        reasons.append(
                            f"alignment matrix {Path(src).name or 'absent'!r} "
                            f"!= {EXPECTED_W[lang]}")
                test = rec.get("test", {})
                need = ["macro_f1", "per_class"] if task == "tc" else ["f1"]
                for key in need:
                    if test.get(key) is None:
                        reasons.append(f"test.{key} missing")
                entry = {"language": lang, "system": system, "task": task,
                         "seed": seed, "record": rec, "path": path,
                         "source_dir": dirname, "reasons": reasons}
                (invalid if reasons else valid).append(entry)
    return valid, invalid


def tc_macro(entry: dict, classes: list[str]) -> float:
    pc = entry["record"]["test"]["per_class"]
    return st.mean(pc[c]["f1"] for c in classes)


def score(entry: dict, metric: str | None = None) -> float:
    """Primary metric by default, on a 0-100 scale.

    TC primary is macro-F1 over the classes present in the test split, not
    the recorded six-class value; see the module docstring.
    """
    task = entry["task"]
    name = metric or ("macro_f1_5" if task == "tc" else "f1")
    if name == "macro_f1_5":
        return tc_macro(entry, tc_classes_in_test(entry))
    if name == "macro_f1_6":
        return tc_macro(entry, TC_CLASSES_RECORDED)
    value = entry["record"]["test"][name]
    return value * 100 if value <= 1 else value


def mean_sd_n(values: list[float]) -> tuple[float, float, int, int]:
    """Mean and sd over converged runs; also returns total and failure count.

    A run scoring exactly zero is a degenerate predictor, not a measurement
    of the system, so it is excluded from the mean and counted separately.
    """
    total = len(values)
    conv = [v for v in values if v > 0]
    if not conv:
        return float("nan"), float("nan"), 0, total
    sd = st.stdev(conv) if len(conv) > 1 else 0.0
    return st.mean(conv), sd, len(conv), total - len(conv)


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    """Mean paired difference with a 95% t-interval, plus exact Wilcoxon.

    The interval is the primary result. The Wilcoxon p is reported with the
    smallest value attainable at the effective n, because at five seeds that
    floor is 0.0625 and at four it is 0.125 -- p < 0.05 is unreachable by
    construction, and a p-value that cannot fall below the threshold carries
    no information about the hypothesis.
    """
    seeds = [s for s in sorted(set(a) & set(b)) if a[s] > 0 and b[s] > 0]
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    out = {"n": n, "seeds": seeds,
           "wins": sum(1 for d in diffs if d > 0),
           "losses": sum(1 for d in diffs if d < 0),
           "ties": sum(1 for d in diffs if d == 0)}
    if n < 2:
        out.update(mean_diff=float("nan"), ci=(float("nan"), float("nan")),
                   p_wilcoxon=float("nan"), floor=float("nan"))
        return out
    mean_d, sd_d = st.mean(diffs), st.stdev(diffs)
    out["mean_diff"], out["sd_diff"] = mean_d, sd_d
    try:
        from scipy import stats as sps
        half = sps.t.ppf(0.975, n - 1) * sd_d / math.sqrt(n)
        out["ci"] = (mean_d - half, mean_d + half)
        try:
            out["p_wilcoxon"] = sps.wilcoxon([a[s] for s in seeds],
                                             [b[s] for s in seeds]).pvalue
        except ValueError:
            out["p_wilcoxon"] = float("nan")
        n_eff = n - out["ties"]
        out["floor"] = (sps.wilcoxon(list(range(1, n_eff + 1)),
                                     [0] * n_eff).pvalue
                        if n_eff >= 1 else float("nan"))
    except ImportError:                                # pragma: no cover
        out.update(ci=(float("nan"), float("nan")),
                   p_wilcoxon=float("nan"), floor=float("nan"))
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_audit(valid, invalid):
    print("=" * 78)
    print("AUDIT")
    print("=" * 78)
    print(f"\n  VALID {len(valid)}    INVALID {len(invalid)}")
    if invalid:
        print("\n  invalid runs:")
        for e in invalid:
            print(f"    {e['language'][:2]}/{e['task']:3s} {e['system']:12s} "
                  f"s{e['seed']}: {'; '.join(e['reasons'])}")
    counts = Counter((e["language"], e["task"], e["system"]) for e in valid)
    print("\n  runs per cell:")
    for lang in LANGUAGES:
        print(f"    {lang}")
        for task in TASKS:
            row = "".join(f"{DISPLAY[s].replace('+LAPT','LAPT'):>13s}="
                          f"{counts[(lang, task, s)]:<3d}" for s in SYSTEMS)
            print(f"      {task:3s}{row}")
    complete = sum(1 for lang in LANGUAGES for t in TASKS for s in SYSTEMS
                   if counts[(lang, t, s)] >= 5)
    print(f"\n  cells with >= 5 seeds: {complete}/30")

    print("\n  sources, with the config checksum each cell's runs carry:")
    seen = {}
    for e in valid:
        key = (e["language"], e["system"])
        seen.setdefault(key, (e["source_dir"],
                              e["record"]["provenance"]["config_sha256"][:16]))
    for (lang, s), (d, sha) in sorted(seen.items()):
        print(f"    {lang[:2]}/{DISPLAY[s]:14s} {d:28s} {sha}")

    print("\n  excluded (protocol differences, not outcomes):")
    for name, reason in EXCLUDED_DIRS:
        print(f"    {name}\n      {reason}")


def print_tables(valid, out_dir: Path | None):
    by = defaultdict(dict)
    for e in valid:
        by[(e["language"], e["task"], e["system"])][e["seed"]] = e

    # ---- Table 2 ------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 2   five systems, three tasks, two languages")
    print("=" * 78)
    for lang in LANGUAGES:
        maj = MAJORITY[lang]
        extra = (f", macro-F1 {maj['macro_f1_5']}"
                 if maj["macro_f1_5"] is not None else "")
        print(f"\n  {lang.upper()}   majority-class baseline: "
              f"accuracy {maj['accuracy']}{extra}")
        for task in TASKS:
            primary = "macro-F1 (classes in test)" if task == "tc" else "F1"
            print(f"\n    {task.upper()}   primary: {primary}")
            head = (f"      {'system':14s}{'mean':>8s}{'sd':>7s}"
                    f"{'n':>4s}{'fail':>6s}")
            if task == "tc":
                head += f"{'acc':>9s}{'sd':>7s}{'mF1(6)':>9s}"
            print(head)
            best = max((mean_sd_n([score(v) for v in by[(lang, task, s)].values()])[0]
                        for s in SYSTEMS
                        if by[(lang, task, s)]), default=float("nan"))
            for s in SYSTEMS:
                runs = by[(lang, task, s)]
                if not runs:
                    print(f"      {DISPLAY[s]:14s}  (no runs)")
                    continue
                m, sd, n, fail = mean_sd_n([score(v) for v in runs.values()])
                line = f"      {DISPLAY[s]:14s}{m:8.2f}{sd:7.2f}{n:4d}{fail:6d}"
                if task == "tc":
                    a = mean_sd_n([score(v, "accuracy") for v in runs.values()])
                    m6 = mean_sd_n([score(v, "macro_f1_6") for v in runs.values()])
                    line += f"{a[0]:9.2f}{a[1]:7.2f}{m6[0]:9.2f}"
                print(line + ("  *" if abs(m - best) < 1e-9 else ""))
    print("\n  * = highest mean in the cell. fail = runs scoring exactly zero,")
    print("  excluded from the mean and counted here (see mean_sd_n).")
    print("  mF1(6) = the value recorded in the run files, which averages over")
    print("  a sixth class with no test instances.")

    # ---- Table 3 ------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE 3   LGSE minus each baseline, matched converged seeds")
    print("=" * 78)
    print(f"\n  {'lang':5s}{'task':5s}{'vs':14s}{'n':>3s}{'delta':>8s}"
          f"{'95% CI':>20s}{'p_wilc':>8s}{'floor':>7s}  W/L/T")
    ahead = total = 0
    excluding = []
    for lang in LANGUAGES:
        for task in TASKS:
            L = {k: score(v) for k, v in by[(lang, task, "lgse_lapt")].items()}
            for opp in BASELINES:
                O = {k: score(v) for k, v in by[(lang, task, opp)].items()}
                r = paired(L, O)
                if r["n"] < 2:
                    continue
                total += 1
                ahead += r["mean_diff"] > 0
                lo, hi = r["ci"]
                if lo * hi > 0:
                    excluding.append(f"{lang[:2]}/{task} vs {DISPLAY[opp]}")
                print(f"  {lang[:4]:5s}{task:5s}{DISPLAY[opp]:14s}{r['n']:3d}"
                      f"{r['mean_diff']:+8.2f}"
                      f"{f'[{lo:+.2f},{hi:+.2f}]':>20s}"
                      f"{r['p_wilcoxon']:8.3f}{r['floor']:7.3f}"
                      f"  {r['wins']}/{r['losses']}/{r['ties']}")
    print(f"\n  LGSE ahead in {ahead} of {total} comparisons.")
    print(f"  intervals excluding zero: {len(excluding)}"
          + (f"  {excluding}" if excluding else ""))
    print("  floor = smallest attainable two-sided exact Wilcoxon p at the")
    print("  effective n; where it exceeds 0.05 no result can be significant.")

    # ---- seed variation ----------------------------------------------
    print("\n" + "=" * 78)
    print("Seed variation against between-system spread (descriptive)")
    print("=" * 78)
    print(f"\n  {'lang':10s}{'task':5s}{'spread':>9s}{'mean sd':>10s}{'ratio':>8s}")
    for lang in LANGUAGES:
        for task in TASKS:
            means, sds = [], []
            for s in SYSTEMS:
                runs = by[(lang, task, s)]
                if not runs:
                    continue
                m, sd, _, _ = mean_sd_n([score(v) for v in runs.values()])
                means.append(m); sds.append(sd)
            if len(means) < 2:
                continue
            spread = max(means) - min(means)
            msd = st.mean(sds)
            print(f"  {lang:10s}{task:5s}{spread:9.2f}{msd:10.2f}"
                  f"{spread / msd if msd else float('nan'):8.2f}")
    print("\n  ratio below 1: changing the seed moves a system more than")
    print("  changing the system does.")

    # ---- per-seed CSV -------------------------------------------------
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "table2_per_seed.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["language", "task", "system", "seed", "primary_metric",
                        "score", "tc_accuracy", "tc_macro_f1_6class",
                        "degenerate", "source_dir", "config_sha256",
                        "commit", "result_file"])
            for lang in LANGUAGES:
                for task in TASKS:
                    for s in SYSTEMS:
                        for seed in sorted(by[(lang, task, s)]):
                            e = by[(lang, task, s)][seed]
                            v = score(e)
                            prov = e["record"]["provenance"]
                            w.writerow([
                                lang, task, DISPLAY[s], seed,
                                "macro_f1_5class" if task == "tc" else "f1",
                                f"{v:.4f}",
                                f"{score(e,'accuracy'):.4f}" if task == "tc" else "",
                                f"{score(e,'macro_f1_6'):.4f}" if task == "tc" else "",
                                int(v == 0.0), e["source_dir"],
                                prov.get("config_sha256", "")[:16],
                                prov.get("commit", "")[:12],
                                str(e["path"].relative_to(ROOT)),
                            ])
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"\nwrote {shown} ({sum(len(by[k]) for k in by)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true",
                    help="print the run audit and exit")
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for the per-seed CSV")
    args = ap.parse_args()

    valid, invalid = load_runs()
    if not valid:
        print("no valid runs found", file=sys.stderr)
        return 1
    print_audit(valid, invalid)
    if args.audit:
        return 0
    print_tables(valid, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
