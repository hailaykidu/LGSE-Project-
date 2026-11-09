#!/usr/bin/env python3
"""Apply the training-loss failure rule to every run behind the paper's tables.

The rule: a fine-tuning run has failed when its relative training-loss
reduction, first to last logged value, is below a threshold placed in the
empty interval between the failed and the healthy runs. No test score is
used to decide the exclusion.

Test scores ARE used for one thing only: to identify which log region
produced a given committed record. A run can appear in several logs
(reruns, superseded sweeps), and those attempts can have different
outcomes, so a region is accepted only when the test score it prints
matches the record to two decimals. Matching never decides an exclusion.

Scope: every one of the 165 records that build_table2.py reads. The rule is
applied per language. Tigrinya text classification is reported but not
used to exclude, because its losses are flat for every run (see the paper,
"Tigrinya text classification behaves differently").

Usage:
    python3 scripts/loss_rule.py                # print the table
    python3 scripts/loss_rule.py --write tables/loss_rule.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

BANNER = re.compile(r"=== ([a-z_]+)__([a-z]+)__(amharic|tigrinya)__seed(\d+)")
LOSS = re.compile(r"'loss': ([0-9.]+)")
# printed by the evaluation runners after fine-tuning
SCORE_LINES = [
    re.compile(r"test F1 ([0-9.]+)"),
    re.compile(r"test macro F1 ([0-9.]+)"),
    re.compile(r"test AC ([0-9.]+)"),
    re.compile(r": test (?:f1|accuracy) ([0-9.]+)"),
]

# Tigrinya TC is reported, never used to exclude (flat losses, no gap).
NOT_RULED = {("tigrinya", "tc")}


def record_scores(rec: dict) -> set[str]:
    """Two-decimal renderings of every test score a record carries."""
    out = set()
    for k, v in rec.get("test", {}).items():
        if isinstance(v, (int, float)) and ("f1" in k.lower() or "acc" in k.lower()):
            x = v * 100 if 0 < v <= 1 else v
            out.add(f"{x:.2f}")
    return out


# Every runner prints, immediately after its own fine-tuning losses, one
# summary line for that run, e.g.
#   seed 46  dev F1 0.00  test F1 0.00
#   seed 42  dev AC 87.12  test AC 83.83  (test macro F1 48.81)
# Logs differ in where the "=== system__task__lang__seedN ===" banner sits
# (before the training output in some job layouts, after it in others), so
# splitting on banners attaches a neighbouring run's curve in one layout or
# the other. The summary line is layout-independent: a run's losses are the
# ones between the previous summary line and its own.
SUMMARY = re.compile(r"seed (\d+)\s+dev \w+ [0-9.]+\s+test \w+ ([0-9.]+)"
                     r"(?:\s+\(test macro F1 ([0-9.]+)\))?")


def log_regions():
    """Yield (path, system, task, language, seed, region text)."""
    files = sorted(set(glob.glob(str(ROOT / "logs/**/*.out"), recursive=True)
                       + glob.glob(str(ROOT / "logs/**/*.err"), recursive=True)
                       + glob.glob(str(ROOT / "logs/**/*.log"), recursive=True)))
    for f in files:
        try:
            text = Path(f).read_text(errors="replace")
        except OSError:
            continue
        if "'loss':" not in text:
            continue
        banners = [(m.start(), m.group(1), m.group(2), m.group(3), int(m.group(4)))
                   for m in BANNER.finditer(text)]
        summaries = list(SUMMARY.finditer(text))
        prev_end = 0
        for j, s in enumerate(summaries):
            seed = int(s.group(1))
            region = text[prev_end:s.end()]          # this run's losses + summary
            nxt = summaries[j + 1].start() if j + 1 < len(summaries) else len(text)
            prev_end = s.end()
            # the run's banner, with the same seed, is the last one before the
            # summary (banner-first layout) or the first one after it, before
            # the next summary (banner-last layout)
            before = [b for b in banners if b[0] < s.start() and b[4] == seed]
            after = [b for b in banners if s.end() <= b[0] < nxt and b[4] == seed]
            ids = {b[1:] for b in ([before[-1]] if before else []) + after[:1]}
            for (sysn, task, lang, sd) in ids:
                # the summary line plus the runner's own "<run>: test ..." line
                tail = text[s.start():nxt]
                yield f, sysn, task, lang, sd, region + tail[:400]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="per-seed CSV from build_table2.py (default: regenerate in memory)")
    ap.add_argument("--write", default=None, help="write the selection as JSON")
    args = ap.parse_args()

    import build_table2 as bt  # the 165 records, from the same source as the paper
    records, invalid = bt.load_runs()
    if invalid:
        raise SystemExit(f"{len(invalid)} invalid records; run build_table2.py --audit")
    for r in records:
        r["system_id"], r["system"] = r["system"], bt.DISPLAY[r["system"]]

    regions = list(log_regions())
    rows, unmatched = [], []
    for r in records:
        key = (r["system_id"], r["task"], r["language"], int(r["seed"]))
        want = record_scores(r["record"])
        cands = []
        for f, *k, region in regions:
            if tuple(k) != key:
                continue
            printed = {f"{float(m.group(1)):.2f}" for p in SCORE_LINES
                       for m in p.finditer(region)}
            losses = [float(x) for x in LOSS.findall(region)]
            if printed & want and len(losses) >= 2:
                cands.append((f, losses))
        if not cands:
            unmatched.append(key)
            continue
        # A run can match more than one log (a rerun that printed the same
        # score, or a copied log). Keep every candidate; the decision below is
        # accepted only if all candidates fall on the same side of the
        # threshold, so the choice of log cannot change an exclusion.
        uniq = {}
        for f, losses in cands:
            uniq.setdefault(tuple(losses), f)
        cand_rows = [{"first": l[0], "last": l[-1],
                      "reduction": (l[0] - l[-1]) / l[0], "n_loss": len(l),
                      "log": str(Path(f).relative_to(ROOT))}
                     for l, f in uniq.items()]
        cand_rows.sort(key=lambda c: c["reduction"])
        best = cand_rows[0]
        rows.append({"language": key[2], "task": key[1], "system": r["system"],
                     "system_id": key[0], "seed": key[3], **best,
                     "candidates": cand_rows if len(cand_rows) > 1 else None})

    print(f"records: {len(records)}   matched to a log: {len(rows)}   "
          f"unmatched: {len(unmatched)}")
    for u in unmatched:
        print("  UNMATCHED", u)

    result = {"rule": "relative training-loss reduction, first to last logged value",
              "languages": {}}
    for lang in ("amharic", "tigrinya"):
        ruled = [x for x in rows if x["language"] == lang
                 and (lang, x["task"]) not in NOT_RULED]
        reds = sorted(x["reduction"] for x in ruled)
        # largest gap between consecutive values defines the two modes
        gaps = [(reds[i + 1] - reds[i], i) for i in range(len(reds) - 1)]
        g, i = max(gaps)
        lo, hi = reds[i], reds[i + 1]
        thr = (lo + hi) / 2
        failed = [x for x in ruled if x["reduction"] < thr]
        # every run with several candidate logs must be decided the same way
        # by each of them
        for x in ruled:
            if x.get("candidates"):
                sides = {c["reduction"] < thr for c in x["candidates"]}
                tag = "consistent" if len(sides) == 1 else "CONFLICT"
                print(f"  {len(x['candidates'])} candidate logs, {tag}: {x['task']} "
                      f"{x['system']} s{x['seed']}  "
                      + ", ".join(f"{100*c['reduction']:.1f}%" for c in x["candidates"]))
                if len(sides) > 1:
                    unmatched.append((x["system_id"], x["task"], lang, x["seed"], "conflict"))
        print(f"\n{lang.upper()}  ({len(ruled)} runs ruled"
              + (f"; {len([x for x in rows if x['language']==lang]) - len(ruled)}"
                 " reported only" if len(ruled) < len([x for x in rows if x['language']==lang]) else "")
              + ")")
        print(f"  failed runs:  {100*min(x['reduction'] for x in failed):.2f}% .. {100*lo:.2f}%"
              if failed else "  failed runs: none")
        print(f"  healthy runs: {100*hi:.2f}% .. {100*reds[-1]:.2f}%")
        print(f"  empty interval: ({100*lo:.2f}%, {100*hi:.2f}%)  midpoint {100*thr:.2f}%")
        for x in sorted(failed, key=lambda x: (x["task"], x["system"], x["seed"])):
            print(f"    FAILED  {x['task']:3s} {x['system']:14s} s{x['seed']}  "
                  f"{100*x['reduction']:5.1f}%  ({x['first']:.4f} -> {x['last']:.4f})  {x['log']}")
        result["languages"][lang] = {
            "ruled_runs": len(ruled), "gap_low": lo, "gap_high": hi, "threshold": thr,
            "failed": [{k: x[k] for k in ("task", "system", "seed", "reduction", "log")}
                       for x in failed]}
    tc = sorted(x["reduction"] for x in rows if (x["language"], x["task"]) in NOT_RULED)
    if tc:
        print(f"\nTIGRINYA TC (reported, not ruled): reduction {100*tc[0]:.1f}% .. {100*tc[-1]:.1f}%")
    result["runs"] = rows
    result["unmatched"] = [list(u) for u in unmatched]

    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.write}")
    return 0 if not unmatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
