"""Regression tests for scripts/build_paper_tables.py log parsing.

Three bugs in the training-log parser each produced a wrong failure
classification while looking perfectly healthy in the output. They are
pinned here because every one of them silently substituted one run's
training curve for another's, and the failure rule is derived from those
curves.

  1. Double banner. The five-seed jobs print two banners per seed -- the
     sbatch echo and the runner's own. Splitting on the FIRST banner and
     cutting at the next banner match yields an empty region, after which
     the parser fell through to a neighbouring seed's losses.

  2. Log shadowing. Superseded jobs from a different protocol wrote the
     same `=== system__task__amharic__seedNN` banners into their own log
     files. A filename glob picked those up in sort order, so a current
     run's curve was read from an old-protocol log. Reported LGSE NER
     seed 45 as a 97.6% loss reduction when the true value is 21.7%.

  3. Trailing banner. Single-seed jobs print the runner's banner AFTER
     training, so the text following the last banner contains no losses at
     all and the run looked unparseable.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_paper_tables.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bpt", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bpt():
    return load_module()


def write_log(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_double_banner_does_not_read_a_neighbouring_seed(bpt, tmp_path, monkeypatch):
    """Bug 1: two banners per seed must not collapse the region to nothing.

    seed 45 fails to train (0.94 -> 0.74) while seed 46 trains normally
    (0.90 -> 0.03). A parser that mishandles the duplicated banner reports
    seed 46's reduction for seed 45 and misclassifies a failed run as
    healthy.
    """
    log = write_log(tmp_path, "wam_ner_lgse_lapt.out", """
=== lgse_lapt__ner__amharic__seed45 (hornmorpho, W_am) ===
=== lgse_lapt__ner__amharic__seed45 ===
{'loss': 0.9433, 'epoch': 1.0}
{'loss': 0.7387, 'epoch': 10.0}
=== lgse_lapt__ner__amharic__seed46 (hornmorpho, W_am) ===
=== lgse_lapt__ner__amharic__seed46 ===
{'loss': 0.9060, 'epoch': 1.0}
{'loss': 0.0339, 'epoch': 10.0}
""")
    monkeypatch.setattr(bpt, "_LOG_INDEX",
                        {"wam_ner_lgse_lapt": [("81558", log)]})

    s45 = bpt.train_curve("ner", "lgse_lapt", 45)
    s46 = bpt.train_curve("ner", "lgse_lapt", 46)
    assert s45 is not None and s46 is not None
    assert s45["first"] == pytest.approx(0.9433)
    assert s45["last"] == pytest.approx(0.7387)
    assert s45["reduction"] == pytest.approx(0.217, abs=0.002)
    assert s46["reduction"] == pytest.approx(0.963, abs=0.002)


def test_trailing_banner_log_is_still_parsed(bpt, tmp_path, monkeypatch):
    """Bug 3: single-seed jobs print the banner after training."""
    log = write_log(tmp_path, "w_tc_xlmr_s42.out", """
  guards OK: rebuilt split, seed not yet done
=== xlmr__tc__amharic__seed42 (rebuilt split) ===
{'loss': 1.0535, 'epoch': 1.0}
{'loss': 0.8224, 'epoch': 10.0}
=== xlmr__tc__amharic__seed42 ===
xlmr__tc__amharic__seed42: test accuracy 70.64
""")
    monkeypatch.setattr(bpt, "_LOG_INDEX", {"w_tc_xlmr_s42": [("95401", log)]})

    curve = bpt.train_curve("tc", "xlmr", 42)
    assert curve is not None, "trailing banner must not hide the curve"
    assert curve["reduction"] == pytest.approx(0.219, abs=0.002)


def test_job_id_mapping_prefers_the_log_with_a_curve(bpt, tmp_path, monkeypatch):
    """Bug 2: an old-protocol log must not shadow the current run.

    Both logs carry the same banner. Only the job the scheduler associates
    with this run may be read; a name- or sort-order match would take the
    superseded curve.
    """
    stale = write_log(tmp_path, "rerun2_amharic_tokenlist_ner.out", """
=== lgse_lapt__ner__amharic__seed45 ===
{'loss': 0.8000, 'epoch': 1.0}
{'loss': 0.0190, 'epoch': 10.0}
""")
    current = write_log(tmp_path, "w_ner_lgse_lapt_s45.out", """
=== lgse_lapt__ner__amharic__seed45 ===
{'loss': 0.9433, 'epoch': 1.0}
{'loss': 0.7387, 'epoch': 10.0}
""")
    assert stale.exists()
    # the index holds only the scheduler's own association
    monkeypatch.setattr(bpt, "_LOG_INDEX",
                        {"w_ner_lgse_lapt_s45": [("99999", current)]})

    curve = bpt.train_curve("ner", "lgse_lapt", 45)
    assert curve["reduction"] == pytest.approx(0.217, abs=0.002), \
        "read the superseded log instead of the scheduled job's own"
    assert curve["job_id"] == "99999"


def test_newest_job_without_a_curve_falls_through(bpt, tmp_path, monkeypatch):
    """A re-submitted job whose guard aborted leaves a log with no losses.

    The parser must fall through to the job that actually trained rather
    than returning None.
    """
    aborted = write_log(tmp_path, "aborted.out", """
=== xlmr__tc__amharic__seed42 (rebuilt split) ===
  guards OK
already complete; not redoing
""")
    real = write_log(tmp_path, "real.out", """
=== xlmr__tc__amharic__seed42 (rebuilt split) ===
{'loss': 1.0535, 'epoch': 1.0}
{'loss': 0.8224, 'epoch': 10.0}
""")
    monkeypatch.setattr(bpt, "_LOG_INDEX",
                        {"w_tc_xlmr_s42": [("99999", aborted), ("83032", real)]})

    curve = bpt.train_curve("tc", "xlmr", 42)
    assert curve is not None
    assert curve["job_id"] == "83032"


def test_failure_rule_selects_the_runs_the_outcomes_flag(bpt):
    """End-to-end: on the real records, the loss-based rule and the
    outcome-side markers must agree.

    The rule is computed from training logs only. That it recovers exactly
    the runs whose test scores are degenerate is the evidence that it
    identifies optimization failure rather than fitting the outcome.
    """
    if not (ROOT / "results_hornmorpho_wam").exists():
        pytest.skip("protocol directory not present")
    valid, _ = bpt.load_runs()
    if not valid:
        pytest.skip("no runs available")
    fa = bpt.failure_analysis(valid)
    if not fa["gap_clear"]:
        pytest.skip("no clear gap in this data")

    by_rule = {(r["task"], r["system"], r["seed"]) for r in fa["selected"]}
    by_outcome = {(e["task"], e["system"], e["seed"]) for e in valid
                  if e["task"] != "qa" and bpt.observed_failure(e)}
    assert by_rule == by_outcome, (
        f"rule selected {sorted(by_rule)}, outcomes flag {sorted(by_outcome)}")


def test_every_run_has_a_locatable_training_curve(bpt):
    """No valid run may be left without a curve: a missing curve silently
    drops that run from the failure analysis."""
    if not (ROOT / "results_hornmorpho_wam").exists():
        pytest.skip("protocol directory not present")
    valid, _ = bpt.load_runs()
    if not valid:
        pytest.skip("no runs available")
    missing = [(e["task"], e["system"], e["seed"]) for e in valid
               if e["task"] != "qa"
               and bpt.train_curve(e["task"], e["system"], e["seed"]) is None]
    assert not missing, f"no training curve found for {missing}"
