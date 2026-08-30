"""AN EXCLUDED BACKTEST ANCHOR MUST CARRY A REASON THAT IS TRUE TODAY.

THE DEFECT.  `CLEAN_BUY_IDS` was a hardcoded set with a prose comment beside it justifying the
exclusion of buy2018/2019/2020 as "DEGENERATE": the scoring pickle's history cap leaving only
a non-representative long-history subset (~1,000-1,250 names) live PIT, with buy2024 named as
"the RICHEST anchor (7,863 names >=8q)".  `-nrperiods 80` lifted that cap.  On the 2026-08-29
run's own log lines every clause of it is false and two are INVERTED: no anchor is near
1,000-1,250 (the thinnest is 1,808); buy2024 has the SMALLEST scored pool of all seven; and
the stated criterion -- top-100 being too large a slice -- applied literally would clear
buy2018 (4.95%) and flag buy2024 (5.53%).  The named risk was a future reader promoting an
anchor on the written reason.

WHAT THESE TESTS CANNOT SEE, and it is the important half:
  * They check that the set and every label derived from it AGREE, and that no surviving
    reason cites the lifted cap.  They CANNOT check that the replacement reason is factually
    right.  The survivorship counts in `ANCHOR_EXCLUSION_REASONS` were taken off one snapshot
    of `delisted_registry.csv` (9,277 rows); a re-fetch moves them and nothing here notices.
    That is a real residual -- the reason can go stale exactly the way the old one did, just
    more slowly.
  * They cannot tell whether the EXCLUSION IS THE RIGHT CALL.  Whether buy2019's 34%-of-
    buy2021 death coverage disqualifies it is a judgement about how much survivorship
    flattery is tolerable, not a fact these tests can settle.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import depth_horizon_grid as dhg
import scoring_compare as sc


def test_an_anchor_cannot_be_excluded_without_a_written_reason():
    """The STRUCTURAL fix: the set is derived from the reason table, so the two cannot drift.
    A hardcoded set beside a prose comment is what let the justification go stale invisibly."""
    all_ids = {wid for wid, _b in dhg.BUY_ANCHORS}
    assert dhg.CLEAN_BUY_IDS == all_ids - set(dhg.ANCHOR_EXCLUSION_REASONS)
    for wid in all_ids:
        excluded = wid not in dhg.CLEAN_BUY_IDS
        assert bool(dhg.exclusion_reason(wid)) == excluded, wid
    for wid, why in dhg.ANCHOR_EXCLUSION_REASONS.items():
        assert len(why) > 60, f"{wid}'s reason is too thin to be a reason"


def test_no_surviving_reason_cites_the_history_cap_that_nrperiods_80_lifted():
    """The exact false claims, named so they cannot come back by copy-paste."""
    text = " ".join(dhg.ANCHOR_EXCLUSION_REASONS.values()).lower()
    for dead_claim in ("history cap", "1,000-1,250", "1000-1250", "richest",
                       "long-history subset", "degenerate"):
        assert dead_claim not in text, f"a reason still cites {dead_claim!r}"


def test_buy2020_is_graded_and_buy2018_is_not():
    """The verified conclusion, pinned so a silent revert is visible.  buy2020's holding
    window carries 3,257 registry deaths against buy2021's 4,557 -- the same order.
    buy2018's carries 658, i.e. 14%, so its losers cannot be graded as losses."""
    assert "buy2020" in dhg.CLEAN_BUY_IDS
    assert "buy2018" not in dhg.CLEAN_BUY_IDS
    assert "survivorship" in dhg.exclusion_reason("buy2018").lower()


def test_every_label_derives_from_the_set_rather_than_restating_it():
    """FOUR separate places hardcoded the clean anchors, not three as this docstring first
    claimed -- and the miscount mattered, because the one it missed was the worst.

      1. the grid report's pooled block title;
      2. `scoring_compare.SCOPE_ORDER` -- ALREADY drifted: "CLEAN anchors 2021/22/23" with no
         buy2024 row at all, weeks after buy2024 joined the set;
      3. two printed headers in `pipeline_analysis`;
      4. `skill_baseline.CLEAN_BUY_ANCHORS` -- MISSED ON THE FIRST PASS.  `skill_baseline`
         runs as a suite stage beside the beat-rate table and `pipeline_analysis` prints
         "the intended difference between the two is the carve".  With buy2020 promoted in
         `dhg` and that literal frozen, the beat-rate graded THREE windows against
         skill_baseline's TWO and the sentence became false -- a reader would have charged
         an anchor-set difference to the carve."""
    import pipeline_analysis as pa
    import skill_baseline as sb
    #  the fourth copy, and the pair that must agree window-for-window
    assert sb.CLEAN_BUY_ANCHORS == [b for w, b in dhg.BUY_ANCHORS if w in dhg.CLEAN_BUY_IDS]
    graded = {b for b, _e in sb.clean_windows(36)}
    assert graded <= set(sb.CLEAN_BUY_ANCHORS)
    scopes = {s for s, _t in sc.SCOPE_ORDER}
    assert scopes == {"POOLED-CLEAN", "POOLED-ALL"} | {w for w, _b in dhg.BUY_ANCHORS}
    for wid, title in sc.SCOPE_ORDER:
        if wid in dhg.CLEAN_BUY_IDS:
            assert "[CLEAN]" in title
        elif wid in dhg.ANCHOR_EXCLUSION_REASONS:
            assert "EXCLUDED" in title
    text = pa._clean_window_text()
    for wid, _b in dhg.BUY_ANCHORS:
        graded = wid in dhg.CLEAN_BUY_IDS and wid in text
        if wid not in dhg.CLEAN_BUY_IDS:
            assert wid not in text, f"{wid} is excluded but named as a graded window"
    assert "buy2020" in text, "the promoted anchor is missing from the printed window list"


def test_the_grid_report_prints_the_reason_where_a_reader_will_see_it():
    """A justification that lives only in a source comment is one no reader of the output
    ever checks -- which is how it survived being false for months."""
    per_anchor = {wid: {"buy": buy, "ranking": [], "rank_depth": 100, "universe_size": 3000,
                        "n_pit_scored": 2000, "n_pit_live": 900, "n_pit_dead": 1100,
                        "basis": "un-vetoed"}
                  for wid, buy in dhg.BUY_ANCHORS}
    cell = {"scope": "buy2021", "clean": True, "buy": "2021-12-31", "eval": "2024-12-31",
            "horizon_m": 36, "depth_N": 20, "n_requested": 20, "n_available_rank": 100,
            "n_included": 20, "n_missing_buy": 0, "n_affected_eval": 0,
            "avg_ret_primary": 0.1, "avg_ret_floor": 0.1, "bench_ret": 0.05,
            "excess_primary": 0.05, "excess_floor": 0.05}
    text = dhg.build_report(per_anchor, [cell], [cell], [cell])
    assert "EXCLUDED anchors and WHY" in text
    for wid in dhg.ANCHOR_EXCLUSION_REASONS:
        assert wid in text
    assert "survivorship" in text
    #  and the stale paragraph is gone
    assert "1,000-1,250" not in text
    assert "RICHEST" not in text


def test_the_pooled_clean_block_title_names_the_pool_it_sits_above():
    per_anchor = {wid: {"buy": buy, "ranking": [], "rank_depth": 100, "universe_size": 3000,
                        "n_pit_scored": 2000, "n_pit_live": 900, "n_pit_dead": 1100,
                        "basis": "un-vetoed"}
                  for wid, buy in dhg.BUY_ANCHORS}
    cell = {"scope": "POOLED-CLEAN", "clean": True, "buy": "2021-12-31", "eval": "2024-12-31",
            "horizon_m": 36, "depth_N": 20, "n_requested": 20, "n_available_rank": 100,
            "n_included": 20, "n_missing_buy": 0, "n_affected_eval": 0,
            "avg_ret_primary": 0.1, "avg_ret_floor": 0.1, "bench_ret": 0.05,
            "excess_primary": 0.05, "excess_floor": 0.05}
    text = dhg.build_report(per_anchor, [cell], [cell], [cell])
    title = [l for l in text.splitlines() if "POOLED -- CLEAN anchors only" in l]
    assert len(title) == 1
    for wid, _b in dhg.BUY_ANCHORS:
        assert (wid in title[0]) == (wid in dhg.CLEAN_BUY_IDS), wid


def test_the_two_paired_stages_grade_THE_SAME_WINDOWS():
    """`pipeline_analysis` prints that the only intended difference between the beat-rate
    table and skill_baseline is the carve.  If their anchor sets differ, that sentence is
    false and a reader charges an anchor-set difference to the carve."""
    import skill_baseline as sb
    import returns_core as rc
    anchors = list(rc.DEFAULT_ANCHORS)
    idx = {a: i for i, a in enumerate(anchors)}
    beat_rate_windows = set()
    for wid, buy in dhg.BUY_ANCHORS:
        if wid not in dhg.CLEAN_BUY_IDS or buy not in idx:
            continue
        j = idx[buy] + 3
        if j < len(anchors):
            beat_rate_windows.add((buy, anchors[j]))
    assert set(sb.clean_windows(36, anchors)) == beat_rate_windows


def test_the_RENDERED_report_carries_no_retired_reason():
    """THE GUARD USED TO SCAN ONLY `ANCHOR_EXCLUSION_REASONS.values()`, so the retired reason
    survived in six rendered places -- including `* DEGENERATE pre-2021 windows: depth cut
    not meaningful` VERBATIM in the grid report's CAVEATS block, which is the part a reader
    keeps.  Scanning the source of truth while the OUTPUT still lies is the exact shape of
    the defect Q-28 was chartered to kill.

    WHAT THIS CANNOT SEE: it renders the grid report and the scoring_compare block.  Text
    printed by `pipeline_analysis` at run time is not rendered here, and a retired phrase
    added to a THIRD report is uncovered."""
    per_anchor = {wid: {"buy": buy, "ranking": [], "rank_depth": 100, "universe_size": 3000,
                        "n_pit_scored": 2000, "n_pit_live": 900, "n_pit_dead": 1100,
                        "basis": "un-vetoed"}
                  for wid, buy in dhg.BUY_ANCHORS}
    cell = {"scope": "POOLED-CLEAN", "clean": True, "buy": "2021-12-31",
            "eval": "2024-12-31", "horizon_m": 36, "depth_N": 20, "n_requested": 20,
            "n_available_rank": 100, "n_included": 20, "n_missing_buy": 0,
            "n_affected_eval": 0, "avg_ret_primary": 0.1, "avg_ret_floor": 0.1,
            "bench_ret": 0.05, "excess_primary": 0.05, "excess_floor": 0.05}
    rendered = [dhg.build_report(per_anchor, [cell], [cell], [cell])]

    import tempfile
    import pandas as pd
    with tempfile.TemporaryDirectory() as w:
        rows = [{"config": "BASELINE", "basis": "un-vetoed", "scope": sc_, "horizon_m": 36,
                 "depth_N": n, "avg_ret_primary": 0.1, "excess_primary": 0.05,
                 "bench_ret": 0.05}
                for sc_ in ("POOLED-CLEAN", "buy2021") for n in (1, 20)]
        pd.DataFrame(rows).to_csv(os.path.join(w, "cells_BASELINE.csv"), index=False)
        out = os.path.join(w, "cmp.out")
        sc.format_comparison(w, out, None, log=lambda *a: None)
        rendered.append(open(out, encoding="utf-8").read())

    for text in rendered:
        low = text.lower()
        for retired in ("degenerate", "history cap", "1,000-1,250", "richest",
                        "non-representative long-history"):
            assert retired not in low, f"the rendered report still says {retired!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
