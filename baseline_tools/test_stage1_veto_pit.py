"""The Stage-1 solvency veto in the PIT backtest: that it runs at all, and that it is
POINT-IN-TIME.

WHY THIS FILE EXISTS.  Until 2026-08-27 the backtest never applied the veto.  `reproduce_pit_top`
went Stage-1 -> Stage-2 with no veto call, and NOTHING under `baseline_tools/` referenced
`stage1_veto` at all -- the gate was wired only in `postBo.py`.  So every PIT number this project
produced (beat-rate, loss diagnostics, depth x horizon grid, the four-cell measurement) graded a
filter WITHOUT the CEO's gates: the veto could have been tightened or loosened and every figure
would have been identical.

THE RISK THE FIX CREATES is a look-ahead.  The veto reads solvency flags over a rolling window,
and `stage1_veto._evaluate` simply sorts newest-first and takes `head(WINDOW_ROWS)` of WHATEVER
FRAME IT IS HANDED -- it has no as-of parameter and no independent data source.  So PIT-safety is
entirely a property of the caller, and handing it the full-history panel would be the
`dollarvol_floor` look-ahead again: a present-day fact applied to a historical pool.  These tests
pin both halves -- that the call site passes the `date <= D` frames, and that a planted FUTURE row
cannot move a verdict.
"""
import ast
import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage1_veto as sv
import stage2_pit as s2
import depth_horizon_grid as dhg


# --------------------------------------------------------------------------- #
#  1. STRUCTURAL -- the veto runs, on the PIT frames, before the head() cut    #
# --------------------------------------------------------------------------- #
def _veto_call():
    """The `apply_veto` Call node inside `reproduce_pit_top`, or None."""
    tree = ast.parse(inspect.getsource(s2.reproduce_pit_top))
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "apply_veto"):
            return n, tree
    return None, tree


def test_the_backtest_path_CALLS_the_veto_at_all():
    """THE DEFECT.  `reproduce_pit_top` had no veto call and no module under baseline_tools/
    referenced `stage1_veto`, so the backtest graded an ungated filter."""
    node, _ = _veto_call()
    assert node is not None, "reproduce_pit_top does not apply the Stage-1 veto"


def test_the_veto_is_handed_the_PIT_frames_and_NOT_the_full_history():
    """THE LOOK-AHEAD GUARD, checked on the AST rather than trusted.

    `_evaluate` takes `head(WINDOW_ROWS)` of whatever frame it gets.  `bm_pit`/`cdx_pit` are
    the `date <= D` slices; `bm`/`cdx` are full history.  Passing the latter would let the
    veto read quarters that had not been filed at the buy anchor -- the same class as the
    `dollarvol_floor` look-ahead, which is exactly how that one shipped.
    """
    node, _ = _veto_call()
    assert node is not None
    panel_arg = node.args[1]
    assert isinstance(panel_arg, ast.Name) and panel_arg.id == "bm_pit", (
        "the veto is handed %r, not the date<=D panel `bm_pit`"
        % ast.unparse(panel_arg))
    kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
    assert kw.get("cdx_df") == "cdx_pit", (
        "the corroborator frame is %r, not the date<=D `cdx_pit`" % kw.get("cdx_df"))


def test_the_veto_runs_BEFORE_the_top100_cut_like_production():
    """Production vetoes `general_scores` (postBo.py:567) and cuts `head(100)` at 686, so an
    ejected name lets the next name PROMOTE.  Vetoing after the cut would instead leave a hole
    and silently shrink the Stage-2 pool."""
    node, tree = _veto_call()
    cut = [n for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "head"
           and any("topn_stage1" in ast.unparse(a) for a in n.args)]
    assert cut, "could not find the head(topn_stage1) cut"
    assert node.lineno < min(c.lineno for c in cut), (
        "the veto runs AFTER the top-100 cut; production vetoes before it")


def test_the_default_keeps_every_existing_caller_bit_for_bit():
    """`tune_run.validate_finish` ABORTS the tuner when its cached fast finish() diverges from
    `reproduce_pit_top`, and the fast path does not veto.  So the default must stay OFF here;
    the BACKTEST turns it on explicitly.  Flipping this default is not a small change."""
    sig = inspect.signature(s2.reproduce_pit_top)
    assert sig.parameters["apply_stage1_veto"].default is False


def test_the_BACKTEST_turns_the_veto_ON_by_default():
    """The instrument meant to judge gates must actually run them."""
    assert inspect.signature(dhg.rank_all_anchors).parameters["stage1_veto"].default is True
    assert inspect.signature(dhg.run_in_pipeline).parameters["stage1_veto"].default is True


def test_the_unvetoed_basis_stays_reproducible():
    """Every historical figure was un-vetoed.  If that basis could not be reproduced, no
    published number could ever be checked again."""
    src = inspect.getsource(dhg.rank_all_anchors)
    assert "apply_stage1_veto=stage1_veto" in src


# --------------------------------------------------------------------------- #
#  2. BEHAVIOURAL -- the PIT truncation actually blinds the veto               #
# --------------------------------------------------------------------------- #
FLAGS = sv.pool_flags("general")
D = pd.Timestamp("2022-12-30")


def _panel(source, rows, extra_future=0):
    """A per-source panel of `rows` quarters ending at D, newest-first.

    `extra_future` adds quarters AFTER D whose values would flip every flag to a clean pass.
    A PIT-correct caller must be unable to see them.
    """
    recs = []
    for i in range(rows):
        d = D - pd.DateOffset(months=3 * i)
        recs.append({"source": source, "date": d,
                     # every flag FAILS: all are `> 0` / `> 1` tests
                     "returnOnAssets": -0.05, "CFOlessEarnings": -1e6,
                     "uCurrentRatio": 0.5, "netDebtToEBITDA": -2.0,
                     "uInterestCoverage": 0.4})
    for i in range(1, extra_future + 1):
        d = D + pd.DateOffset(months=3 * i)
        recs.append({"source": source, "date": d,
                     "returnOnAssets": 0.20, "CFOlessEarnings": 1e6,
                     "uCurrentRatio": 5.0, "netDebtToEBITDA": 5.0,
                     "uInterestCoverage": 9.0})
    return pd.DataFrame(recs)


def test_a_healthy_name_is_not_ejected_and_a_sick_one_is():
    """Baseline sanity, so the look-ahead test below cannot pass vacuously."""
    sick = _panel("SICK", 8)
    healthy = sick.copy()
    healthy["source"] = "OK"
    for c, v in (("returnOnAssets", 0.2), ("CFOlessEarnings", 1e6),
                 ("uCurrentRatio", 5.0), ("netDebtToEBITDA", 5.0),
                 ("uInterestCoverage", 9.0)):
        healthy[c] = v
    panel = pd.concat([sick, healthy], ignore_index=True)
    scores = pd.DataFrame({"source": ["SICK", "OK"], "score": [2.0, 1.0]})
    kept, rep = sv.apply_veto(scores, panel, pool_label="general", enabled=True)
    assert rep["applies"] and rep["enabled"]
    assert set(rep["ejected"]) == {"SICK"}
    assert kept["source"].tolist() == ["OK"]


def test_FUTURE_rows_cannot_change_the_verdict_when_the_panel_is_PIT_TRUNCATED():
    """THE PROOF that the wiring is point-in-time.

    The same name is vetoed twice: once on a panel that stops at D, once on a panel that also
    carries four post-D quarters good enough to clear every flag.  A caller that truncates to
    `date <= D` -- which is what `bm_pit` is -- must reach the SAME verdict both times.  If
    this ever fails, the backtest is grading history with information from the future.
    """
    without_future = _panel("SICK", 8)
    with_future = _panel("SICK", 8, extra_future=4)
    scores = pd.DataFrame({"source": ["SICK"], "score": [1.0]})

    a = sv.apply_veto(scores, without_future[without_future["date"] <= D],
                      pool_label="general", enabled=True)[1]
    b = sv.apply_veto(scores, with_future[with_future["date"] <= D],
                      pool_label="general", enabled=True)[1]
    assert a["ejected"] == b["ejected"] == ["SICK"]
    assert a["by_flag"] == b["by_flag"]


def test_the_SAME_future_rows_DO_change_the_verdict_without_the_truncation():
    """The other half, and the reason the test above is not vacuous: those planted rows are
    genuinely verdict-changing.  Handed the untruncated panel the veto clears the name -- so
    the truncation is doing real work, not describing a no-op."""
    with_future = _panel("SICK", 8, extra_future=4)
    scores = pd.DataFrame({"source": ["SICK"], "score": [1.0]})
    leaky = sv.apply_veto(scores, with_future, pool_label="general", enabled=True)[1]
    assert leaky["ejected"] == [], (
        "the planted future rows do not change the verdict, so the PIT test above proves "
        "nothing -- strengthen the fixture")


def test_the_window_is_the_EIGHT_NEWEST_ROWS_AVAILABLE_AT_D():
    """The mechanism behind the two tests above, stated directly: `_evaluate` sorts
    newest-first and takes head(WINDOW_ROWS) of the frame it is given.  Nothing else."""
    deep = _panel("SICK", 20)
    pit = deep[deep["date"] <= D].sort_values("date", ascending=False)
    assert len(pit) == 20
    assert pit["date"].iloc[0] == D
    assert sv.WINDOW_ROWS == 8


# --------------------------------------------------------------------------- #
#  3. A DECLINED VETO IS NOT A CLEAN ONE                                       #
# --------------------------------------------------------------------------- #
def test_a_panel_missing_veto_columns_DECLINES_and_says_so():
    """THE TRAP THIS PROJECT IS SITTING IN.  The saved 2026-07-17 panel -- the one every
    existing PIT number was computed on -- lacks `netDebtToEBITDA` and `uInterestCoverage`, two
    of the five general-pool flags.  `apply_veto` then declines to gate that pool.

    "Declined to gate" and "gated and found nothing" are completely different statements and
    must never surface as the same number: the first means the measurement did not happen.
    """
    panel = _panel("SICK", 8).drop(columns=["netDebtToEBITDA", "uInterestCoverage"])
    scores = pd.DataFrame({"source": ["SICK"], "score": [1.0]})
    kept, rep = sv.apply_veto(scores, panel, pool_label="general", enabled=True)
    assert rep["applies"] is False
    assert set(rep["missing_columns"]) == {"netDebtToEBITDA", "uInterestCoverage"}
    assert rep["n_ejected"] == 0
    assert kept is scores          # unchanged, not merely equal


def test_the_basis_string_distinguishes_declined_from_clean():
    """The stamp the CEO reads.  A declined veto must not print as a vetoed basis."""
    import pipeline_analysis as pa
    declined = {"w": {"basis": "un-vetoed (veto DECLINED: panel missing netDebtToEBITDA)"}}
    gated = {"w": {"basis": "VETOED (stage-1 solvency gate applied, 3 ejected)"}}
    assert "DECLINED" in pa._basis_of(declined)
    assert pa._basis_of(gated).startswith("VETOED")
    #  and a disagreement across anchors is never silently collapsed
    mixed = {"a": declined["w"], "b": gated["w"]}
    assert pa._basis_of(mixed).startswith("MIXED")


def test_an_unstamped_ranking_reads_as_unvetoed_not_as_unknown_good():
    """A per_anchor built by an older code path carries no basis.  The safe reading is the one
    that was true for every historical figure: un-vetoed."""
    import pipeline_analysis as pa
    assert "un-vetoed" in pa._basis_of({"w": {"ranking": []}})


# --------------------------------------------------------------------------- #
#  4. ACCEPTANCE -- the live TRTN case, on real saved data                     #
#                                                                             #
#  From the 14-name post-mortem: at buy2022 TRTN fails `netDebtToEBITDA` 0 of  #
#  8 and is the ONLY one of 14 delisted picks the gate would have removed.     #
#  It is also arguably the healthiest of them -- a container lessor whose      #
#  leverage IS its industry's capital structure -- so the first thing wiring   #
#  the veto shows is a name being ejected that arguably should not be.         #
#                                                                             #
#  SKIPPED, NOT WEAKENED, when the data is absent: the panel must carry all    #
#  five flag columns, and only the CUR3K 2026-08-11 pickle does.               #
# --------------------------------------------------------------------------- #
_CUR3K = os.path.join(
    _REPO, "Boresults_dic-fmp_stock_CUR3K_all_2026-08-11_len2624_manelim0_fails433.pickle")
_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive\delisted_out"
_REG = os.path.join(_HOME, "delisted_registry.csv")


def _dead_pickle():
    if not os.path.isdir(_HOME):
        return None
    for f in sorted(os.listdir(_HOME)):
        if f.startswith("dead_fundamentals") and f.endswith(".pickle"):
            return os.path.join(_HOME, f)
    return None


@pytest.fixture(scope="module")
def trtn_pit_panel():
    """The `date <= buy2022` panel with TRTN merged in -- built ONCE, and for TRTN ALONE.

    A first draft merged all 7,194 dead entities in EACH of the two tests below and cost
    10m54s of a ~29-minute suite for information neither test used.  A second draft merged
    `entities=['TRTN']` alone -- and `dead_merge` correctly REFUSED it: with one dead name the
    Stage-1 column `dSalesToInventory` ends up populated on one side of the merge only, which
    is its content-mismatch guard doing its job on a too-thin fixture.

    TRTN + 40 COMPANIONS satisfies the guard in ~2.4s, and 40 is the slice
    `test_dead_merge.test_real_override_none_top20_bit_identical` already uses.  Note what is
    NOT done here: `ALLOW_MERGE_CONTENT_MISMATCH=1` would also have made this pass, and it is
    the wrong move -- the guard's own message calls that "a known-invalid basis", and silencing
    a correctness guard to green a test is the failure this whole change set exists to correct.
    If a future snapshot refuses even with companions, this SKIPS loudly rather than forcing.
    """
    import dead_merge as dm
    res = pd.read_pickle(_CUR3K)
    missing_live = sv.missing_columns(res["BoMetric_df"], FLAGS)
    if missing_live:
        pytest.skip("panel predates the veto columns: %s" % missing_live)
    dead = pd.read_pickle(_dead_pickle())
    if "TRTN" not in dead:
        pytest.skip("TRTN absent from the delisted fundamentals snapshot")
    companions = [k for k in dead if k != "TRTN"][:40]
    try:
        merged, _stats = dm.merge_dead_into_dmdic(
            res, dead, dm.load_registry(_REG), as_of="2021-12-31",
            entities=["TRTN"] + companions)
    except SystemExit as e:
        #  dead_merge refuses via SystemExit.  Skipping is the honest response; forcing it
        #  with ALLOW_MERGE_CONTENT_MISMATCH would score a panel the repo says is invalid.
        pytest.skip("dead_merge refused this snapshot: %s" % str(e)[:160])
    bm = merged["BoMetric_df"].copy()
    bm["date"] = pd.to_datetime(bm["date"], errors="coerce")
    if "TRTN" not in set(bm["source"]):
        pytest.skip("TRTN absent from the merged panel on this data snapshot")
    return bm[bm["date"] <= D]


@pytest.mark.skipif(not (os.path.exists(_CUR3K) and os.path.exists(_REG)
                         and _dead_pickle()),
                    reason="real panel / delisted registry not present (dev-only)")
def test_ACCEPTANCE_TRTN_is_ejected_at_buy2022(trtn_pit_panel):
    """The live acceptance case.  If this does not reproduce, the wiring is wrong."""
    #  EXACTLY what the wired call does: the date<=D panel, general pool.
    bm_pit = trtn_pit_panel
    scores = pd.DataFrame({"source": sorted(set(bm_pit["source"])), "score": 1.0})
    kept, rep = sv.apply_veto(scores, bm_pit, pool_label="general", enabled=True)

    assert rep["applies"], rep.get("not_applicable_reason")
    assert "TRTN" in set(rep["ejected"]), (
        "TRTN is NOT ejected at buy2022 -- the wiring does not reproduce the post-mortem")
    assert "TRTN" not in set(kept["source"])
    #  and on the flag the post-mortem named, not some other one
    assert "netDebtToEBITDA" in sv.failed_flags(
        bm_pit[bm_pit["source"] == "TRTN"], FLAGS)["TRTN"]


@pytest.mark.skipif(not (os.path.exists(_CUR3K) and os.path.exists(_REG)
                         and _dead_pickle()),
                    reason="real panel / delisted registry not present (dev-only)")
def test_ACCEPTANCE_TRTN_leverage_is_the_MARGIN_not_the_raw_ratio(trtn_pit_panel):
    """Why TRTN fails, recorded so the ejection is not mistaken for a data error.

    The panel column is NOT net-debt/EBITDA: `calcMetrics.net_debt_three_branch` stores
    `NET_DEBT_TO_EBITDA_BAR - annualised_ratio`, a MARGIN that passes when `> 0`.  TRTN's
    ~-1.9 therefore means ~4.9x annualised (~19.7x on quarterly EBITDA, which is the "~20x"
    the post-mortem quotes) against a flat 3.0x bar.  Same fact, three normalisations -- and a
    reader who assumes the column is the raw ratio will read -1.9 as net cash and conclude the
    gate is broken when it is behaving exactly as designed.
    """
    import calcMetrics as cm
    import reporting_period as rp
    bm = trtn_pit_panel
    win = (bm[bm["source"] == "TRTN"]
           .sort_values("date", ascending=False).head(sv.WINDOW_ROWS))
    margins = pd.to_numeric(win["netDebtToEBITDA"], errors="coerce")

    assert (margins < 0).all(), "premise: TRTN fails the leverage flag in every window row"
    assert int((margins > 0).sum()) <= sv.FAIL_MAX_PASSES
    annualised = cm.NET_DEBT_TO_EBITDA_BAR - margins
    assert (annualised > cm.NET_DEBT_TO_EBITDA_BAR).all()
    factor = rp.stage1_flow_factor("netDebtToEBITDA", 4)
    raw = annualised / factor
    assert 15.0 < float(raw.median()) < 30.0, float(raw.median())


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
