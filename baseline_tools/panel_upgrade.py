"""PANEL UPGRADE -- put a SAVED (pre-fix) panel onto TODAY's metric basis, offline.

WHY THIS EXISTS
---------------
The 2026-07-17 panel (`Boresults_dic-...2026-07-17...pickle`) was fetched BEFORE the
2026-07-19..07-27 correctness arc, so it carries the OLD basis and today's code REFUSES to
score it, in two places and for good reasons:

  * `run_target_test.assert_panel_basis` refuses a beat-rate on it -- `marketCap /
    (price * shares)` is ~4.0 on that panel, i.e. `price` is the old quarterly-PE
    derivation at ~1/4 of the real share price, and `grahamNumber` is FMP's quarterly one;
  * `calcScore.simpleScore_fromDict` refuses it outright -- its `BoMetric_df` has no
    `CFOlessEarnings` column, the Tier-S criterion that replaced `uIncomeQuality`.

Both refusals say the same thing: THE SAVED DERIVED FRAMES ARE STALE.  But the panel's
`cdx_df` holds the raw statement quantities every corrected metric is a function of, so the
corrections are RECOMPUTABLE without a re-fetch -- which is what this module does.  It is
NOT a re-implementation: it drives the SAME production functions the live fetch drives
(`getData_fmp.stamp_frequency_and_graham`, `getData_fmp.build_bometric_rows`,
`calcMetrics`, `createDicts`, `reporting_period`, `getData_gen.fixAfterGetData`), which is
why those two were extracted from the fetch loop rather than copied here.

NO network I/O.  Pure function of the saved panel.

WHAT IT UPGRADES  (everything else follows downstream for free)
--------------------------------------------------------------
  1. `price`  ->  marketCap / weightedAverageShsOut   (audit C-2 fix, 2026-07-19)
  2. `reportingFrequency` stamp + in-pipeline `grahamNumber` + `grahamUndefinedReason`
     (review H2 fix, 2026-07-25) -- via the live `stamp_frequency_and_graham`
  3. `BoMetric_df` rebuilt from the upgraded `cdx_df` -- which is what delivers
     `CFOlessEarnings`, the `dAssetsToLongTermLiabilities` sign basis, the per-source
     `rpy` flow-scale corrections, the rpy-aware (not fixed-4) history trim, and the
     Stage-1 window revert, because all of those live in the dicts/calc functions this
     rebuild calls.
Stage-2 metrics, the forensics, Altman/Piotroski/Beneish/Montier/Sloan, the moat window and
the winsorizer are NOT touched here: they are computed downstream from `cdx_df` at scoring
time, so they are corrected by construction once the panel is upgraded.

FOUR FIDELITY GAPS.  The first two are what the SHIPPED code already does on a saved panel.
The last two are introduced BY THIS MODULE and were found by review (2026-07-27); they are
recorded here because a rebuilt panel is not the ingest's panel and any comparison across
the rebuild boundary carries them:

  C. TIE ORDER IS INVERTED for duplicate snapped quarters.  `_groups_newest_first` reverses
     the saved row order, and the saved order came from `data_quality.py:235`'s multi-column
     STABLE lexsort -- so reversing it inverts each tie pair rather than restoring the
     ingest's order.  MEASURED: rebuilding the 282 duplicate-quarter sources under both tie
     orders differs on 1,639 of 5,501 rows = 0.92% of the panel, concentrated in
     `dEPS` / `dPbRatio` / `dReturnOn*`.  UNFIXABLE on this panel: the tiebreakers the
     ingest had (`periodEndDate`, `period`) are both absent from it.  This is the mechanism
     behind the ~0.4% residual disagreement in verify_against_saved.
  D. THE SAVED `cdx_df` IS THE DATA-QUALITY-PRUNED FRAME.  `apply_data_quality_filter`
     removed 3,522 rows across 551 sources on this exact run, so the rebuild's rolling and
     diff windows straddle holes the ingest never saw.  The ingest computed Stage-1 BEFORE
     pruning; the rebuild necessarily computes it after.
  => CONSEQUENCE FOR EXPERIMENTS: never contrast a REBUILT-panel arm against an
     ORIGINAL-panel arm and attribute the difference to metric definitions alone.  Hold the
     panel fixed and vary only the code (baseline_tools/attribution_arms.py arm A4 does
     this).

TWO DOCUMENTED APPROXIMATIONS (both are what the SHIPPED code already does on a saved
panel -- they are not inconsistencies introduced here):
  A. `period` is ABSENT from the saved panel, so the reporting-frequency classification
     falls back to date CADENCE.  Worse than the live path in one specific way: at ingest
     the cadence is read off RAW period-end dates, whereas a saved panel only has dates
     already SNAPPED to quarter starts, and snapping can make a semi-annual filer look
     quarterly.  Measured on 07-17: 1,118 / 7,752 = 14.42% semi-annual, which is the same
     base rate the shipped run reports.
  B. `reportedCurrency` is ABSENT, so any USD market-cap view degrades to suffix FX with
     IOB lines excluded (carveOut's documented fallback).  Not used by this module; noted
     because anything banding the upgraded panel inherits it.

THE PANEL IS STILL THE OLD GATES' UNIVERSE.  Upgrading the metrics does NOT widen the
universe: the ~523 pricefails, the ~72% non-US skew and the lenfail 16->8 cohort were never
FETCHED, so no offline recompute can add them.  Label every figure from an upgraded panel
`on the 07-17 universe`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import createDicts as cdic
import getData_gen as gdg
import getData_fmp as gdf
import reporting_period as rp
import utils as utils


# --------------------------------------------------------------------------- #
#  Row-order contract                                                         #
# --------------------------------------------------------------------------- #
# The SAVED cdx_df is ASCENDING per source (data_quality.py:235 sorts it) while the SAVED
# BoMetric_df is DESCENDING (verified on 07-17: 7,752/7,752 each way).  The ingest builds
# both from a NEWEST-FIRST `tempfund`, so every function here that reproduces an ingest step
# must be fed newest-first -- `calc_diff`'s shift(-1) means "one period OLDER" and the
# history trim drops the frame's TAIL.
#
# CORRECTION (review, 2026-07-27).  This used to claim the reversal reproduces the ingest
# order "EXACTLY (including ties)".  IT DOES NOT.  `data_quality.py:235` is a STABLE
# multi-column lexsort, so within a duplicated snapped quarter the saved order preserves the
# ingest's arrival order -- and reversing the frame INVERTS each such tie pair instead of
# restoring it.  It is exact for the 96.4% of sources with no duplicate quarter and wrong
# within ties for the other 282; see fidelity gap C in the module docstring for the measured
# 0.92%-of-panel effect.  A `sort_values` on `date` would be WORSE (it would reorder ties
# arbitrarily rather than just inverting them), and the fields needed to break the ties
# correctly are absent from this panel, so the reversal stands as the best available -- but
# it is an APPROXIMATION, not an identity.
def _groups_newest_first(cdx_df):
    """Yield (source, frame) with each source's rows in reversed SAVED order == the
    ingest's newest-first order.  Index is preserved so results can be written back."""
    for src, g in cdx_df.iloc[::-1].groupby("source", sort=False):
        yield src, g


# --------------------------------------------------------------------------- #
#  1 + 2 : cdx_df upgrade                                                     #
# --------------------------------------------------------------------------- #
def upgrade_cdx(cdx_df, verbose=True):
    """Return a COPY of `cdx_df` on today's basis: corrected `price`, stamped
    `reportingFrequency`, in-pipeline `grahamNumber` + `grahamUndefinedReason`.

    Row order, index and every other column are preserved exactly.
    """
    out = cdx_df.copy()

    # ---- 1. PRICE = marketCap / weightedAverageShsOut -------------------------------
    # LOCKSTEP with getData_fmp.fillPreReqdf's C-2 fix (getData_fmp.py:305-314).  That site
    # cannot be called here: it reads the raw km/inc statement frames, which a saved panel
    # does not carry -- only their merged product, `cdx_df`.  Both inputs ARE in cdx_df, and
    # the expression is the whole fix: same two fields, same +-inf -> NaN normalisation
    # (an undefined price must read as MISSING, never as a real extreme value).
    _mc = pd.to_numeric(out["marketCap"], errors="coerce")
    _sh = pd.to_numeric(out["weightedAverageShsOut"], errors="coerce")
    price_old = pd.to_numeric(out.get("price"), errors="coerce")
    out["price"] = (_mc / _sh).replace([np.inf, -np.inf], np.nan)

    # ---- 2. frequency stamp + in-pipeline Graham (LIVE function, per source) --------
    n_src = 0
    graham = pd.Series(np.nan, index=out.index, dtype="float64")
    freq = pd.Series(rp.UNKNOWN, index=out.index, dtype=object)
    reason = pd.Series("", index=out.index, dtype=object)
    for src, g in _groups_newest_first(out):
        tf = g.reset_index(drop=True)           # positional index, as the live frame has
        tf = gdf.stamp_frequency_and_graham(tf)
        graham.loc[g.index] = pd.to_numeric(tf["grahamNumber"], errors="coerce").values
        freq.loc[g.index] = tf[rp.FREQ_COLUMN].values
        reason.loc[g.index] = tf["grahamUndefinedReason"].values
        n_src += 1
    out["grahamNumber"] = graham
    out[rp.FREQ_COLUMN] = freq
    out["grahamUndefinedReason"] = reason

    if verbose:
        r = (_mc / (price_old * _sh)).replace([np.inf, -np.inf], np.nan).dropna()
        print("panel_upgrade.upgrade_cdx: %d sources, %d rows" % (n_src, len(out)))
        # NOT EVIDENCE, and must not be quoted as such (reviewer, 2026-07-27): the
        # post-upgrade 1.0 is a TAUTOLOGY -- detect_price_basis computes
        # marketCap/(price*shares) and this function sets price = marketCap/shares, so it
        # reads 1.0 whatever production does.  What the BEFORE figure shows is real (the
        # saved panel is on the old ~4x basis); what the AFTER figure shows is only that
        # the substitution happened.  Correctness of the formula rests on it matching
        # fillPreReqdf (getData_fmp.py:271-274) textually -- same two fields, same
        # coercion, same +-inf -> NaN -- not on this ratio.
        print("  price   : BEFORE marketCap/(price*shares) median = %.4f (the real check: "
              "the saved panel IS on the old ~4x basis).  AFTER = 1.0 TAUTOLOGICALLY "
              "(same expression both sides) -- not evidence."
              % (float(r.median()) if len(r) else np.nan))
        _fc = freq.groupby(out["source"]).first().value_counts().to_dict()
        print("  frequency (cadence fallback, `period` absent): %s" % _fc)
        _rc = reason[reason != ""].value_counts().to_dict()
        print("  graham defined on %d/%d rows; undefined reasons: %s"
              % (int(graham.notna().sum()), len(graham), _rc))
    return out


# --------------------------------------------------------------------------- #
#  3 : BoMetric_df rebuild                                                    #
# --------------------------------------------------------------------------- #
def rebuild_bometric(cdx_up, n=1, verbose=True):
    """Rebuild `BoMetric_df` from an UPGRADED cdx_df using the live Stage-1 construction.

    `cdx_up` must already carry the `reportingFrequency` stamp (run `upgrade_cdx` first):
    the per-source `rpy` is READ from that stamp, never re-derived from snapped dates
    (review item 9 -- two sites deriving it independently could disagree).
    """
    if rp.FREQ_COLUMN not in cdx_up.columns:
        raise ValueError("rebuild_bometric: cdx_df carries no %r stamp -- run upgrade_cdx "
                         "first, so Stage-1 and the Graham site read the SAME "
                         "classification (review item 9)." % rp.FREQ_COLUMN)
    bm_cols = list(utils.initBoMetric_fromDict()["BoMetric_df"].columns)
    dicts = cdic.getDicts()
    packed = (dicts[2], dicts[3], dicts[5], dicts[4], dicts[6])   # base, mean, unity, diff, special

    frames = []
    for src, g in _groups_newest_first(cdx_up):
        tf = g.reset_index(drop=True)
        tmp = pd.DataFrame(columns=bm_cols)
        tmp["date"] = tf["date"].values
        tmp["source"] = src
        tmp = utils.setDatesToQuarterly(tmp)     # idempotent on already-snapped dates
        _rpy = rp.rows_per_year(tf[rp.FREQ_COLUMN].iloc[0])
        frames.append(gdf.build_bometric_rows(tf, tmp, _rpy, n=n, dicts=packed))
    bm = pd.concat(frames, ignore_index=True)

    # The SAME post-ingest fixup production applies once at the end of the fetch loop
    # (getData_fmp.py:170) -- inf -> NaN on both frames, so a zero-denominator ratio cannot
    # corrupt the Stage-1 pool median (getAves2 does not scrub inf, calcScore.py:137).
    bm, _ = gdg.fixAfterGetData(bm, cdx_up.copy())
    if verbose:
        print("panel_upgrade.rebuild_bometric: %d rows x %d cols over %d sources"
              % (len(bm), len(bm.columns), bm["source"].nunique()))
    return bm


def upgrade_panel(dmdic, verbose=True):
    """Return a NEW dmdic with `cdx_df` upgraded and `BoMetric_df` rebuilt from it.

    Every other key is shallow-copied.  `BoMetric_ave` / `BoMetric_dateAve` are DROPPED,
    not carried: they are cross-sectional Stage-1 baselines over the OLD BoMetric frame,
    and carrying them would score the new frame against old medians (the audit H-1 defect).
    Recompute them with calcScore.getAves2 on the frame you actually score.
    """
    new = dict(dmdic)
    new["cdx_df"] = upgrade_cdx(dmdic["cdx_df"], verbose=verbose)
    new["BoMetric_df"] = rebuild_bometric(new["cdx_df"], verbose=verbose)
    for k in ("BoMetric_ave", "BoMetric_dateAve"):
        new.pop(k, None)
    return new


# --------------------------------------------------------------------------- #
#  VERIFICATION                                                               #
# --------------------------------------------------------------------------- #
def main():
    """CLI, so the runbook's instructions are executable (review, 2026-07-27: the runbook
    told a stranger to point arguments at this module and it accepted none)."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Put a SAVED pre-fix panel onto today's metric basis (offline).")
    ap.add_argument("--panel", required=True,
                    help="saved Boresults_dic-*.pickle to upgrade")
    ap.add_argument("--out", required=True,
                    help="where to write the upgraded panel (cdx_df + BoMetric_df + "
                         "Tickers_df); *.pickle is gitignored under baseline_tools/")
    ap.add_argument("--verify", action="store_true",
                    help="also print the column-by-column comparison against the panel's "
                         "own saved BoMetric_df (the acceptance test for the rebuild)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import run_target_test as rtt
    src = pd.read_pickle(args.panel)
    dmdic = dict(src)
    print("panel_upgrade: %s" % args.panel)
    print("  basis BEFORE: %s" % (rtt.detect_price_basis(dmdic["cdx_df"]),))
    new = upgrade_panel(dmdic, verbose=not args.quiet)
    print("  basis AFTER : %s   (note: the AFTER ratio is TAUTOLOGICAL -- see upgrade_cdx)"
          % (rtt.detect_price_basis(new["cdx_df"]),))
    if args.verify:
        verify_against_saved(dmdic["BoMetric_df"], new["BoMetric_df"])
    pd.to_pickle({"cdx_df": new["cdx_df"], "BoMetric_df": new["BoMetric_df"],
                  "Tickers_df": new.get("Tickers_df")}, args.out)
    print("  wrote %s (%.0f MB)" % (args.out, os.path.getsize(args.out) / 1e6))
    print("  NOTE: a panel from a FRESH fetch on current code needs no upgrade -- run this "
          "only on a pre-2026-07-19 panel.")


def verify_against_saved(saved_bm, rebuilt_bm, rtol=1e-9, verbose=True):
    """Column-by-column agreement between the SAVED and the REBUILT BoMetric frame.

    This is the acceptance test for the whole rebuild.  A column the correctness arc did
    NOT touch must come back IDENTICAL -- that is what proves the offline path really is
    the production path (row order, per-source rpy, trim end, index alignment).  A column
    that DISAGREES is either a metric the arc changed or a defect, and the two are told
    apart by name, never by tolerance.

    Compared on the INNER join of (source, date) so the rpy-aware trim difference (the
    saved frame dropped a fixed 4 rows per source, today's drops `rpy`) does not read as a
    value disagreement.
    """
    a = saved_bm.copy()
    b = rebuilt_bm.copy()
    for d in (a, b):
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    key = ["source", "date"]
    a = a.drop_duplicates(subset=key).set_index(key)
    b = b.drop_duplicates(subset=key).set_index(key)
    common_rows = a.index.intersection(b.index)
    cols_both = [c for c in a.columns if c in b.columns]
    only_saved = [c for c in a.columns if c not in b.columns]
    only_new = [c for c in b.columns if c not in a.columns]

    rows = []
    for c in cols_both:
        x = pd.to_numeric(a.loc[common_rows, c], errors="coerce")
        y = pd.to_numeric(b.loc[common_rows, c], errors="coerce")
        both = x.notna() & y.notna()
        n_both = int(both.sum())
        if n_both:
            close = np.isclose(x[both], y[both], rtol=rtol, atol=0)
            frac_same = float(close.mean())
            med_ratio = float((y[both] / x[both]).replace(
                [np.inf, -np.inf], np.nan).median())
        else:
            frac_same, med_ratio = float("nan"), float("nan")
        rows.append({
            "column": c,
            "n_compared": n_both,
            "frac_identical": frac_same,
            "median_new_over_old": med_ratio,
            "nan_saved": int(x.isna().sum()),
            "nan_rebuilt": int(y.isna().sum()),
        })
    rep = pd.DataFrame(rows).sort_values("frac_identical")
    if verbose:
        print("panel_upgrade.verify_against_saved: %d common (source,date) rows, "
              "%d shared columns" % (len(common_rows), len(cols_both)))
        print("  columns only in SAVED (retired by the arc): %s" % only_saved)
        print("  columns only in REBUILT (added by the arc) : %s" % only_new)
        print(rep.to_string(index=False, float_format=lambda v: "%.6f" % v))
    return rep, only_saved, only_new


if __name__ == "__main__":
    main()
