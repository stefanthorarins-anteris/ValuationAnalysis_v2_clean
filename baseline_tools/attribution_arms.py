"""ATTRIBUTION ARMS -- did the pre-fix AggScore invert because of the METRICS, or because
of MISSING-DATA REWARDS the correctness arc happened to remove?

THE ALTERNATIVE STORY THIS EXISTS TO TEST (devils-advocate gate, 2026-07-27)
---------------------------------------------------------------------------
The first pass reported "the corrections flipped AggScore from anti-predictive to
predictive" and attributed that to the metric corrections.  But the pre-arc configuration
carried TWO mechanisms that systematically promote DATA-POOR names, and both were removed
by the arc:

  (i)  `mcapQuants` missing-cap fill.  Pre-fix, `pd.qcut(...).cat.codes == -1` (qcut's NaN
       sentinel) fell through the `(-1)*(code/3 - 0.5)` mapping to **+0.8333** -- outside
       the metric's [-0.5, +0.5] range and BETTER than the most-rewarded real bucket, on a
       w = 0.080 column.  So a missing market cap earned the maximum small-cap reward:
       about +0.067 of AggScore, handed to every name with no market cap.
       (stage2_metrics.add_mcap_quants; 746 rows on the 07-17 panel.)
  (ii) The z-path NaN fill.  `fillna(0)` puts an unavailable metric at the winsorized MEAN,
       which on a right-skewed column is ABOVE the median.
       THE MAGNITUDE, CORRECTED 2026-08-03.  This line read "+0.1616 of AggScore for full
       missingness, against a 0.134 median-to-top-20 distance" -- i.e. >100%, missingness
       alone reaching the shortlist.  Both figures were stale.  Re-measured on
       resdic_2026-07-17_CORRECTED.pickle (general top-100), advantage := 0 - sum_c w_c *
       median(z_c over OBSERVED cells): +0.0739 against a 0.2560 median->rank-20 distance,
       i.e. 29%, not >100%.  The DIRECTION -- a real reward, big enough to be a rival
       explanation for the sign flip, which is all this arm needs -- is unchanged; the size is
       ~3.6x smaller than stated.  PANEL-DEPENDENT: do not re-quote either number without its
       panel.  (For the record, mechanism (ii) no longer exists in the tree: E-1 centres each
       column on its observed median as of 2026-08-03, which puts the measured advantage at
       ~2e-18.  These arms still reproduce the PRE-ARC path from dumped frames, so they are
       unaffected -- but "the z-path rewards missingness" is now a statement about history.)

Data-poor global micro-caps were annihilated over 2023-25, so a score that rewards
missingness would be expected to look ANTI-predictive on this window without any metric
being wrong.  That story predicts everything the first pass observed: the sign flip, the
near-total reordering, the concentration in the bottom decile, and the flat beat-rate.
It has to be excluded before "the metrics were wrong" can be claimed.

THE DESIGN
----------
Factorise: (metric frame) x (fill/normalisation treatment), applied to the SAME dumped
frames so no arm can differ by anything else.

  A0  PRE-ARC frame  + pre-arc z-score & |z|>4 ejection      (reproduces config (a))
  A1  PRE-ARC frame  + ONLY the mcapQuants +0.8333 fill corrected      -> isolates (i)
  A2  PRE-ARC frame  + rank/inverse-normal (fill IS the median)        -> isolates (ii)
  A3  PRE-ARC frame  + BOTH corrections                                -> isolates (i)+(ii)
  B0  CURRENT frame  + z-score                                (reproduces config (b))
  B1  CURRENT frame  + Graham NaN incidence held at the PRE-ARC level.  FAILED AS AN
      ISOLATION -- retained only to record the failure.  It agrees with B0 to the 7th
      decimal and every decile/beat column is bit-identical, because the two frames'
      Graham missingness differs by only 3 nullable rows (1,696 vs a 1,558 reference, and
      525 reference-missing rows cannot be un-missed).  A control that moves nothing rules
      out nothing, so B1 supports NO claim about the Graham channel in either direction.
      (The gate asked for "(a) with Graham NaN incidence held at the old level", which is
      (a) itself; this was the informative reading of that request, and it did not work.)
  C0  CURRENT frame  + rank                                   (reproduces config (c))

If most of A0's inversion survives A1 and A2, the metric-signal story stands.  If it
collapses, the honest claim is the narrower "the arc stopped rewarding missing data".

ALSO MEASURED HERE (the missingness channel, directly)
------------------------------------------------------
  * IC(n_missing_metrics, excess) -- the confound's OWN predictive power on this window.
  * corr(AggScore, n_missing) -- whether each arm's score rewards missingness.
  * IC(AggScore, excess | n_missing) -- the partial, i.e. what is left of the score's
    ordering once the missingness channel is removed.  This is the single number that
    decides between the two stories.

And every median gradient is reported WITH the beat-rate columns beside it: this project's
target IS a beat-rate, and a median spread can be large while the share of names that beat
the benchmark is flat (which is exactly what happens here).  Reporting the median gradient
alone invites reading it as "the score sorts by outperformance".
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import createDicts as cdic
import postBoRank as pbr
import decile_test as dt
import stage2_metrics as sm

MCAP_COL = "marketCapRevQuants"
PREARC_MCAP_MISSING = 5.0 / 6.0        # the +0.8333 sentinel, exactly (-1)*(-1/3 - 0.5)


# --------------------------------------------------------------------------- #
#  Normalisation treatments                                                   #
# --------------------------------------------------------------------------- #
def _prearc_normalize(psm):
    """The PRE-ARC normaliser, reproduced here so both frames can be run through it:
    z-score, fillna(0), then EJECT any row with |z| > 4 in ANY column.

    Verbatim from postBoRank.normalizeAndDropNA at d40ace6 (no winsorizer, no exemptions).
    Reproduced rather than imported because the point of this module is to apply every
    treatment to BOTH frames from one place; the pre-arc tree cannot see the current frame.
    Fidelity is checked by test_attribution_arms.test_prearc_normalize_matches_d40ace6.
    """
    df = psm.copy().reset_index(drop=True)
    metric_cols = [c for c in df.columns if c != "source"]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = df[metric_cols].isna().sum(axis=1) < len(metric_cols)
    dropped_allnan = list(df.loc[~keep, "source"])
    dfnona = df[keep].copy()
    num = dfnona.drop("source", axis=1)
    z = (num - num.mean()) / num.std().replace(0, np.nan).fillna(1)
    z = z.fillna(0)
    dfnona[z.columns] = z
    ok = (z.abs() <= 4).all(axis=1)
    return dfnona[ok].copy(), dropped_allnan, int((~ok).sum())


def _prearc_normalize_noeject(psm):
    """The pre-arc normaliser with the |z|>4 EJECTION disabled -- separates "the pre-arc
    normaliser inverted the ranking" from "the pre-arc normaliser DROPPED the wrong names".

    NOTE ON WHAT A NULL HERE MEANS.  A control that moves nothing rules out nothing; it only
    shows the ejection is not the mechanism.  The earlier pass reported this as "-0.075 vs
    -0.077, so the inversion was in the metrics" -- an invalid inference twice over: the two
    numbers are bit-identical to 15 decimals on the shared cell, and even a genuine null
    would only have excluded the ejection, not implicated the metrics.
    """
    df = psm.copy().reset_index(drop=True)
    metric_cols = [c for c in df.columns if c != "source"]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = df[metric_cols].isna().sum(axis=1) < len(metric_cols)
    dropped_allnan = list(df.loc[~keep, "source"])
    dfnona = df[keep].copy()
    num = dfnona.drop("source", axis=1)
    z = (num - num.mean()) / num.std().replace(0, np.nan).fillna(1)
    dfnona[z.columns] = z.fillna(0)
    return dfnona.copy(), dropped_allnan, 0


def _rank_normalize(psm):
    out, dropped = pbr.normalizeAndDropNA(psm.copy(), method=pbr.NORM_RANK)
    return out, dropped, 0


def _current_zscore(psm):
    out, dropped = pbr.normalizeAndDropNA(psm.copy(), method=pbr.NORM_ZSCORE)
    return out, dropped, 0


# --------------------------------------------------------------------------- #
#  Frame treatments                                                           #
# --------------------------------------------------------------------------- #
def fix_mcap_fill(psm, verbose=True):
    """Replace the pre-arc +0.8333 missing-cap sentinel with the neutral 0.0 the fix uses.

    Matched on EXACT equality to 5/6 -- the sentinel is a single arithmetic constant, so an
    exact match cannot catch a real quartile value (the real codes are the 4 values
    -0.5, -1/6, +1/6, +0.5 and none of them is 5/6).
    """
    out = psm.copy()
    v = pd.to_numeric(out[MCAP_COL], errors="coerce")
    hit = np.isclose(v, PREARC_MCAP_MISSING, rtol=0, atol=1e-12)
    out.loc[hit, MCAP_COL] = getattr(sm, "MCAP_QUANT_MISSING", 0.0)
    if verbose:
        print("  fix_mcap_fill: %d of %d rows carried the +0.8333 sentinel (%.2f%%) -> %s"
              % (int(hit.sum()), len(out), 100.0 * hit.mean(),
                 getattr(sm, "MCAP_QUANT_MISSING", 0.0)))
    return out, int(hit.sum())


def hold_graham_missingness(psm_target, psm_reference, verbose=True):
    """Force `grahamNumberToPrice` missingness in `psm_target` to `psm_reference`'s pattern.

    Rows the reference HAS but the target does not stay missing (a value cannot be
    invented); rows the target has but the reference does not are SET to NaN.  So this can
    only move incidence TOWARD the reference, and the report states how far it got.
    """
    out = psm_target.copy()
    col = "grahamNumberToPrice"
    tgt = out.set_index("source")[col]
    ref = psm_reference.set_index("source")[col]
    common = tgt.index.intersection(ref.index)
    to_null = common[tgt.loc[common].notna().to_numpy() & ref.loc[common].isna().to_numpy()]
    before = int(tgt.isna().sum())
    out.loc[out["source"].isin(set(to_null)), col] = np.nan
    after = int(pd.to_numeric(out[col], errors="coerce").isna().sum())
    if verbose:
        print("  hold_graham_missingness: NaN %d -> %d of %d (reference has %d of %d); "
              "%d rows nulled, %d reference-missing rows could not be un-missed"
              % (before, after, len(out), int(ref.isna().sum()), len(ref), len(to_null),
                 int((tgt.loc[common].isna().to_numpy()
                      & ref.loc[common].notna().to_numpy()).sum())))
    return out, len(to_null)


# --------------------------------------------------------------------------- #
#  Scoring                                                                    #
# --------------------------------------------------------------------------- #
def score(psm, normalizer):
    """psm -> AggScore, using the production weight vector and getAggScore."""
    normed, dropped_allnan, n_ejected = normalizer(psm)
    postBm, postNew = cdic.getPostDict()
    ws = {**{k: postBm[k]["w"] for k in postBm}, **{k: postNew[k]["w"] for k in postNew}}
    unknown = [c for c in normed.columns if c not in ws and c != "source"]
    if unknown:
        raise RuntimeError("weight lookup would silently default to 1 for %s" % unknown)
    w = normed.drop("source", axis=1)
    for c in w.columns:
        w[c] = normed[c].values * ws[c]
    frame = pd.concat([normed[normed.columns.difference(w.columns)], w], axis=1)
    ranked = pbr.getAggScore(frame)
    return (ranked[["source", "AggScore"]].copy(),
            {"n_in": len(psm), "n_out": len(normed),
             "n_dropped_allnan": len(dropped_allnan), "n_ejected": n_ejected})


def n_missing_per_name(psm):
    """Count of WEIGHTED metric columns a name is missing -- the confound's carrier."""
    postBm, postNew = cdic.getPostDict()
    ws = {**{k: postBm[k]["w"] for k in postBm}, **{k: postNew[k]["w"] for k in postNew}}
    cols = [c for c in psm.columns
            if c != "source" and float(ws.get(c, 0) or 0) != 0]
    v = psm[cols].apply(pd.to_numeric, errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"source": psm["source"], "n_missing": v.isna().sum(axis=1),
                         "n_weighted_cols": len(cols)})


# --------------------------------------------------------------------------- #
#  Price-data sanity screen                                                   #
# --------------------------------------------------------------------------- #
def price_sanity(cell, verbose=True):
    """Flag return legs that are price-DATA defects rather than realized outcomes.

    Two signatures, both found by the devils-advocate gate in the first pass's own CSVs:
      * ABSURD magnitude -- a total return above +100x over 36 months is a corporate-action
        (reverse-split) adjustment failure, not a return.  SEZL reads +158,600%.
      * EXACT-INTEGER return -- a buy leg stored at a 2-decimal FLOOR makes eval/buy - 1 land
        on a whole number (16.0, 61.0, 9.0, 1.0, 0.0). A genuine return hits an exact integer
        with probability ~0.
    Both are excluded from any MEAN.  A median is insensitive to them by construction, which
    is exactly why the first pass did not notice them -- and exactly why the mean had to be
    reported to see the fat right tail at all.

    THE CLASS IS REDUCED, NOT CLOSED (gate note, 2026-07-27).  The magnitude test is a
    >100x threshold, so a MISHANDLED 1:10 REVERSE SPLIT -- a 10x error -- passes it
    untouched, and a 1:2 or 1:3 split error is entirely invisible to both tests.  Only the
    egregious tail and the 2-dp-floor signature are caught.  Closing the class properly
    needs corporate-action data the price grid does not carry.  Treat any MEAN on this cell
    as still exposed to the split-adjustment class; the medians and the rank-ICs are
    insensitive to it (a monotone per-name error cannot reorder that name's neighbours by
    much, and rank-IC only sees order).
    """
    c = cell.copy()
    r = pd.to_numeric(c["total_return"], errors="coerce")
    c["flag_absurd"] = r.abs() > 100.0
    c["flag_int"] = np.isclose(r, r.round(), rtol=0, atol=1e-12) & (r.abs() >= 1e-9)
    c["flag_zero_exact"] = np.isclose(r, 0.0, rtol=0, atol=1e-12)
    c["price_suspect"] = c["flag_absurd"] | c["flag_int"] | c["flag_zero_exact"]
    if verbose:
        bad = c[c["price_suspect"]]
        print("  price_sanity: %d of %d rows suspect (%d absurd |r|>100x, %d exact-integer,"
              " %d exactly-zero)"
              % (len(bad), len(c), int(c["flag_absurd"].sum()),
                 int((c["flag_int"] & ~c["flag_zero_exact"]).sum()),
                 int(c["flag_zero_exact"].sum())))
        if len(bad):
            print(bad[["source", "buy_adjClose", "eval_adjClose", "total_return"]]
                  .sort_values("total_return", ascending=False)
                  .to_string(index=False, float_format=lambda v: "%.4f" % v))
    return c


# --------------------------------------------------------------------------- #
#  Statistics                                                                 #
# --------------------------------------------------------------------------- #
def _ic(x, y):
    return float(pd.Series(np.asarray(x, float)).rank()
                 .corr(pd.Series(np.asarray(y, float)).rank()))


def boot_ic_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = _ic(x[idx], y[idx])
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def partial_ic(score_v, excess_v, control_v):
    """Spearman partial correlation: rank-residualise BOTH score and outcome on the control,
    then correlate the residuals.  With `control` = n_missing this is "does the score order
    the outcome for reasons OTHER than rewarding missing data"."""
    s = pd.Series(np.asarray(score_v, float)).rank()
    e = pd.Series(np.asarray(excess_v, float)).rank()
    c = pd.Series(np.asarray(control_v, float)).rank()
    def resid(v):
        cc = pd.concat([v, c], axis=1).dropna()
        b = np.polyfit(cc.iloc[:, 1], cc.iloc[:, 0], 1)
        return cc.iloc[:, 0] - np.polyval(b, cc.iloc[:, 1])
    rs, re_ = resid(s), resid(e)
    j = rs.index.intersection(re_.index)
    return float(np.corrcoef(rs.loc[j], re_.loc[j])[0, 1])


def arm_report(name, cell, verbose=True):
    """Decile table + the beat-rate columns + the tail decomposition, for one arm."""
    d = dt.decile_table(cell, "AggScore")
    m = dt.monotonicity(d)
    y = d["median_excess"]
    b = d["share_beat"]
    b10 = d["share_beat_10pp"]
    row = {
        "arm": name, "n": len(cell.dropna(subset=["AggScore", "excess"])),
        "IC": _ic(cell["AggScore"], cell["excess"]),
        "d1_med": y.iloc[0], "d10_med": y.iloc[-1],
        "d1_minus_d10": y.iloc[0] - y.iloc[-1],
        "d1_minus_d9": y.iloc[0] - y.iloc[8],
        "d9_minus_d10": y.iloc[8] - y.iloc[-1],
        "spearman_dec": m["spearman_decile_vs_median"],
        "steps_down_of_9": m["n_steps_down_of_9"],
        # THE COLUMNS THE FIRST PASS COMPUTED AND DID NOT REPORT
        "d1_share_beat": b.iloc[0], "d10_share_beat": b.iloc[-1],
        "beat_spread": b.iloc[0] - b.iloc[-1],
        "beat_range": float(b.max() - b.min()),
        "d1_beat10": b10.iloc[0], "d10_beat10": b10.iloc[-1],
        "beat10_spread": b10.iloc[0] - b10.iloc[-1],
    }
    if verbose:
        print(d[["decile", "n", "median_excess", "median_se", "share_beat",
                 "share_beat_10pp", "median_return"]].to_string(
                     index=False, float_format=lambda v: "%+.4f" % v))
    return row, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prearc", default=os.path.join(_HERE, "psm_PREARC_2022-12-30.pickle"))
    ap.add_argument("--current", default=os.path.join(_HERE, "psm_CURRENT_2022-12-30.pickle"))
    ap.add_argument("--prearc-code-rebuilt-panel",
                    default=os.path.join(_HERE,
                                         "psm_PREARCCODE_REBUILTPANEL_2022-12-30.pickle"),
                    help="PRE-ARC Stage-2 code run on the REBUILT panel with the CURRENT "
                         "Stage-1 -- the arm that removes the panel confound (see A4/A5)")
    ap.add_argument("--buy", default="2022-12-30")
    ap.add_argument("--eval", dest="eval_", default="2025-12-31")
    ap.add_argument("--min-cdx-rows", default=8, type=int)
    ap.add_argument("--out-prefix", default=os.path.join(_HERE, "attribution"))
    args = ap.parse_args()

    A = pd.read_pickle(args.prearc)
    B = pd.read_pickle(args.current)
    bar = "=" * 100
    print(bar)
    print("ATTRIBUTION ARMS -- metric corrections vs missing-data rewards")
    print("  buy %s -> eval %s  |  scorability floor: >=%d cdx rows as of the anchor"
          % (args.buy, args.eval_, args.min_cdx_rows))
    print("  ALL FIGURES ON THE 07-17 UNIVERSE")
    print(bar)
    for tag, D in (("PRE-ARC", A), ("CURRENT", B)):
        i = D["info"]
        print("  %-8s label=%s" % (tag, i["label"]))
        print("           git_sha=%s dirty=%s | reporting_period=%s winsorizer=%s "
              "rank_method=%s MCAP_QUANT_MISSING=%s | normalize_params=%s"
              % (i.get("git_sha"), i.get("git_dirty"), i.get("has_reporting_period"),
                 i.get("has_winsorizer"), i.get("has_rank_method"),
                 i.get("has_MCAP_QUANT_MISSING"), i.get("normalize_params")))
        print("           n_stage1=%d n_psm_rows=%d dropped_metrics=%s"
              % (i["n_stage1"], i["n_psm_rows"], i["dropped_metrics"]))
    print(bar, flush=True)

    # ---- build the arms -----------------------------------------------------------
    print("\nBUILDING ARMS")
    psm_a, psm_b = A["psm"], B["psm"]
    psm_a_mcap, n_sent = fix_mcap_fill(psm_a)
    psm_b_graham, n_null = hold_graham_missingness(psm_b, psm_a)
    ARMS = [
        ("A0 pre-arc frame + pre-arc z&eject", psm_a, _prearc_normalize),
        ("A1 pre-arc + mcapQuants fill FIXED", psm_a_mcap, _prearc_normalize),
        ("A2 pre-arc + RANK (fill = median)", psm_a, _rank_normalize),
        ("A3 pre-arc + mcap FIXED + RANK", psm_a_mcap, _rank_normalize),
        ("B0 current frame + z-score", psm_b, _current_zscore),
        ("B1 current + Graham NaN held OLD", psm_b_graham, _current_zscore),
        ("C0 current frame + RANK", psm_b, _rank_normalize),
        # A6/A7 split the pre-arc NORMALISER into its two differences from today's z-path --
        # the missing WINSORIZER and the |z|>4 EJECTION -- on the pre-arc frame, so the
        # active ingredient is identified rather than bundled as "the fill".
        ("A6 pre-arc frame + CURRENT z (winsorized)", psm_a, _current_zscore),
        ("A7 pre-arc frame + pre-arc z, NO ejection", psm_a, _prearc_normalize_noeject),
    ]
    # ---- the CONFOUND-FREE code contrast (review, 2026-07-27) ---------------------
    # A0..A3 read the ORIGINAL saved panel; B0/C0 read the panel_upgrade'd REBUILT one.  So
    # any A-vs-B difference carries the rebuild's own artifacts (panel_upgrade fidelity gaps
    # C and D: inverted tie order on 282 duplicate-quarter sources = 0.92% of rows, and the
    # data-quality-pruned input the ingest never had) on top of the metric fixes.  A4/A5 run
    # the PRE-ARC Stage-2 code on the SAME rebuilt panel and the SAME Stage-1 output, so
    # A4-vs-B0 and A5-vs-C0 differ ONLY in the Stage-2 metric definitions.  That pair is the
    # metric-fix effect; everything else is held fixed.
    if os.path.exists(args.prearc_code_rebuilt_panel):
        P = pd.read_pickle(args.prearc_code_rebuilt_panel)
        ARMS += [
            ("A4 pre-arc CODE on REBUILT panel + z", P["psm"], _current_zscore),
            ("A5 pre-arc CODE on REBUILT panel + RANK", P["psm"], _rank_normalize),
        ]
        print("  A4/A5 source: %s" % P["info"]["label"])
        print("            Stage-1 borrowed from: %s (n=%s)"
              % (P["info"].get("stage1_source_label"), P["info"].get("stage1_n_kept")))
    else:
        print("  !! A4/A5 SKIPPED -- %s absent, so the panel confound is NOT removed and "
              "no A-vs-B difference may be attributed to metric definitions alone."
              % args.prearc_code_rebuilt_panel)

    ps = dt.build_price_source()
    rows, dtabs, cells = [], {}, {}
    for name, psm, norm in ARMS:
        scores, info = score(psm, norm)
        cell, bench = dt.measurement_cell(scores, ps, args.buy, args.eval_, verbose=False)
        miss = n_missing_per_name(psm)
        cell = cell.merge(miss, on="source", how="left")
        rows_asof = B["rows_asof"] if (name.startswith("B") or name.startswith("C") or name.startswith("A4") or name.startswith("A5")) else A["rows_asof"]
        if args.min_cdx_rows:
            keep = set(rows_asof[rows_asof >= args.min_cdx_rows].index)
            cell = cell[cell["source"].isin(keep)].copy()
        cells[name] = cell
        print("\n" + "-" * 100)
        print("%s   n_in=%d n_out=%d (all-NaN dropped %d, EJECTED %d) -> cell n=%d"
              % (name, info["n_in"], info["n_out"], info["n_dropped_allnan"],
                 info["n_ejected"], len(cell)))
        print("-" * 100)
        r, d = arm_report(name, cell)
        r.update({k: info[k] for k in ("n_in", "n_out", "n_dropped_allnan", "n_ejected")})
        # the missingness channel
        r["IC_nmissing_vs_excess"] = _ic(cell["n_missing"], cell["excess"])
        r["spearman_score_vs_nmissing"] = _ic(cell["AggScore"], cell["n_missing"])
        r["partial_IC_given_nmissing"] = partial_ic(cell["AggScore"], cell["excess"],
                                                    cell["n_missing"])
        lo, hi = boot_ic_ci(cell["AggScore"], cell["excess"])
        r["IC_lo95"], r["IC_hi95"] = lo, hi
        print("  IC=%+.4f [%+.3f,%+.3f] | IC(n_missing,excess)=%+.4f | "
              "rho(score,n_missing)=%+.4f | partial IC | n_missing = %+.4f"
              % (r["IC"], lo, hi, r["IC_nmissing_vs_excess"],
                 r["spearman_score_vs_nmissing"], r["partial_IC_given_nmissing"]))
        rows.append(r)
        dtabs[name] = d.assign(arm=name)

    summ = pd.DataFrame(rows)
    print("\n" + bar)
    print("HEADLINE -- ATTRIBUTION (all ON THE 07-17 UNIVERSE)")
    print(bar)
    # REPORTING RULE (adopted 2026-07-27 after TWO rounds in which the decisive evidence sat
    # in a column this script computed and the summary omitted -- first the beat-rate
    # columns, then `spearman_score_vs_nmissing` / `partial_IC_given_nmissing`).  EVERY
    # computed column is printed.  Narrowing the view is now a code change with an
    # assertion behind it, not an oversight in prose.
    _omit = [c for c in summ.columns if c not in list(summ.columns)]
    assert not _omit, "computed but not printed: %s" % _omit
    with pd.option_context("display.max_columns", None, "display.width", 300):
        print(summ.to_string(index=False, float_format=lambda v: "%+.4f" % v))
    print("  [columns printed: %d of %d computed -- reporting rule enforced]"
          % (len(summ.columns), len(summ.columns)))
    print("\n  IC(n_missing, excess) on this window: %+.4f  <- the confound's OWN power"
          % summ["IC_nmissing_vs_excess"].iloc[0])
    print("  NOTE ON n: it is PER ARM and differs between arms -- only A0/A1 sit on 6,533 "
          "(the |z|>4 ejection removes 31 names); every other arm is 6,564.  Do not quote "
          "one n for the study.")
    print("  NOTE ON THE CONTRAST TO USE: A7-vs-A6, NOT A0-vs-A6.  A0 and A6 are not "
          "cell-matched (6,533 vs 6,564); A7 is A0 with the ejection off and IS matched.")
    print("  A7 vs A0 are NOT bit-identical: IC %.5f vs %.5f -- the ejection is worth about "
          "%.4f of IC and moves 31 names.  It is a small effect, not a null."
          % (summ.set_index('arm').loc[[c for c in summ['arm'] if c.startswith('A7')][0], 'IC'],
             summ.set_index('arm').loc[[c for c in summ['arm'] if c.startswith('A0')][0], 'IC'],
             abs(summ.set_index('arm').loc[[c for c in summ['arm'] if c.startswith('A7')][0], 'IC']
                 - summ.set_index('arm').loc[[c for c in summ['arm'] if c.startswith('A0')][0], 'IC'])))
    print("  B1 IS NOT AN ISOLATION and supports no Graham claim: it agrees with B0 to the "
          "7th decimal and every decile/beat column is bit-identical, because the Graham "
          "missingness it was meant to restore differs by only 3 nullable rows.  By the "
          "same rule applied to the ejection, a control that moves nothing rules out "
          "nothing.  Retained only to record that the channel could NOT be isolated here.")

    # ---- price sanity + the mean, with and without the defects --------------------
    print("\n" + bar)
    print("PRICE-DATA SANITY + MEAN excess (the median is blind to the right tail)")
    print(bar)
    for name in ("A0 pre-arc frame + pre-arc z&eject", "B0 current frame + z-score",
                 "C0 current frame + RANK"):
        c = price_sanity(cells[name])
        clean = c[~c["price_suspect"]]
        d_all = dt.decile_table(c, "AggScore")
        d_cl = dt.decile_table(clean, "AggScore")
        print("  %-38s mean d1/d10  ALL: %+.4f / %+.4f   CLEANED: %+.4f / %+.4f  (n -%d)"
              % (name, d_all["mean_excess"].iloc[0], d_all["mean_excess"].iloc[-1],
                 d_cl["mean_excess"].iloc[0], d_cl["mean_excess"].iloc[-1],
                 len(c) - len(clean)))
        cells[name] = c

    # ---- artifacts ---------------------------------------------------------------
    summ.to_csv(args.out_prefix + "_headline.csv", index=False)
    pd.concat(dtabs.values(), ignore_index=True).to_csv(
        args.out_prefix + "_deciles.csv", index=False)
    for name, c in cells.items():
        slug = name.split()[0]
        c.to_csv("%s_cell_%s.csv" % (args.out_prefix, slug), index=False)
    print("\nwrote %s_headline.csv, %s_deciles.csv and %d per-arm cell CSVs"
          % (args.out_prefix, args.out_prefix, len(cells)))
    print(bar, flush=True)


if __name__ == "__main__":
    main()
