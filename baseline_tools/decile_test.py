"""DECILE MONOTONICITY of AggScore on a realized 36-month hold  (OFFLINE, no network).

THE QUESTION
------------
Does the score ORDER the universe by subsequent excess return at all?  Not "do the top 20
beat the index" -- that question has ~40 outcomes and a +-11pp standard error, so it cannot
distinguish a real change from noise.  Cutting the whole scorable universe into deciles and
reading MEDIAN 36-month excess return per decile puts ~650 names in each bucket, which is a
~5x tightening of the same underlying data.  A gradient is a property of the ranking
function; a top-20 count is a property of 20 draws from it.

MEASUREMENT CELL (the reason this is worth running now)
-------------------------------------------------------
buy 2022-12-30 -> evaluate 2025-12-31 (year-end anchors on the real_prices grid; the eval
anchor carries the 2025-12-30 fill for exchanges closed on the 31st -- returns_core's
documented union rule).  It is the only anchor where the saved panel has BOTH enough
statement history to score a wide universe AND real prices at both legs.

WHAT THIS IS NOT
----------------
  * NOT the deployed two-stage configuration.  The shipped filter applies Stage-2 ONLY to
    the Stage-1 top-100, so its cross-sectional normalisation is over ~100 names; here it
    is over ~7k.  This measures "AggScore as a universe-wide ranking function", which is
    the object a monotonicity test needs and the object a weight re-fit would fit.  The
    Stage-1 BoScore gradient IS reported alongside it, and that one IS the deployed first
    cut, universe-wide by construction.
  * NOT a fit.  It consumes zero degrees of freedom: the weights are the shipped ones and
    nothing here is tuned against the outcome.  Treat the cell as a held-out test set.
  * NOT the universe a fresh fetch produces.  The panel is the OLD gates' universe (~523
    pricefails, ~72% non-US, the lenfail 16->8 cohort never fetched).  Every number this
    module prints is `on the 07-17 universe`.

CONFIGURATIONS
--------------
  a) shipped/old metrics + z-scoring        -- run this file from the PRE-FIX tree against
                                               the SAVED panel (the historical baseline)
  b) corrected metrics + z-scoring          -- current tree + panel_upgrade'd panel
  c) corrected metrics + rank-normalisation -- as (b) with --norm rank
(a) is the only one that needs the other tree, because "old metrics" means the old
functions, not just the old panel.
"""

import argparse
import inspect
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
import calcScore as csf
import postBoRank as pbr
import stage2_pit as s2
import returns_core as rc

BUY = "2022-12-30"
EVAL = "2025-12-31"
N_BOOT = 2000


# --------------------------------------------------------------------------- #
#  Universe-wide AggScore                                                     #
# --------------------------------------------------------------------------- #
def _stage1(bm_pit, cdx_pit, nq_stage1=8):
    """Universe-wide Stage-1 BoScore, through the tree's own stage2_pit helper.

    The pre-fix tree's stage1_boscore takes no cdx_pit (the `period`-derived frequency map
    landed 2026-07-26), so the parameter is passed only when the signature has it -- each
    tree therefore runs ITS OWN Stage-1, which is what a before/after comparison requires.
    """
    if "cdx_pit" in inspect.signature(s2.stage1_boscore).parameters:
        bs = s2.stage1_boscore(bm_pit, nq_stage1=nq_stage1, cdx_pit=cdx_pit)
    else:
        bs = s2.stage1_boscore(bm_pit, nq_stage1=nq_stage1)
    return bs.sort_values("score", ascending=False).reset_index(drop=True)


def universe_aggscore(dmdic, D=BUY, nq_stage1=8, nq_stage2=16, norm="zscore",
                      rank_bounded=True, stage1_cache=None, verbose=True):
    """AggScore for EVERY Stage-1-scorable name as-of D (no top-100 cut).

    Returns (frame[source, BoScore, AggScore], info dict).
    """
    bm_pit, cdx_pit = s2.prepare_pit(dmdic, D, na1_only=False)
    bs = stage1_cache if stage1_cache is not None else _stage1(bm_pit, cdx_pit, nq_stage1)

    # Stage-2 over the WHOLE Stage-1 output.  Same production metric loop the top-100 path
    # uses -- only the pool it is handed is wider.
    cdxtop = cdx_pit[cdx_pit["source"].isin(bs["source"])].reset_index(drop=True)
    psm = s2._stage2_metric_loop_offline(bs, cdxtop, nq=nq_stage2)

    kw = {}
    sig = inspect.signature(pbr.normalizeAndDropNA).parameters
    if "method" in sig:
        kw["method"] = norm
        if "rank_bounded" in sig:
            kw["rank_bounded"] = rank_bounded
    elif norm != "zscore":
        raise SystemExit("this tree's normalizeAndDropNA has no `method` parameter -- "
                         "--norm %s is only available on the post-2026-07-27 tree" % norm)
    n_in = len(psm)
    psm_norm, dropped = pbr.normalizeAndDropNA(psm, **kw)
    n_out = len(psm_norm)

    postBm, postNew = cdic.getPostDict()
    ws = {**{k: postBm[k]["w"] for k in postBm}, **{k: postNew[k]["w"] for k in postNew}}
    w = psm_norm.drop("source", axis=1)
    for col in w.columns:
        w[col] = psm_norm[col].values * ws.get(col, 1)
    psmdf = pd.concat([psm_norm[psm_norm.columns.difference(w.columns)], w], axis=1)
    postRank = pbr.getAggScore(psmdf)

    out = postRank[["source", "AggScore"]].copy()
    out = out.merge(bs.rename(columns={"score": "BoScore_stage1"}), on="source", how="left")
    info = {
        "n_stage1": len(bs),
        "n_into_norm": n_in,
        "n_out_of_norm": n_out,
        "n_dropped_by_norm": n_in - n_out,
        "dropped_sample": sorted(map(str, dropped))[:15],
        "norm": norm,
        "rank_bounded": rank_bounded,
        "has_method_param": "method" in sig,
    }
    if verbose:
        print("universe_aggscore: Stage-1 n=%d -> Stage-2 in=%d out=%d (normaliser dropped "
              "%d) | norm=%s" % (info["n_stage1"], n_in, n_out, info["n_dropped_by_norm"],
                                 norm), flush=True)
    return out, info


# --------------------------------------------------------------------------- #
#  Returns + the cell                                                         #
# --------------------------------------------------------------------------- #
def build_price_source(prices_csv=None, prices_2025_csv=None):
    pd_dir = os.path.join(_HERE, "price_data")
    return rc.PriceSource(prices_csv or os.path.join(pd_dir, "real_prices.csv"),
                          supp_csv=prices_2025_csv or os.path.join(pd_dir,
                                                                   "real_prices_2025.csv"))


def measurement_cell(scores, price_source, buy=BUY, eval_=EVAL, cdx_df=None,
                     min_cdx_rows=0, verbose=True):
    """Join realized 36-month total returns; keep only names PRICED AT BOTH ENDS.

    `status == 'ok'` is exactly "real buy leg and real eval leg" -- the terminal-value
    policy (used when the eval leg is missing) is deliberately NOT invoked here: a decile
    median must not be a mixture of realized returns and exit approximations.  The excluded
    counts are reported so the exclusion is visible, not silent.

    min_cdx_rows : optional SCORABILITY floor -- require >= this many cdx rows as of the buy
        anchor.  Stage-1 will score a name off a single row, and a score built on one
        statement is not the same measurement as one built on ten, so the floor is the
        honest definition of "scorable".  0 = no floor (report both; the gradient's
        robustness to the floor is itself informative).
    """
    ret = rc.compute_returns(list(scores["source"]), buy, eval_, price_source)
    bench = rc.benchmark_return(price_source, buy, eval_, require_exact=True)
    df = scores.merge(ret, left_on="source", right_on="ticker", how="left")
    cell = df[df["status"] == "ok"].copy()
    n_priced = len(cell)
    n_thin = 0
    if min_cdx_rows and cdx_df is not None:
        d = pd.to_datetime(cdx_df["date"], errors="coerce")
        cnt = cdx_df.loc[d <= pd.Timestamp(buy)].groupby("source").size()
        keep = set(cnt[cnt >= min_cdx_rows].index)
        n_thin = int((~cell["source"].isin(keep)).sum())
        cell = cell[cell["source"].isin(keep)].copy()
    cell["excess"] = cell["total_return"] - bench
    if verbose:
        print("measurement_cell: %s -> %s | URTH %+.4f | scored %d, priced-both-ends %d "
              "(no_buy %d, eval-missing %d)%s"
              % (buy, eval_, bench, len(df), n_priced,
                 int((df["status"] == "no_buy").sum()),
                 int((df["status"] == "terminal").sum()),
                 ("; dropped %d with <%d cdx rows as of the anchor -> CELL n=%d"
                  % (n_thin, min_cdx_rows, len(cell))) if min_cdx_rows else ""), flush=True)
    return cell, bench


# --------------------------------------------------------------------------- #
#  Statistics                                                                 #
# --------------------------------------------------------------------------- #
def _boot_median_se(x, n_boot=N_BOOT, seed=0):
    """Bootstrap SE + percentile CI of the MEDIAN.  Nonparametric on purpose: 36-month
    equity returns are strongly right-skewed, so a normal-theory median SE
    (1.2533*sd/sqrt(n)) is not trustworthy here."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    return float(meds.std(ddof=1)), (float(np.percentile(meds, 2.5)),
                                     float(np.percentile(meds, 97.5)))


def _wilson(k, n, z=1.96):
    """Wilson score interval -- correct at the small n and near-0/1 rates a top-5 count
    produces, where a normal +-1.96*sqrt(p(1-p)/n) band runs outside [0,1]."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def decile_table(cell, score_col="AggScore", n_deciles=10, threshold=0.10, seed=0):
    """Median/mean excess return per score decile.  Decile 1 = HIGHEST score."""
    c = cell.dropna(subset=[score_col, "excess"]).copy()
    c = c.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(c)
    # Equal-COUNT buckets by rank position: qcut on the score itself mis-buckets a tied /
    # quantized score (BoScore is quantized by construction), and a decile test needs equal
    # n per bucket for its SEs to be comparable.
    c["decile"] = (np.arange(n) * n_deciles // n) + 1
    rows = []
    for d, g in c.groupby("decile"):
        se, ci = _boot_median_se(g["excess"], seed=seed + d)
        k = int((g["excess"] >= threshold).sum())
        p, lo, hi = _wilson(k, len(g))
        rows.append({
            "decile": int(d), "n": len(g),
            "median_excess": float(g["excess"].median()),
            "median_se": se, "median_lo95": ci[0], "median_hi95": ci[1],
            "mean_excess": float(g["excess"].mean()),
            "share_beat": float((g["excess"] > 0).mean()),
            "share_beat_10pp": p, "beat10_lo95": lo, "beat10_hi95": hi,
            "median_return": float(g["total_return"].median()),
        })
    return pd.DataFrame(rows)


def topn_beat_rates(cell, score_col="AggScore", ns=(5, 10, 20), threshold=0.10):
    c = cell.dropna(subset=[score_col, "excess"]).sort_values(score_col, ascending=False)
    rows = []
    for n in ns:
        g = c.head(n)
        k = int((g["excess"] >= threshold).sum())
        p, lo, hi = _wilson(k, len(g))
        rows.append({"top_n": n, "n_eval": len(g), "n_beat_10pp": k, "beat_rate": p,
                     "lo95": lo, "hi95": hi, "median_excess": float(g["excess"].median())})
    return pd.DataFrame(rows)


def rank_ic(cell, score_col="AggScore"):
    """Spearman rank correlation between score and realized excess return -- the whole-
    distribution summary the decile table is a coarsening of."""
    c = cell.dropna(subset=[score_col, "excess"])
    if len(c) < 10:
        return float("nan"), 0
    r = c[score_col].rank().corr(c["excess"].rank())
    return float(r), len(c)


def monotonicity(dtab):
    """Spearman of (decile index, median excess) plus the top-minus-bottom spread.
    Reported instead of eyeballing the column: a gradient claim needs a number."""
    x = dtab["decile"].astype(float)
    y = dtab["median_excess"].astype(float)
    rho = float(x.rank().corr(y.rank()))
    n_desc = int((y.diff().dropna() < 0).sum())
    return {"spearman_decile_vs_median": rho,          # -1.0 == perfectly monotone-good
            "n_steps_down_of_9": n_desc,
            "top_minus_bottom": float(y.iloc[0] - y.iloc[-1])}


# --------------------------------------------------------------------------- #
#  Missing-data bias (finding N1) -- measured on the SAME frame                #
# --------------------------------------------------------------------------- #
def missingness_reward(dmdic, D=BUY, nq_stage2=16, norm="zscore", rank_bounded=True,
                       stage1_cache=None, topn=100, verbose=True):
    """The N1 number: what a name GAINS in AggScore purely by having a metric missing.

    Method: normalise the pool, then for each WEIGHTED column read the percentile that the
    fill value (0) occupies in that column's own normalised distribution, and the AggScore
    a fully-missing name would receive (= sum of w * 0 = 0) against the pool's MEDIAN
    AggScore.  Under z-scoring 0 is the winsorized MEAN (above the median on a
    right-skewed column); under the rank map 0 is the median by construction.

    Measured on the STAGE-2 POOL SIZE the finding was measured on (top-100), not the whole
    universe, so the before/after is comparable to the reviewer's figure.
    """
    bm_pit, cdx_pit = s2.prepare_pit(dmdic, D, na1_only=False)
    bs = stage1_cache if stage1_cache is not None else _stage1(bm_pit, cdx_pit)
    top = bs.head(topn).reset_index(drop=True)
    cdxtop = cdx_pit[cdx_pit["source"].isin(top["source"])].reset_index(drop=True)
    psm = s2._stage2_metric_loop_offline(top, cdxtop, nq=nq_stage2)

    kw = {}
    sig = inspect.signature(pbr.normalizeAndDropNA).parameters
    if "method" in sig:
        kw["method"] = norm
        if "rank_bounded" in sig:
            kw["rank_bounded"] = rank_bounded
    psm_norm, _ = pbr.normalizeAndDropNA(psm, **kw)

    postBm, postNew = cdic.getPostDict()
    ws = {**{k: postBm[k]["w"] for k in postBm}, **{k: postNew[k]["w"] for k in postNew}}
    rows, total = [], 0.0
    for col in psm_norm.columns:
        if col == "source":
            continue
        w = float(ws.get(col, 1) or 0)
        if w == 0:
            continue
        v = pd.to_numeric(psm_norm[col], errors="coerce").dropna()
        if v.empty:
            continue
        pct = float((v < 0).mean() + 0.5 * (v == 0).mean())      # percentile of the fill
        contrib = w * (0.0 - float(v.median()))    # what fill(0) gains vs the median name
        total += contrib
        rows.append({"column": col, "weight": w, "pct_of_fill_value": pct,
                     "median_normed": float(v.median()), "gain_vs_median": contrib})
    tab = pd.DataFrame(rows).sort_values("gain_vs_median", ascending=False)
    n_above = int((tab["pct_of_fill_value"] > 0.50).sum())
    if verbose:
        print("\nmissingness_reward [%s]: fill value 0 sits ABOVE the column median on "
              "%d of %d weighted columns; full-missingness AggScore advantage = %+.4f"
              % (norm, n_above, len(tab), total), flush=True)
        print(tab.to_string(index=False, float_format=lambda v: "%.4f" % v))
    return {"total_advantage": total, "n_above_median": n_above,
            "n_weighted_cols": len(tab), "table": tab}


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def _fmt(dtab):
    return dtab.to_string(index=False, float_format=lambda v: "%+.4f" % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="saved panel pickle (raw or upgraded)")
    ap.add_argument("--label", default="")
    ap.add_argument("--norm", default="zscore", choices=["zscore", "rank"])
    ap.add_argument("--rank-bounded", default=1, type=int)
    ap.add_argument("--buy", default=BUY)
    ap.add_argument("--eval", dest="eval_", default=EVAL)
    ap.add_argument("--n-deciles", default=10, type=int)
    ap.add_argument("--stage1-cache", default=None)
    ap.add_argument("--min-cdx-rows", default=8, type=int,
                    help="scorability floor: cdx rows required as of the buy anchor "
                         "(8 = the brief's cell definition; 0 = no floor)")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--skip-n1", action="store_true")
    args = ap.parse_args()

    p = pd.read_pickle(args.panel)
    dmdic = {"cdx_df": p["cdx_df"], "BoMetric_df": p["BoMetric_df"],
             "Tickers_df": p.get("Tickers_df")}
    s1 = pd.read_pickle(args.stage1_cache) if args.stage1_cache else None

    bar = "=" * 84
    print(bar)
    print("DECILE MONOTONICITY -- %s" % (args.label or os.path.basename(args.panel)))
    print("  panel  : %s" % args.panel)
    print("  norm   : %s (rank_bounded=%s)" % (args.norm, bool(args.rank_bounded)))
    print("  window : buy %s -> eval %s   [ON THE 07-17 UNIVERSE]" % (args.buy, args.eval_))
    print(bar, flush=True)

    scores, info = universe_aggscore(dmdic, D=args.buy, norm=args.norm,
                                     rank_bounded=bool(args.rank_bounded),
                                     stage1_cache=s1)
    ps = build_price_source()
    cell, bench = measurement_cell(scores, ps, args.buy, args.eval_,
                                   cdx_df=dmdic["cdx_df"],
                                   min_cdx_rows=args.min_cdx_rows)

    print("\n--- AggScore deciles (1 = highest score), median 36mo excess vs URTH ---")
    dtab = decile_table(cell, "AggScore", n_deciles=args.n_deciles)
    print(_fmt(dtab))
    print("  monotonicity: %s" % monotonicity(dtab))
    ric, nic = rank_ic(cell, "AggScore")
    print("  Spearman rank-IC(AggScore, excess) = %+.4f  (n=%d)" % (ric, nic))

    print("\n--- Stage-1 BoScore deciles (the DEPLOYED first cut) ---")
    btab = decile_table(cell, "BoScore_stage1", n_deciles=args.n_deciles)
    print(_fmt(btab))
    print("  monotonicity: %s" % monotonicity(btab))
    bric, _ = rank_ic(cell, "BoScore_stage1")
    print("  Spearman rank-IC(BoScore, excess) = %+.4f" % bric)

    print("\n--- SECONDARY (wide bands -- do NOT read a 5pp move as a result) ---")
    ttab = topn_beat_rates(cell, "AggScore")
    print(ttab.to_string(index=False, float_format=lambda v: "%.4f" % v))

    if not args.skip_n1:
        missingness_reward(dmdic, D=args.buy, norm=args.norm,
                           rank_bounded=bool(args.rank_bounded), stage1_cache=s1)

    if args.out_csv:
        dtab.assign(score="AggScore", config=args.label).to_csv(args.out_csv, index=False)
        btab.assign(score="BoScore", config=args.label).to_csv(
            args.out_csv.replace(".csv", "_boscore.csv"), index=False)
        cell[["source", "AggScore", "BoScore_stage1", "total_return", "excess"]].to_csv(
            args.out_csv.replace(".csv", "_cell.csv"), index=False)
        print("\nwrote %s" % args.out_csv)
    print(bar, flush=True)


if __name__ == "__main__":
    main()
