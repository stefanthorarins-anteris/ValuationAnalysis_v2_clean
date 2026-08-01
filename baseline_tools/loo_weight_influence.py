"""LEAVE-ONE-OUT REALISED-RANKING INFLUENCE of every non-zero Stage-2 weight.

WHAT THIS MEASURES, AND WHY IT IS NOT A VARIANCE SHARE
-----------------------------------------------------
design/metric-rationale.md Part 1.1 ranks the Stage-2 metrics by their share of
AggScore VARIANCE, cov(w_i z_i, Agg)/var(Agg).  That quantity is dominated by
rho(z_i, Agg-minus-i) and can be NEGATIVE, so it is not a share of influence.  This
script measures the thing the CEO actually reviews instead: for each metric, zero its
weight, RENORMALISE the survivors to sum|w| = 1, re-score through the DEPLOYED code
path, and report how the REALISED ranking moves.

READ-ONLY.  No pipeline file is modified, no weight is changed on any shipped path, no
network call is made.  Every weight change lives in a local dict handed to the deployed
functions.

THE SCORING PATH USED (and how it is verified)
----------------------------------------------
Stage-2 weights enter the pipeline at exactly one place: postBoRank.postBoScoreRanking
lines 106-119.  Upstream of that, the 100-name pool is Stage-1 BoScore
(postBo.py:190  BoS_dftop100 = general_scores.head(100)) and is weight-INDEPENDENT;
the per-ticker raw metrics are weight-independent too.  So this script replays the
DEPLOYED functions from the saved raw metric matrix onward:

    postBoRank.normalizeAndDropNA(raw, weight_series=W)   # winsorize + z-score
      -> z * W                                            # the pipeline's own multiply
      -> postBoRank.getAggScore
      -> postBoRank.getRankOfRanks
      -> postBoRank._dedup_issuers_in_ranking

verify_baseline() asserts this reproduces resdic[postRank] EXACTLY (identical source
order, |dAggScore| = 0) before any perturbation is applied.  It aborts otherwise.

PRE- OR POST-WINSORIZATION?
---------------------------
POST, i.e. the winsorizer is re-run inside every arm, because that is what the deployed
path does: normalizeAndDropNA is called with the weight_series, and a w=0 column is
EXEMPT from winsorization (postBoRank.py:838-839).  Re-running it means the exemption
lands exactly as it would in production.  It is also provably harmless here: the
winsorizer clips column-by-column on the RAW column, so zeroing metric k changes only
column k clipping, and column k is then multiplied by 0.  CycleHeat / Piotroski /
marketCapRevQuants are winsor-EXEMPT already (postBoRank.py:524), so the asymmetry the
brief warns about is present in the BASELINE too and is not introduced by the test.

THE THREE NULLS
---------------
F1  negligible-weight-perturbation: scale one weight by (1 +- eps), eps in {1%, 5%},
    renormalise, re-rank.  "What does a change nobody would argue about do?"
F2  matched-magnitude random-direction (the PRIMARY floor): perturb AggScore by
    -|w_k| * e, e ~ N(0,1) standardised to sd 1 across the pool -- the same magnitude as
    removing metric k, but pointing in a random direction.  R draws.  If a metric
    realised churn sits INSIDE this distribution its displacement is a generic
    magnitude effect; BELOW it the metric is aligned with the rest of the score; ABOVE
    it, the metric reorders more than a random column of its size.
F3  pool jackknife: drop one pool NAME, re-z-score over the 99, re-rank.  Measures the
    top-20 boundary intrinsic fragility under a change that carries no weight
    information at all.

POPULATION LABELS.  Everything here is [pool] = the deployed general top-100 of the
2026-07-17 CORRECTED panel (n=100).  Nothing in this file is a [universe] or [panel]
number.  The four limits stamped on run_corrected_current.py apply unchanged: this is
the 07-17 universe under OLD acquisition gates, currency/frequency are fallbacks, the mu
weights were fitted PRE-winsorizer, and the panel rebuild has known tie/window gaps.
"""

import argparse
import os
import sys

os.environ.setdefault("VA_OFFLINE_NO_DCF", "1")     # BEFORE postBoRank is imported

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import createDicts as cdic
import postBoRank as pbr

DEFAULT_RESDIC = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")

CANDIDATES = ["Piotroski", "revenueGrowth", "Altman-Z", "grossProfitMargin",
              "incomeQuality", "CycleHeat"]


# --------------------------------------------------------------------------- #
#  weights                                                                    #
# --------------------------------------------------------------------------- #
def deployed_weights():
    a, b = cdic.getPostDict()
    return {**{k: float(a[k]["w"]) for k in a}, **{k: float(b[k]["w"]) for k in b}}


def renormalise(W):
    """Scale every weight so sum|w| = 1.  Sign preserved."""
    s = sum(abs(v) for v in W.values())
    if s == 0:
        raise ValueError("all weights zero")
    return {k: v / s for k, v in W.items()}


def loo_weights(W, metric, renorm=True):
    Wn = dict(W)
    Wn[metric] = 0.0
    return renormalise(Wn) if renorm else Wn


# --------------------------------------------------------------------------- #
#  the deployed scoring replay                                                #
# --------------------------------------------------------------------------- #
def score(raw, W, cdxtop, names, quiet=True):
    """DEPLOYED Stage-2 path from the raw metric matrix to the emitted ranking."""
    if quiet:
        import io
        _old, sys.stdout = sys.stdout, io.StringIO()
    try:
        psm, outlierlist = pbr.normalizeAndDropNA(raw.copy(), weight_series=W)
        wt = psm.drop("source", axis=1)
        for col in wt.columns:
            wt[col] = psm[col].values * W.get(col, 1)
        wn = pd.concat([psm[psm.columns.difference(wt.columns)], wt], axis=1)
        postRank = pbr.getAggScore(wn)
        postRank = pbr.getRankOfRanks(postRank)
        postRank, dupes = pbr._dedup_issuers_in_ranking(postRank, cdxtop, names, True)
    finally:
        if quiet:
            sys.stdout = _old
    return postRank.reset_index(drop=True)


def verify_baseline(resdic, raw, W, cdxtop, names):
    """HARD GATE: the replay must reproduce the deployed postRank exactly."""
    dep = resdic["postRank"].reset_index(drop=True)
    got = score(raw, W, cdxtop, names)
    same_order = list(dep["source"]) == list(got["source"])
    dmax = float(np.abs(dep["AggScore"].to_numpy(float)
                        - got["AggScore"].to_numpy(float)).max())
    dep_z = resdic["postScoreMetric"].set_index("source")
    zcols = list(dep_z.columns)
    import io
    _old, sys.stdout = sys.stdout, io.StringIO()
    try:
        psm, _ = pbr.normalizeAndDropNA(raw.copy(), weight_series=W)
    finally:
        sys.stdout = _old
    zmax = float((dep_z[zcols].astype(float)
                  - psm.set_index("source").loc[dep_z.index, zcols].astype(float))
                 .abs().max().max())
    print("BASELINE REPRODUCTION GATE")
    print("  source order identical to deployed postRank : %s" % same_order)
    print("  max abs dAggScore                           : %.3e" % dmax)
    print("  max abs dz vs deployed postScoreMetric      : %.3e" % zmax)
    ok = same_order and dmax < 1e-12 and zmax < 1e-12
    print("  VERDICT: %s" % ("EXACT -- proceed" if ok else "MISMATCH -- ABORT"))
    if not ok:
        raise SystemExit("baseline reproduction failed; the replay is not the deployed "
                         "path. Reported as a finding, no perturbation run.")
    return got


# --------------------------------------------------------------------------- #
#  displacement metrics                                                       #
# --------------------------------------------------------------------------- #
def spearman(a, b):
    ra = pd.Series(list(a)).rank()
    rb = pd.Series(list(b)).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(base, new, top=(5, 20)):
    """Realised-ranking displacement of new against base."""
    bs, ns = list(base["source"]), list(new["source"])
    br = {s: i + 1 for i, s in enumerate(bs)}
    nr = {s: i + 1 for i, s in enumerate(ns)}
    common = [s for s in bs if s in nr]
    d = np.array([abs(br[s] - nr[s]) for s in common], dtype=float)
    out = {"n_common": len(common),
           "max_abs_disp": float(d.max()) if len(d) else np.nan,
           "median_abs_disp": float(np.median(d)) if len(d) else np.nan,
           "mean_abs_disp": float(d.mean()) if len(d) else np.nan,
           "spearman": spearman([br[s] for s in common], [nr[s] for s in common])}
    for n in top:
        b_set, n_set = set(bs[:n]), set(ns[:n])
        out["top%d_changed" % n] = len(b_set - n_set)
        out["top%d_in" % n] = sorted(n_set - b_set)
        out["top%d_out" % n] = sorted(b_set - n_set)
    return out


def score_gaps(postRank):
    """AggScore landmarks [pool]: the median-to-top-20 distance the brief asks us to
    verify rather than assume."""
    a = postRank["AggScore"].to_numpy(float)
    n = len(a)
    med = float(np.median(a))
    return {"agg_rank1": float(a[0]), "agg_rank5": float(a[4]),
            "agg_rank20": float(a[19]), "agg_rank21": float(a[20]),
            "agg_median": med, "median_to_top20": float(a[19] - med),
            "median_to_top5": float(a[4] - med),
            "agg_sd": float(a.std(ddof=1)), "n": n}


def variance_shares(resdic, W):
    """The document own quantity, recomputed here so the comparison column is not taken
    on trust: cov(w_i z_i, AggScore) / var(AggScore) [pool]."""
    wn = resdic["psmdf_normalized"]
    agg = wn[[c for c in wn.columns if c in W]].sum(axis=1).to_numpy(float)
    v = agg.var(ddof=1)
    out = {}
    for k in W:
        if W[k] == 0 or k not in wn.columns:
            continue
        col = wn[k].to_numpy(float)
        out[k] = float(np.cov(col, agg, ddof=1)[0, 1] / v)
    return out


# --------------------------------------------------------------------------- #
#  nulls                                                                      #
# --------------------------------------------------------------------------- #
def null_random_direction(base, W, mags, R=2000, seed=20260731):
    """F2 -- matched-magnitude, random-direction perturbation of AggScore.

    Removing metric k subtracts w_k * z_k, a vector of per-name sd |w_k| (z is unit
    variance over the pool by construction).  Here we subtract |w| * e instead, with e a
    standardised Gaussian draw -- SAME magnitude, random direction -- and renormalise the
    result by 1/(1-|w|) exactly as the leave-one-out arm does (rank-neutral either way).
    """
    rng = np.random.default_rng(seed)
    bs = list(base["source"])
    a0 = base["AggScore"].to_numpy(float)
    n = len(bs)
    res = {}
    for mag in mags:
        rec = []
        for _ in range(R):
            e = rng.standard_normal(n)
            e = (e - e.mean()) / e.std(ddof=1)
            a = (a0 - abs(mag) * e) / (1.0 - abs(mag))
            order = np.argsort(-a, kind="stable")
            ns = [bs[i] for i in order]
            nr = {s: i + 1 for i, s in enumerate(ns)}
            d = np.array([abs(i + 1 - nr[s]) for i, s in enumerate(bs)], float)
            rec.append({"top20_changed": len(set(bs[:20]) - set(ns[:20])),
                        "top5_changed": len(set(bs[:5]) - set(ns[:5])),
                        "max_abs_disp": d.max(),
                        "median_abs_disp": float(np.median(d)),
                        "spearman": spearman(range(1, n + 1), [nr[s] for s in bs])})
        res[mag] = pd.DataFrame(rec)
    return res


def null_weight_jitter(base, raw, W, cdxtop, names, eps_list=(0.01, 0.05)):
    """F1 -- scale ONE weight by (1 +- eps), renormalise, re-rank."""
    rows = []
    for k, w in W.items():
        if w == 0:
            continue
        for eps in eps_list:
            for sgn in (+1, -1):
                Wj = dict(W)
                Wj[k] = w * (1.0 + sgn * eps)
                pr = score(raw, renormalise(Wj), cdxtop, names)
                c = compare(base, pr)
                rows.append({"metric": k, "eps": sgn * eps,
                             "top20_changed": c["top20_changed"],
                             "top5_changed": c["top5_changed"],
                             "max_abs_disp": c["max_abs_disp"],
                             "median_abs_disp": c["median_abs_disp"],
                             "spearman": c["spearman"]})
    return pd.DataFrame(rows)


def null_pool_jackknife(base, raw, W, cdxtop, names):
    """F3 -- drop one pool NAME, re-z-score over the remaining 99, re-rank.  Churn is
    counted among the SURVIVING names only (the dropped name own slot is excluded, so a
    mechanical -1 is not mistaken for churn)."""
    bs = list(base["source"])
    rows = []
    for drop in bs:
        sub = raw[raw["source"] != drop].copy()
        pr = score(sub, W, cdxtop, names)
        ns = list(pr["source"])
        nr = {s: i + 1 for i, s in enumerate(ns)}
        surv = [s for s in bs if s != drop]
        br = {s: i + 1 for i, s in enumerate(surv)}        # re-indexed baseline
        d = np.array([abs(br[s] - nr[s]) for s in surv], float)
        rows.append({"dropped": drop,
                     "top20_changed": len(set(surv[:20]) - set(ns[:20])),
                     "top5_changed": len(set(surv[:5]) - set(ns[:5])),
                     "max_abs_disp": d.max(),
                     "median_abs_disp": float(np.median(d)),
                     "spearman": spearman([br[s] for s in surv],
                                          [nr[s] for s in surv])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdic", default=DEFAULT_RESDIC)
    ap.add_argument("--outdir", default=_HERE)
    ap.add_argument("--tag", default="loo-2026-07-31")
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--skip-jackknife", action="store_true")
    args = ap.parse_args()

    bar = "=" * 100
    resdic = pd.read_pickle(args.resdic)
    raw = resdic["postScoreMetric_raw"].copy()
    cdxtop = resdic["cdx_dftop100"]
    tdf = resdic.get("Tickers_df")
    names = (dict(zip(tdf["symbol"], tdf["name"]))
             if tdf is not None and "symbol" in getattr(tdf, "columns", []) else {})
    W = deployed_weights()

    print(bar)
    print("LEAVE-ONE-OUT WEIGHT INFLUENCE -- [pool] deployed general top-100, "
          "2026-07-17 CORRECTED")
    print("  weights: createDicts.getPostDict  sum abs w = %.4f  (%d non-zero of %d)"
          % (sum(abs(v) for v in W.values()),
             sum(1 for v in W.values() if v != 0), len(W)))
    print(bar)

    base = verify_baseline(resdic, raw, W, cdxtop, names)

    g = score_gaps(base)
    print("")
    print("AGGSCORE LANDMARKS [pool, n=%d]" % g["n"])
    for k in ("agg_rank1", "agg_rank5", "agg_rank20", "agg_rank21", "agg_median",
              "median_to_top20", "median_to_top5", "agg_sd"):
        print("  %-18s %+.4f" % (k, g[k]))

    vs = variance_shares(resdic, W)

    print("")
    print("RENORMALISATION RANK-NEUTRALITY CHECK")
    for k in CANDIDATES[:3]:
        a = score(raw, loo_weights(W, k, True), cdxtop, names)
        b = score(raw, loo_weights(W, k, False), cdxtop, names)
        print("  %-20s renorm vs no-renorm order identical: %s"
              % (k, list(a["source"]) == list(b["source"])))

    nz = [k for k, v in W.items() if v != 0]
    order = CANDIDATES + [k for k in sorted(nz, key=lambda x: -abs(W[x]))
                          if k not in CANDIDATES]
    rows = []
    for k in order:
        Wl = loo_weights(W, k, True)
        pr = score(raw, Wl, cdxtop, names)
        c = compare(base, pr)
        rows.append({"metric": k, "w_nominal": W[k],
                     "survivor_rescale": 1.0 / (1.0 - abs(W[k])),
                     "per_sigma_frac_of_med_to_top20":
                         abs(W[k]) / g["median_to_top20"],
                     "var_share": vs.get(k, np.nan),
                     "top20_changed": c["top20_changed"],
                     "top5_changed": c["top5_changed"],
                     "max_abs_disp": c["max_abs_disp"],
                     "median_abs_disp": c["median_abs_disp"],
                     "mean_abs_disp": c["mean_abs_disp"],
                     "spearman": c["spearman"],
                     "top20_in": " ".join(c["top20_in"]),
                     "top20_out": " ".join(c["top20_out"]),
                     "top5_in": " ".join(c["top5_in"]),
                     "top5_out": " ".join(c["top5_out"])})
    loo = pd.DataFrame(rows)

    print("")
    print(bar)
    print("LEAVE-ONE-OUT -- REALISED DISPLACEMENT [pool]")
    print(bar)
    show = ["metric", "w_nominal", "var_share", "top20_changed", "top5_changed",
            "max_abs_disp", "median_abs_disp", "mean_abs_disp", "spearman",
            "per_sigma_frac_of_med_to_top20"]
    print(loo[show].to_string(index=False, float_format=lambda v: "%+.4f" % v))
    print("")
    print("NAMES MOVING IN/OUT OF THE TOP-20 (and top-5)")
    for _, r in loo.iterrows():
        print("  %-28s t20 in[%s] out[%s]   t5 in[%s] out[%s]"
              % (r["metric"], r["top20_in"], r["top20_out"],
                 r["top5_in"], r["top5_out"]))

    mags = sorted({round(abs(W[k]), 6) for k in nz})
    f2 = null_random_direction(base, W, mags, R=args.R)
    f2sum = pd.DataFrame([{"abs_w": m,
                           "t20_p05": float(np.percentile(d["top20_changed"], 5)),
                           "t20_median": float(np.median(d["top20_changed"])),
                           "t20_mean": float(d["top20_changed"].mean()),
                           "t20_p95": float(np.percentile(d["top20_changed"], 95)),
                           "t5_median": float(np.median(d["top5_changed"])),
                           "t5_p95": float(np.percentile(d["top5_changed"], 95)),
                           "max_disp_median": float(np.median(d["max_abs_disp"])),
                           "med_disp_median": float(np.median(d["median_abs_disp"])),
                           "spearman_median": float(np.median(d["spearman"]))}
                          for m, d in f2.items()])
    print("")
    print(bar)
    print("F2 NULL -- matched-magnitude RANDOM-DIRECTION perturbation (R=%d) [pool]"
          % args.R)
    print(bar)
    print(f2sum.to_string(index=False, float_format=lambda v: "%.3f" % v))

    f1 = null_weight_jitter(base, raw, W, cdxtop, names)
    print("")
    print(bar)
    print("F1 NULL -- one weight scaled by (1 +- eps), renormalised [pool]")
    print(bar)
    print(f1.groupby("eps")[["top20_changed", "top5_changed", "max_abs_disp",
                             "median_abs_disp", "spearman"]]
          .agg(["mean", "max"]).to_string(float_format=lambda v: "%.3f" % v))
    print("")
    print("  rows with ANY top-20 churn:")
    nzj = f1[f1["top20_changed"] > 0]
    print("    " + (nzj.to_string(index=False) if len(nzj) else "(none)"))
    print("  rows with ANY rank movement (max_abs_disp > 0): %d of %d"
          % (int((f1["max_abs_disp"] > 0).sum()), len(f1)))

    f3 = pd.DataFrame()
    if not args.skip_jackknife:
        f3 = null_pool_jackknife(base, raw, W, cdxtop, names)
        print("")
        print(bar)
        print("F3 NULL -- pool jackknife (drop 1 name, re-z-score over 99) [pool]")
        print(bar)
        print(f3[["top20_changed", "top5_changed", "max_abs_disp",
                  "median_abs_disp", "spearman"]]
              .describe(percentiles=[.5, .9, .95]).to_string(
                  float_format=lambda v: "%.3f" % v))

    pv = []
    for _, r in loo.iterrows():
        m = round(abs(r["w_nominal"]), 6)
        d = f2[m]
        pv.append({"metric": r["metric"], "abs_w": m,
                   "realised_t20": r["top20_changed"],
                   "null_t20_median": float(np.median(d["top20_changed"])),
                   "null_t20_p95": float(np.percentile(d["top20_changed"], 95)),
                   "null_frac_below_realised":
                       float((d["top20_changed"] < r["top20_changed"]).mean()),
                   "null_frac_at_or_above_realised":
                       float((d["top20_changed"] >= r["top20_changed"]).mean()),
                   "realised_spearman": r["spearman"],
                   "null_spearman_median": float(np.median(d["spearman"])),
                   "realised_max_disp": r["max_abs_disp"],
                   "null_max_disp_median": float(np.median(d["max_abs_disp"]))})
    pvdf = pd.DataFrame(pv)
    print("")
    print(bar)
    print("REALISED vs F2 NULL  (same abs w, random direction) [pool]")
    print(bar)
    print(pvdf.to_string(index=False, float_format=lambda v: "%.3f" % v))

    t = args.tag
    loo.to_csv(os.path.join(args.outdir, "loo_influence-%s.csv" % t), index=False)
    f2sum.to_csv(os.path.join(args.outdir, "loo_null_f2-%s.csv" % t), index=False)
    f1.to_csv(os.path.join(args.outdir, "loo_null_f1-%s.csv" % t), index=False)
    if len(f3):
        f3.to_csv(os.path.join(args.outdir, "loo_null_f3-%s.csv" % t), index=False)
    pvdf.to_csv(os.path.join(args.outdir, "loo_vs_null-%s.csv" % t), index=False)
    pd.DataFrame([g]).to_csv(os.path.join(args.outdir, "loo_landmarks-%s.csv" % t),
                             index=False)
    print("")
    print("wrote loo_influence / loo_null_f1 / loo_null_f2 / loo_null_f3 / "
          "loo_vs_null / loo_landmarks -%s.csv" % t)


if __name__ == "__main__":
    main()
