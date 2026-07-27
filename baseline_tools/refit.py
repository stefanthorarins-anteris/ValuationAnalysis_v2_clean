"""TURNKEY WEIGHT RE-FIT with a PRE-REGISTERED acceptance test.

POST-FETCH USAGE -- one command, no editing:

    python baseline_tools/refit.py --panel <new_panel.pickle>

Everything else has a default.  The panel's metric basis is auto-detected and an OLD-basis
panel is REFUSED, so a post-fetch run cannot silently mix generations.  Every run writes a
dated run-log artifact that a reader who was not present can audit.

WHY 4 PARAMETERS AND NOT 21
---------------------------
There are ~1.3 independent 36-month windows and ~40-80 top-20 outcomes in this data.  Fitting
21 free weights against that interpolates noise: any move the data "supports" sits inside its
own standard error.  So the fit is over ONE MULTIPLIER PER FACTOR CLUSTER, sum-constrained --
4 fitted numbers, 3 effective degrees of freedom.

CLUSTERS (from the deployed mu vector; see CLUSTERS below for the exact membership)
  cheapness            earnYield + grahamNumberToPrice + bVpRatio + tbVpRatio   (0.1595)
  profitability        RoA + returnOnEquity + returnOnCapitalEmployed           (0.1500)
  eps_mean_reversion   EPStoEPSmean                                             (0.0560)
  gross_margin         grossProfitMargin                                        (0.1000)
CycleHeat is deliberately NOT in a cluster -- see the refuse list.

WHAT IS REFUSED, AND WHY (standing list -- do not quietly fit these)
-------------------------------------------------------------------
  marketCapRevQuants (w 0.080)  The training anchors are SURVIVORSHIP-SELECTED against
      exactly the small / short-history names this metric exists to reward: a name that was
      small in 2018 and died before the eval leg is absent from the panel entirely.  Fitting
      a size reward on a sample that deleted the size cohort's failures is fitting the
      selection, not the effect.
  Altman-Z (w 0.062)  66.5% of our Altman is the single 0.6*MVE/TL term, correlation +0.997
      with it, and that term anti-correlates with cheapness BY CONSTRUCTION.  The quantity is
      MISLABELLED relative to what the name implies; fitting a weight onto a mislabelled
      quantity assigns it importance the label cannot bear.
  CycleHeat, SIGN AND MAGNITUDE (w -0.080)  A LATE-CYCLE PENALTY estimated on a single
      2021-25 supercycle -- the worst possible window for it.  The sign is theory-derived and
      stays negative, and because the stated reason is that the MAGNITUDE is unidentifiable
      on this sample, the magnitude is frozen too.  It is therefore held OUTSIDE every fitted
      cluster, so no cluster multiplier can scale it.  (It used to sit inside
      `mean_reversion`, which scaled it while the label claimed it was frozen -- code and
      label disagreed and the label was the one worth keeping.  Resolved 2026-07-27; asserted
      in main() and tested.)
  ALL SIGNS, everywhere.  Theory-derived and audit-verified; the optimiser cannot flip one.

REGULARISATION
  L2 toward the prior (the prior is the shipped mu vector), plus a ONE-STANDARD-ERROR
  parsimony tie-break: among candidates within 1 SE of the best score, take the one CLOSEST
  TO THE PRIOR.  No L1 -- there are no zeroing decisions left at cluster level, and L1 on 4
  parameters would only add a knob with no decision behind it.
  THE SE IS sd(per-anchor scores) / sqrt(n_anchors).  It was briefly computed as the sd of
  JACKKNIFE MEANS, which equals sd/(n-1) -- 3x too narrow at n=4 -- and that silently
  DISABLED this safeguard: the band excluded the prior and the pick drifted off it. Fixed
  2026-07-27; the corrected band holds every in-box candidate on the 07-17 dry run and the
  pick is exactly the prior.

VALIDATION
  GENUINE leave-one-window-out: for each fold the WHOLE selection is re-run on the other
  anchors and the resulting vector is scored on the anchor that fold never saw.  An earlier
  version scored each candidate on the fold's own TRAINING anchors, which is algebraically
  the in-sample mean (max deviation 5.6e-17) -- it produced no out-of-sample number at all.
  The holdout cell is in NO fold, and that is enforced by a test.

FIT ON THE DEPLOYED CONFIGURATION
  Stage-1 -> cohort carve + $25M floor -> general top-100 -> Stage-2 re-normalised OVER THAT
  POOL -> issuer-dedup -> top-N.  Not universe-wide: the measured behaviour of the two
  differs, and the weights only ever operate inside the pool.
  Normaliser = the CURRENT WINSORIZED Z.  Re-matching the weights to the scorer that actually
  ships is the entire point: the mu vector was fitted 2026-07-14 (38621fd) when
  normalizeAndDropNA was un-winsorized z + |z|>4 ejection, and `_winsorize_raw` landed
  2026-07-25 (69c3671).  Rank-normalisation is NOT fitted against -- it was measured worse on
  completeness-and-size-independent information at 3 of 3 anchors.

EXACTNESS NOTE (this is what makes the fit fast enough to be turnkey)
  Stage-1, the carve, the top-100 cut and the Stage-2 normalisation are ALL independent of
  the cluster multipliers: no multiplier can zero a column (boxes are [0.75, 1.33]), so the
  winsorization exemption set -- the only way weights touch normalisation -- is identical for
  every candidate.  So the pool, its normalised frame and its issuer groupings are computed
  ONCE per anchor and only the weight-sum / re-rank / dedup-survivor step runs per candidate.
  This is an exact refactor of the deployed path, not an approximation.

HELD-OUT CELL
  2022-12-30 -> 2025-12-31 is excluded from training and from EVERY CV fold.  It is the only
  wide cell this project has and it is the test set.

ALL FIGURES ON WHATEVER UNIVERSE THE PANEL CARRIES.  On the 07-17 panel that is the OLD
GATES' universe (~523 pricefails ~72% non-US, plus the lenfail 16->8 cohort never fetched).
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

os.environ.setdefault("VA_OFFLINE_NO_DCF", "1")     # BEFORE postBoRank import

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calcScore as cs
import carveOut as co
import createDicts as cdic
import data_quality as dq
import postBoRank as pbr
import reporting_period as rp
import returns_core as rc
import run_target_test as rtt
import stage2_pit as s2
import decile_test as dt
import confound_controls as cc
import attribution_arms as aa

# --------------------------------------------------------------------------- #
#  PRE-REGISTRATION -- written BEFORE the first run, read by the code         #
# --------------------------------------------------------------------------- #
# The acceptance rule is DATA-INDEPENDENT and is read from here, never recomputed from the
# result.  `noise_floor_pp` is the standard error of a top-20 beat-rate at p ~ 0.40:
#     SE = sqrt(0.40 * 0.60 / 20) = 0.1095 -> 10.95pp, quoted as 11pp.
# A re-fit is ADOPTED only if it beats the shipped prior on the HELD-OUT cell by MORE than
# that.  Anything smaller is inside the measurement's own noise and is not evidence.
#
# EXPECTED OUTCOME: FAILURE.  Prior work puts the re-weighting-alone ceiling at 35-45% and
# every deployed variant measured so far sits at 37-45%, all overlapping.  A pre-registered
# failure IS a result: it says the weights are not the binding constraint, and it must be
# reported as such -- never re-specified into a pass.
PREREG = {
    "registered_on": "2026-07-27",
    # STATED EXACTLY (review, 2026-07-27): the count is over the PRICED members of the
    # deployed pool, because a name with no price at both legs has no outcome to score.  At
    # the thin early anchors the pool has 100 members but only ~51 priced, so "top-20" is
    # top-20 OF THE PRICED SUBSET and must not be read as top-20 of the pool.  The
    # min_priced_per_anchor guard below is what stops that subset getting so small that the
    # objective is meaningless.
    "primary_metric": "deployed top-20 OF PRICED POOL MEMBERS: beat-rate vs URTH at "
                      ">= +10pp over 36 months",
    "secondary_metric": "deployed-pool rank-IC vs realized excess return (priced members)",
    "min_priced_per_anchor": 40,
    "holdout_cell": {"buy": "2022-12-30", "eval": "2025-12-31"},
    "train_anchors": [("2018-12-31", "2021-12-31"), ("2019-12-31", "2022-12-30"),
                      ("2020-12-31", "2023-12-29"), ("2021-12-31", "2024-12-31")],
    "cv": "GENUINE leave-one-window-out: per fold, re-run the whole selection on the other "
          "anchors and score the resulting vector on the held-out anchor. The holdout cell "
          "is in NO fold.",
    "one_se_definition": "sd(per-anchor scores) / sqrt(n_anchors) -- NOT the sd of jackknife "
                         "means, which is sd/(n-1) and 3x too narrow at n=4",
    "noise_floor_pp": 10.95,
    "adopt_if": "holdout_beat_rate(refit) - holdout_beat_rate(prior) > noise_floor_pp",
    "box": [0.75, 1.3333],
    "box_binds_on": "REALISED per-metric multipliers (after the sum-constraint "
                    "renormalisation), not the nominal grid values",
    "regularisation": "L2 toward prior; one-SE parsimony tie-break toward prior; no L1",
    "signs": "FIXED (theory-derived, audit-verified) -- the optimiser cannot flip one",
    "expected_outcome": "FAILURE (re-weighting-alone ceiling 35-45%)",
    # The prior is displaced only if the best candidate's advantage exceeds BOTH the sampling
    # SE and a SELECTION-NOISE FLOOR calibrated by shuffling the outcomes.  The floor is a
    # PERCENTILE, so it carries an explicit false-adoption rate: at q=99 about 1% of pure-noise
    # runs would still displace the prior.  99 rather than 95 because the costs are asymmetric
    # -- a false adoption ships a wrong weight vector, a false retention only keeps today's.
    "selection_noise_gate_percentile": 99.0,
    # 200, NOT 20.  At n_perm=20 `percentile(.., 99)` interpolates against the top of 20
    # draws and is a badly biased estimator of the true 99th: measured against a
    # 400-shuffle reference on the real anchors it landed on the ~95th, i.e. it DELIVERED
    # the percentile this pre-registration explicitly rejected on its asymmetric-cost
    # argument, with a ~70% spread about its own median.  The delivered percentile is
    # re-measured by baseline_tools/test_refit.py and reported in the runbook; if it ever
    # drifts from the registered value again, raise n_perm or re-register honestly -- do
    # NOT leave the label and the behaviour disagreeing.
    "selection_noise_permutations": 200,
    "implied_false_adoption_rate": "~1% on pure noise IF the delivered percentile matches "
                                   "the registered one -- verified, not assumed",
}

# CycleHeat is NOT in a cluster (review, 2026-07-27).  It used to sit inside
# `mean_reversion`, so the cluster multiplier scaled its MAGNITUDE while the refuse-to-fit
# key claimed sign AND magnitude were frozen -- the label and the code disagreed, and the
# stated rationale (the magnitude is unidentifiable on a single supercycle) argued for the
# code being wrong, not the label.  Resolved in favour of the rationale: CycleHeat is now
# fully frozen, and the remaining cluster is EPStoEPSmean alone and named for it.
CLUSTERS = {
    "cheapness": ["earnYield", "grahamNumberToPrice", "bVpRatio", "tbVpRatio"],
    "profitability": ["RoA", "returnOnEquity", "returnOnCapitalEmployed"],
    "eps_mean_reversion": ["EPStoEPSmean"],
    "gross_margin": ["grossProfitMargin"],
}
REFUSE_TO_FIT = {
    "marketCapRevQuants": "training anchors are survivorship-selected against the small / "
                          "short-history names this metric exists to reward",
    "Altman-Z": "66.5% one term (corr +0.997) that anti-correlates with cheapness by "
                "construction -- a mislabelled quantity",
    "CycleHeat": "late-cycle penalty estimated on one 2021-25 supercycle, so BOTH its sign "
                 "(theory-derived, stays negative) and its MAGNITUDE are frozen; it is "
                 "deliberately outside every fitted cluster so nothing scales it",
}
BOX_LO, BOX_HI = PREREG["box"]
GRID = [0.75, 0.875, 1.0, 1.15, 1.3333]      # per-cluster candidate multipliers


def prior_weights():
    a, b = cdic.getPostDict()
    return {**{k: float(a[k]["w"]) for k in a}, **{k: float(b[k]["w"]) for k in b}}


def apply_multipliers(prior, mult):
    """Scale each fitted cluster, then renormalise the FITTED BUDGET back to its prior total.

    This is the sum-to-1 constraint, implemented so it cannot leak into the frozen weights:
    the fitted clusters keep their combined budget exactly, every other weight is untouched,
    and the total |w| is therefore unchanged.  It also means the 4 multipliers carry 3
    effective degrees of freedom, which is stated in the log.
    """
    w = dict(prior)
    fitted = [m for c in CLUSTERS.values() for m in c]
    budget = sum(abs(prior[m]) for m in fitted)
    for cname, members in CLUSTERS.items():
        for m in members:
            w[m] = prior[m] * float(mult[cname])
    scaled = sum(abs(w[m]) for m in fitted)
    if scaled > 0:
        k = budget / scaled
        for m in fitted:
            w[m] *= k
    # SIGN GUARD -- non-negotiable, and cheap enough to assert every single evaluation.
    for m, v in w.items():
        assert np.sign(v) == np.sign(prior[m]) or prior[m] == 0, \
            "sign flip on %s: prior %+.4f -> %+.4f" % (m, prior[m], v)
    return w


# --------------------------------------------------------------------------- #
#  Per-anchor pool, computed ONCE (weight-independent)                        #
# --------------------------------------------------------------------------- #
def build_anchor(dmdic, buy, eval_, price_source, topn_pool=100):
    """Deployed pool as-of `buy`: Stage-1 -> carve -> top-100 -> Stage-2 normalised over the
    pool.  Returns everything the per-candidate step needs, plus the realized excess return.
    """
    bm = dmdic["BoMetric_df"].copy(); cdx = dmdic["cdx_df"].copy()
    bm["date"] = pd.to_datetime(bm["date"], errors="coerce")
    cdx["date"] = pd.to_datetime(cdx["date"], errors="coerce")
    D = pd.Timestamp(buy)
    bm_pit = bm[bm["date"] <= D].sort_values(["source", "date"], ascending=[True, False])
    cdx_pit = cdx[cdx["date"] <= D].sort_values(["source", "date"], ascending=[True, False])
    if bm_pit.empty:
        return None

    md = cs.getAves2(bm_pit)
    bs = cs.simpleScore_fromDict(bm_pit, md["BoMetric_ave"], md["BoMetric_dateAve"],
                                 dmdic.get("nrScorePeriods", 8),
                                 freq_map=rp.frequency_by_source(cdx_pit))
    bs = bs.sort_values("score", ascending=False).reset_index(drop=True)

    # NO SILENT FALLBACK (H-BLOCKER 3, 2026-07-27).  The previous version caught a carve
    # failure, printed one line and continued on an UN-CARVED, NON-DEDUPED universe -- a
    # DIFFERENT universe scored under the same filenames, exit 0, nothing in the run-log.
    # The trigger was real: carveOut resolved its sector pickle against the CWD, so merely
    # running from a subdirectory produced it.  A fit on a substituted universe is worthless
    # and indistinguishable from a good one, so refuse.
    try:
        carve = co.partition_universe(bs, cdx_pit, dmdic.get("Tickers_df"),
                                      mcap_floor=25e6, cohort_head=25)
        general = carve["general"]
        carve_ok = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(
            "\n" + "!" * 78 +
            ("\n!!! REFUSING TO FIT: the cohort carve FAILED at anchor %s (%s: %s)."
             % (buy, type(e).__name__, e)) +
            "\n!!! Continuing would score an UN-CARVED, NON-DEDUPED universe -- a different"
            "\n!!! universe under the same outputs. Most likely cause: the sector/industry"
            "\n!!! pickles did not resolve (they live at the repo root). Fix the inputs and"
            "\n!!! re-run; never accept a substituted universe.\n" + "!" * 78)
    if len(general) < 100:
        print("  [%s] WARNING: general pool has only %d names (<100)" % (buy, len(general)))
    pool = general.head(topn_pool).reset_index(drop=True)
    cdxtop = cdx_pit[cdx_pit["source"].isin(set(pool["source"]))].reset_index(drop=True)
    psm = s2._stage2_metric_loop_offline(pool, cdxtop, nq=16)

    prior = prior_weights()
    normed, _drop = pbr.normalizeAndDropNA(psm.copy(), weight_series=prior)

    # Issuer GROUPS are order-independent (fingerprints key on shares/fundamentals); only
    # WHICH member survives depends on rank order.  So group once, pick per candidate.
    tdf = dmdic.get("Tickers_df")
    names = (dict(zip(tdf["symbol"], tdf["name"]))
             if tdf is not None and "name" in getattr(tdf, "columns", []) else {})
    kept, dropped = co.dedup_ranked(list(normed["source"]), cdxtop, names)
    group_of = {}
    for d, surv in (dropped or []):
        group_of[d] = surv
    for s in kept:
        group_of.setdefault(s, s)

    ret = rc.compute_returns(list(normed["source"]), buy, eval_, price_source)
    bench = rc.benchmark_return(price_source, buy, eval_, require_exact=True)
    ret = ret[ret["status"] == "ok"].copy()
    ret["excess"] = ret["total_return"] - bench
    excess = dict(zip(ret["ticker"], ret["excess"]))

    return {"buy": buy, "eval": eval_, "normed": normed, "group_of": group_of,
            "excess": excess, "bench": bench, "n_pool": len(normed),
            "n_priced": len(ret), "carve_ok": carve_ok,
            "n_general_pool": int(len(general))}


def score_anchor(anchor, w, topn=20, threshold=0.10):
    """Deployed top-N beat-rate + pool rank-IC for one weight vector.  Exact re-run of the
    weighting / aggregation / dedup / head(N) steps on the pre-normalised pool."""
    nd = anchor["normed"]
    cols = [c for c in nd.columns if c != "source"]
    M = nd[cols].to_numpy(dtype="float64")
    wv = np.array([w.get(c, 0.0) for c in cols], dtype="float64")
    agg = M @ wv
    order = np.argsort(-agg, kind="stable")
    srcs = nd["source"].to_numpy()[order]

    seen, deduped = set(), []
    for s in srcs:
        g = anchor["group_of"].get(s, s)
        if g in seen:
            continue
        seen.add(g)
        deduped.append(s)

    ex = anchor["excess"]
    top = [s for s in deduped if s in ex][:topn]
    if not top:
        return {"beat_rate": float("nan"), "n_eval": 0, "ic": float("nan")}
    flags = [(ex[s] - 0.0) >= threshold for s in top]
    priced = [(s, ex[s]) for s in deduped if s in ex]
    # pool rank-IC on every priced pool member (secondary metric)
    aggmap = dict(zip(nd["source"], agg))
    ps = pd.Series({s: aggmap[s] for s, _ in priced})
    es = pd.Series({s: v for s, v in priced})
    ic = float(ps.rank().corr(es.rank())) if len(ps) >= 10 else float("nan")
    return {"beat_rate": float(np.mean(flags)), "n_eval": len(top), "ic": ic,
            "top": top, "median_excess": float(np.median([ex[s] for s in top]))}


# --------------------------------------------------------------------------- #
#  Fit                                                                        #
# --------------------------------------------------------------------------- #
def l2_penalty(mult, lam):
    return lam * float(np.sum((np.array([mult[c] for c in CLUSTERS]) - 1.0) ** 2))


def realised_multipliers(prior, mult):
    """Per-metric multiplier ACTUALLY APPLIED to the score, i.e. after the sum-constraint
    renormalisation.  This -- not the nominal grid value -- is what the box must bind on.

    (Defect M1, 2026-07-27: the box was enforced on the NOMINAL grid only, while the
    renormalisation factor k = budget/scaled pushed realised multipliers to 0.6116x-1.5559x
    against a registered [0.75, 1.3333].  A constraint that does not bind on what reaches
    the score is not a constraint.)
    """
    w = apply_multipliers(prior, mult)
    return {m: (w[m] / prior[m]) for c in CLUSTERS.values() for m in c if prior[m] != 0}


def in_box(prior, mult, lo=BOX_LO, hi=BOX_HI, tol=1e-9):
    r = realised_multipliers(prior, mult)
    return all(lo - tol <= v <= hi + tol for v in r.values())


def _cv_score(anchors, w, topn=20):
    """Mean beat-rate over a set of anchors, plus the SE OF THAT MEAN.

    THE SE IS THE SD ACROSS ANCHORS / sqrt(n_anchors).  It is NOT the sd of jackknife
    means: for n anchors those have sd = sd(anchors)/(n-1), i.e. 3x too NARROW at n=4, which
    is exactly the bug that disabled the one-SE parsimony safeguard (H-BLOCKER 1,
    2026-07-27).  Verified by test_refit.test_one_se_is_sd_over_sqrt_n_not_jackknife.
    """
    vals = np.array([score_anchor(a, w, topn)["beat_rate"] for a in anchors], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), vals
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    return float(np.mean(vals)), se, vals


def selection_noise_floor(anchors, prior, lam, topn=20, n_perm=None, seed=0,
                          q=None):
    """How much "improvement" this grid search finds when there is NOTHING to find.

    WHY THIS IS NEEDED, and why a one-SE band alone is not enough.  The pick is the MAXIMUM
    over ~380 in-box candidates, so it is an extreme order statistic: on pure noise the best
    candidate beats the prior by construction, and comparing that maximum against one
    standard error of a SINGLE estimate ignores the multiplicity entirely.  Measured: with
    the corrected sd/sqrt(n) SE and a one-SE band, pure-noise anchors still drifted off the
    prior in 1 of 8 trials (8 of 30 with a median-SE band).  That is selection noise, not
    signal, and no amount of adjusting the SE definition addresses it -- the SE is not the
    quantity that is wrong.

    So calibrate directly: SHUFFLE the outcome across names within each anchor (destroying
    any relationship between score and return while preserving both marginal distributions),
    re-run the WHOLE selection, and record how much the best candidate beats the prior.  The
    q-th percentile of that is the floor a real improvement has to clear.

    Returns (floor, samples).
    """
    if n_perm is None:
        n_perm = int(PREREG["selection_noise_permutations"])
    if q is None:
        q = float(PREREG["selection_noise_gate_percentile"])
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_perm):
        shuffled = []
        for a in anchors:
            keys = list(a["excess"].keys())
            vals = list(a["excess"].values())
            rng.shuffle(vals)
            b = dict(a)
            b["excess"] = dict(zip(keys, vals))
            shuffled.append(b)
        rows = []
        for combo in itertools.product(GRID, repeat=len(CLUSTERS)):
            mult = dict(zip(CLUSTERS, combo))
            if not in_box(prior, mult):
                continue
            m, _se, _v = _cv_score(shuffled, apply_multipliers(prior, mult), topn)
            rows.append(m - l2_penalty(mult, lam))
        pri, _se, _v = _cv_score(shuffled, prior, topn)
        gains.append(float(np.nanmax(rows)) - (pri - 0.0))
    return float(np.nanpercentile(gains, q)), gains


def _select(anchors, prior, lam, topn=20, verbose=False, tag="", noise_floor=None):
    """One complete selection: grid -> L2-penalised objective -> one-SE parsimony pick.

    This is THE PROCEDURE.  It is what the outer leave-one-window-out loop must re-run per
    fold, because what needs validating is the procedure's generalisation, not a fixed
    weight vector's.
    """
    names = list(CLUSTERS)
    rows = []
    for combo in itertools.product(GRID, repeat=len(names)):
        mult = dict(zip(names, combo))
        if not in_box(prior, mult):
            continue                          # M1: the box binds on REALISED multipliers
        w = apply_multipliers(prior, mult)
        mean, se, _vals = _cv_score(anchors, w, topn)
        rows.append({**mult, "score_mean": mean, "score_se": se,
                     "objective": mean - l2_penalty(mult, lam),
                     "dist_from_prior": float(np.sum((np.array(combo) - 1.0) ** 2))})
    g = pd.DataFrame(rows)
    if g.empty:
        raise SystemExit("no grid point satisfies the realised-multiplier box -- the box or "
                         "the grid is misconfigured")
    best = float(g["objective"].max())
    # BAND WIDTH = the MEDIAN per-candidate SE across in-box candidates, not the SE of the
    # winner.  Breiman's one-SE rule canonically uses the winner's own SE, but that assumes
    # the SE is well estimated; here it has 3 degrees of freedom (4 anchors), so a candidate
    # can win partly BY BEING A LOW-VARIANCE FLUKE and then supply a narrow band -- which
    # narrows the safeguard exactly when it is most needed.  Measured: with the winner's SE,
    # pure-noise anchors returned the prior in 7 of 8 trials; with the median SE, 8 of 8.
    # The quantity being guarded against is a property of having only 4 anchors, which is a
    # property of the DATA, not of the candidate.
    se = float(g["score_se"].median())
    if not np.isfinite(se):
        se = 0.0

    # THE PRIOR IS THE DEFAULT.  It is displaced only if the best candidate beats it by more
    # than BOTH the sampling SE and the measured selection-noise floor.
    prior_row = g[np.all([np.isclose(g[c], 1.0) for c in names], axis=0)]
    prior_obj = float(prior_row["objective"].iloc[0]) if len(prior_row) else float("nan")
    gate = max(se, noise_floor if noise_floor is not None else 0.0)
    improvement = best - prior_obj if np.isfinite(prior_obj) else float("inf")

    if np.isfinite(improvement) and improvement <= gate:
        pick_mult = {c: 1.0 for c in names}
        reason = ("improvement %.4f <= gate %.4f (SE %.4f, selection-noise floor %s) "
                  "-> PRIOR RETAINED"
                  % (improvement, gate, se,
                     "n/a" if noise_floor is None else "%.4f" % noise_floor))
        n_within = int((g["objective"] >= best - se).sum())
    else:
        within = g[g["objective"] >= best - se]
        pick = within.sort_values(["dist_from_prior", "objective"],
                                  ascending=[True, False]).iloc[0]
        pick_mult = {c: float(pick[c]) for c in names}
        n_within = len(within)
        reason = ("improvement %.4f > gate %.4f -> parsimony pick within the one-SE band "
                  "(dist %.4f)" % (improvement, gate, pick["dist_from_prior"]))
    if verbose:
        print("  %sgrid=%d in-box of %d | best objective %.4f | prior objective %.4f | "
              "one-SE band -> %d candidates (%.1f%%)"
              % (tag, len(g), len(GRID) ** len(names), best, prior_obj, n_within,
                 100.0 * n_within / len(g)))
        print("  %s%s" % (tag, reason))
    return pick_mult, g, se, n_within


def fit(anchors_train, prior, lam=0.02, verbose=True, topn=20, n_perm=None):
    """FINAL fit on all training anchors (the vector that gets tested on the holdout)."""
    floor, samples = selection_noise_floor(anchors_train, prior, lam, topn, n_perm=n_perm)
    if verbose:
        print("  selection-noise floor (%d outcome shuffles, %.0fth pct of best-minus-prior)"
              ": %.4f   [range %.4f .. %.4f]"
              % (len(samples), PREREG["selection_noise_gate_percentile"], floor,
                 min(samples), max(samples)))
    mult, g, se, n_within = _select(anchors_train, prior, lam, topn, verbose, tag="",
                                    noise_floor=floor)
    return mult, g, se, floor


def lowo(anchors_train, prior, lam=0.02, topn=20, verbose=True, n_perm=None):
    """GENUINE leave-one-window-out: for each fold, re-run the whole selection on the OTHER
    anchors and score the resulting vector on the HELD-OUT anchor.

    The previous implementation scored each candidate on the fold's TRAINING anchors, which
    is algebraically the in-sample mean (max deviation 5.6e-17) -- so it produced no
    out-of-sample number at all and must not have been called CV (H-BLOCKER 2, 2026-07-27).
    """
    out = []
    for held in range(len(anchors_train)):
        tr = [a for i, a in enumerate(anchors_train) if i != held]
        ha = anchors_train[held]
        _fl, _sm = selection_noise_floor(tr, prior, lam, topn, n_perm=n_perm, seed=100 + held)
        m_f, _g, _se, _n = _select(tr, prior, lam, topn, verbose=False, noise_floor=_fl)
        w_f = apply_multipliers(prior, m_f)
        s_fit = score_anchor(ha, w_f, topn)
        s_pri = score_anchor(ha, prior, topn)
        in_fit, _, _ = _cv_score(tr, w_f, topn)
        out.append({
            "fold_held_out": ha["buy"], "n_priced_held_out": ha["n_priced"],
            "n_eval_held_out": s_fit["n_eval"],
            "multipliers_fitted_on_other_folds": m_f,
            "IN_sample_mean_on_other_folds": in_fit,
            "OUT_of_sample_fit": s_fit["beat_rate"],
            "OUT_of_sample_prior": s_pri["beat_rate"],
            "OUT_of_sample_gain_pp": 100.0 * (s_fit["beat_rate"] - s_pri["beat_rate"]),
        })
        if verbose:
            print("  fold %s (n_priced=%d, n_eval=%d): fitted %s on the other 3 -> "
                  "OUT-OF-SAMPLE fit %.4f vs prior %.4f (%+.2fpp) | in-sample on those 3 "
                  "%.4f"
                  % (ha["buy"], ha["n_priced"], s_fit["n_eval"],
                     {k: round(v, 4) for k, v in m_f.items()},
                     s_fit["beat_rate"], s_pri["beat_rate"],
                     out[-1]["OUT_of_sample_gain_pp"], in_fit))
    return out


# --------------------------------------------------------------------------- #
#  Run-log                                                                    #
# --------------------------------------------------------------------------- #
def git_sha():
    """(sha, dirty) -- the dirty flag is not optional: a SHA alone can name a tree that is
    not the tree that ran."""
    try:
        sha = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", _REPO, "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    timeout=20).stdout.strip())
        return (sha or None), dirty
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--anchors", default=None,
                    help="semicolon list buy>eval; default = the pre-registered train set")
    ap.add_argument("--holdout", default=None, help="buy>eval; default = pre-registered")
    ap.add_argument("--lam", default=0.02, type=float)
    ap.add_argument("--topn", default=20, type=int)
    ap.add_argument("--allow-basis-mismatch", action="store_true")
    ap.add_argument("--outdir", default=_HERE)
    args = ap.parse_args()

    def _parse(s):
        return [tuple(p.split(">")) for p in s.split(";")] if s else None

    train = _parse(args.anchors) or [tuple(t) for t in PREREG["train_anchors"]]
    hold = (_parse(args.holdout) or
            [(PREREG["holdout_cell"]["buy"], PREREG["holdout_cell"]["eval"])])[0]

    bar = "=" * 100
    print(bar)
    print("WEIGHT RE-FIT -- turnkey, pre-registered")
    print("  panel        : %s" % args.panel)
    print("  train anchors: %s" % train)
    print("  HELD OUT     : %s (in no fold)" % (hold,))
    print("  PRE-REG bar  : adopt only if holdout beat-rate gain > %.2fpp"
          % PREREG["noise_floor_pp"])
    print("  expected     : %s" % PREREG["expected_outcome"])
    print(bar, flush=True)

    if hold in train:
        raise SystemExit("REFUSING: the holdout %s is in the training anchors." % (hold,))

    p = pd.read_pickle(args.panel)
    dmdic = dict(p)
    # BASIS GUARD -- reuse the shipped detector; refuse an old-basis panel outright.
    rtt.assert_panel_basis(dmdic, args.panel, allow_mismatch=args.allow_basis_mismatch)
    if "BoMetric_ave" in dmdic:
        dmdic.pop("BoMetric_ave"); dmdic.pop("BoMetric_dateAve", None)
    dmdic["nrScorePeriods"] = dmdic.get("nrScorePeriods", 8)

    ps = dt.build_price_source()
    prior = prior_weights()

    print("\nBuilding deployed pools (weight-independent; computed once per anchor)")
    A_train, skipped = [], []
    minp = PREREG["min_priced_per_anchor"]
    for b, e in train:
        a = build_anchor(dmdic, b, e, ps, topn_pool=100)
        if a is None:
            print("  %s -> SKIPPED (no Stage-1 rows as of the anchor)" % b)
            skipped.append({"anchor": b, "reason": "no Stage-1 rows"}); continue
        # PRICE-THINNESS GUARD: an objective computed on a top-20 drawn from a handful of
        # priced names is not a measurement.  Pre-registered, not chosen after seeing this.
        if a["n_priced"] < minp:
            print("  %s -> SKIPPED: only %d priced pool members (< pre-registered minimum "
                  "%d)" % (b, a["n_priced"], minp))
            skipped.append({"anchor": b, "reason": "n_priced=%d < %d" % (a["n_priced"], minp)})
            continue
        print("  %s -> %s : general_pool=%d, scored_pool=%d, PRICED=%d, carve_ok=%s"
              % (b, e, a["n_general_pool"], a["n_pool"], a["n_priced"], a["carve_ok"]))
        A_train.append(a)
    if len(A_train) < 2:
        raise SystemExit("REFUSING: only %d usable training anchor(s) after the price-"
                         "thinness guard; a fit needs at least 2." % len(A_train))
    A_hold = build_anchor(dmdic, hold[0], hold[1], ps, topn_pool=100)
    print("  HOLDOUT %s -> %s : pool=%d, priced=%d"
          % (hold[0], hold[1], A_hold["n_pool"], A_hold["n_priced"]))

    print("\nFitting 4 cluster multipliers (3 effective df; boxes [%.2f, %.4f])"
          % (BOX_LO, BOX_HI))
    mult, grid, se, noise_floor = fit(A_train, prior, lam=args.lam, topn=args.topn)
    w_fit = apply_multipliers(prior, mult)

    print("\n" + bar); print("FITTED CLUSTER MULTIPLIERS vs PRIOR"); print(bar)
    for c in CLUSTERS:
        print("  %-16s x%.4f   members: %s" % (c, mult[c], ", ".join(CLUSTERS[c])))
    print("\n  weight changes (prior -> fitted):")
    for m in [x for c in CLUSTERS.values() for x in c]:
        print("    %-28s %+.4f -> %+.4f  (%+.1f%%)"
              % (m, prior[m], w_fit[m], 100 * (w_fit[m] / prior[m] - 1)))
    # BE EXACT ABOUT WHAT "FROZEN" MEANS PER METRIC -- an earlier version of this line listed
    # CycleHeat as frozen while the mean_reversion multiplier had in fact scaled it, which is
    # precisely the kind of label that gets quoted and is wrong.
    print("  FULLY FROZEN (weight untouched): %s"
          % ", ".join("%s=%+.4f" % (k, prior[k])
                      for k in ("marketCapRevQuants", "Altman-Z", "CycleHeat")))
    assert w_fit["CycleHeat"] == prior["CycleHeat"], "CycleHeat must be FULLY frozen"
    print("  CycleHeat FULLY FROZEN (sign AND magnitude): %+.4f -- outside every fitted "
          "cluster, so nothing scales it." % prior["CycleHeat"])
    print("  realised per-metric multiplier range: %.4f x .. %.4f x  (registered box "
          "[%.2f, %.4f]; the box binds HERE, on what reaches the score)"
          % (min(realised_multipliers(prior, mult).values()),
             max(realised_multipliers(prior, mult).values()), BOX_LO, BOX_HI))

    print("\n" + bar)
    print("GENUINE LEAVE-ONE-WINDOW-OUT (the selection is re-run per fold; the reported")
    print("number is scored on the anchor that fold never saw)")
    print(bar)
    folds = lowo(A_train, prior, lam=args.lam, topn=args.topn, verbose=True)
    oos_fit = float(np.mean([f["OUT_of_sample_fit"] for f in folds]))
    oos_pri = float(np.mean([f["OUT_of_sample_prior"] for f in folds]))
    ins_mean = float(np.mean([f["IN_sample_mean_on_other_folds"] for f in folds]))
    # THE OVERFIT DIAGNOSTIC IS max|IN - OUT| ACROSS FOLDS, not the difference of the means.
    # mean(IN) - mean(OUT) is an ALGEBRAIC IDENTITY equal to 0 whenever every fold retains
    # the prior: with a common score S, mean_h (S - b_h)/3 = S/4 = mean_h b_h.  So the mean
    # version is structurally incapable of firing on exactly the outcome this run produces
    # -- measured 0.0000000000 while the per-fold gaps were 0.0667/0.0667/0.2/0.2.  A
    # diagnostic that cannot fire in the case that occurred is not a diagnostic.
    per_fold_gap = [abs(f["IN_sample_mean_on_other_folds"] - f["OUT_of_sample_fit"])
                    for f in folds]
    max_gap = float(np.max(per_fold_gap))
    print("  MEAN OUT-OF-SAMPLE: fit %.4f vs prior %.4f (%+.2fpp) | mean IN-sample %.4f"
          % (oos_fit, oos_pri, 100 * (oos_fit - oos_pri), ins_mean))
    print("  OVERFIT DIAGNOSTIC (the one that can fire): max |IN - OUT| across folds = "
          "%.4f   per fold %s" % (max_gap, [round(x, 4) for x in per_fold_gap]))
    print("  (mean(IN) - mean(OUT) = %+.10f -- an IDENTITY at 0 when every fold retains the"
          " prior, so it is reported but NOT used as the diagnostic)" % (ins_mean - oos_fit))
    if max_gap > 0.15:
        print("  !! max |IN - OUT| = %.4f exceeds 0.15: the per-fold selections generalise"
              " poorly and the fitted vector should not be trusted even on a PASS."
              % max_gap)

    print("\n" + bar); print("HELD-OUT CELL %s -> %s  (never in any fold)" % hold); print(bar)
    H_fit = score_anchor(A_hold, w_fit, args.topn)
    H_pri = score_anchor(A_hold, prior, args.topn)
    gain_pp = 100.0 * (H_fit["beat_rate"] - H_pri["beat_rate"])
    verdict = "PASS" if gain_pp > PREREG["noise_floor_pp"] else "FAIL"
    print("  top-%d beat-rate  PRIOR %.4f (n=%d)   REFIT %.4f (n=%d)   gain %+.2fpp"
          % (args.topn, H_pri["beat_rate"], H_pri["n_eval"], H_fit["beat_rate"],
             H_fit["n_eval"], gain_pp))
    print("  pool rank-IC     PRIOR %+.4f            REFIT %+.4f" % (H_pri["ic"], H_fit["ic"]))
    print("  median excess    PRIOR %+.4f            REFIT %+.4f"
          % (H_pri["median_excess"], H_fit["median_excess"]))
    print("  PRE-REGISTERED BAR: gain > %.2fpp  ->  ***%s***"
          % (PREREG["noise_floor_pp"], verdict))
    if verdict == "FAIL":
        print("  => THE SHIPPED PRIOR STANDS. This is a pre-registered negative result, not a")
        print("     failed run: it says the weights are not the binding constraint here.")

    # partial-IC controls on the holdout pool, so a size/coverage bet cannot be mistaken
    # for skill (the round-2 lesson, applied by default rather than on request)
    print("\n" + bar); print("HOLDOUT partial-IC controls (n_missing + market-cap decile)")
    print(bar)
    ctrl_rows = []
    try:
        mc = cc.mcap_by_source(args.panel, hold[0])
        nd = A_hold["normed"]
        cols = [c for c in nd.columns if c != "source"]
        M = nd[cols].to_numpy(dtype="float64")
        nm = aa.n_missing_per_name(nd)
        for lab, wv in (("prior", prior), ("refit", w_fit)):
            agg = M @ np.array([wv.get(c, 0.0) for c in cols])
            cell = pd.DataFrame({"source": nd["source"], "AggScore": agg})
            cell["excess"] = cell["source"].map(A_hold["excess"])
            cell = cell.dropna(subset=["excess"]).merge(nm, on="source", how="left")
            cell["mcap"] = cell["source"].map(mc)
            raw, n = cc.partial_ic(cell)
            both, _ = cc.partial_ic(cell, use_nmissing=True, use_mcap_decile=True)
            ctrl_rows.append({"weights": lab, "n": n, "IC_raw": raw,
                              "IC_partial_BOTH": both,
                              "frac_surviving": (both / raw) if raw else np.nan})
        print(pd.DataFrame(ctrl_rows).to_string(index=False,
              float_format=lambda v: "%+.4f" % v))
    except Exception as e:
        print("  controls unavailable: %s: %s" % (type(e).__name__, e))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _sha, _dirty = git_sha()
    log = {
        "run_utc": datetime.utcnow().isoformat() + "Z",
        # a SHA without the dirty flag is meaningless -- the tree it names may not be the
        # tree that ran (H4, 2026-07-27)
        "git_sha": _sha, "git_tree_dirty": _dirty,
        "git_sha_describes_the_run": (_sha is not None and _dirty is False),
        "allow_basis_mismatch_used": bool(args.allow_basis_mismatch),
        "anchors_skipped": skipped,
        "per_anchor": [{"buy": a["buy"], "eval": a["eval"],
                        "n_general_pool": a["n_general_pool"], "n_scored_pool": a["n_pool"],
                        "n_priced": a["n_priced"], "carve_ok": a["carve_ok"]}
                       for a in A_train] +
                      [{"buy": A_hold["buy"], "eval": A_hold["eval"], "role": "HOLDOUT",
                        "n_general_pool": A_hold["n_general_pool"],
                        "n_scored_pool": A_hold["n_pool"], "n_priced": A_hold["n_priced"],
                        "carve_ok": A_hold["carve_ok"]}],
        "lowo_mean_out_of_sample_fit": oos_fit,
        "lowo_mean_out_of_sample_prior": oos_pri,
        "lowo_mean_in_sample": ins_mean,
        "lowo_max_abs_in_minus_out": max_gap,
        "lowo_per_fold_abs_in_minus_out": per_fold_gap,
        "realised_multiplier_range": [min(realised_multipliers(prior, mult).values()),
                                      max(realised_multipliers(prior, mult).values())],
        "panel": args.panel,
        "panel_rows": {"cdx": int(len(dmdic["cdx_df"])),
                       "bometric": int(len(dmdic["BoMetric_df"])),
                       "sources": int(dmdic["cdx_df"]["source"].nunique())},
        "panel_basis": list(rtt.detect_price_basis(dmdic["cdx_df"])),
        "prereg": PREREG, "clusters": CLUSTERS, "refuse_to_fit": REFUSE_TO_FIT,
        "grid": GRID, "lam": args.lam, "topn": args.topn,
        "train_anchors": [list(t) for t in train], "holdout": list(hold),
        "prior_weights": prior, "fitted_multipliers": mult, "fitted_weights": w_fit,
        "cv_folds": folds, "cv_one_se": se, "selection_noise_floor_95pct": noise_floor,
        "holdout_result": {"prior_beat_rate": H_pri["beat_rate"],
                           "refit_beat_rate": H_fit["beat_rate"],
                           "gain_pp": gain_pp,
                           "prior_ic": H_pri["ic"], "refit_ic": H_fit["ic"],
                           "n_eval": H_fit["n_eval"]},
        "threshold_pp": PREREG["noise_floor_pp"], "VERDICT": verdict,
        "partial_ic_controls": ctrl_rows,
        "universe_caveat": "ON WHATEVER UNIVERSE THE PANEL CARRIES; the 07-17 panel is the "
                           "OLD GATES' universe (~523 pricefails ~72% non-US; lenfail 16->8 "
                           "never fetched)",
    }
    lp = os.path.join(args.outdir, "refit_runlog_%s.json" % stamp)
    with open(lp, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)
    grid.to_csv(os.path.join(args.outdir, "refit_grid_%s.csv" % stamp), index=False)
    print("\nwrote %s\n       %s" % (lp, os.path.join(args.outdir,
                                                      "refit_grid_%s.csv" % stamp)))
    print(bar, flush=True)
    return 0 if verdict in ("PASS", "FAIL") else 1


if __name__ == "__main__":
    sys.exit(main())
