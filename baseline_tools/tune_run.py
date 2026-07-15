"""
REAL TUNE RUN  --  regularized weight tune + reoptimization-window sweep on the LIVE
scoring space (getPostDict 21 raw weights + carveOut.COHORT_WEIGHTS), offline.

Executes the tune the machinery in tuner.py / rebalance_engine.py was built for, with
the valuation-specialist's real priors (MD directive 2026-07-14).  Deliverables:
  (1) prior-only (mu) beat-rate  [sanity vs the certified 30% default-weights baseline]
  (2) TUNED beat-rate at k=H (Strategy A) + lift vs the 30% baseline
  (3) tuned metric weights (what moved off mu) + tuned cohort vectors
  (4) lambda-selection (LOWO CV, one-SE-toward-stronger) + sensitivity curve
  (5) k-sweep A(36mo) vs k=12mo  (frictionless + 20bps-tx-charged + turnover) -> optimal-k
  (6) k=6mo / k=3mo reported UNEVALUABLE (annual price grid -- a real data limit)

FEASIBILITY (the enabler): Stage-1 BoScore is INDEPENDENT of the postBo weights -- only
Stage-2 (re-weight the ~100-name pool + re-aggregate) depends on w.  So the expensive
Stage-1 is cached ONCE per as-of (4 anchors) and every weight-vector eval is a fast
Stage-2 finish().  Faithfulness of the cached finish() to stage2_pit.reproduce_pit_top
is asserted at startup (validate_finish) so nothing drifts from the certified path.

SEARCH: BeatRate is piecewise-constant with flat plateaus, so the pattern search is
driven on a SMOOTH SURROGATE (median top-20 excess-return-vs-URTH); the TRUE rc.beat_rate
primitive is evaluated on the surrogate's chosen weights (never re-implemented).

No network.  Never prints any api_key.  Reads price_data read-only (canonical untouched).
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import returns_core as rc
import rebalance_engine as reb
import tuner as T

import createDicts as cdic
import calcScore as csf
import postBoRank as pbr
import dead_merge as dm
import stage2_pit as s2

_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
PICKLE = os.path.join(
    _HOME, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-13_len7879_manelim3692_fails1966.pickle")
DEAD = os.path.join(_HOME, "delisted_out", "dead_fundamentals_20260713_104350.pickle")
REGISTRY = os.path.join(_HOME, "delisted_out", "delisted_registry.csv")
PRICES = os.path.join(_HERE, "price_data", "real_prices.csv")
PRICES_2025 = os.path.join(_HERE, "price_data", "real_prices_2025.csv")

# --- real priors (MD 2026-07-14, mapped from the LOCKED effective weights) -------
MU_GENERAL = {
    "RoA": 0.060, "returnOnCapitalEmployed": 0.060, "returnOnEquity": 0.030,
    "grossProfitMargin": 0.100, "earnYield": 0.0605, "freeCashFlowYield": 0.0605,
    "bVpRatio": 0.033, "tbVpRatio": 0.033, "grahamNumberToPrice": 0.033,
    "incomeQuality": 0.072, "Piotroski": 0.072, "EPStoEPSmean": 0.056,
    "marketCapRevQuants": 0.080, "Altman-Z": 0.062, "currentRatio": 0.038,
    "freeCashFlowPerShareGrowth": 0.043, "revenueGrowth": 0.027, "CycleHeat": -0.080,
    "DcfToPrice": 0.000, "BoScore": 0.000, "priceGrowth": 0.000,
}
MU_GP_MODERATED = dict(MU_GENERAL, grossProfitMargin=0.070)   # GP/assets-vs-GP/revenue lineage variant

# L1 partition (enforced as box constraints so L1 respects the floors)
L1_HARD_ZEROABLE = {"DcfToPrice", "BoScore", "priceGrowth"}
L1_SOFT_FLOORED = {"Altman-Z", "currentRatio", "revenueGrowth"}
L1_ELIGIBLE = L1_HARD_ZEROABLE | L1_SOFT_FLOORED

N_METRICS = 21
SOFT_FLOOR = 0.4 / N_METRICS      # ~0.019; SOFT-FLOORED metrics never driven below this

ANCHORS = list(rc.DEFAULT_ANCHORS)
CLEAN_WINDOWS = ["2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31"]
# available horizons (months) per buy window on the anchor grid
WINDOW_HORIZONS = {
    "2021-12-31": [12, 24, 36], "2022-12-30": [12, 24, 36],
    "2023-12-29": [12, 24], "2024-12-31": [12],
}
TARGET_WINDOWS_36 = ["2021-12-31", "2022-12-30"]   # the certified 36mo target cell
CERT_BASELINE_36 = 0.30                            # default-getPostDict certified baseline


def log(m):
    print(m, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
#  Box + L1 mask from a prior                                                 #
# --------------------------------------------------------------------------- #
def make_boxes(prior, keys):
    """[0.5x, 2x]*mu per metric; SOFT-FLOORED floored at SOFT_FLOOR; sign-safe."""
    boxes = []
    for k in keys:
        mu = prior[k]
        lo, hi = min(0.5 * mu, 2.0 * mu), max(0.5 * mu, 2.0 * mu)
        if k in L1_SOFT_FLOORED:
            lo = max(lo, SOFT_FLOOR)
            hi = max(hi, lo)
        boxes.append((lo, hi))
    return boxes


def l1_mask_for(keys):
    return np.array([k in L1_ELIGIBLE for k in keys])


# --------------------------------------------------------------------------- #
#  Fast cached PIT context (Stage-1 cached; Stage-2 finish per weight)        #
# --------------------------------------------------------------------------- #
class FastPitContext:
    def __init__(self, dmdic, merged, registry, carve="off"):
        self.dmdic, self.merged, self.registry = dmdic, merged, registry
        self.carve = carve
        self._uni = {}
        self._norm = {}      # as_of -> psm_norm (cached Stage-1 + metric loop)
        self.price_source = rc.PriceSource(PRICES, supp_csv=PRICES_2025)
        pb, pn = cdic.getPostDict()
        self._base_ws = {**{k: pb[k]["w"] for k in pb}, **{k: pn[k]["w"] for k in pn}}

    def universe(self, as_of):
        if as_of not in self._uni:
            uni = dm.pit_universe(self.dmdic, self.registry, as_of=as_of)
            if self.carve == "on":
                import depth_horizon_grid as dh
                uni = sorted(dh.carve_general_universe(
                    uni, self.merged["cdx_df"], self.dmdic.get("Tickers_df"),
                    lambda *a: None))
            self._uni[as_of] = uni
        return self._uni[as_of]

    def _prep(self, as_of):
        """Stage-1 + Stage-2 metric loop up to psm_norm -- EXPENSIVE, cached.
        Mirrors stage2_pit.reproduce_pit_top (universe_override path) exactly up to
        normalizeAndDropNA; validate_finish asserts the finish path reproduces it."""
        if as_of in self._norm:
            return self._norm[as_of]
        D = pd.Timestamp(as_of)
        uni = set(self.universe(as_of))
        bm = self.merged["BoMetric_df"].copy()
        cdx = self.merged["cdx_df"].copy()
        bm["date"] = pd.to_datetime(bm["date"], errors="coerce")
        cdx["date"] = pd.to_datetime(cdx["date"], errors="coerce")
        bm = bm[bm["source"].isin(uni)]
        cdx = cdx[cdx["source"].isin(uni)]
        bm_pit = s2._sort_newest_first(bm[bm["date"] <= D])
        cdx_pit = s2._sort_newest_first(cdx[cdx["date"] <= D])
        meandic = csf.getAves2(bm_pit)
        BoScore_df = csf.simpleScore_fromDict(
            bm_pit, meandic["BoMetric_ave"], meandic["BoMetric_dateAve"], 8)
        BoScore_df = BoScore_df.sort_values("score", ascending=False)
        BoS_top = BoScore_df.head(100).reset_index(drop=True)
        cdxtop = cdx_pit[cdx_pit["source"].isin(BoS_top["source"])].reset_index(drop=True)
        psm = s2._stage2_metric_loop_offline(BoS_top, cdxtop, nq=16)
        psm_norm, _ = pbr.normalizeAndDropNA(psm)
        self._norm[as_of] = psm_norm
        return psm_norm

    def _finish(self, psm_norm, weight_override):
        """Stage-2 weighting + getAggScore -- FAST.  Lockstep with reproduce_pit_top
        (stage2_pit.py:285-305); if you touch that block, touch this."""
        ws = dict(self._base_ws)
        if weight_override:
            ws.update(weight_override)
        weighted = psm_norm.drop("source", axis=1).copy()
        for col in weighted.columns:
            weighted[col] = psm_norm[col].values * ws.get(col, 1)
        psmdf_norm = pd.concat(
            [psm_norm[psm_norm.columns.difference(weighted.columns)], weighted], axis=1)
        postRank = pbr.getAggScore(psmdf_norm)
        return postRank["source"].tolist()

    def make_rank_fn(self, weight_override):
        return lambda as_of: self._finish(self._prep(as_of), weight_override)


def validate_finish(ctx, as_of="2021-12-31"):
    """Assert the cached finish(prep, None) reproduces reproduce_pit_top's pool
    exactly (proves the fast path is bit-faithful to the certified scorer)."""
    fast = ctx._finish(ctx._prep(as_of), None)
    res = s2.reproduce_pit_top(ctx.merged, as_of,
                               universe_override=ctx.universe(as_of))
    ref = res["pool_after_norm"]
    ok = (fast == ref)
    log(f"validate_finish @ {as_of}: fast==reproduce_pit_top -> {ok} "
        f"(fast_depth={len(fast)}, ref_depth={len(ref)})")
    if not ok:
        # localize the first divergence for the report
        for i, (a, b) in enumerate(zip(fast, ref)):
            if a != b:
                log(f"  first divergence at rank {i}: fast={a} ref={b}")
                break
    return ok


# --------------------------------------------------------------------------- #
#  Objectives                                                                 #
# --------------------------------------------------------------------------- #
def top20_excess_returns(rank_fn, buy, eval_anchor, ps, N=20):
    """Per-name excess-vs-URTH over [buy, eval] for the top-N at `buy` (single rank)."""
    top = rank_fn(buy)[:N]
    rdf = rc.compute_returns(top, buy, eval_anchor, ps)
    bench = rc.benchmark_return(ps, buy, eval_anchor, require_exact=True)
    inc = rc.included(rdf)
    return (inc["total_return"] - bench).to_numpy(), bench


def surrogate_score(rank_fn, window_horizons, ps, N=20):
    """Smooth search surrogate: mean over (window,horizon) cells of the MEDIAN top-N
    excess return.  Moves ~continuously with w -> good search direction on the flat
    beat-rate plateaus."""
    meds = []
    for buy, horizons in window_horizons.items():
        b_idx = ANCHORS.index(buy)
        for h in horizons:
            ev = ANCHORS[b_idx + h // 12]
            ex, _ = top20_excess_returns(rank_fn, buy, ev, ps, N=N)
            if len(ex):
                meds.append(float(np.median(ex)))
    return float(np.mean(meds)) if meds else -1e9


def true_beat_rate(rank_fn, window_horizons, ps, N=20, threshold=0.10):
    """Pooled TRUE per-name beat-rate over the given (window,horizon) cells, routed
    through rc.beat_rate (count-weighted pool).  Returns (rate, n)."""
    num, den = 0.0, 0
    for buy, horizons in window_horizons.items():
        b_idx = ANCHORS.index(buy)
        for h in horizons:
            ev = ANCHORS[b_idx + h // 12]
            top = rank_fn(buy)[:N]
            rdf = rc.compute_returns(top, buy, ev, ps)
            bench = rc.benchmark_return(ps, buy, ev, require_exact=True)
            r, nn = rc.beat_rate(rdf, bench, threshold=threshold, missing="fail")
            if r == r and nn:
                num += r * nn
                den += nn
    return (num / den if den else float("nan")), den


# --------------------------------------------------------------------------- #
#  Fit weights at a fixed lambda (surrogate-driven pattern search)            #
# --------------------------------------------------------------------------- #
def fit_weights(ctx, train_window_horizons, keys, prior_vec, base_dict, mu_vec,
                l1_mask, boxes, lam_r, lam_l, ps, search_kw):
    def score(w_vec):
        w_dict = T.dict_of(w_vec, keys, base_dict)
        rf = ctx.make_rank_fn(w_dict)
        surr = surrogate_score(rf, train_window_horizons, ps)
        return T.penalized_objective(surr, w_vec, mu_vec, l1_mask, lam_r, lam_l)

    w_vec, _, n_ev = T.pattern_search(score, prior_vec, boxes, **search_kw)
    return T.dict_of(w_vec, keys, base_dict), w_vec, n_ev


# --------------------------------------------------------------------------- #
#  LOWO CV over the lambda grid                                               #
# --------------------------------------------------------------------------- #
def lowo_cv(ctx, keys, prior_vec, base_dict, mu_vec, l1_mask, boxes,
            lam_r_grid, lam_l_grid, ps, search_kw):
    windows = list(WINDOW_HORIZONS.keys())
    curve = []
    for lr, ll in itertools.product(sorted(lam_r_grid), sorted(lam_l_grid)):
        held = []
        for hold in windows:
            train = {w: WINDOW_HORIZONS[w] for w in windows if w != hold}
            w_dict, _, _ = fit_weights(ctx, train, keys, prior_vec, base_dict, mu_vec,
                                       l1_mask, boxes, lr, ll, ps, search_kw)
            rf = ctx.make_rank_fn(w_dict)
            r, n = true_beat_rate(rf, {hold: WINDOW_HORIZONS[hold]}, ps)
            if r == r:
                held.append(r)
        cv = float(np.mean(held)) if held else float("nan")
        se = float(np.std(held) / np.sqrt(len(held))) if len(held) > 1 else 0.0
        curve.append({"lambda_r": lr, "lambda_l": ll, "cv": cv, "se": se,
                      "n_folds": len(held)})
        log(f"  CV lr={lr} ll={ll}: cv={cv:.4f} se={se:.4f} folds={len(held)}")
    valid = [c for c in curve if c["cv"] == c["cv"]]
    best = None
    if valid:
        top = max(valid, key=lambda c: c["cv"])
        thresh = top["cv"] - top["se"]                    # one-SE rule
        within = [c for c in valid if c["cv"] >= thresh]
        # strongest lambda within 1 SE of the best (parsimony)
        best = max(within, key=lambda c: (c["lambda_r"], c["lambda_l"]))
    return best, curve


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def run_general(ctx, prior, ps, search_kw, lam_r_grid, lam_l_grid, tag):
    keys = sorted(prior.keys())
    base_dict = dict(prior)
    prior_vec = T.vec_of(prior, keys)
    mu_vec = prior_vec.copy()
    l1_mask = l1_mask_for(keys)
    boxes = make_boxes(prior, keys)

    # Scale the pattern-search step to THIS prior's magnitude so the search is equally
    # powered across knob-sets (general mu ~0.05 scale; raw cohort weights ~0.25-2.0
    # scale). A fixed step would leave a raw-scale cohort search under-powered ->
    # spurious "no move off prior". step ~ 0.35 * median(|nonzero prior|).
    mags = np.abs([v for v in prior.values() if abs(v) > 1e-9])
    scale = float(np.median(mags)) if len(mags) else 0.05
    search_kw = dict(search_kw)
    search_kw["init_step"] = 0.35 * scale
    search_kw["min_step"] = 0.35 * scale / 8.0
    log(f"[{tag}] search step scaled to prior magnitude: init_step="
        f"{search_kw['init_step']:.4f} (median|mu|={scale:.4f})")

    out = {"tag": tag}
    # (1) prior-only (mu) beat-rate at the 36mo target cell + all-cell
    rf_mu = ctx.make_rank_fn(prior)
    br_mu_36, n36 = true_beat_rate(rf_mu, {w: [36] for w in TARGET_WINDOWS_36}, ps)
    br_mu_all, nall = true_beat_rate(rf_mu, WINDOW_HORIZONS, ps)
    out["mu_beat_rate_36"] = (br_mu_36, n36)
    out["mu_beat_rate_all"] = (br_mu_all, nall)
    log(f"[{tag}] mu prior-only beat-rate 36mo-target={br_mu_36:.4f}({n36})  "
        f"all-cell={br_mu_all:.4f}({nall})")

    # (4) lambda selection by LOWO CV
    best, curve = lowo_cv(ctx, keys, prior_vec, base_dict, mu_vec, l1_mask, boxes,
                          lam_r_grid, lam_l_grid, ps, search_kw)
    out["cv_curve"] = curve
    out["lambda_best"] = best
    log(f"[{tag}] CV-selected lambda: {best}")

    # (2)/(3) refit on ALL windows at selected lambda
    if best:
        w_dict, w_vec, n_ev = fit_weights(
            ctx, WINDOW_HORIZONS, keys, prior_vec, base_dict, mu_vec, l1_mask, boxes,
            best["lambda_r"], best["lambda_l"], ps, search_kw)
    else:
        w_dict = dict(prior); n_ev = 0
    out["tuned_weights"] = w_dict
    out["search_evals"] = n_ev
    out["moved_off_mu"] = {k: (round(prior[k], 4), round(w_dict[k], 4))
                           for k in keys if abs(w_dict[k] - prior[k]) > 1e-6}
    rf_t = ctx.make_rank_fn(w_dict)
    br_t_36, _ = true_beat_rate(rf_t, {w: [36] for w in TARGET_WINDOWS_36}, ps)
    br_t_all, _ = true_beat_rate(rf_t, WINDOW_HORIZONS, ps)
    out["tuned_beat_rate_36"] = br_t_36
    out["tuned_beat_rate_all"] = br_t_all
    out["lift_vs_cert_baseline"] = br_t_36 - CERT_BASELINE_36
    out["lift_vs_mu"] = br_t_36 - br_mu_36
    log(f"[{tag}] TUNED beat-rate 36mo-target={br_t_36:.4f}  all-cell={br_t_all:.4f}  "
        f"lift(vs30%)={br_t_36 - CERT_BASELINE_36:+.4f}  lift(vs mu)={br_t_36 - br_mu_36:+.4f}")

    # (5)/(6) k-sweep at the tuned weights (and mu) over the 36mo target windows
    out["ksweep_tuned"] = T.sweep_k(rf_t, TARGET_WINDOWS_36, 36, [36, 12, 6, 3], ps,
                                    tx_cost_bps=20.0)
    out["ksweep_mu"] = T.sweep_k(rf_mu, TARGET_WINDOWS_36, 36, [36, 12, 6, 3], ps,
                                 tx_cost_bps=20.0)
    return out


class CohortPartition:
    """Caches carveOut.partition_universe labels per anchor so each cohort's PIT
    member set is resolved once and shared across cohorts."""

    def __init__(self, dmdic, merged, registry):
        self.dmdic, self.merged, self.registry = dmdic, merged, registry
        self._labels = {}   # as_of -> Series source->label

    def labels(self, as_of):
        if as_of not in self._labels:
            import carveOut as co
            uni = dm.pit_universe(self.dmdic, self.registry, as_of=as_of)
            bs = pd.DataFrame({"source": sorted(uni), "score": 0.0})
            part = co.partition_universe(bs, self.merged["cdx_df"],
                                         self.dmdic.get("Tickers_df"),
                                         mcap_floor=25e6, cohort_head=25)
            self._labels[as_of] = part["labels"]
        return self._labels[as_of]

    def members(self, as_of, cohort):
        lab = self.labels(as_of)
        return sorted(lab[lab == cohort].index.tolist())


class CohortCtx(FastPitContext):
    """FastPitContext whose per-anchor universe is a single cohort's PIT members."""

    def __init__(self, dmdic, merged, registry, partition, cohort):
        super().__init__(dmdic, merged, registry, carve="off")
        self.partition, self.cohort = partition, cohort

    def universe(self, as_of):
        if as_of not in self._uni:
            self._uni[as_of] = self.partition.members(as_of, self.cohort)
        return self._uni[as_of]


def format_report(results):
    L = ["=" * 100,
         "REAL TUNE RESULT -- regularized weight tune + reopt-window sweep (offline)",
         f"generated {pd.Timestamp.utcnow().isoformat()}Z  |  no network  |  price_data read-only",
         "=" * 100,
         "Baseline anchor: certified DEFAULT-getPostDict 36mo top-20 pooled-clean "
         "beat-rate = 30.0% (12/40).",
         "mu = valuation-specialist theory prior (MD 2026-07-14); box [0.5x,2x]*mu; "
         "L1 partition (hard-zero / soft-floor / not-eligible); shrink toward mu.",
         "Search: smooth surrogate (median top-20 excess vs URTH) drives the pattern "
         "search; TRUE rc.beat_rate evaluated on the chosen weights.",
         ""]
    for out in results:
        tag = out["tag"]
        L += ["#" * 100, f"KNOB-SET: {tag}", "#" * 100]
        if "member_counts" in out:
            L.append(f"    cohort PIT member counts (small pools -> beat-rate NOISY): "
                     f"{out['member_counts']}")
            L.append("    NOTE: the 30% baseline is the GENERAL-pool anchor; for a cohort "
                     "the meaningful comparison is TUNED vs its own mu (lift-vs-mu).")
        mu36, n36 = out["mu_beat_rate_36"]
        L.append(f"(1) mu prior-only beat-rate: 36mo-target={mu36*100:.1f}% (n={n36})  "
                 f"| all-cell={out['mu_beat_rate_all'][0]*100:.1f}% (n={out['mu_beat_rate_all'][1]})")
        L.append(f"(2) TUNED beat-rate:        36mo-target={out['tuned_beat_rate_36']*100:.1f}%  "
                 f"| all-cell={out['tuned_beat_rate_all']*100:.1f}%")
        L.append(f"    LIFT vs 30% certified baseline: {out['lift_vs_cert_baseline']*100:+.1f}pp  "
                 f"| LIFT vs mu prior: {out['lift_vs_mu']*100:+.1f}pp")
        L.append(f"    (pattern-search evals at refit: {out['search_evals']})")
        L.append("(3) weights moved off mu (metric: mu -> tuned):")
        if out["moved_off_mu"]:
            for k, (a, b) in sorted(out["moved_off_mu"].items()):
                L.append(f"      {k:32} {a:+.4f} -> {b:+.4f}")
        else:
            L.append("      (none -- tuned weights == mu; data did not earn any move off prior)")
        L.append(f"(4) lambda selected (one-SE toward stronger): {out['lambda_best']}")
        L.append("    lambda sensitivity curve (flat => low-leverage, as expected):")
        for c in out["cv_curve"]:
            L.append(f"      lr={c['lambda_r']:<6} ll={c['lambda_l']:<6} "
                     f"CV_beat={c['cv']*100:5.1f}%  se={c['se']*100:4.1f}pp  folds={c['n_folds']}")
        for label, key in (("tuned", "ksweep_tuned"), ("mu-prior", "ksweep_mu")):
            sw = out[key]
            L.append(f"(5) reoptimization-window sweep [{label} weights] "
                     f"(target = buy2021+buy2022, 36mo, top-20):")
            L.append(f"    OPTIMAL REOPT WINDOW k = {sw['optimal_k']} months")
            L.append(f"      {'k':>6} | {'beat_fr':>8} | {'beat_tx20bps':>12} | "
                     f"{'turnover':>9} | note")
            for r in sw["per_k"]:
                if not r["evaluable"]:
                    L.append(f"      {r['k_months']:>4}mo | {'--':>8} | {'--':>12} | "
                             f"{'--':>9} | UNEVALUABLE: {r['reason']}")
                else:
                    L.append(f"      {r['k_months']:>4}mo | "
                             f"{r['beat_rate_frictionless']*100:6.1f}% | "
                             f"{r['beat_rate_txcharged']*100:10.1f}% | "
                             f"{r['mean_turnover_oneway']:8.2f} | {r['strategy']}")
        L.append("(6) k=6mo / k=3mo are UNEVALUABLE on the annual-only price grid "
                 "(real_prices year-ends). A quarterly price grid would resolve them "
                 "with no code change (anchor_step_months=3). REAL DATA LIMIT -- not hidden.")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_HERE, "tune_result.out"))
    ap.add_argument("--scratch-out", default=None)
    ap.add_argument("--max-evals", type=int, default=120)
    ap.add_argument("--mode", choices=["general", "cohorts"], default="general")
    ap.add_argument("--gp-variant", action="store_true",
                    help="also run the moderated grossProfitMargin=0.070 variant")
    args = ap.parse_args()

    for p in (PICKLE, DEAD, REGISTRY, PRICES, PRICES_2025):
        if not os.path.exists(p):
            log(f"FATAL missing input: {p}"); sys.exit(2)

    log("Loading pickle + dead-merge (slow, once) ...")
    dmdic = pd.read_pickle(PICKLE)
    dead = pd.read_pickle(DEAD)
    registry = dm.load_registry(REGISTRY)
    merged, _ = dm.merge_dead_into_dmdic(dmdic, dead, registry, as_of="2018-12-31")

    search_kw = {"init_step": 0.02, "min_step": 0.0025, "max_evals": args.max_evals}
    lam_r_grid = [0.0, 0.5, 1.0, 2.0, 5.0]
    lam_l_grid = [0.0, 0.5]
    results = []

    if args.mode == "general":
        ctx = FastPitContext(dmdic, merged, registry, carve="off")
        ps = ctx.price_source
        if not validate_finish(ctx, "2021-12-31"):
            log("ABORT: cached finish() diverges from reproduce_pit_top -- not faithful.")
            sys.exit(3)
        log("Warming Stage-1 caches for the 4 clean anchors ...")
        for a in CLEAN_WINDOWS:
            ctx._prep(a)
        results.append(run_general(ctx, MU_GENERAL, ps, search_kw, lam_r_grid,
                                   lam_l_grid, "GENERAL (mu, GP=0.100)"))
        if args.gp_variant:
            results.append(run_general(ctx, MU_GP_MODERATED, ps, search_kw, lam_r_grid,
                                       lam_l_grid, "GENERAL (mu, GP=0.070 moderated)"))
    else:  # cohorts
        import carveOut as co
        partition = CohortPartition(dmdic, merged, registry)
        cohort_priors = co.COHORT_WEIGHTS  # ratified per-cohort vectors = per-cohort mu
        for cohort, prior in cohort_priors.items():
            log(f"=== COHORT {cohort} ===")
            cctx = CohortCtx(dmdic, merged, registry, partition, cohort)
            # cohort pools are small/noisy -> report member counts for the caveat
            sizes = {a: len(cctx.universe(a)) for a in CLEAN_WINDOWS}
            log(f"  {cohort} PIT member counts: {sizes}")
            for a in CLEAN_WINDOWS:
                if cctx.universe(a):
                    cctx._prep(a)
            res = run_general(cctx, dict(prior), cctx.price_source, search_kw,
                              lam_r_grid, lam_l_grid, f"COHORT {cohort}")
            res["member_counts"] = sizes
            results.append(res)

    text = format_report(results)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    log(f"result written: {args.out}")
    if args.scratch_out:
        with open(args.scratch_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        log(f"scratch copy: {args.scratch_out}")
    print(text)


if __name__ == "__main__":
    main()
