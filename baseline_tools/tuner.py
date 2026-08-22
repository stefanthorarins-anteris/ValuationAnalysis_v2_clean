"""
REGULARIZED WEIGHT TUNER + REOPTIMIZATION-WINDOW SWEEP  (offline, network-free).

Searches the scoring weight space to maximise a SHRINKAGE-PENALISED beat-rate:

    maximise over w:  BeatRate(w)  -  lambda_r * ||w - mu||^2  -  lambda_l * ||w||_1
                                     (L1 taken over L1-ELIGIBLE metrics only)

  * mu               : the THEORY prior (default = current getPostDict weights).
  * cohort priors    : per-cohort prior weight vectors (default = carveOut.COHORT_WEIGHTS).
  * L1-eligible list : metrics whose weight the L1 term may drive toward 0 (default []).
  * lambda grids     : ridge-toward-prior and L1 strengths swept by CV.
All four are INPUTS (the valuation-specialist's real priors land later); until then the
machinery runs against the default-weights placeholder prior + empty L1 list so the real
run is one command.

BeatRate is a NON-SMOOTH step function of w -> a deterministic DERIVATIVE-FREE pattern
search (Hooke-Jeeves) around mu, seeded and reproducible.  (lambda_r, lambda_l) are
selected by LEAVE-ONE-WINDOW-OUT CV across the clean buy windows; ties break toward the
STRONGER lambda (parsimony).

BeatRate itself is the rebalance engine's pooled per-sleeve beat-rate, which flows
through the certified rc.beat_rate primitive.  The reoptimization window k is swept
(k == horizon is Strategy A / buy-hold); the OPTIMAL-K is argmax of the target beat-rate
NET of turnover cost, reported alongside the frictionless read.

TWO KNOB-SETS, one machine: the general-pool metric weights (tuned against the target
top-20 beat-rate) and each per-cohort weight vector (tuned against that cohort's own
top-N beat-rate) are both `PoolSpec`s handed to the SAME tune routine -- both shrunk to
their own prior.

STOP POINT (2026-07-14): this module is built to run but the expensive full tune is
GATED on the real priors + the CEO 'go'.  `python tuner.py --dry-run` exercises the
whole search/CV/sweep pipeline on a FAST STUB ranker (no scorer) to prove the machinery;
the real ranker is used only by the A(k=H,tx=0) self-check in test_rebalance_engine.py.
"""

import argparse
import itertools
import json
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


# --------------------------------------------------------------------------- #
#  Weight-vector <-> dict plumbing                                            #
# --------------------------------------------------------------------------- #
def default_prior():
    """The theory-prior placeholder: current getPostDict weights (all 21 metrics)."""
    import createDicts as cdic
    postBm, postNew = cdic.getPostDict()
    return {**{k: float(postBm[k]["w"]) for k in postBm},
            **{k: float(postNew[k]["w"]) for k in postNew}}


def vec_of(w_dict, keys):
    return np.array([w_dict[k] for k in keys], dtype=float)


def dict_of(vec, keys, base):
    d = dict(base)
    for k, v in zip(keys, vec):
        d[k] = float(v)
    return d


# --------------------------------------------------------------------------- #
#  Penalised objective                                                        #
# --------------------------------------------------------------------------- #
def penalized_objective(beat_rate, w_vec, mu_vec, l1_mask, lambda_r, lambda_l):
    """BeatRate - lambda_r ||w-mu||^2 - lambda_l ||w||_1 (L1 over eligible metrics)."""
    ridge = lambda_r * float(np.sum((w_vec - mu_vec) ** 2))
    lasso = lambda_l * float(np.sum(np.abs(w_vec[l1_mask]))) if l1_mask.any() else 0.0
    return beat_rate - ridge - lasso


# --------------------------------------------------------------------------- #
#  Deterministic derivative-free search (Hooke-Jeeves pattern search)         #
# --------------------------------------------------------------------------- #
def pattern_search(score_fn, x0, bounds, init_step=0.5, min_step=0.0625,
                   shrink=0.5, max_evals=2000):
    """Maximise score_fn over a box.  score_fn(x)->float.  Deterministic (no RNG):
    coordinate exploratory moves +/- step, accept improvements, shrink step on a
    stall.  Returns (best_x, best_score, n_evals).  Suited to the non-smooth
    step-function BeatRate objective.
    """
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    x = np.clip(np.array(x0, float), lo, hi)
    best = score_fn(x)
    n = 1
    step = float(init_step)
    while step >= min_step and n < max_evals:
        improved = False
        base_x, base_score = x.copy(), best
        for i in range(len(x)):
            for sgn in (+1.0, -1.0):
                cand = base_x.copy()
                cand[i] = min(hi[i], max(lo[i], cand[i] + sgn * step))
                if cand[i] == base_x[i]:
                    continue
                s = score_fn(cand); n += 1
                if s > best:
                    best, x, improved = s, cand, True
                if n >= max_evals:
                    break
            if n >= max_evals:
                break
        if not improved:
            step *= shrink
    return x, best, n


# --------------------------------------------------------------------------- #
#  Real PIT ranking context (the EXPENSIVE path -- gated on 'go')             #
# --------------------------------------------------------------------------- #
class PitRankContext:
    """Builds + caches the survivorship-clean PIT ranking, parametrised by weight
    override.  Wraps dead_merge + stage2_pit exactly as the certified
    compute_beat_rate_grid path does, so a rank_fn from here is bit-compatible with
    the certified baseline.  Caches rankings by (weights-key, as_of)."""

    def __init__(self, pickle_path, dead_path, registry_path, carve="off"):
        import dead_merge as dm
        import stage2_pit as s2
        self.dm, self.s2 = dm, s2
        self.dmdic = pd.read_pickle(pickle_path)
        dead = pd.read_pickle(dead_path)
        self.registry = dm.load_registry(registry_path)
        self.merged, _ = dm.merge_dead_into_dmdic(
            self.dmdic, dead, self.registry, as_of="2018-12-31")
        self.carve = carve
        self._uni_cache = {}
        self._rank_cache = {}

    def _universe(self, as_of):
        if as_of not in self._uni_cache:
            uni = self.dm.pit_universe(self.dmdic, self.registry, as_of=as_of)
            if self.carve == "on":
                import depth_horizon_grid as dh
                tickers_df = self.dmdic.get("Tickers_df")
                #  coverage_scope = the LIVE sources (see carve_general_universe): the
                #  PIT universe includes delisted entities the profile-derived sector map
                #  structurally cannot cover.
                uni = sorted(dh.carve_general_universe(
                    uni, self.merged["cdx_df"], tickers_df, lambda *a: None,
                    coverage_scope=set(self.dmdic["cdx_df"]["source"].dropna().unique())))
            self._uni_cache[as_of] = uni
        return self._uni_cache[as_of]

    def rank_fn(self, weight_override):
        wkey = None if not weight_override else tuple(sorted(weight_override.items()))

        def _rank(as_of):
            ck = (wkey, as_of)
            if ck not in self._rank_cache:
                uni = self._universe(as_of)
                res = self.s2.reproduce_pit_top(
                    self.merged, as_of, universe_override=uni,
                    weight_override=weight_override)
                self._rank_cache[ck] = [] if res is None else res["pool_after_norm"]
            return self._rank_cache[ck]
        return _rank


# --------------------------------------------------------------------------- #
#  BeatRate(w) for a set of windows at one k  (the objective's data term)     #
# --------------------------------------------------------------------------- #
def beat_rate_for_weights(rank_fn, windows, horizon_m, k_months, price_source,
                          N=20, tx_cost_bps=20.0, use_net=True,
                          anchors=None, anchor_step_months=12):
    """Pooled per-sleeve beat-rate over `windows` (list of buy anchors) at one k.
    use_net=True -> tx-charged; False -> frictionless.  Returns (rate, n)."""
    results = []
    for buy in windows:
        try:
            res = reb.evaluate_strategy(
                buy, horizon_m, k_months, rank_fn, price_source, N=N,
                tx_cost_bps=tx_cost_bps, anchors=anchors,
                anchor_step_months=anchor_step_months)
        except reb.UnevaluableK:
            continue
        results.append(res)
    if not results:
        return float("nan"), 0
    return reb.pooled_beat_rate(results, use_net)


# --------------------------------------------------------------------------- #
#  Leave-one-window-out CV over the (lambda_r, lambda_l) grid                  #
# --------------------------------------------------------------------------- #
def tune_at_lambda(make_rank_fn, tune_windows, horizon_m, k_months, price_source,
                   keys, prior_vec, base_dict, l1_mask, lambda_r, lambda_l,
                   bounds, N=20, tx_cost_bps=0.0, use_net=False, search_kw=None):
    """Fit w at a fixed (lambda_r, lambda_l) on `tune_windows` (maximise the penalised
    beat-rate).  Returns (w_dict, w_vec, train_beat_rate).  Weight-tuning uses the
    frictionless read by default (tx is a k-property, not a weight-property)."""
    search_kw = search_kw or {}

    def score(w_vec):
        w_dict = dict_of(w_vec, keys, base_dict)
        rf = make_rank_fn(w_dict)
        br, _ = beat_rate_for_weights(rf, tune_windows, horizon_m, k_months,
                                      price_source, N=N, tx_cost_bps=tx_cost_bps,
                                      use_net=use_net)
        if br != br:
            return -1e9
        return penalized_objective(br, w_vec, prior_vec, l1_mask, lambda_r, lambda_l)

    w_vec, _, _ = pattern_search(score, prior_vec, bounds, **search_kw)
    w_dict = dict_of(w_vec, keys, base_dict)
    rf = make_rank_fn(w_dict)
    br, _ = beat_rate_for_weights(rf, tune_windows, horizon_m, k_months,
                                  price_source, N=N, tx_cost_bps=tx_cost_bps,
                                  use_net=use_net)
    return w_dict, w_vec, br


def leave_one_window_out_cv(make_rank_fn, windows, horizon_m, k_months, price_source,
                            keys, prior_vec, base_dict, l1_mask,
                            lambda_r_grid, lambda_l_grid, bounds,
                            N=20, tx_cost_bps=0.0, use_net=False, search_kw=None):
    """Select (lambda_r, lambda_l) by LOWO CV.  For each lambda pair: for each held-out
    window, fit on the rest, evaluate the UNPENALISED beat-rate on the held-out window;
    CV score = mean held-out beat-rate.  Returns (best_lambda, cv_curve) where cv_curve
    is a list of dicts.  Ties break toward the STRONGER lambda (larger lambda_r then
    lambda_l -> parsimony)."""
    cv_curve = []
    for lr, ll in itertools.product(sorted(lambda_r_grid), sorted(lambda_l_grid)):
        held_scores = []
        for held in windows:
            train = [w for w in windows if w != held]
            if not train:
                continue
            w_dict, _, _ = tune_at_lambda(
                make_rank_fn, train, horizon_m, k_months, price_source, keys,
                prior_vec, base_dict, l1_mask, lr, ll, bounds, N=N,
                tx_cost_bps=tx_cost_bps, use_net=use_net, search_kw=search_kw)
            rf = make_rank_fn(w_dict)
            br, _ = beat_rate_for_weights(rf, [held], horizon_m, k_months,
                                          price_source, N=N, tx_cost_bps=tx_cost_bps,
                                          use_net=use_net)
            if br == br:
                held_scores.append(br)
        cv = float(np.mean(held_scores)) if held_scores else float("nan")
        cv_curve.append({"lambda_r": lr, "lambda_l": ll, "cv_beat_rate": cv,
                         "n_folds": len(held_scores)})
    valid = [c for c in cv_curve if c["cv_beat_rate"] == c["cv_beat_rate"]]
    best = None
    if valid:
        # max CV; tie-break toward stronger lambda (parsimony).
        best = max(valid, key=lambda c: (round(c["cv_beat_rate"], 6),
                                         c["lambda_r"], c["lambda_l"]))
    return best, cv_curve


# --------------------------------------------------------------------------- #
#  Reoptimization-window sweep                                                #
# --------------------------------------------------------------------------- #
def sweep_k(rank_fn, windows, horizon_m, k_list, price_source, N=20,
            tx_cost_bps=20.0, anchors=None, anchor_step_months=12):
    """For a FIXED weight set (rank_fn), pooled beat-rate at each k -- frictionless
    AND tx-charged -- plus the optimal-k (argmax of the tx-charged/target beat-rate).
    k unevaluable on the anchor grid is reported as such, not silently skipped."""
    rows = []
    for k in k_list:
        try:
            results = [reb.evaluate_strategy(
                buy, horizon_m, k, rank_fn, price_source, N=N,
                tx_cost_bps=tx_cost_bps, anchors=anchors,
                anchor_step_months=anchor_step_months) for buy in windows]
        except reb.UnevaluableK as e:
            rows.append({"k_months": k, "evaluable": False, "reason": str(e),
                         "beat_rate_frictionless": float("nan"),
                         "beat_rate_txcharged": float("nan"),
                         "mean_turnover_oneway": float("nan"), "n": 0})
            continue
        br_fr, n = reb.pooled_beat_rate(results, use_net=False)
        br_net, _ = reb.pooled_beat_rate(results, use_net=True)
        turn = float(np.mean([r["turnover_oneway_total"] for r in results]))
        rows.append({"k_months": k, "evaluable": True, "reason": "",
                     "beat_rate_frictionless": br_fr,
                     "beat_rate_txcharged": br_net,
                     "mean_turnover_oneway": turn, "n": n,
                     "strategy": "A (buy-hold)" if k == horizon_m else f"B k={k}mo"})
    evaluable = [r for r in rows if r["evaluable"]
                 and r["beat_rate_txcharged"] == r["beat_rate_txcharged"]]
    optimal_k = None
    if evaluable:
        optimal_k = max(evaluable, key=lambda r: r["beat_rate_txcharged"])["k_months"]
    return {"per_k": rows, "optimal_k": optimal_k}


# --------------------------------------------------------------------------- #
#  STUB ranker (dry-run only): deterministic, fast, no scorer                 #
# --------------------------------------------------------------------------- #
class StubRankContext:
    """A fast deterministic stand-in for PitRankContext to exercise the search/CV/
    sweep plumbing without the (slow) real scorer.  Ranking is a seeded pseudo-random
    permutation that shifts with both the as-of date AND the weight vector, so the
    objective is a genuine (non-constant, non-smooth) function of w and k -- enough to
    prove the machinery end-to-end.  NOT a valuation model; dry-run scaffolding only."""

    def __init__(self, universe_size=120, seed=0):
        self.universe = [f"T{i:04d}" for i in range(universe_size)]
        self.seed = seed

    def price_source(self):
        # synthetic year-end anchor grid + URTH; deterministic geometric-ish paths.
        anchors = list(rc.DEFAULT_ANCHORS)
        lut = {}
        rng = np.random.default_rng(self.seed)
        for t in self.universe + ["URTH"]:
            lvl = 100.0
            for a in anchors:
                lut[(t, a)] = lvl
                lvl *= float(1.0 + rng.normal(0.08, 0.25))
                lvl = max(lvl, 1.0)
        ps = rc.PriceSource.__new__(rc.PriceSource)
        ps.anchors = anchors
        ps._idx = {a: i for i, a in enumerate(anchors)}
        ps._lut = lut
        return ps

    def rank_fn(self, weight_override):
        wsig = 0.0 if not weight_override else float(sum(
            (i + 1) * v for i, (_, v) in enumerate(sorted(weight_override.items()))))

        def _rank(as_of):
            h = (hash((as_of, round(wsig, 3))) & 0xFFFFFFFF)
            rng = np.random.default_rng((self.seed + h) % (2 ** 32))
            order = list(self.universe)
            rng.shuffle(order)
            return order
        return _rank


# --------------------------------------------------------------------------- #
#  Dry-run driver                                                             #
# --------------------------------------------------------------------------- #
def dry_run(out_path):
    """Exercise search + CV + k-sweep on the stub ranker; write a machinery-proof."""
    prior = default_prior()
    keys = sorted(prior.keys())
    base_dict = dict(prior)
    prior_vec = vec_of(prior, keys)
    l1_eligible = []                       # placeholder: empty until specialist input
    l1_mask = np.array([k in l1_eligible for k in keys])
    bounds = [(min(0.0, prior[k]) - 2.0, prior[k] + 2.0) for k in keys]

    ctx = StubRankContext(universe_size=120, seed=1)
    ps = ctx.price_source()
    make_rank_fn = ctx.rank_fn

    windows = ["2021-12-31", "2022-12-30"]      # 36mo-supporting clean windows
    horizon = 36
    lambda_r_grid = [0.0, 0.01, 0.05]
    lambda_l_grid = [0.0, 0.01]
    search_kw = {"init_step": 1.0, "min_step": 0.5, "max_evals": 60}   # tiny for speed

    lines = ["DRY-RUN: tuner machinery proof (STUB ranker -- NOT a valuation result)",
             "=" * 78]

    # 1. baseline (prior-only) beat-rate at k=H
    rf0 = make_rank_fn(None)
    br0_fr, n0 = beat_rate_for_weights(rf0, windows, horizon, horizon, ps,
                                       tx_cost_bps=0.0, use_net=False)
    lines.append(f"baseline prior-only beat-rate (k=H, frictionless): {br0_fr:.3f} (n={n0})")

    # 2. CV over lambda grid (36mo folds: LOWO across the 2 windows)
    best, cv_curve = leave_one_window_out_cv(
        make_rank_fn, windows, horizon, horizon, ps, keys, prior_vec, base_dict,
        l1_mask, lambda_r_grid, lambda_l_grid, bounds, search_kw=search_kw)
    lines.append(f"CV selected lambda: {best}")
    lines.append("CV curve:")
    for c in cv_curve:
        lines.append(f"  lr={c['lambda_r']} ll={c['lambda_l']} "
                     f"cv={c['cv_beat_rate']:.3f} folds={c['n_folds']}")

    # 3. refit at selected lambda on ALL windows
    if best:
        w_dict, w_vec, train_br = tune_at_lambda(
            make_rank_fn, windows, horizon, horizon, ps, keys, prior_vec, base_dict,
            l1_mask, best["lambda_r"], best["lambda_l"], bounds, search_kw=search_kw)
        moved = {k: round(w_dict[k] - prior[k], 3) for k in keys
                 if abs(w_dict[k] - prior[k]) > 1e-9}
        lines.append(f"refit train beat-rate: {train_br:.3f}; weights moved off prior: {moved}")
    else:
        w_dict = dict(prior)

    # 4. k-sweep at the tuned weights
    rf = make_rank_fn(w_dict)
    sweep = sweep_k(rf, windows, horizon, [36, 12, 6, 3], ps, tx_cost_bps=20.0)
    lines.append(f"optimal reoptimization window (k): {sweep['optimal_k']} months")
    lines.append("k-sweep (beat-rate frictionless / tx-charged; turnover):")
    for r in sweep["per_k"]:
        if not r["evaluable"]:
            lines.append(f"  k={r['k_months']:>2}mo  UNEVALUABLE: {r['reason']}")
        else:
            lines.append(f"  k={r['k_months']:>2}mo  fr={r['beat_rate_frictionless']:.3f}  "
                         f"tx={r['beat_rate_txcharged']:.3f}  "
                         f"turnover={r['mean_turnover_oneway']:.2f}  ({r['strategy']})")

    lines.append("=" * 78)
    lines.append("MACHINERY OK: search + LOWO-CV + k-sweep ran end-to-end on the stub.")
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the machinery on the fast stub ranker (default)")
    ap.add_argument("--out", default=os.path.join(_HERE, "tuner_dryrun.out"))
    args = ap.parse_args()
    # Only the dry-run is wired to main; the real tune is gated on priors + CEO go.
    dry_run(args.out)


if __name__ == "__main__":
    main()
