"""
BENCH-VALIDATION of the PROPOSED new scorer (OFFLINE, no network, NOT the build).

Research reproduction of the redesign composite
    S_i = Phi_i * sum_C W_C * sum_{m in C} omega_m * rho_{i,m} * s_{i,m}
on the saved 2026-01-09 pickle panel, and three checks:
  CHECK 1  founding test  -- IC(new) vs IC(old) vs IC(best-single) vs IC(equal-wt)
  CHECK 2  turnover/churn -- top-20 overlap across consecutive quarters, new vs old,
                             + fast-sleeve vs durable-core noisiness
  CHECK 3  factor identity -- new composite exposure vs value / size / quality proxies

Reuses baseline_tools/model_vs_metric.py (build_panel, WEIGHTS) and stage2_pit.py
(the OLD two-stage top-20 engine).  ALL reconstructed-return / lookahead /
survivorship caveats from those modules carry over.

DEFAULTED PARAMETERS (underdetermined in the proposal -- documented, not blocking):
  kappa=1.6, c=1.4826 (proposal A);  phi_min=0.5, theta=2 (proposal E);
  variance penalty delta=1, u* = universe p85 of u (proposal F2/P-F4);
  d_i = fraction of the 18 EWMA levels missing for name i (proposal E: d_i defn deferred);
  hard investability floor = data-completeness only (>=8 of 18 metrics non-null)
    -- mkt-cap/ADV levels deferred to economy/CEO;
  size metric fed CONTINUOUS: level = log10(EWMA marketCap), sign=-1 (small=good)
    -- the redesign wants no quantization, so marketCapRevQuants' quartile is replaced
    by continuous log-mcap (same economic direction);
  priceGrowth: NOT in the new composite (M = CycleHeat only, weights-proposal s6.3
    RESOLVE-1 pending);  the OLD composite keeps priceGrowth AS-CODED (negated, w=0.5).
  EWMA h per channel (weights-proposal s3): SLOW=7, SLOWEST(size)=8, DENOISE=5,
    DENOISE-long(FCFyield)=6, FAST h1 (Piotroski/incomeQuality/FCFpsGrowth)=1,
    FAST h2 (EPStoEPSmean/revenueGrowth/CycleHeat)=2.  T = min(3h, available).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):        # _REPO added so `scoringWeights` resolves directly
    if _p not in sys.path:       # rather than as a side effect of another import
        sys.path.insert(0, _p)
import model_vs_metric as mvm
import scoringWeights as sw

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_data", "real_prices.csv")

KAPPA = 1.6
C_MAD = 1.4826
PHI_MIN = 0.5
THETA = 2.0
DELTA = 1.0
USTAR_PCTL = 85

# ---- KEEP-18 metric config: metric -> (cluster, sign, effW = W_C*omega_m, h) ----
CFG = {
    # P  (0.22)
    "RoA":                        ("P", +1, 0.044, 5),
    "returnOnCapitalEmployed":    ("P", +1, 0.044, 5),
    "returnOnEquity":             ("P", +1, 0.044, 5),
    "grossProfitMargin":          ("P", +1, 0.088, 7),
    # V  (0.20)
    "bVpRatio":                   ("V", +1, 0.033, 7),
    "tbVpRatio":                  ("V", +1, 0.033, 7),
    "grahamNumberToPrice":        ("V", +1, 0.033, 7),
    "earnYield":                  ("V", +1, 0.050, 5),
    "freeCashFlowYield":          ("V", +1, 0.050, 6),
    # Q  (0.17)
    "incomeQuality":              ("Q", +1, 0.057, 1),
    "Piotroski":                  ("Q", +1, 0.057, 1),
    "EPStoEPSmean":               ("Q", +1, 0.057, 2),
    # Z  (0.12)  -- continuous log-mcap, sign -1 (small=good)
    "_logmcap":                   ("Z", -1, 0.120, 8),
    # S  (0.12)
    "Altman-Z":                   ("S", +1, 0.072, 7),
    "currentRatio":               ("S", +1, 0.048, 7),
    # G  (0.09)
    "freeCashFlowPerShareGrowth": ("G", +1, 0.054, 1),
    "revenueGrowth":              ("G", +1, 0.036, 2),
    # M  (0.08)
    "CycleHeat":                  ("M", -1, 0.080, 2),
}
METRICS18 = list(CFG.keys())
CORE_CLUSTERS = {"V", "P", "S", "Z"}      # durable core
FAST_CLUSTERS = {"Q", "G", "M"}           # fast sleeve

# Prior CLUSTER-AND-CAP effective weights (benched earlier; failed do-no-harm).
W_CLUSTERCAP = {m: CFG[m][2] for m in CFG}

# THEORY-SET effective weights (CEO-steered, valuation-theory; Sigma=1). Signs stay
# in CFG (CycleHeat -1, _logmcap -1). These OVERRIDE effW for the scheme under test.
#
# DERIVED from `scoringWeights.DEPLOYED`, not copied (single-source refactor,
# 2026-08-02).  W_THEORY was never an independent vector: it is the DEPLOYED Stage-2
# vector with three mechanical adjustments, each of which is now stated in code instead
# of being silently baked into 18 literals that had to track a re-weighting by hand.
#   1. MAGNITUDES ONLY.  The bench carries each metric's sign in CFG (CycleHeat -1,
#      _logmcap -1), so the weight here is |w| -- deployed CycleHeat -0.080 -> 0.080.
#      Applying the sign twice would flip the metric.
#   2. RENAME.  marketCapRevQuants (a quartile) is replaced in this bench by the
#      CONTINUOUS log10(marketCap) channel `_logmcap` -- the redesign wants no
#      quantization -- at the SAME weight and the same economic direction (small=good).
#   3. THREE METRICS EXCLUDED.  The bench has 18 channels, not 21.  Each excluded metric
#      is asserted to be 0.000 in the deployed vector, so dropping it is score-neutral:
#      a re-weighting that resurrects one REFUSES here rather than quietly benching a
#      different scheme than the pipeline ships.
# KEY ORDER IS PART OF THE BEHAVIOUR HERE, so it is pinned rather than inherited from
# scoringWeights.METRIC_KEYS: `_weighted()` builds `pd.DataFrame({m: w[m]*equalc[m] for m
# in wdict})`, whose COLUMN order is this dict's key order, and then row-sums it.  Float
# addition is not associative, so re-ordering the keys moves S_th in the last bit or two
# -- immaterial to an IC, but enough to flip an exact tie inside a
# `.rank(method="first")`.  This is the bench's original hand-set order (descending
# effective weight, with the 0.062 / 0.0605 pair as originally written), preserved so the
# refactor cannot perturb a printed number.
_BENCH_KEY_ORDER = (
    "grossProfitMargin", "Piotroski", "incomeQuality", "earnYield", "freeCashFlowYield",
    "Altman-Z", "RoA", "returnOnCapitalEmployed", "CycleHeat", "_logmcap",
    "EPStoEPSmean", "freeCashFlowPerShareGrowth", "currentRatio", "bVpRatio",
    "tbVpRatio", "grahamNumberToPrice", "returnOnEquity", "revenueGrowth",
)
_BENCH_RENAME = {"marketCapRevQuants": "_logmcap"}
_BENCH_EXCLUDED = {
    "DcfToPrice":  "no point-in-time DCF exists offline (stage2_pit.DROP_METRICS)",
    "BoScore":     "a composite of the other metrics, not an independent channel",
    "priceGrowth": "no CFG channel; M = CycleHeat only (weights-proposal s6.3 RESOLVE-1)",
}
_deployed = sw.deployed_weights()
for _k, _why_excluded in _BENCH_EXCLUDED.items():
    assert float(_deployed[_k]) == 0.0, (
        "new_scorer_bench excludes %r (%s) on the premise that it carries w = 0.000 in "
        "the deployed vector, but it now carries %r. The bench has no channel for it, so "
        "it cannot silently drop it -- add a CFG channel or re-decide the exclusion."
        % (_k, _why_excluded, _deployed[_k]))
_derived = {_BENCH_RENAME.get(k, k): abs(float(w))
            for k, w in _deployed.items() if k not in _BENCH_EXCLUDED}
assert set(_derived) == set(_BENCH_KEY_ORDER), (
    "_BENCH_KEY_ORDER has drifted off the derived channel set: only-in-order=%s "
    "only-in-derived=%s" % (sorted(set(_BENCH_KEY_ORDER) - set(_derived)),
                            sorted(set(_derived) - set(_BENCH_KEY_ORDER))))
W_THEORY = {k: _derived[k] for k in _BENCH_KEY_ORDER}
assert abs(sum(W_THEORY.values()) - 1.0) < 1e-9, sum(W_THEORY.values())
assert set(W_THEORY) == set(METRICS18), (
    "W_THEORY no longer covers the bench's 18 channels: only-in-weights=%s "
    "only-in-CFG=%s" % (sorted(set(W_THEORY) - set(METRICS18)),
                        sorted(set(METRICS18) - set(W_THEORY))))

REAL_DATES = ["2018-12-31", "2019-12-31", "2020-12-28", "2021-12-31",
              "2022-12-27", "2023-12-29", "2024-12-31"]  # 2024-12-28 Saturday excluded


# --------------------------------------------------------------------------- #
def augmented_panel(cdx):
    """mvm.build_panel + continuous log10(marketCap) level (index-aligned)."""
    panel = mvm.build_panel(cdx)
    df = cdx.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["source", "date"])
    mcap = pd.to_numeric(df["marketCap"], errors="coerce").replace(0, np.nan)
    panel["_logmcap"] = np.log10(mcap).reindex(panel.index)
    panel["_mcap"] = mcap.reindex(panel.index)
    return panel


def ewma_levels(panel, D):
    """Return DataFrame indexed by source: for each metric  <m>=xbar, <m>__sig=sigma.
    EWMA over trailing quarters (newest first), per-channel h, T=min(3h,avail),
    NaN-skipping with weight renormalization."""
    D = pd.Timestamp(D)
    sub = panel[panel["date"] <= D].copy()
    sub = sub.sort_values(["source", "date"], ascending=[True, False])
    sub["_q"] = sub.groupby("source").cumcount()          # 0 = newest
    src = sub["source"]
    out = {}
    hcache = {}
    for metric, (_, _, _, h) in CFG.items():
        if h not in hcache:
            alpha = 1.0 - 2.0 ** (-1.0 / h)
            w = (1.0 - alpha) ** sub["_q"].values
            w = np.where(sub["_q"].values < 3 * h, w, 0.0)
            hcache[h] = w
        w = hcache[h]
        x = pd.to_numeric(sub[metric], errors="coerce").values
        ok = np.isfinite(x)
        we = w * ok
        wf = pd.Series(we, index=sub.index)
        xf = pd.Series(np.where(ok, x, 0.0) * we, index=sub.index)
        den = wf.groupby(src).sum()
        num = xf.groupby(src).sum()
        xbar = (num / den.replace(0, np.nan))
        # weighted variance
        xb_row = src.map(xbar).values
        dev2 = (np.where(ok, x, 0.0) - xb_row) ** 2 * we
        var = pd.Series(dev2, index=sub.index).groupby(src).sum() / den.replace(0, np.nan)
        out[metric] = xbar
        out[metric + "__sig"] = np.sqrt(var)
    res = pd.DataFrame(out)
    return res


def robust_z_tanh(levels):
    """Cross-sectional robust-z -> tanh per metric.  Returns s (DataFrame) and
    signed-level DataFrame (for per-metric IC / best-single)."""
    s = pd.DataFrame(index=levels.index)
    signed = pd.DataFrame(index=levels.index)
    for metric, (_, sign, _, _) in CFG.items():
        x = levels[metric]
        med = x.median()
        mad = (x - med).abs().median()
        if not np.isfinite(mad) or mad == 0:
            z = pd.Series(0.0, index=x.index)
        else:
            z = sign * (x - med) / (C_MAD * mad)
        s[metric] = np.tanh(z / KAPPA)
        signed[metric] = sign * x
    return s, signed


def variance_penalty(levels):
    """rho_{i,m} = (1 + (u/u*)^2)^(-delta), u = sigma/median_i(sigma), u* = p85(u)."""
    rho = pd.DataFrame(index=levels.index)
    for metric in CFG:
        sig = levels[metric + "__sig"]
        med = sig.median()
        if not np.isfinite(med) or med == 0:
            rho[metric] = 1.0
            continue
        u = sig / med
        ustar = np.nanpercentile(u.dropna(), USTAR_PCTL)
        if not np.isfinite(ustar) or ustar == 0:
            ustar = 1.0
        rho[metric] = (1.0 + (u / ustar) ** 2) ** (-DELTA)
        rho[metric] = rho[metric].fillna(1.0)
    return rho


def new_composites(panel, D, min_metrics=8):
    """Compute the new composite (+ equal-wt variants + fast/core) at as-of D.
    Returns dict of Series indexed by source, plus s / rho / signed / levels."""
    levels = ewma_levels(panel, D)
    # eligibility (hard floor = data completeness only)
    nnz = levels[METRICS18].notna().sum(axis=1)
    elig = nnz[nnz >= min_metrics].index
    levels = levels.loc[elig]
    s, signed = robust_z_tanh(levels)
    rho = variance_penalty(levels)
    # Phi
    d = 1.0 - levels[METRICS18].notna().sum(axis=1) / len(METRICS18)
    phi = 1.0 - (1.0 - PHI_MIN) * d ** THETA

    equalc = pd.DataFrame(index=levels.index)     # rho * s  per metric (weight-free term)
    for m in CFG:
        equalc[m] = rho[m] * s[m]

    def _weighted(wdict):
        acc = pd.DataFrame({m: wdict[m] * equalc[m] for m in wdict})
        return phi * acc.sum(axis=1, min_count=1)

    S_th = _weighted(W_THEORY)                     # THEORY scheme (under test)
    S_cc = _weighted(W_CLUSTERCAP)                 # prior cluster-and-cap
    S_new = S_th                                   # "new" = scheme under test
    S_eq18 = phi * equalc.mean(axis=1)            # equal 1/18
    # equal 1/7 across clusters
    clusters = {}
    for m, (cl, _, _, _) in CFG.items():
        clusters.setdefault(cl, []).append(m)
    eq7 = pd.Series(0.0, index=levels.index)
    for cl, members in clusters.items():
        eq7 = eq7 + (1.0 / len(clusters)) * equalc[members].mean(axis=1)
    S_eq7 = phi * eq7
    # fast-sleeve-only and core-only (reweighted within group, phi applied)
    fast_m = [m for m in CFG if CFG[m][0] in FAST_CLUSTERS]
    core_m = [m for m in CFG if CFG[m][0] in CORE_CLUSTERS]
    fw = sum(CFG[m][2] for m in fast_m)
    cw = sum(CFG[m][2] for m in core_m)
    S_fast = phi * sum(CFG[m][2] / fw * rho[m] * s[m] for m in fast_m)
    S_core = phi * sum(CFG[m][2] / cw * rho[m] * s[m] for m in core_m)
    return {
        "S_new": S_new, "S_th": S_th, "S_cc": S_cc,
        "S_eq18": S_eq18, "S_eq7": S_eq7,
        "S_fast": S_fast, "S_core": S_core,
        "s": s, "rho": rho, "signed": signed, "levels": levels, "phi": phi,
    }


def old_composite(panel, D, lag_days=0):
    """OLD production-style composite: trailing 16q simple mean per metric ->
    cross-sectional z * production weight -> sum.  Single-stage over the eligible
    universe (two-stage funnel collapsed -- IC compares ranking signal, not the
    100-name selection).  Includes marketCapRevQuants + priceGrowth as-coded."""
    D = pd.Timestamp(D) - pd.Timedelta(days=lag_days)
    sub = panel[panel["date"] <= D].copy()
    sub = sub.sort_values(["source", "date"], ascending=[True, False])
    sub["_q"] = sub.groupby("source").cumcount()
    src = sub["source"]
    keep16 = sub["_q"] < 16
    # 18 panel metrics incl priceGrowth (as-coded), excl size.
    # NOT single-sourced, deliberately, and FLAGGED (2026-08-02): mvm.WEIGHTS is the
    # pre-2026-07-14 LEGACY vector despite its comment claiming it is "from
    # createDicts.getPostDict" -- see the PRE-EXISTING DISAGREEMENT note in
    # scoringWeights.py.  For THIS arm ("OLD") the legacy numbers are plausibly what is
    # intended, so nothing is changed here; the 0.25 below is likewise the LEGACY
    # marketCapRevQuants weight (deployed is 0.080).  Whether the OLD arm should be
    # legacy or deployed is a research call for the CEO, not a refactor.
    weights = dict(mvm.WEIGHTS)
    means = {}
    for m in weights:
        x = pd.to_numeric(sub[m], errors="coerce")
        xm = x.where(keep16)
        means[m] = xm.groupby(src).mean()
    M = pd.DataFrame(means)
    # marketCapRevQuants: quartile of trailing-mean marketCap (small=high), w=0.25
    mc = pd.to_numeric(sub["_mcap"], errors="coerce").where(keep16).groupby(src).mean()
    try:
        q = -1.0 * ((pd.qcut(mc, 4, labels=False, duplicates="drop") / 3.0) - 0.5)
        M["marketCapRevQuants"] = q
        weights["marketCapRevQuants"] = 0.25
    except Exception:
        pass
    z = pd.DataFrame(index=M.index)
    for m in weights:
        if m not in M:
            continue
        col = M[m]
        z[m] = (col - col.mean()) / col.std() * weights[m]
    return z.sum(axis=1, min_count=1)


def parallel_candidate_scores(panel, D, decisional="old", topn=20, new_key="S_new"):
    """Run BOTH scorers at as-of D and return a labelled side-by-side frame.

    The EXISTING (old) scorer is the DECISIONAL / pick-generating instrument; the
    NEW scorer is computed ALONGSIDE it as a NON-DECISIONAL parallel CANDIDATE, so
    the backtest this fetch enables can compare them on evidence.  Switching the
    decisional scorer to the new one is a SEPARATE, evidence-gated decision AFTER
    that backtest -- deliberately NOT done here (CEO directive 2026-07-12: do not
    make the new scorer decisional on faith, pre-evidence).  `decisional` is pinned
    to "old" and this function refuses any other value so the candidate can never
    silently become the pick-generator.

    Returns a DataFrame indexed by source:
      S_decisional        -- old composite (drives picks)
      rank_decisional     -- 1 = best by the decisional score
      S_candidate_new     -- new composite (candidate only, non-decisional)
      rank_candidate_new
      is_pick             -- True for the top-`topn` by the DECISIONAL score
    """
    if decisional != "old":
        raise ValueError(
            "The old scorer stays decisional through the first backtest (CEO "
            "directive 2026-07-12).  The new scorer runs only as a parallel "
            "candidate; making it decisional is a separate evidence-gated step.")
    D = pd.Timestamp(D)
    old = old_composite(panel, D).rename("S_decisional")
    new = new_composites(panel, D)[new_key].rename("S_candidate_new")
    out = pd.concat([old, new], axis=1)
    out["rank_decisional"] = out["S_decisional"].rank(ascending=False, method="first")
    out["rank_candidate_new"] = out["S_candidate_new"].rank(ascending=False, method="first")
    picks = out["S_decisional"].dropna().sort_values(ascending=False).head(topn).index
    out["is_pick"] = out.index.isin(picks)
    out.attrs["decisional"] = "old"
    out.attrs["as_of"] = str(D.date())
    return out


# --------------------------------------------------------------------------- #
def load_real():
    df = pd.read_csv(REAL)
    df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
    piv = df.pivot_table(index="symbol", columns="date_actual",
                         values="adjClose", aggfunc="last")
    return piv


def real_fwd(real, buy, ev):
    if buy not in real.columns or ev not in real.columns:
        return None
    return (real[ev] / real[buy] - 1).replace([np.inf, -np.inf], np.nan)


def horizon_pairs():
    d = REAL_DATES
    return {
        "12mo": [(d[i], d[i + 1]) for i in range(len(d) - 1)],
        "24mo": [(d[i], d[i + 2]) for i in range(len(d) - 2)],
        "36mo": [(d[i], d[i + 3]) for i in range(len(d) - 3)],
    }


def _ic(vals, ret, min_n=50):
    j = pd.concat([vals.rename("v"), ret.rename("r")], axis=1).dropna()
    if len(j) < min_n:
        return np.nan, len(j)
    ic, _ = spearmanr(j["v"], j["r"])
    return ic, len(j)


def check1(panel, real, use_real=True, lag_days=0):
    """Returns (pooled results, per-metric IC, per-window IC) for all schemes."""
    pairs = horizon_pairs()
    buy_dates = sorted({b for hs in pairs.values() for (b, _) in hs})
    cache = {}
    for b in buy_dates:
        nc = new_composites(panel, b)
        oldc = old_composite(panel, b, lag_days=lag_days)
        cache[b] = (nc, oldc)
    schemes = {"THEORY": "S_th", "CLUST-CAP": "S_cc", "EQ18": "S_eq18", "EQ7": "S_eq7"}
    results = {}
    perwin = {h: [] for h in pairs}       # list of (window_label, {scheme:IC}, best_single_ic, best_m)
    permetric_ic = {h: {m: [] for m in METRICS18} for h in pairs}
    for h, hs in pairs.items():
        agg = {k: [] for k in list(schemes) + ["OLD"]}
        for (b, ev) in hs:
            nc, oldc = cache[b]
            if use_real:
                ret = real_fwd(real, b, ev)
            else:
                px = panel[panel["date"] <= pd.Timestamp(b)].groupby("source").tail(1).set_index("source")["_price"]
                pxe = panel[panel["date"] <= pd.Timestamp(ev)].groupby("source").tail(1).set_index("source")["_price"]
                ret = (pxe / px - 1).replace([np.inf, -np.inf], np.nan)
            if ret is None:
                continue
            wic = {}
            for name, key in schemes.items():
                ic = _ic(nc[key], ret)[0]
                agg[name].append(ic); wic[name] = ic
            oic = _ic(oldc, ret)[0]
            agg["OLD"].append(oic); wic["OLD"] = oic
            signed = nc["signed"]
            bm_ic, bm_m = -9, None
            for m in METRICS18:
                mic = _ic(signed[m], ret)[0]
                permetric_ic[h][m].append(mic)
                if pd.notna(mic) and mic > bm_ic:
                    bm_ic, bm_m = mic, m
            perwin[h].append((f"{b[:4]}->{ev[:4]}", wic, bm_ic, bm_m))
        results[h] = {k: (np.nanmean(v) if v else np.nan) for k, v in agg.items()}
    pm = {h: {m: np.nanmean(v) if v else np.nan for m, v in d.items()}
          for h, d in permetric_ic.items()}
    return results, pm, perwin


def print_check1(res, pm, perwin, tag):
    print("\n" + "=" * 78)
    print(f"CHECK 1 -- DO-NO-HARM FOUNDING TEST  [{tag}]")
    print("=" * 78)
    print(f"{'horizon':8s} {'THEORY':>8s} {'CLUST-CAP':>9s} {'OLD':>8s} "
          f"{'EQ-1/18':>8s} {'EQ-1/7':>8s}  {'BEST-SINGLE':>22s}")
    for h in res:
        best_m = max(pm[h], key=lambda m: -1e9 if pd.isna(pm[h][m]) else pm[h][m])
        best = pm[h][best_m]
        r = res[h]
        print(f"{h:8s} {r['THEORY']:+8.3f} {r['CLUST-CAP']:+9.3f} {r['OLD']:+8.3f} "
              f"{r['EQ18']:+8.3f} {r['EQ7']:+8.3f}   {best:+.3f} ({best_m})")
        v = []
        v.append("beats EQ18" if r['THEORY'] > r['EQ18'] else "LOSES to EQ18")
        v.append("beats OLD" if r['THEORY'] > r['OLD'] else "loses to OLD")
        v.append("beats CLUST-CAP" if r['THEORY'] > r['CLUST-CAP'] else "below CLUST-CAP")
        v.append("beats best-single" if r['THEORY'] > best else "below best-single")
        print(f"         THEORY -> {'; '.join(v)}")


def print_perwindow(perwin, tag):
    print("\n" + "-" * 78)
    print(f"CHECK 2 -- PER-WINDOW IC DISPERSION (regime stability)  [{tag}]")
    print("-" * 78)
    for h, rows in perwin.items():
        print(f"\n  {h} windows:")
        print(f"    {'window':10s} {'THEORY':>8s} {'CLUST-CAP':>9s} {'OLD':>8s} "
              f"{'EQ18':>8s}  {'best-single':>20s}")
        for label, wic, bic, bm in rows:
            note = ""
            if label.startswith("2019"):
                note = " <- COVID"
            elif "2022" in label.split("->")[1] or label.startswith("2022"):
                note = " <- 2022 val-rot"
            print(f"    {label:10s} {wic['THEORY']:+8.3f} {wic['CLUST-CAP']:+9.3f} "
                  f"{wic['OLD']:+8.3f} {wic['EQ18']:+8.3f}   {bic:+.3f} ({bm}){note}")
        th = [w['THEORY'] for _, w, _, _ in rows if pd.notna(w['THEORY'])]
        eq = [w['EQ18'] for _, w, _, _ in rows if pd.notna(w['EQ18'])]
        if th:
            print(f"    THEORY  mean {np.mean(th):+.3f}  min {np.min(th):+.3f}  "
                  f"max {np.max(th):+.3f}  std {np.std(th):.3f}")
            print(f"    EQ18    mean {np.mean(eq):+.3f}  min {np.min(eq):+.3f}  "
                  f"max {np.max(eq):+.3f}  std {np.std(eq):.3f}")


def print_diagnosis(pm):
    print("\n" + "-" * 78)
    print("DIAGNOSIS (36mo): THEORY effective weight  vs  per-metric 36mo IC")
    print("-" * 78)
    h = "36mo"
    tot_w = sum(W_THEORY[m] for m in METRICS18)
    neg_w = 0.0
    rows = sorted(METRICS18, key=lambda m: -W_THEORY[m])
    print(f"{'metric':30s} {'sign':>4s} {'THEORY':>7s} {'CLUST-CAP':>9s} {'IC36':>8s}")
    for m in rows:
        _, sign, cc, _ = CFG[m]
        w = W_THEORY[m]
        ic = pm[h][m]
        if pd.notna(ic) and ic < 0:
            neg_w += w
        ics = f"{ic:+8.3f}" if pd.notna(ic) else "     nan"
        print(f"{m:30s} {sign:+4d} {w:7.3f} {cc:9.3f} {ics}")
    print(f"\n  weight on NEGATIVE-36mo-IC metrics (THEORY) = {neg_w:.3f} "
          f"({100*neg_w/tot_w:.0f}%)   [was 36% under cluster-and-cap]")


def print_concentration():
    print("\n" + "-" * 78)
    print("CHECK 3 -- CONCENTRATION PROFILE (effective weights)")
    print("-" * 78)
    for name, w in (("THEORY", W_THEORY), ("CLUST-CAP", W_CLUSTERCAP)):
        vals = np.array([w[m] for m in METRICS18], float)
        vals = vals / vals.sum()
        hhi = float((vals ** 2).sum())
        top_m = max(w, key=lambda k: w[k])
        print(f"  {name:10s}  max-weight={w[top_m]:.3f} ({top_m})  "
              f"HHI={hhi:.4f}  effective-N={1.0/hhi:.1f} / 18")


# --------------------------------------------------------------------------- #
def new_top20(panel, D, key="S_new", n=20):
    nc = new_composites(panel, D)
    return nc[key].sort_values(ascending=False).head(n).index.tolist()


def check2(panel, dmdic):
    import io, contextlib
    import stage2_pit as s2
    print("\n" + "=" * 74)
    print("CHECK 2 -- TURNOVER / CHURN  (consecutive quarters, top-20 overlap)")
    print("=" * 74)
    qdates = ["2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
              "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]

    def ov(a, b):
        return len(set(a) & set(b))

    caches = {d: new_composites(panel, d) for d in qdates}
    th_tops = {d: caches[d]["S_th"].sort_values(ascending=False).head(20).index.tolist() for d in qdates}
    cc_tops = {d: caches[d]["S_cc"].sort_values(ascending=False).head(20).index.tolist() for d in qdates}
    fast_tops = {d: caches[d]["S_fast"].sort_values(ascending=False).head(20).index.tolist() for d in qdates}
    core_tops = {d: caches[d]["S_core"].sort_values(ascending=False).head(20).index.tolist() for d in qdates}
    old_tops = {}
    for d in qdates:
        with contextlib.redirect_stdout(io.StringIO()):
            bm, cdx = s2.prepare_pit(dmdic, d, na1_only=False)
            old_tops[d] = s2.stage2_top(s2.stage1_boscore(bm, cdx_pit=cdx), cdx)

    print(f"{'q0->q1':22s} {'THEORY':>7s} {'CLUST-CAP':>9s} {'OLD':>7s} {'FAST':>7s} {'CORE':>7s}")
    tch, cch2, och, fch, cch = [], [], [], [], []
    for i in range(len(qdates) - 1):
        d0, d1 = qdates[i], qdates[i + 1]
        t = ov(th_tops[d0], th_tops[d1]); cc = ov(cc_tops[d0], cc_tops[d1])
        o = ov(old_tops[d0], old_tops[d1])
        f = ov(fast_tops[d0], fast_tops[d1]); c = ov(core_tops[d0], core_tops[d1])
        tch.append((20 - t) / 20); cch2.append((20 - cc) / 20); och.append((20 - o) / 20)
        fch.append((20 - f) / 20); cch.append((20 - c) / 20)
        print(f"{d0}->{d1[5:]:12s} {t:4d}/20 {cc:6d}/20 {o:4d}/20 {f:4d}/20 {c:4d}/20")
    print(f"\n  mean quarterly churn: THEORY={np.mean(tch)*100:.0f}%  "
          f"CLUST-CAP={np.mean(cch2)*100:.0f}%  OLD={np.mean(och)*100:.0f}%  "
          f"FAST-sleeve={np.mean(fch)*100:.0f}%  CORE={np.mean(cch)*100:.0f}%")
    print("  (snapshot price + EWMA => sub-quarterly churn is 0 by construction;")
    print("   this measures QUARTER-TO-QUARTER fundamental turnover.)")


def check3(panel):
    print("\n" + "=" * 78)
    print("CHECK 4 -- FACTOR IDENTITY (top-20 exposure) THEORY vs CLUST-CAP")
    print("=" * 78)
    dates = ["2020-12-31", "2021-12-31", "2022-12-31"]
    acc = {sc: {"vc": [], "zc": [], "qc": [], "tv": [], "tz": [], "tq": []}
           for sc in ("S_th", "S_cc")}
    for D in dates:
        nc = new_composites(panel, D)
        lv = nc["levels"]
        value = lv["bVpRatio"]              # book-to-price (high=cheap=value)
        size = -lv["_logmcap"]              # small-cap tilt (high=smaller)
        qual = lv["grossProfitMargin"]      # profitability/quality
        for sc in ("S_th", "S_cc"):
            S = nc[sc]
            jn = pd.concat([S.rename("S"), value.rename("v"), size.rename("z"),
                            qual.rename("q")], axis=1).dropna()
            acc[sc]["vc"].append(spearmanr(jn["S"], jn["v"])[0])
            acc[sc]["zc"].append(spearmanr(jn["S"], jn["z"])[0])
            acc[sc]["qc"].append(spearmanr(jn["S"], jn["q"])[0])
            top = S.sort_values(ascending=False).head(20).index
            acc[sc]["tv"].append(value.rank(pct=True).reindex(top).mean())
            acc[sc]["tz"].append(size.rank(pct=True).reindex(top).mean())
            acc[sc]["tq"].append(qual.rank(pct=True).reindex(top).mean())
    print(f"  (mean over {dates})")
    print(f"  {'':26s} {'THEORY':>8s} {'CLUST-CAP':>10s}")
    labels = [("x-sec corr  value (B/P)", "vc"), ("x-sec corr  size (small)", "zc"),
              ("x-sec corr  quality(GM)", "qc"),
              ("top-20 pctile value", "tv"), ("top-20 pctile size (small)", "tz"),
              ("top-20 pctile quality", "tq")]
    for lab, k in labels:
        a = np.nanmean(acc["S_th"][k]); b = np.nanmean(acc["S_cc"][k])
        print(f"  {lab:26s} {a:+8.2f} {b:+10.2f}")
    print("  (size pctile high => small-cap tilt; 0.5 = neutral)")


def _factor_pctile(lv, name):
    """percentile of name on each factor proxy (0..1), for the 'why' note."""
    def pct(col):
        r = col.rank(pct=True)
        return r.get(name, np.nan)
    return {
        "GM": pct(lv["grossProfitMargin"]),
        "RoA": pct(lv["RoA"]),
        "Piotr": pct(lv["Piotroski"]),
        "BtP": pct(lv["bVpRatio"]),
        "FCFy": pct(lv["freeCashFlowYield"]),
        "small": pct(-lv["_logmcap"]),
        "Altman": pct(lv["Altman-Z"]),
    }


def _why(lv, name):
    p = _factor_pctile(lv, name)
    hi = [f"{k} {p[k]:.2f}" for k in ("GM", "RoA", "Piotr", "FCFy", "BtP", "small", "Altman")
          if pd.notna(p[k]) and p[k] >= 0.80]
    lo = [f"{k} {p[k]:.2f}" for k in ("GM", "RoA", "Piotr", "FCFy", "BtP", "small", "Altman")
          if pd.notna(p[k]) and p[k] <= 0.25]
    s = "high: " + (", ".join(hi) if hi else "-")
    if lo:
        s += " | low: " + ", ".join(lo)
    return s


def check5(panel):
    """Decision-relevant: THEORY top-20 vs EQ-1/18 top-20 (and vs prior CLUST-CAP)."""
    print("\n" + "=" * 78)
    print("CHECK 5 -- SHORTLIST COMPARISON: THEORY vs EQUAL-1/18 (and vs CLUST-CAP)")
    print("=" * 78)
    dates = [d for d in ["2020-12-31", "2021-12-31", "2022-12-31"]]
    jac_te, jac_tc, common_te, common_tc = [], [], [], []
    detail_date = dates[-1]                      # most recent for the name-by-name detail
    for D in dates:
        nc = new_composites(panel, D)
        th = nc["S_th"].sort_values(ascending=False).head(20).index.tolist()
        eq = nc["S_eq18"].sort_values(ascending=False).head(20).index.tolist()
        cc = nc["S_cc"].sort_values(ascending=False).head(20).index.tolist()
        te = len(set(th) & set(eq)); tc = len(set(th) & set(cc))
        common_te.append(te); common_tc.append(tc)
        jac_te.append(te / len(set(th) | set(eq)))
        jac_tc.append(tc / len(set(th) | set(cc)))
        print(f"  {D}:  THEORY&EQ18 common {te}/20 (Jaccard {jac_te[-1]:.2f})   "
              f"THEORY&CLUST-CAP common {tc}/20 (Jaccard {jac_tc[-1]:.2f})")
    print(f"\n  mean over {dates}: THEORY vs EQ18 common {np.mean(common_te):.1f}/20 "
          f"(Jaccard {np.mean(jac_te):.2f});  THEORY vs CLUST-CAP common "
          f"{np.mean(common_tc):.1f}/20 (Jaccard {np.mean(jac_tc):.2f})")

    # name-by-name difference on the most recent date
    nc = new_composites(panel, detail_date)
    lv = nc["levels"]
    th = nc["S_th"].sort_values(ascending=False).head(20).index.tolist()
    eq = nc["S_eq18"].sort_values(ascending=False).head(20).index.tolist()
    only_th = [n for n in th if n not in set(eq)]
    only_eq = [n for n in eq if n not in set(th)]
    print(f"\n  --- name-by-name difference as-of {detail_date} "
          f"(THEORY {len(th)}, EQ18 {len(eq)}, differ by {len(only_th)}) ---")
    print(f"\n  IN THEORY, NOT EQ18 ({len(only_th)}):")
    for n in only_th:
        print(f"    + {n:12s} {_why(lv, n)}")
    print(f"\n  IN EQ18, NOT THEORY ({len(only_eq)}):")
    for n in only_eq:
        print(f"    - {n:12s} {_why(lv, n)}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("Loading pickle + building augmented panel (offline) ...", flush=True)
    dmdic = pd.read_pickle(mvm.PICKLE)
    panel = augmented_panel(dmdic["cdx_df"])
    print(f"panel rows: {len(panel)}, names: {panel['source'].nunique()}", flush=True)
    real = load_real()
    print(f"real price matrix: {real.shape[0]} symbols x {len(real.columns)} dates", flush=True)

    if which in ("all", "1"):
        res_r, pm_r, pw_r = check1(panel, real, use_real=True, lag_days=0)
        print_check1(res_r, pm_r, pw_r, "REAL returns, quarter-START stamp (default)")
        print_concentration()
        print_diagnosis(pm_r)
        print_perwindow(pw_r, "REAL returns")
        res_x, pm_x, pw_x = check1(panel, real, use_real=False, lag_days=0)
        print_check1(res_x, pm_x, pw_x, "RECONSTRUCTED returns (within-data cross-check)")
        res_l, pm_l, pw_l = check1(panel, real, use_real=True, lag_days=150)
        print_check1(res_l, pm_l, pw_l, "REAL returns, +150d LAG (lookahead-corrected sensitivity)")
    if which in ("all", "2"):
        check2(panel, dmdic)
    if which in ("all", "3"):
        check3(panel)
    if which in ("all", "5"):
        check5(panel)


if __name__ == "__main__":
    main()
