"""
SKILL BASELINE  --  oracle-best-N ceiling + random-N Monte-Carlo baseline (offline).

WHAT THIS ANSWERS
-----------------
The filter's top-20 beats URTH-by->=10pp at some rate (e.g. ~38.5% on the 36mo cell).
Is that SELECTION SKILL, or just where a random pick from the same universe would land?
This module frames the filter between two reference points, on the SAME survivorship-
clean, dead-merged, point-in-time (PIT) universe the filter itself is scored on:

  * ORACLE-BEST-N  (perfect-hindsight CEILING, NOT operational): from the filter's
    shortlist (the Stage-2 ranked pool the top-20 is chosen FROM) pick, WITH PERFECT
    HINDSIGHT, the N names that maximize the outcome.  A ceiling label, never a claim
    the winners are humanly identifiable ex-ante.

  * RANDOM-N       (Monte-Carlo FLOOR, isolates selection skill from regime tilt):
    >=`n_draws` random N-picks drawn from the EXACT scored/ranked universe (and from
    the filter's own top-200 / top-100 rungs), reporting the distribution.

  * DECOMPOSITION LADDER: universe -> random-top-200 -> random-top-100 -> filter top-20,
    for BOTH beat-rate AND dollar return.  Splits "shortlist / tilt-membership value"
    (universe -> top-100) from "fine Stage-2 ranking skill" (top-100 -> top-20).

EVERYTHING is a thin view over the CERTIFIED primitives -- NOTHING is re-implemented:
  stage2_pit.reproduce_pit_top   -> filter top-20 + Stage-2 pool (the oracle shortlist)
  stage2_pit.stage1_boscore      -> full Stage-1 BoScore ranking (universe / 200 / 100 sets)
  dead_merge.pit_universe        -> the exact as-of-D scored-universe scope
  returns_core.compute_returns   -> per-ticker total return (dividend-inclusive)
  returns_core.beat_rate         -> the count beat-rate (missing='fail' symmetry)
  returns_core.benchmark_return  -> URTH window return (require_exact)

THREE CORRECTIONS baked in (these were bugs in the first ad-hoc run -- do NOT reintroduce):
  (1) UNIVERSE-MATCH: random draws come from the EXACT scored universe (dm.pit_universe
      + Stage-1 BoScore membership), NOT a broader mcap>=$25M set.  A consistency guard
      asserts our derived Stage-1 top-100 set == reproduce_pit_top's stage1_top100.
  (2) DEAD-NAME PARITY: filter AND random both draw from the same dead-merged PIT
      universe and both score via returns_core.beat_rate(missing='fail') -- symmetric
      missing=fail with a VARIABLE denominator (no_buy excluded both sides).
  (3) MEAN-RETURN CONTAMINATION: a near-zero-buy-price name yields an astronomical ratio
      (the ~403,382% artifact) that corrupts the universe-level mean return.  Beat-rate is
      immune (it's just one more 'beater'); dollar means DROP names above `contam_return_cap`
      and additionally report a winsorized mean.  A contaminated raw universe mean is NEVER
      reported.

Deterministic: a single seeded numpy Generator drives every draw -> same seed, same result.

No network.  Reads price CSVs only through returns_core.PriceSource (never dumps them).
Never prints an api_key.  Does NOT edit Sbocker / postBo / postBoRank / stage2_pit.
"""

import contextlib
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import returns_core as rc          # PriceSource, compute_returns, beat_rate, benchmark_return
import stage2_pit as s2            # reproduce_pit_top, stage1_boscore
import dead_merge as dm            # merge_dead_into_dmdic, pit_universe, load_registry
import stage2_metrics as sm        # shared Stage-2 metrics (used only for the short-history guard)

# Buy anchors with a non-degenerate PIT universe (pre-2021 anchors are universe-degenerate
# -- thin dead-registry coverage + price-coverage gaps; see run_target_test caveats).
CLEAN_BUY_ANCHORS = ["2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31"]

# +10,000% return.  A real 36mo winner never nears this; the near-zero-buy-price artifact
# (~4,033x) does.  Names above the cap are dropped from dollar means (correction #3).
DEFAULT_CONTAM_RETURN_CAP = 100.0
DEFAULT_WINSOR = (0.01, 0.99)      # percentile clip for the robust dollar mean


# --------------------------------------------------------------------------- #
#  SHORT-HISTORY EPS GUARD  (workaround for a defect in a file we must NOT edit) #
# --------------------------------------------------------------------------- #
# FINDING / HISTORY (surface to reviewer):
#   stage2_metrics.eps_to_eps_mean USED to crash for any frame with < 4 rows -- its guard
#   `all(eps.iloc[0:4] > 0)` passed on a SHORT slice (e.g. 2 positive quarters -> True) and
#   the body then read eps.iloc[2]/eps.iloc[3] -> IndexError.  On the LIVE survivor universe
#   every name is mature (>=4 quarters) so it never fired; on the SURVIVORSHIP-CLEAN dead-
#   merged universe a short-history dead name reaching the Stage-1 top-100 crashed
#   stage2_pit.reproduce_pit_top (and, latently, production postBoRank.py:192).
#   OBSERVED MID-SESSION (2026-07-15): the parallel stage2_metrics refactor ADDED the fix
#   (`if len(eps) >= 4 and all(...)`) at the source, so the root cause is resolved upstream
#   and this guard is now REDUNDANT.  It is retained as an OPT-IN (default OFF) safety net
#   -- stage2_metrics.py is under concurrent edit, so a caller who hits a regression can set
#   guard_short_history_eps=True.  When enabled it returns 0 for a <4-row frame (exactly the
#   fixed function's own no-EWMA value) and hands >=4-row frames to the original untouched.
_short_history_eps_sources = set()


def _guarded_eps_to_eps_mean(orig):
    # **kwargs is LOAD-BEARING (review R-N1, 2026-07-25): callers now pass rpy= (the
    # per-source rows-per-year, reporting_period).  A fixed (tempcdx)-only signature made
    # this guard -- the documented escape hatch for a stage2_metrics regression -- itself
    # raise TypeError the moment it was enabled, i.e. it would have failed exactly when it
    # was needed.  Forward everything through untouched so the wrapper stays transparent
    # to any future signature change too.
    def wrapped(tempcdx, *args, **kwargs):
        eps = tempcdx["netIncome"] / tempcdx["weightedAverageShsOut"]
        if len(eps) < 4:
            if "source" in getattr(tempcdx, "columns", []) and len(tempcdx):
                _short_history_eps_sources.add(str(tempcdx["source"].iloc[0]))
            return 0
        return orig(tempcdx, *args, **kwargs)
    return wrapped


@contextlib.contextmanager
def short_history_eps_guard(enabled=True):
    """Temporarily wrap stage2_metrics.eps_to_eps_mean with a <4-row length guard.
    stage2_pit / postBoRank call it via the module attribute, so patching the module
    attribute is seen by both; restored on exit.  No file on disk is modified."""
    if not enabled:
        yield
        return
    orig = sm.eps_to_eps_mean
    sm.eps_to_eps_mean = _guarded_eps_to_eps_mean(orig)
    try:
        yield
    finally:
        sm.eps_to_eps_mean = orig


# --------------------------------------------------------------------------- #
#  Window construction                                                        #
# --------------------------------------------------------------------------- #
def clean_windows(cadence_months, anchors=None):
    """(buy_anchor, eval_anchor) pairs for the given hold length, restricted to the
    clean buy set and to anchors that actually exist on the price grid.

    36mo -> [(2021-12-31, 2024-12-31), (2022-12-30, 2025-12-31)]   (2 windows)
    12mo -> 4 windows buy2021..buy2024                              (annual cadence)
    """
    anchors = list(anchors) if anchors is not None else list(rc.DEFAULT_ANCHORS)
    idx = {a: i for i, a in enumerate(anchors)}
    step = cadence_months // 12
    out = []
    for buy in CLEAN_BUY_ANCHORS:
        i = idx.get(buy)
        if i is None:
            continue
        j = i + step
        if j < len(anchors):
            out.append((buy, anchors[j]))
    return out


# --------------------------------------------------------------------------- #
#  Per-window set + rank extraction (all via certified interfaces)            #
# --------------------------------------------------------------------------- #
def _pit_bm(merged, universe, D):
    """Universe-filtered, date<=D, NEWEST-FIRST BoMetric slice -- mirrors the PIT prep
    inside stage2_pit.reproduce_pit_top (universe_override path) so the Stage-1 BoScore
    we derive matches it bit-for-bit.  (The sort key is the same as s2._sort_newest_first;
    reproduced locally rather than importing a private symbol.)"""
    bm = merged["BoMetric_df"].copy()
    bm["date"] = pd.to_datetime(bm["date"], errors="coerce")
    bm = bm[bm["source"].isin(universe)]
    bm = bm[bm["date"] <= pd.Timestamp(D)]
    return bm.sort_values(["source", "date"], ascending=[True, False])


def window_sets(dmdic, merged, registry, buy, nq_stage1=8, exchange_filter=None):
    """Return (rung_sets, filter_res, universe) for one buy anchor.

    rung_sets  : {'universe': [...], 'top200': [...], 'top100': [...]}  (Stage-1 order)
    filter_res : the reproduce_pit_top result dict (its 'top20' = filter pick, its
                 'pool_after_norm' = the oracle shortlist).
    universe   : the exact as-of-D scored-universe scope (dm.pit_universe).
    exchange_filter : passed to dm.pit_universe.  None = the NA1 default this baseline has
                 always run on; dm.ALL_EXCHANGES opens it to the scored universe.  Threaded
                 2026-08-22; before that the random/oracle rungs were drawn from an
                 NA1-only universe while the filter they are benchmarked against picks from
                 a much wider one -- so the "random floor" was a floor for a different game.
    """
    D = pd.Timestamp(buy)
    universe = dm.pit_universe(dmdic, registry, as_of=buy,
                               exchange_filter=exchange_filter)
    bm_pit = _pit_bm(merged, set(universe), D)
    # cdx PIT slice passed so Stage-1 reads the `period`-derived frequency map, in
    # LOCKSTEP with live postBo (review S3): cdx is the only frame that carries `period`.
    _cdx_pit = merged['cdx_df']
    _cdx_pit = _cdx_pit[_cdx_pit['source'].isin(set(universe))]
    _cdx_pit = _cdx_pit[pd.to_datetime(_cdx_pit['date'], errors='coerce') <= D]
    bo = s2.stage1_boscore(bm_pit, nq_stage1=nq_stage1,
                           cdx_pit=_cdx_pit)   # full ranked BoScore (public)
    scored = bo["source"].tolist()
    rung_sets = {"universe": scored, "top200": scored[:200], "top100": scored[:100]}

    res = s2.reproduce_pit_top(merged, buy, universe_override=universe)
    if res is None:
        raise RuntimeError(f"reproduce_pit_top returned None at {buy} (empty PIT frame)")

    # CORRECTION #1 consistency guard: our derived Stage-1 top-100 set must equal the
    # certified reproduce_pit_top stage1_top100 -- proves the random rungs are drawn from
    # exactly the filter's own scored/ranked universe, not a divergent set.
    if set(rung_sets["top100"]) != set(res["stage1_top100"]):
        raise AssertionError(
            f"stage1 top-100 mismatch at {buy}: derived set != reproduce_pit_top "
            f"(|derived|={len(rung_sets['top100'])}, |cert|={len(res['stage1_top100'])}, "
            f"sym_diff={len(set(rung_sets['top100']) ^ set(res['stage1_top100']))})")
    return rung_sets, res, universe


# --------------------------------------------------------------------------- #
#  Return / beat helpers (thin views over returns_core)                       #
# --------------------------------------------------------------------------- #
def _name_flags_and_returns(rdf, bench, threshold, missing):
    """Per-INCLUDED-name (buy leg present) beat flag + primary total return, applying the
    SAME missing-eval policy as returns_core.beat_rate.  Returns (flags[bool], rets[float])
    aligned; 'drop' skips terminal names from both."""
    inc = rc.included(rdf)
    flags, rets = [], []
    for _, row in inc.iterrows():
        r = float(row["total_return"])
        if row["terminal_flag"]:
            if missing == "drop":
                continue
            if missing == "zero":
                flags.append((0.0 - bench) >= threshold)
                rets.append(r)
                continue
            # 'fail'
            flags.append(False)
            rets.append(r)
            continue
        flags.append((r - bench) >= threshold)
        rets.append(r)
    return np.array(flags, dtype=bool), np.array(rets, dtype=float)


def _clean_and_winsor(returns, contam_cap, winsor):
    """(mean_dropped, mean_winsor, n_contam, contam_returns) for a return vector.
    mean_dropped : mean after DROPPING names above contam_cap (correction #3).
    mean_winsor  : mean after percentile-clipping the dropped vector (robust).
    """
    r = np.asarray([x for x in returns if x == x], dtype=float)
    if len(r) == 0:
        return float("nan"), float("nan"), 0, np.array([])
    contam = r[r > contam_cap]
    r_clean = r[r <= contam_cap]
    if len(r_clean) == 0:
        return float("nan"), float("nan"), int(len(contam)), contam
    mean_dropped = float(r_clean.mean())
    lo, hi = np.quantile(r_clean, winsor[0]), np.quantile(r_clean, winsor[1])
    mean_winsor = float(np.clip(r_clean, lo, hi).mean())
    return mean_dropped, mean_winsor, int(len(contam)), contam


def _percentile_of(value, samples):
    """Where `value` sits within `samples` (share of samples strictly below it), in %."""
    s = np.sort(np.asarray([x for x in samples if x == x], dtype=float))
    if len(s) == 0 or value != value:
        return float("nan")
    return 100.0 * float(np.searchsorted(s, value, side="left")) / len(s)


def _dist(samples):
    s = np.asarray([x for x in samples if x == x], dtype=float)
    if len(s) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p5": float("nan"),
                "p95": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(s.mean()), "median": float(np.median(s)),
            "p5": float(np.quantile(s, 0.05)), "p95": float(np.quantile(s, 0.95)),
            "std": float(s.std()), "n": int(len(s))}


# --------------------------------------------------------------------------- #
#  Filter (the deterministic actual pick)                                     #
# --------------------------------------------------------------------------- #
def _filter_metrics(per_window_ctx, ps, threshold, missing, contam_cap, winsor):
    num = den = 0.0
    all_rets = []
    per_window = {}
    for buy, ev, res, _sets in per_window_ctx:
        top20 = res["top20"]
        rdf = rc.compute_returns(top20, buy, ev, ps)
        bench = rc.benchmark_return(ps, buy, ev, require_exact=True)
        r, n = rc.beat_rate(rdf, bench, threshold=threshold, missing=missing)
        flags, rets = _name_flags_and_returns(rdf, bench, threshold, missing)
        all_rets.extend(rets.tolist())
        if r == r and n:
            num += r * n
            den += n
        per_window[buy] = {"beat_rate": r, "n": int(n), "n_picked": len(top20),
                           "bench_return": bench}
    mean_dropped, mean_winsor, n_contam, _ = _clean_and_winsor(all_rets, contam_cap, winsor)
    return {
        "beat_rate": {"pooled": (num / den if den else float("nan")), "n": int(den),
                      "per_window": per_window},
        "dollar_return": {"mean_dropped": mean_dropped, "mean_winsor": mean_winsor,
                          "n_contam": n_contam},
    }


# --------------------------------------------------------------------------- #
#  Oracle-best-N (perfect-hindsight ceiling)                                  #
# --------------------------------------------------------------------------- #
def _oracle_metrics(per_window_ctx, ps, oracle_ns, threshold, missing, contam_cap, winsor):
    """For each N: pooled ceiling beat-rate (pick the N best-by-beat names per window)
    and pooled ceiling dollar mean (pick the N best-by-return names per window)."""
    out = {}
    for N in oracle_ns:
        num = den = 0.0
        picked_rets = []
        pool_sizes = []
        for buy, ev, res, _sets in per_window_ctx:
            # D3 CAVEAT: the oracle shortlist is the UN-deduped pool_after_norm, while the
            # filter pick (top20) is issuer-DEDUPED.  So the oracle ceiling is drawn from a
            # slightly larger set than the filter chooses from -> the ceiling is a touch
            # LOOSE, and the top100->filter ladder step conflates the dedup collapse with
            # fine-ranking skill.  Directional, not exact.
            pool = res["pool_after_norm"]
            rdf = rc.compute_returns(pool, buy, ev, ps)
            bench = rc.benchmark_return(ps, buy, ev, require_exact=True)
            flags, rets = _name_flags_and_returns(rdf, bench, threshold, missing)
            m = len(flags)
            pool_sizes.append(m)
            if m == 0:
                continue
            k = min(N, m)
            # beat ceiling: pick the k names most likely to beat -> min(#beaters, k)
            num += float(min(int(flags.sum()), k))
            den += k
            # dollar ceiling: the k highest realized total returns (contaminants dropped
            # first so a near-zero-buy artifact can't fake an oracle dollar ceiling)
            rets_clean = np.sort(rets[rets <= contam_cap])[::-1]
            picked_rets.extend(rets_clean[:k].tolist())
        mean_dropped, mean_winsor, _n_contam, _ = _clean_and_winsor(
            picked_rets, contam_cap, winsor)
        out[N] = {
            "beat_rate": (num / den if den else float("nan")),
            "n": int(den),
            "dollar_return": {"mean_dropped": mean_dropped, "mean_winsor": mean_winsor},
            "pool_sizes": pool_sizes,
        }
    return out


# --------------------------------------------------------------------------- #
#  Random-N Monte-Carlo over one rung                                         #
# --------------------------------------------------------------------------- #
def _random_rung(rung_name, per_window_ctx, ps, pick_n, n_draws, threshold, missing,
                 rng, contam_cap, winsor):
    """Monte-Carlo distribution of a random pick_n-pick drawn from `rung_name`'s set in
    each window, pooled across windows per draw (mirrors the filter's pooled beat-rate)."""
    # Precompute the rung's per-window return table ONCE (draws just index into it).
    per_window = []
    set_sizes = {}
    for buy, ev, _res, sets in per_window_ctx:
        names = sets[rung_name]
        rdf = rc.compute_returns(names, buy, ev, ps).reset_index(drop=True)
        bench = rc.benchmark_return(ps, buy, ev, require_exact=True)
        per_window.append((rdf, bench))
        set_sizes[buy] = len(names)

    beat_samples, dollar_dropped, dollar_winsor = [], [], []
    for _ in range(n_draws):
        num = den = 0.0
        rets_pool = []
        for rdf, bench in per_window:
            m = len(rdf)
            if m == 0:
                continue
            k = min(pick_n, m)
            idx = rng.choice(m, size=k, replace=False)
            sub = rdf.iloc[idx]
            r, n = rc.beat_rate(sub, bench, threshold=threshold, missing=missing)
            if r == r and n:
                num += r * n
                den += n
            _flags, rr = _name_flags_and_returns(sub, bench, threshold, missing)
            rets_pool.extend(rr.tolist())
        beat_samples.append(num / den if den else float("nan"))
        md, mw, _nc, _ = _clean_and_winsor(rets_pool, contam_cap, winsor)
        dollar_dropped.append(md)
        dollar_winsor.append(mw)

    # contaminants present in the rung as a whole (for the correction-#3 report)
    all_rets = []
    for rdf, _bench in per_window:
        inc = rc.included(rdf)
        all_rets.extend(inc["total_return"].tolist())
    _md, _mw, n_contam, contam = _clean_and_winsor(all_rets, contam_cap, winsor)

    return {
        "set_sizes_per_window": set_sizes,
        "beat_rate": _dist(beat_samples),
        "dollar_return_dropped": _dist(dollar_dropped),
        "dollar_return_winsor": _dist(dollar_winsor),
        "beat_samples": beat_samples,
        "dollar_dropped_samples": dollar_dropped,
        "n_contam_in_rung": n_contam,
        "contam_returns": [float(x) for x in np.sort(contam)[::-1][:10]],
    }


# --------------------------------------------------------------------------- #
#  ENTRY POINT                                                                #
# --------------------------------------------------------------------------- #
def run_skill_baseline(dmdic, merged, registry, price_source, *,
                       cadence_months=36, windows=None, pick_n=20,
                       oracle_ns=(3, 20), ladder_rungs=("universe", "top200", "top100"),
                       n_draws=1000, threshold=0.10, missing="fail", seed=0,
                       contam_return_cap=DEFAULT_CONTAM_RETURN_CAP,
                       winsor=DEFAULT_WINSOR, nq_stage1=8,
                       guard_short_history_eps=False, exchange_filter=None,
                       log=None):
    """In-memory skill baseline.  All inputs are OBJECTS (no file paths, no pickle load).

    Parameters
    ----------
    dmdic         : the LIVE scoring pickle dict (survivor-only; used for pit_universe /
                    Tickers_df).  Loaded by the caller (or by `load_inputs`).
    merged        : the dead-MERGED dmdic (dm.merge_dead_into_dmdic(dmdic, dead, registry,
                    as_of=<=earliest buy)) -- the frames the filter is scored on.  Pass
                    `dmdic` unchanged only if you deliberately want the survivor-only run.
    registry      : delisted-registry DataFrame (dm.load_registry) for per-window universes.
    price_source  : a returns_core.PriceSource.
    cadence_months: hold length; 36 (buy-hold) or 12 (annual).  Ignored if `windows` given.
    windows       : explicit [(buy, eval), ...]; overrides cadence_months.
    pick_n        : pick size N for filter / random / the common-N sanity ordering.
    oracle_ns     : the N values reported for the oracle ceiling.
    n_draws       : Monte-Carlo draws per rung (>=1000 recommended).
    threshold     : beat margin (0.10 = +10pp vs URTH).
    missing       : missing-eval policy ('fail' = delisted-at-eval counts as not beating).
    seed          : RNG seed (fixed -> reproducible).
    exchange_filter : PIT universe exchange scope (None = NA1 default, dm.ALL_EXCHANGES =
                    no restriction, or an explicit exchangeShortName iterable).

    Returns a structured dict (see module docstring); every number is derived from the
    certified primitives.
    """
    log = log or (lambda *a: None)
    rng = np.random.default_rng(seed)
    if windows is None:
        windows = clean_windows(cadence_months)
    if not windows:
        raise ValueError(f"no clean windows for cadence_months={cadence_months}")

    # Resolve per-window sets + filter result ONCE (Stage-1 is the expensive step).
    # The short-history-EPS guard is active only for this scoring block (see finding above).
    _short_history_eps_sources.clear()
    per_window_ctx = []      # (buy, eval, filter_res, rung_sets)
    with short_history_eps_guard(enabled=guard_short_history_eps):
        for buy, ev in windows:
            log(f"[skill_baseline] window {buy} -> {ev}: sets + filter top-{pick_n} ...")
            sets, res, _uni = window_sets(dmdic, merged, registry, buy,
                                          nq_stage1=nq_stage1,
                                          exchange_filter=exchange_filter)
            per_window_ctx.append((buy, ev, res, sets))
    if _short_history_eps_sources:
        log(f"[skill_baseline] short-history-EPS guard fired for "
            f"{len(_short_history_eps_sources)} name(s): "
            f"{sorted(_short_history_eps_sources)[:10]}")

    log("[skill_baseline] filter metrics ...")
    filt = _filter_metrics(per_window_ctx, price_source, threshold, missing,
                           contam_return_cap, winsor)

    log(f"[skill_baseline] oracle ceilings N={list(oracle_ns)} ...")
    oracle = _oracle_metrics(per_window_ctx, price_source, list(oracle_ns), threshold,
                             missing, contam_return_cap, winsor)

    random = {}
    for rung in ladder_rungs:
        log(f"[skill_baseline] random-{pick_n} Monte-Carlo x{n_draws} on rung '{rung}' ...")
        rr = _random_rung(rung, per_window_ctx, price_source, pick_n, n_draws,
                          threshold, missing, rng, contam_return_cap, winsor)
        rr["filter_beat_percentile"] = _percentile_of(
            filt["beat_rate"]["pooled"], rr["beat_samples"])
        rr["filter_dollar_percentile"] = _percentile_of(
            filt["dollar_return"]["mean_dropped"], rr["dollar_dropped_samples"])
        rr["oracle_beat_percentile"] = {
            N: _percentile_of(oracle[N]["beat_rate"], rr["beat_samples"])
            for N in oracle_ns}
        random[rung] = rr

    # ---- decomposition ladders (headline views) ----
    beat_ladder = [(rung, random[rung]["beat_rate"]["mean"]) for rung in ladder_rungs]
    beat_ladder.append((f"filter_top{pick_n}", filt["beat_rate"]["pooled"]))
    dollar_ladder = [(rung, random[rung]["dollar_return_dropped"]["mean"])
                     for rung in ladder_rungs]
    dollar_ladder.append((f"filter_top{pick_n}", filt["dollar_return"]["mean_dropped"]))

    # ---- sanity: oracle >= filter >= random at the common pick_n ----
    base_rung = "universe"
    rnd_beat = random[base_rung]["beat_rate"]["mean"]
    flt_beat = filt["beat_rate"]["pooled"]
    orc_beat = oracle[pick_n]["beat_rate"] if pick_n in oracle else float("nan")
    sanity = {
        "pick_n": pick_n,
        "oracle_beat": orc_beat, "filter_beat": flt_beat, "random_beat_universe": rnd_beat,
        "oracle_ge_filter": (orc_beat >= flt_beat) if orc_beat == orc_beat else None,
        "filter_ge_random": (flt_beat >= rnd_beat) if flt_beat == flt_beat else None,
        "ordering_holds": (orc_beat >= flt_beat >= rnd_beat)
                          if (orc_beat == orc_beat and flt_beat == flt_beat) else None,
        # D4 CAVEAT: oracle and filter use DIFFERENT denominators (oracle den = per-window
        # min(N, priceable pool); filter den = priceable deduped top-N).  In a low-
        # priceability edge these bases diverge enough that ordering_holds can read False
        # SPURIOUSLY even when the true ordering holds.  It is a REPORTED diagnostic flag,
        # NOT a guard -- nothing branches on it.
        "note": ("ordering_holds is a reported flag, not a guard; oracle/filter "
                 "denominator asymmetry can make it read False spuriously in a "
                 "low-priceability edge."),
    }

    return {
        "config": {
            "cadence_months": cadence_months, "windows": windows, "pick_n": pick_n,
            "oracle_ns": list(oracle_ns), "ladder_rungs": list(ladder_rungs),
            "n_draws": n_draws, "threshold": threshold, "missing": missing, "seed": seed,
            "contam_return_cap": contam_return_cap, "winsor": winsor,
            "clean_buy_anchors": CLEAN_BUY_ANCHORS,
            #  RECORDED because it silently changes what "the universe" means, and a
            #  report that does not say which exchanges it drew from cannot be compared
            #  with one that drew from a different set.
            #  isinstance-guarded: a bare `== dm.ALL_EXCHANGES` against a numpy array or a
            #  pandas Index is ELEMENTWISE and the conditional raises "truth value of an
            #  array is ambiguous", which would kill the whole report at the last line.
            "exchange_filter": (
                "na1 (NYSE/NASDAQ/TSX)" if exchange_filter is None
                else ("all" if (isinstance(exchange_filter, str)
                                and exchange_filter == dm.ALL_EXCHANGES)
                      else list(exchange_filter))),
            "note": ("pre-2021 buy anchors EXCLUDED (universe-degenerate). "
                     "oracle shortlist = the Stage-2 pool the top-N is chosen from "
                     "(reproduce_pit_top pool_after_norm)."),
        },
        "filter": filt,
        "oracle": oracle,
        "random": random,
        "ladder": {"beat_rate": beat_ladder, "dollar_return": dollar_ladder},
        "sanity": sanity,
        "short_history_eps_guard": {
            "enabled": guard_short_history_eps,
            "fired_for": sorted(_short_history_eps_sources),
            "note": ("stage2_metrics.eps_to_eps_mean crashes on <4-row frames; guarded to "
                     "return 0 (its own no-EWMA value). PERMANENT fix belongs in that file."),
        },
        "corrections_applied": {
            "universe_match": "random rungs derived from Stage-1 BoScore; top-100 set "
                              "asserted == reproduce_pit_top.stage1_top100",
            "dead_name_parity": f"symmetric missing='{missing}', variable denom "
                                "(no_buy excluded both sides), dead-merged PIT universe",
            "mean_return_contamination": f"dollar means drop returns > {contam_return_cap} "
                                         f"(={contam_return_cap*100:.0f}%) and report a "
                                         f"winsorized mean; no raw contaminated mean reported",
        },
    }


# --------------------------------------------------------------------------- #
#  Convenience loader (for __main__ / tests -- keeps run_skill_baseline pure) #
# --------------------------------------------------------------------------- #
def load_inputs(pickle_path, dead_path, registry_path, prices_csv, prices_2025_csv=None,
                merge_as_of="2018-12-31", log=None):
    """Load the objects run_skill_baseline needs, from disk (read-only).

    Returns (dmdic, merged, registry, price_source).  Prices are read only through
    PriceSource (contents never dumped); no api_key is ever printed.
    """
    log = log or (lambda *a: None)
    log("loading base scoring pickle (offline) ...")
    dmdic = pd.read_pickle(pickle_path)
    log("loading dead-fundamentals pickle + registry ...")
    dead = pd.read_pickle(dead_path)
    registry = dm.load_registry(registry_path)
    log(f"dead-merge as-of {merge_as_of} (once; slow) ...")
    merged, _stats = dm.merge_dead_into_dmdic(dmdic, dead, registry, as_of=merge_as_of)
    ps = rc.PriceSource(prices_csv, supp_csv=prices_2025_csv)
    return dmdic, merged, registry, ps


# --------------------------------------------------------------------------- #
#  Report formatter                                                           #
# --------------------------------------------------------------------------- #
def format_report(result):
    c = result["config"]
    L = ["=" * 92,
         "SKILL BASELINE  --  oracle-best-N ceiling + random-N floor (offline)",
         "=" * 92,
         f"cadence={c['cadence_months']}mo  pick_n={c['pick_n']}  draws={c['n_draws']}  "
         f"seed={c['seed']}  threshold=+{c['threshold']*100:.0f}pp vs URTH  "
         f"missing='{c['missing']}'",
         f"windows: {c['windows']}",
         f"PIT universe exchange scope: {c['exchange_filter']}",
         ""]
    f = result["filter"]
    L.append(f"FILTER  top-{c['pick_n']}  beat-rate = {f['beat_rate']['pooled']*100:5.1f}% "
             f"(n={f['beat_rate']['n']})   "
             f"mean dollar return = {f['dollar_return']['mean_dropped']*100:5.1f}% "
             f"(winsor {f['dollar_return']['mean_winsor']*100:.1f}%)")
    L.append("")
    L.append("ORACLE-BEST-N (perfect-hindsight CEILING -- NOT operational):")
    for N, o in result["oracle"].items():
        L.append(f"  N={N:<3} beat-rate = {o['beat_rate']*100:5.1f}% (n={o['n']})   "
                 f"mean dollar = {o['dollar_return']['mean_dropped']*100:6.1f}%")
    L.append("")
    L.append("RANDOM-N Monte-Carlo per rung (beat-rate distribution):")
    L.append(f"  {'rung':10} {'mean':>6} {'p5':>6} {'p95':>6}  {'filter_pctile':>13} "
             f"{'n_contam':>9}")
    for rung, rr in result["random"].items():
        b = rr["beat_rate"]
        L.append(f"  {rung:10} {b['mean']*100:5.1f}% {b['p5']*100:5.1f}% {b['p95']*100:5.1f}%"
                 f"  {rr['filter_beat_percentile']:12.1f}% {rr['n_contam_in_rung']:>9}")
    L.append("")
    L.append("DECOMPOSITION LADDER (beat-rate):")
    for name, v in result["ladder"]["beat_rate"]:
        L.append(f"  {name:16} {v*100:5.1f}%")
    L.append("DECOMPOSITION LADDER (mean dollar return, contaminants dropped):")
    for name, v in result["ladder"]["dollar_return"]:
        L.append(f"  {name:16} {v*100:6.1f}%")
    L.append("")
    s = result["sanity"]
    L.append(f"SANITY (common N={s['pick_n']}): oracle {s['oracle_beat']*100:.1f}% >= "
             f"filter {s['filter_beat']*100:.1f}% >= random(universe) "
             f"{s['random_beat_universe']*100:.1f}%  -> ordering_holds={s['ordering_holds']}")
    L.append("=" * 92)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  __main__ (thin; the module is importable and callable without it)          #
# --------------------------------------------------------------------------- #
def _default_paths():
    home = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
    return {
        "pickle": os.path.join(
            home, "Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-13_len7879_manelim3692_fails1966.pickle"),
        "dead": os.path.join(home, "delisted_out", "dead_fundamentals_20260713_104350.pickle"),
        "registry": os.path.join(home, "delisted_out", "delisted_registry.csv"),
        "prices": os.path.join(_HERE, "price_data", "real_prices.csv"),
        "prices_2025": os.path.join(_HERE, "price_data", "real_prices_2025.csv"),
    }


def main():
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    p = _default_paths()
    ap = argparse.ArgumentParser(description="Run the skill baseline on local data.")
    ap.add_argument("--pickle", default=p["pickle"])
    ap.add_argument("--dead", default=p["dead"])
    ap.add_argument("--registry", default=p["registry"])
    ap.add_argument("--prices", default=p["prices"])
    ap.add_argument("--prices-2025", default=p["prices_2025"])
    ap.add_argument("--cadence", type=int, default=36, choices=[12, 36])
    ap.add_argument("--pick-n", type=int, default=20)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    #  Default 'na1' = the scope this baseline has always run; see dm.pit_universe.
    ap.add_argument("--exchanges", default="na1",
                    help="PIT universe exchange scope: 'na1' (default, unchanged), 'all', "
                         "or a comma-separated exchangeShortName list")
    args = ap.parse_args()

    log = lambda *a: print(*a, file=sys.stderr, flush=True)
    for lbl, path in (("pickle", args.pickle), ("dead", args.dead),
                      ("registry", args.registry), ("prices", args.prices)):
        if not os.path.exists(path):
            log(f"FATAL missing {lbl}: {path}")
            sys.exit(2)

    dmdic, merged, registry, ps = load_inputs(
        args.pickle, args.dead, args.registry, args.prices, args.prices_2025, log=log)
    exch = dm.resolve_exchange_filter(args.exchanges)
    log(f"[universe] exchange scope = {args.exchanges!r} -> "
        f"{'NA1 (NYSE/NASDAQ/TSX)' if exch is None else exch}")
    result = run_skill_baseline(dmdic, merged, registry, ps, cadence_months=args.cadence,
                                pick_n=args.pick_n, n_draws=args.draws, seed=args.seed,
                                oracle_ns=(3, args.pick_n), exchange_filter=exch, log=log)
    print(format_report(result))


if __name__ == "__main__":
    main()
