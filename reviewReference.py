"""
reviewReference.py  --  READ-ONLY human-review reference DATA artifacts.

Emits three reference artifacts for the CEO's manual review of a run's shortlist.
They are DECISION-SUPPORT DATA ONLY: nothing here feeds scoring or the ranking.

  ARTIFACT 1  RawMetricsTop100-<date>_<datasource>.csv
              One row per shipped general top-100 name (carve-on / deduped general):
              source, carve_label, AggScore, rankOfRanks, the RAW value of each
              playbook metric (returnOnCapitalEmployed ... CycleHeat), and -- folded
              in as ARTIFACT 3 -- a <metric>_pct column giving that name's percentile
              within its OWN cohort's full distribution.

  ARTIFACT 2  CohortMetricStats-<date>_<datasource>.csv   (LONG form)
              One row per (cohort, metric): count, mean, std, min, p10, p25, p50, p75,
              p90, max over the FULL carved + size-floored cohort membership this run --
              EVERY peer (every miner, every REIT, ...), including names that never
              reached Stage-2 scoring -- so it is a true peer benchmark, not a stat over
              the 25 top-scored names.  Median + quartiles are the headline; mean is kept
              but is fragile on small/skewed cohorts.

  ARTIFACT 3  per-name cohort percentile within that FULL cohort distribution -- folded
              into ARTIFACT 1 as <metric>_pct.

CRITICAL BOUNDARY (do not violate -- CEO-ratified):
  These artifacts are computed from a COPY of the RAW metrics captured BEFORE
  normalizeAndDropNA (postBoRank.postBoScoreRanking -> rankdic['postScoreMetric_raw'])
  and emitted AFTER scoring.  NOTHING here is EVER read back into postScoreMetric_df /
  getAggScore / normalizeAndDropNA / the ranking.  Feeding cohort means/percentiles
  into the score would be cross-sectional sector-neutralization (a cheap miner would be
  re-centered against other miners instead of scored on absolute terms), which is
  CEO-ratified OFF (risk-premium principle).  This module only READS a result dict and
  WRITES CSVs.  It is also DISTINCT from carveOut.COHORT_WEIGHTS (within-cohort side-list
  reweighting) -- it must never become a backdoor to cross-sectional neutralization.

CURRENCY GUARD (ARTIFACT 2/3):
  Distributions are computed ONLY over currency-INVARIANT ratios/scores -- the
  PLAYBOOK_METRICS allow-list (margins / returns / yields / growth / Piotroski /
  Altman-Z / moatScore / CycleHeat).  Absolute-currency fields (price, marketCap) are
  never even present in the metric frame, and currency-denominated scorer inputs that
  ARE present (EPStoEPSmean in per-share earnings units; marketCapRevQuants, a pool
  rank) are deliberately EXCLUDED by the allow-list.  A cohort mixes GBp / USD / CAD, so
  aggregating an absolute-currency field would be meaningless.
"""

import os
import sys

import numpy as np
import pandas as pd

# The playbook metrics, in ARTIFACT-1 column order.  Every one is currency-INVARIANT.
# This IS the allow-list: ARTIFACT 2/3 only ever touch these, so absolute-currency /
# currency-denominated scorer inputs (price, marketCap, EPStoEPSmean, marketCapRevQuants,
# BoScore, DcfToPrice, priceGrowth, grahamNumberToPrice) can never leak into a cohort
# distribution.
PLAYBOOK_METRICS = [
    'returnOnCapitalEmployed', 'returnOnEquity', 'RoA', 'grossProfitMargin',
    'freeCashFlowYield', 'currentRatio', 'earnYield', 'revenueGrowth',
    'incomeQuality', 'Altman-Z', 'Piotroski', 'bVpRatio', 'tbVpRatio',
    'freeCashFlowPerShareGrowth', 'moatScore', 'CycleHeat',
]

# Fields that must NEVER be aggregated as a cohort distribution (absolute-currency or
# currency-denominated).  Kept as an explicit assertion target so a future edit that
# widens the metric set trips the guard rather than silently neutralizing by currency.
_CURRENCY_ABSOLUTE_DENY = {
    'price', 'marketCap', 'EPStoEPSmean', 'marketCapRevQuants',
}

_ALL_COHORTS = ['general', 'REIT', 'Mining', 'InvestmentVehicle',
                'FinManager', 'BalanceSheetFin']

# postBmRankingDict keys among the currency-invariant playbook set (grahamNumberToPrice
# is a postBm key but is NOT in PLAYBOOK_METRICS, so it is excluded here).
_CI_BM_KEYS = ['RoA', 'earnYield', 'bVpRatio', 'revenueGrowth', 'incomeQuality',
               'returnOnEquity', 'returnOnCapitalEmployed', 'currentRatio',
               'grossProfitMargin']


# --------------------------------------------------------------------------- #
#  Full-membership raw-metric computation (from cdx_df; scales to the universe) #
# --------------------------------------------------------------------------- #
def _pool_raw_fast(sources, cdx_df, nq=16):
    """Compute the currency-invariant playbook metrics (all of PLAYBOOK_METRICS except
    moatScore, which is merged separately) for `sources`, straight from cdx_df.

    Uses the SAME stage2_metrics functions as the live scorer and the certified offline
    PIT loop (stage2_pit._stage2_metric_loop_offline) -- so the values are bit-for-bit
    identical (verified: max abs diff 0.0) -- but builds the frame in O(n) via a single
    groupby('source') instead of that loop's O(n^2) per-row .loc assignment.  This is
    what lets the cohort distributions cover the FULL carved+size-floored membership
    (thousands of names) rather than just the Stage-2-scored subset.

    None of these metrics is pool-relative (each is a per-ticker reduction over the
    name's own history), so a name's value is identical whether computed over the scored
    pool or the full membership -- only marketCapRevQuants would differ, and that
    pool-rank field is deliberately NOT a playbook metric.
    """
    import stage2_metrics as sm
    import createDicts as cdic

    postBm, _postNew = cdic.getPostDict()
    sub = cdx_df[cdx_df['source'].isin(set(sources))].copy()
    if sub.empty:
        return pd.DataFrame(columns=['source'])
    sub['date'] = pd.to_datetime(sub['date'], errors='coerce')
    sub = sub.sort_values(['source', 'date'], ascending=[True, False])   # newest-first
    rows = []
    for src, t in sub.groupby('source', sort=False):
        r = {'source': src}
        for k in _CI_BM_KEYS:
            r[k] = sm.postbm_metric(k, postBm[k]['eqMet'], t, nq)
        r['freeCashFlowYield'] = sm.free_cash_flow_yield(t.freeCashFlow, t.marketCap, nq)
        r['freeCashFlowPerShareGrowth'] = sm.free_cash_flow_per_share_growth(
            t.freeCashFlow, t.weightedAverageShsOut, nq)
        r['tbVpRatio'] = sm.tbv_p_ratio(t, nq)
        r['Altman-Z'] = sm.altman_z(t)
        r['Piotroski'] = sm.piotroski(t)
        r['CycleHeat'] = sm.cycleheat(t)
        rows.append(r)
    return pd.DataFrame(rows)


def full_membership_pools(membership, cdx_df, moatdf=None, nq=16):
    """dist_pools = {cohort_label: full-membership raw metric df} computed from cdx_df.

    membership : {cohort_label: [source, ...]} -- the FULL carved + size-floored
                 membership per cohort (every peer, incl. names never Stage-2-scored).
    Returns pools ready for cohort_stats_long / percentile distributions, with moatScore
    merged on where available.
    """
    pools = {}
    for label, sources in membership.items():
        if sources:
            pools[label] = _pool_raw_fast(sources, cdx_df, nq=nq)
    return _merge_moat(pools, moatdf)


# --------------------------------------------------------------------------- #
#  Percentile helper (scipy-free; matches scipy.percentileofscore kind='mean') #
# --------------------------------------------------------------------------- #
def _percentile_of(dist, value):
    """Percentile of `value` within 1-D array-like `dist` (NaNs dropped).

    Equivalent to scipy.stats.percentileofscore(dist, value, kind='mean'):
    the average of the 'weak' (<=) and 'strict' (<) ranks, in [0, 100].
    Returns NaN when `value` is NaN or the distribution is empty.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    a = np.asarray(dist, dtype='float64')
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return np.nan
    strict = np.count_nonzero(a < value)
    weak = np.count_nonzero(a <= value)
    return 100.0 * (strict + weak) / (2.0 * n)


# --------------------------------------------------------------------------- #
#  Core builders (path-agnostic: driven by raw_pools + the shipped ranking)   #
# --------------------------------------------------------------------------- #
def _finite_series(s):
    """Coerce to numeric and map +/-inf to NaN.

    The RAW capture is taken BEFORE normalizeAndDropNA, which is exactly where the
    scorer replaces inf with NaN (an undefined ratio, e.g. bVpRatio = 1/pbRatio when
    pbRatio == 0).  The reference distributions and displayed raw values mirror that:
    inf becomes blank, so a single divide-by-zero row can't turn a cohort's mean / max /
    p90 into inf.  This is the SAME inf handling the scorer applies -- not a new rule.
    """
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _present_metrics(df):
    """PLAYBOOK_METRICS actually present in `df` (order preserved). Also asserts no
    currency-absolute field sneaks into the working set."""
    cols = set(df.columns)
    present = [m for m in PLAYBOOK_METRICS if m in cols]
    leaked = _CURRENCY_ABSOLUTE_DENY & set(present)
    assert not leaked, ("reviewReference currency guard: absolute-currency field(s) %s "
                        "must never be aggregated" % sorted(leaked))
    return present


def cohort_stats_long(raw_pools):
    """ARTIFACT 2: LONG-form (cohort, metric, count, mean, std, min, p10, p25, p50,
    p75, p90, max) over each pool's raw values.

    raw_pools : dict {cohort_label: DataFrame with 'source' + raw metric columns}.
    Only currency-invariant PLAYBOOK_METRICS present in a pool are aggregated.
    """
    rows = []
    for cohort in _ALL_COHORTS:
        df = raw_pools.get(cohort)
        if df is None or df.empty:
            continue
        for metric in _present_metrics(df):
            s = _finite_series(df[metric]).dropna()
            if s.empty:
                continue
            arr = s.to_numpy(dtype='float64')
            rows.append({
                'cohort': cohort,
                'metric': metric,
                'count': int(arr.size),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan,
                'min': float(np.min(arr)),
                'p10': float(np.percentile(arr, 10)),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'p90': float(np.percentile(arr, 90)),
                'max': float(np.max(arr)),
            })
    return pd.DataFrame(rows, columns=['cohort', 'metric', 'count', 'mean', 'std',
                                       'min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'])


def raw_top_table(top_rank_df, labels, dist_pools, top_raw_pool=None,
                  with_percentiles=True):
    """ARTIFACT 1 (+ ARTIFACT 3 folded in as <metric>_pct).

    top_rank_df  : the SHIPPED general ranking (postRank head(100)); must carry
                   'source', 'AggScore', 'rankOfRanks' and optionally 'moatScore'.
    labels       : Series source -> carve_label (carve['labels']); None (pre-carve).
    dist_pools   : dict cohort_label -> FULL-membership raw metric DataFrame.  Used ONLY
                   for the ARTIFACT-3 percentile distributions (every peer).
    top_raw_pool : the raw metric DataFrame that supplies the DISPLAYED top-100 RAW
                   values (the live scorer's capture, postScoreMetric_raw).  When None,
                   falls back to dist_pools['general'] (which contains the top-100 with
                   identical values -- the metrics are per-ticker, pool-independent).
    with_percentiles : if True, add a <metric>_pct column per playbook metric giving the
                   name's percentile within its OWN cohort's FULL distribution (ARTIFACT 3).

    The RAW metric values come from top_raw_pool / dist_pools (never from top_rank_df,
    whose metric columns are the post-normalize z-scores).
    """
    general_raw = top_raw_pool if top_raw_pool is not None else dist_pools.get('general')
    if general_raw is None or general_raw.empty:
        raise ValueError("raw_top_table: need a general raw pool "
                         "(top_raw_pool or dist_pools['general'])")

    # NB: moatScore is deliberately NOT taken here -- it is sourced from the raw pool
    # (below) when present, with a top_rank_df fallback, so it never collides in the
    # merge that follows.
    keep = ['source']
    for c in ('AggScore', 'rankOfRanks'):
        if c in top_rank_df.columns and c not in keep:
            keep.append(c)
    out = top_rank_df[keep].copy().reset_index(drop=True)

    # carve_label per name (pre-carve -> a single sentinel so the column is never blank).
    if labels is not None:
        out.insert(1, 'carve_label', out['source'].map(labels).fillna('general'))
    else:
        out.insert(1, 'carve_label', '(pre-carve)')

    # RAW playbook metric values, joined from the general raw pool by source.
    metrics_present = _present_metrics(general_raw)
    raw_block = general_raw[['source'] + metrics_present].copy()
    for m in metrics_present:                      # mirror the scorer's inf -> NaN
        raw_block[m] = _finite_series(raw_block[m])
    out = out.merge(raw_block, on='source', how='left')

    # moatScore raw column: the raw-pool merge above already brought it in when the pool
    # carried it (moatScore is in PLAYBOOK_METRICS).  Otherwise fall back to the shipped
    # ranking's merged moatScore, and finally to NaN -- so ARTIFACT 1 ALWAYS carries a
    # moatScore column (schema stability across live / recompute paths).
    if 'moatScore' not in out.columns:
        if 'moatScore' in top_rank_df.columns:
            out = out.merge(top_rank_df[['source', 'moatScore']].drop_duplicates('source'),
                            on='source', how='left')
        else:
            out['moatScore'] = np.nan

    if with_percentiles:
        # ARTIFACT 3: percentile of each name's raw metric within its OWN cohort's FULL
        # distribution (from dist_pools[label] -- every carved+floored peer).  Top-100
        # names are general-pool names, so the distribution is the full general pool; the
        # per-cohort lookup stays general-keyed so it remains correct if a non-general
        # name ever appears.
        dist_cache = {}

        def _dist(cohort, metric):
            key = (cohort, metric)
            if key not in dist_cache:
                pool = dist_pools.get(cohort)
                if pool is None or metric not in pool.columns:
                    dist_cache[key] = np.array([], dtype='float64')
                else:
                    dist_cache[key] = _finite_series(
                        pool[metric]).dropna().to_numpy('float64')
            return dist_cache[key]

        for metric in metrics_present:
            pct_vals = []
            for _, r in out.iterrows():
                cohort = r['carve_label'] if r['carve_label'] in dist_pools else 'general'
                pct_vals.append(_percentile_of(_dist(cohort, metric), r.get(metric)))
            out[metric + '_pct'] = pct_vals

    return out


def build(dist_pools, top_rank_df, labels, top_raw_pool=None, with_percentiles=True):
    """Build (artifact1_df, artifact2_df).

    dist_pools   : FULL-membership raw pools -> ARTIFACT 2 stats + ARTIFACT 3 percentiles.
    top_raw_pool : the scored-capture general pool -> ARTIFACT 1 displayed raw values
                   (None -> use dist_pools['general']).
    """
    art2 = cohort_stats_long(dist_pools)
    art1 = raw_top_table(top_rank_df, labels, dist_pools, top_raw_pool=top_raw_pool,
                         with_percentiles=with_percentiles)
    return art1, art2


# --------------------------------------------------------------------------- #
#  moatScore merge helper                                                     #
# --------------------------------------------------------------------------- #
def _merge_moat(raw_pools, moatdf):
    """Merge moatScore (full-universe, from moatIdentifier) onto each raw pool so it
    joins the currency-invariant metric set.  No-op if moatdf is missing/empty."""
    if moatdf is None or getattr(moatdf, 'empty', True) or 'moatScore' not in getattr(
            moatdf, 'columns', []):
        return raw_pools
    mm = moatdf[['source', 'moatScore']].dropna(subset=['source']).drop_duplicates('source')
    merged = {}
    for label, df in raw_pools.items():
        if df is None or df.empty or 'moatScore' in df.columns:
            merged[label] = df
        else:
            merged[label] = df.merge(mm, on='source', how='left')
    return merged


# --------------------------------------------------------------------------- #
#  LIVE emission (called from postBo.writeResWrapper)                         #
# --------------------------------------------------------------------------- #
def emit_live(resdic, fidag, datasource, tickerfilter, out_dir='.'):
    """Write ARTIFACTS 1-3 for a LIVE run.

    ARTIFACT 1 displayed raw values come from the live scorer's capture
    (resdic['postScoreMetric_raw']).  ARTIFACT 2/3 distributions are computed over the
    FULL carved+size-floored membership (resdic['carve_full_membership']), recomputed
    from cdx_df -- every peer, not just the Stage-2-scored subset.

    GATE: needs cohort labels + full membership -> returns [] with a printed note when
    carve is absent (a pre-carve run cannot produce cohort labels).  Returns the list of
    written filenames (for the Drive-transfer sync).
    """
    import time
    labels = resdic.get('carveout_labels')
    if labels is None:
        print("reviewReference: carve absent (pre-carve run) -> skipping cohort "
              "reference artifacts (no cohort labels to produce them).", flush=True)
        return []

    top_raw_pool = resdic.get('postScoreMetric_raw')
    if top_raw_pool is None or top_raw_pool.empty:
        print("reviewReference: no captured raw metrics (postScoreMetric_raw absent) "
              "-> skipping reference artifacts.", flush=True)
        return []

    membership = resdic.get('carve_full_membership')
    if not membership:
        print("reviewReference: full cohort membership absent (carve_full_membership) "
              "-> skipping reference artifacts.", flush=True)
        return []

    moatdf = resdic.get('moatdf')
    # ARTIFACT 2/3 distributions: FULL carved+size-floored membership, recomputed from
    # cdx_df (every peer, not just the Stage-2-scored subset).  Timed so the added
    # wall-clock is visible to the CEO.
    t0 = time.time()
    dist_pools = full_membership_pools(membership, resdic['cdx_df'], moatdf)
    dt = time.time() - t0
    n_full = sum(len(v) for v in membership.values())
    print(f"reviewReference: full-membership raw metrics computed for {n_full} names "
          f"across {len(membership)} cohorts in {dt:.1f}s (added wall-clock).", flush=True)

    # ARTIFACT 1 displayed raw values: the live scorer's capture (unchanged).
    top_raw_pool = _merge_moat({'general': top_raw_pool}, moatdf)['general']
    top_rank_df = resdic['postRank'].head(100)
    art1, art2 = build(dist_pools, top_rank_df, labels, top_raw_pool=top_raw_pool,
                       with_percentiles=True)

    f1 = os.path.join(out_dir, f'RawMetricsTop100-{fidag}_{datasource}.csv')
    f2 = os.path.join(out_dir, f'CohortMetricStats-{fidag}_{datasource}.csv')
    art1.to_csv(f1, index=False)
    art2.to_csv(f2, index=False)
    print(f'Review-reference artifacts written: {os.path.basename(f1)}, '
          f'{os.path.basename(f2)} (READ-ONLY; not fed back into scoring).', flush=True)
    return [f1, f2]


# --------------------------------------------------------------------------- #
#  RECOMPUTE-FROM-SAVED emission (called by run_analysis_on_saved_run.py)      #
# --------------------------------------------------------------------------- #
def _saved_membership(dmdic):
    """FULL carved+size-floored membership {cohort: [source,...]} for a saved pickle.

    Prefers the membership STORED by the run (carve_full_membership -- exact to that
    run); else re-derives it by re-running the carve partition from BoScore_df + cdx_df
    (uses local sector/industry pickles, so it can differ slightly from the original run
    if those changed).  Returns None if the pickle is pre-carve.
    """
    m = dmdic.get('carve_full_membership')
    if m:
        return m
    if dmdic.get('carveout_labels') is None:
        return None
    import carveOut as co
    carve = co.partition_universe(dmdic['BoScore_df'], dmdic['cdx_df'],
                                  dmdic.get('Tickers_df'), mcap_floor=25e6, cohort_head=25)
    return {'general': list(carve['general']['source']),
            **{lab: list(cs['source']) for lab, cs in carve['cohorts'].items()}}


def emit_from_saved(dmdic, out_dir='.', date_str=None, datasource=None):
    """Produce ARTIFACTS 1-3 from a SAVED run's pickle (offline; no network, no re-run).

    Same FULL-membership semantics as the live path: ARTIFACT 2/3 distributions are over
    every carved+size-floored peer, recomputed from cdx_df via the pure stage2_metrics
    functions.  Cases, auto-detected:
      (1/2) carve-on pickle: ARTIFACTS 1-3.  Membership from the stored
            carve_full_membership when present (exact to the run), else re-derived by
            re-running the carve partition.  ARTIFACT-1 displayed raw values come from the
            captured postScoreMetric_raw when present, else from the recomputed full
            general pool (identical values -- the metrics are per-ticker).
      (3)   pre-carve pickle (no cohort labels, e.g. the Jan snapshots): recompute raw
            metrics for the top-100 and emit ARTIFACT 1 only, reporting clearly that
            ARTIFACTS 2-3 need a carve-on run.

    Returns {'files': [...], 'case': int, 'note': str, 'added_wall_s': float}.
    """
    import time
    date_str = date_str or dmdic.get('date_created') or 'saved'
    datasource = datasource or dmdic.get('datasource', 'fmp')
    carve_on = dmdic.get('carveout_labels') is not None
    top_rank_df = dmdic['postRank'].head(100)
    moatdf = dmdic.get('moatdf')

    files = []
    added = 0.0
    if carve_on:
        membership = _saved_membership(dmdic)
        t0 = time.time()
        dist_pools = full_membership_pools(membership, dmdic['cdx_df'], moatdf)
        added = time.time() - t0
        stored = dmdic.get('carve_full_membership') is not None
        has_raw = dmdic.get('postScoreMetric_raw') is not None
        top_raw_pool = None
        if has_raw:
            top_raw_pool = _merge_moat({'general': dmdic['postScoreMetric_raw']},
                                       moatdf)['general']
        art1, art2 = build(dist_pools, top_rank_df, dmdic['carveout_labels'],
                           top_raw_pool=top_raw_pool, with_percentiles=True)
        n_full = sum(len(v) for v in membership.values())
        note = ("case %d: carve-on. FULL-membership stats over %d names%s; ARTIFACT-1 raw "
                "from %s." % (1 if stored else 2, n_full,
                             "" if stored else " (carve RE-DERIVED from local pickles)",
                             "captured raw" if has_raw else "recomputed full general pool"))
        case = 1 if stored else 2
    else:
        # ---- Case 3: pre-carve -> ARTIFACT 1 only ----------------------------
        t0 = time.time()
        general_raw = _pool_raw_fast(list(top_rank_df['source']), dmdic['cdx_df'])
        added = time.time() - t0
        dist_pools = _merge_moat({'general': general_raw}, moatdf)
        art1 = raw_top_table(top_rank_df, labels=None, dist_pools=dist_pools,
                             with_percentiles=False)
        art2 = None
        note = ("case 3: PRE-CARVE pickle (no cohort labels). ARTIFACT 1 (raw metrics) "
                "produced; ARTIFACTS 2-3 (cohort stats + percentiles) NEED A CARVE-ON RUN "
                "-- they cannot be produced without cohort membership.")
        case = 3

    f1 = os.path.join(out_dir, f'RawMetricsTop100-{date_str}_{datasource}.csv')
    art1.to_csv(f1, index=False)
    files.append(f1)
    if art2 is not None:
        f2 = os.path.join(out_dir, f'CohortMetricStats-{date_str}_{datasource}.csv')
        art2.to_csv(f2, index=False)
        files.append(f2)

    print("reviewReference.emit_from_saved: " + note, flush=True)
    print("  added wall-clock (full-membership recompute): %.1fs" % added, flush=True)
    print("  wrote: " + ", ".join(os.path.basename(f) for f in files), flush=True)
    return {'files': files, 'case': case, 'note': note, 'added_wall_s': added}
