"""
Stage-2 point-in-time reproduction (OFFLINE) -- the validation-gate engine.

Reproduces the filter's TWO-STAGE shortlist as-of a historical date D, using
ONLY the saved pickle (no network):
  Stage-1  BoScore  -> top-100 pool   (reuses calcScore.simpleScore_fromDict)
  Stage-2  AggScore -> top-20         (reproduces the metric loop offline, then
           reuses production's EXACT normalizeAndDropNA / weighting / getAggScore)

FIDELITY CHOICES (baked in per the fresh-context pre-mortem):
  * AggScore is a CROSS-SECTIONAL Z-NORMALISED weighted sum over the ~100-name
    pool -- means/stds computed over the pool present at D (postBoRank.py:603),
    THEN weighted (:530-532), THEN summed (getAggScore :620-623). We reuse the
    production functions verbatim so this is exact.
  * The metric loop (postBoRank.py:172-441) is reproduced offline. The only two
    metrics that need live APIs are handled thus:
        - DcfToPrice (w=0.35): DROPPED (no PIT DCF; BoDCF.py is broken). The
          column is simply absent -> not weighted, not summed. Recorded.
        - CycleHeat beta: fixed at 1.0 (production's own API-failure fallback,
          postBoRank.py:419-427).
  * Full-history-as-of-D median (getAves2 over date<=D) + SYNTHETIC price for
    SCORING -- faithful to what the filter itself computes (R5).
  * Column names match production's weight_series keys exactly (the postBm/postNew
    dict keys) so weight_series.get(col,1) never silently defaults (:531).

This module imports the repo's own createDicts / calcScore / postBoRank. Importing
postBoRank pulls in `requests`/`matplotlib` but we NEVER call its API paths.
"""

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import createDicts as cdic
import calcScore as csf
import postBoRank as pbr
import stage2_metrics as sm
import reporting_period as rp

NA1_EXCHANGES = ["NYSE", "NASDAQ", "TSX"]
DROP_METRICS = ["DcfToPrice"]  # not reconstructable PIT offline


def na1_symbols(tickers_df):
    return set(tickers_df.loc[
        tickers_df["exchangeShortName"].isin(NA1_EXCHANGES), "symbol"])


def _sort_newest_first(df):
    """Production relies on newest-first ingestion order for .head(nq); enforce it."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d.sort_values(["source", "date"], ascending=[True, False])


def _stage2_metric_loop_offline(bstop, cdxtop, nq=16):
    """Offline reproduction of postBoScoreRanking's per-ticker metric loop
    (postBoRank.py:98-441), minus DcfToPrice, with beta=1.0 for CycleHeat.

    Returns postScoreMetric_df with columns = getPostDict keys (minus dropped)
    + 'source', matching production column names exactly.
    """
    postBm, postNew = cdic.getPostDict()
    cols = ["source"] + list(postBm.keys()) + list(postNew.keys())
    cols = [c for c in cols if c not in DROP_METRICS]
    out = pd.DataFrame(columns=cols)
    out["source"] = bstop["source"].values

    cdxtop = cdxtop.copy()
    # pool-level marketCap quartile code (stage2_metrics.add_mcap_quants)
    cdxtop["mcapQuants"] = sm.add_mcap_quants(cdxtop)
    # PER-SOURCE REPORTING FREQUENCY -- LOCKSTEP with postBoRank.postBoScoreRanking
    # (review H4, 2026-07-25).  Without this the PIT reproduction inherited the new
    # DEFAULTS (Altman full-year sums, Graham EPS_ttm) but NOT the rpy branching, so it
    # was a HYBRID: neither the old nor the new scorer, and non-comparable to live for
    # the 14.4% of names that report semi-annually -- on the very path that produces the
    # charter PRIMARY >=60% beat-rate criterion.
    freq_map = rp.frequency_by_source(cdxtop)

    for ticker in bstop["source"]:
        tempcdx = cdxtop.loc[cdxtop["source"] == ticker]
        if tempcdx.empty:
            continue
        _rpy = rp.rows_per_year(freq_map, ticker)
        tempfcf = tempcdx.freeCashFlow
        tempshares = tempcdx.weightedAverageShsOut
        tempmcapQuants = tempcdx.mcapQuants.iloc[0]

        def setv(col, val):
            out.loc[out["source"] == ticker, col] = val

        # ---- postBmRankingDict metrics ----
        for key1 in postBm:
            setv(key1, sm.postbm_metric(key1, postBm[key1]["eqMet"], tempcdx, nq,
                                        rpy=_rpy))

        # ---- postNewRankingDict metrics (shared with production via stage2_metrics) ----
        setv("freeCashFlowYield", sm.free_cash_flow_yield(tempfcf, tempcdx.marketCap,
                                                         nq, rpy=_rpy))
        setv("freeCashFlowPerShareGrowth",
             sm.free_cash_flow_per_share_growth(tempfcf, tempshares, nq, rpy=_rpy))
        # DcfToPrice: DROPPED offline (no point-in-time DCF; weight 0 in the live vector)
        setv("marketCapRevQuants", tempmcapQuants)
        setv("tbVpRatio", sm.tbv_p_ratio(tempcdx, nq, rpy=_rpy))
        setv("BoScore", float(bstop.loc[bstop["source"] == ticker, "score"].iloc[0]))
        setv("EPStoEPSmean", sm.eps_to_eps_mean(tempcdx, rpy=_rpy))
        setv("priceGrowth", sm.price_growth(tempcdx, nq, rpy=_rpy))
        setv("Altman-Z", sm.altman_z(tempcdx, rpy=_rpy))
        setv("Piotroski", sm.piotroski(tempcdx, rpy=_rpy))
        # CycleHeat: the SAME shared function the live scorer uses.  Its canonical
        # EPS prep (stage2_metrics.prepare_eps_series) collapses duplicate-dated
        # (restated) quarters to the last-ingested figure, so the live and offline
        # paths now agree by construction -- including on the previously-divergent
        # ILMN-type restatement names.
        # rpy passed for LOCKSTEP with live postBoRank (fix 2026-07-30): the window is
        # frequency-scaled, so omitting it here would silently re-introduce the
        # 2x-calendar-span divergence for semi-annual filers in the offline path.
        setv("CycleHeat", sm.cycleheat(tempcdx, rpy=_rpy))

    return out


def reproduce_pit_top(dmdic, D, na1_only=True, nq_stage1=8, nq_stage2=16,
                      topn_stage1=100, topn_final=20, cycleheat_beta=1.0,
                      boscore_noise=0.0, price_noise_frac=0.0, rng=None,
                      universe_override=None, weight_override=None,
                      dedup_issuers=True):
    """Full PIT reproduction as-of date D. Returns a result dict.

    Controlled-noise hooks (for the churn diagnostic, both default OFF):
      boscore_noise   : std of Gaussian added to each BoScore before the top-100
                        cut -> isolates the QUANTIZATION / pool-boundary-flip
                        channel (Stage-1 -> pool membership).
      price_noise_frac: multiplicative Gaussian noise on the NEWEST quarter's
                        synthetic price & marketCap per name -> isolates the
                        Stage-2 price-metric channel (pool held fixed).

    universe_override : optional iterable of `source` values defining the as-of-D
                        universe.  DEFAULT None -> exactly today's behaviour
                        (na1_symbols(Tickers_df) when na1_only).  When supplied (the
                        SURVIVORSHIP-CLEAN path: dead names merged in via
                        dead_merge.merge_dead_into_dmdic, universe from its
                        `pit_universe`), it REPLACES the na1 filter so dead-but-
                        alive-at-D names are scored.  The live path (override=None)
                        is bit-for-bit unchanged.

    weight_override   : optional {metric_name: weight} dict merged over the default
                        getPostDict weight vector (mirrors postBoRank.py:107-108's
                        cohort-weight hook).  DEFAULT None -> default weights, live
                        path bit-for-bit unchanged.  Used by the scoring-comparison
                        grid to run an equal-weight variant (all metric weights = 1).

    dedup_issuers     : collapse same-issuer lines in the ranked pool, keeping the
                        HIGHEST-RANKED line per issuer, BEFORE the head(topn_final)
                        cut (CEO standing principle: no duplicate issuers in the
                        emitted top-N -- the TFPM/TFPM.TO case).  Reuses
                        carveOut.dedup_ranked (same issuer-fingerprint as the carve).
                        DEFAULT True.  It shapes ONLY the emitted `top20`; the returned
                        `pool_after_norm` is the UNDEDUPED full pool (so the tune's
                        validate_finish / bit-for-bit anchor is unaffected).  Set False
                        to reproduce the pre-dedup top-N.  The pre-dedup top-N is always
                        returned as `top20_predupe`, and the dropped lines as
                        `issuer_dupes_dropped`, for the slot-change audit.
    """
    D = pd.Timestamp(D)
    if rng is None:
        rng = np.random.default_rng(0)
    bm = dmdic["BoMetric_df"].copy()
    cdx = dmdic["cdx_df"].copy()
    bm["date"] = pd.to_datetime(bm["date"], errors="coerce")
    cdx["date"] = pd.to_datetime(cdx["date"], errors="coerce")

    universe = None
    if universe_override is not None:
        # SURVIVORSHIP-CLEAN path: membership already resolved (live survivors + dead
        # entities alive_as_of D) upstream; do NOT apply the na1 survivor filter, which
        # would drop the dead names we just merged in.
        universe = set(universe_override)
        bm = bm[bm["source"].isin(universe)]
        cdx = cdx[cdx["source"].isin(universe)]
    elif na1_only:
        universe = na1_symbols(dmdic["Tickers_df"])
        bm = bm[bm["source"].isin(universe)]
        cdx = cdx[cdx["source"].isin(universe)]

    bm_pit = _sort_newest_first(bm[bm["date"] <= D])
    cdx_pit = _sort_newest_first(cdx[cdx["date"] <= D])
    if bm_pit.empty:
        return None

    # optional: jitter the newest-quarter price/marketCap (daily-input channel)
    if price_noise_frac > 0:
        idx_newest = cdx_pit.groupby("source", sort=False).head(1).index
        for col in ("price", "marketCap"):
            if col in cdx_pit.columns:
                mult = 1.0 + rng.normal(0.0, price_noise_frac, size=len(idx_newest))
                cdx_pit.loc[idx_newest, col] = cdx_pit.loc[idx_newest, col].values * mult

    # ---- Stage-1: BoScore (production function, offline) ----
    meandic = csf.getAves2(bm_pit)
    bmav, bmda = meandic["BoMetric_ave"], meandic["BoMetric_dateAve"]
    # LOCKSTEP with live postBo: Stage-1 frequency comes from cdx (the `period` carrier),
    # never from BoMetric_df (review S3).
    BoScore_df = csf.simpleScore_fromDict(bm_pit, bmav, bmda, nq_stage1,
                                         freq_map=rp.frequency_by_source(cdx_pit))
    if boscore_noise > 0:
        BoScore_df = BoScore_df.copy()
        BoScore_df["score"] = (BoScore_df["score"].astype(float)
                               + rng.normal(0.0, boscore_noise, size=len(BoScore_df)))
    BoScore_df = BoScore_df.sort_values("score", ascending=False)
    BoS_top = BoScore_df.head(topn_stage1).reset_index(drop=True)

    # ---- Stage-2: AggScore over the pool ----
    cdxtop = cdx_pit[cdx_pit["source"].isin(BoS_top["source"])].reset_index(drop=True)
    psm = _stage2_metric_loop_offline(BoS_top, cdxtop, nq=nq_stage2)

    # reuse production normalization + weighting + aggregation (exact)
    psm_norm, outliers = pbr.normalizeAndDropNA(psm)
    postBm, postNew = cdic.getPostDict()
    weight_series = {**{k: postBm[k]["w"] for k in postBm},
                     **{k: postNew[k]["w"] for k in postNew}}
    # Optional weight override (mirrors postBoRank.py:107-108's cohort-weight hook).
    # weight_override=None leaves the default vector -> the live path is bit-identical.
    if weight_override:
        weight_series = {**weight_series, **weight_override}
    weighted = psm_norm.drop("source", axis=1)
    for col in weighted.columns:
        weighted[col] = psm_norm[col].values * weight_series.get(col, 1)
    psmdf_norm = pd.concat(
        [psm_norm[psm_norm.columns.difference(weighted.columns)], weighted], axis=1)
    postRank = pbr.getAggScore(psmdf_norm)

    ranked_sources = postRank["source"].tolist()
    top20_predupe = ranked_sources[:topn_final]
    dedup_dropped = []
    if dedup_issuers:
        # Reuse the carve-out's issuer-fingerprint grouping; keep the highest-ranked
        # line per issuer BEFORE the head cut (CEO: no duplicate issuers in the top-N).
        # Use the UNIVERSE-filtered FULL cdx (all dates), NOT the date<=D slice: issuer
        # identity is a STRUCTURAL fact, and the fingerprint keys on EXACT shares +
        # near-equal fundamentals. A PIT slice breaks it on trivial reporting drift --
        # e.g. buy2022 TFPM vs TFPM.TO have identical revenue/NI/TA but shares differing
        # by 0.001% (1,563 of 155.8M) as-of-2022, escaping every edge; on full history
        # the share counts match exactly and edge C catches the dual-listing. This is
        # NOT return-inflating lookahead: it only collapses same-issuer lines to one
        # slot, never touches a score or a return.
        import carveOut as _co
        tdf = dmdic.get("Tickers_df")
        _cols = getattr(tdf, "columns", [])
        names = (dict(zip(tdf["symbol"], tdf["name"]))
                 if tdf is not None and "symbol" in _cols and "name" in _cols else {})
        deduped, dedup_dropped = _co.dedup_ranked(ranked_sources, cdx, names)
    else:
        deduped = ranked_sources
    top20 = deduped[:topn_final]
    return {
        "date": str(D.date()),
        "na1_only": na1_only,
        "cycleheat_beta": cycleheat_beta,
        "dropped_metrics": DROP_METRICS,
        "stage1_top100": BoS_top["source"].tolist(),
        "pool_after_norm": postRank["source"].tolist(),   # UNDEDUPED (validate_finish anchor)
        "top20": top20,
        "top20_predupe": top20_predupe,
        "issuer_dupes_dropped": dedup_dropped,
        "postRank": postRank,
        "universe_size": (len(universe) if universe else cdx["source"].nunique()),
    }


def prepare_pit(dmdic, D, na1_only=True):
    """PIT-filtered, newest-first bm/cdx frames as-of D (the slow-agnostic prep)."""
    D = pd.Timestamp(D)
    bm = dmdic["BoMetric_df"].copy()
    cdx = dmdic["cdx_df"].copy()
    bm["date"] = pd.to_datetime(bm["date"], errors="coerce")
    cdx["date"] = pd.to_datetime(cdx["date"], errors="coerce")
    if na1_only:
        u = na1_symbols(dmdic["Tickers_df"])
        bm = bm[bm["source"].isin(u)]
        cdx = cdx[cdx["source"].isin(u)]
    return _sort_newest_first(bm[bm["date"] <= D]), _sort_newest_first(cdx[cdx["date"] <= D])


def stage1_boscore(bm_pit, nq_stage1=8, cdx_pit=None):
    """Stage-1 BoScore over the full PIT universe (the EXPENSIVE step; cache it).

    cdx_pit: pass it so Stage-1 reads the `period`-derived frequency map, in LOCKSTEP
    with live postBo (review S3).  Omitting it falls back to cadence on bm_pit, which is
    the pre-2026-07-26 behaviour and is flagged rather than silent."""
    meandic = csf.getAves2(bm_pit)
    if cdx_pit is None:
        print("stage1_boscore: no cdx_pit passed -- Stage-1 frequency falls back to "
              "BoMetric cadence, NOT the authoritative `period` field (review S3).",
              flush=True)
    bs = csf.simpleScore_fromDict(bm_pit, meandic["BoMetric_ave"],
                                  meandic["BoMetric_dateAve"], nq_stage1,
                                  freq_map=(rp.frequency_by_source(cdx_pit)
                                            if cdx_pit is not None else None))
    return bs.sort_values("score", ascending=False).reset_index(drop=True)


def stage2_top(BoScore_df, cdx_pit, nq_stage2=16, topn_stage1=100, topn_final=20,
               boscore_noise=0.0, price_noise_frac=0.0, rng=None):
    """Cheap Stage-2 from a cached Stage-1 BoScore (+ optional noise channels)."""
    if rng is None:
        rng = np.random.default_rng(0)
    bs = BoScore_df
    if boscore_noise > 0:
        bs = bs.copy()
        bs["score"] = bs["score"].astype(float) + rng.normal(0.0, boscore_noise, len(bs))
        bs = bs.sort_values("score", ascending=False)
    BoS_top = bs.head(topn_stage1).reset_index(drop=True)

    cdxtop = cdx_pit[cdx_pit["source"].isin(BoS_top["source"])].reset_index(drop=True)
    if price_noise_frac > 0:
        idx_newest = cdxtop.groupby("source", sort=False).head(1).index
        for col in ("price", "marketCap"):
            if col in cdxtop.columns:
                mult = 1.0 + rng.normal(0.0, price_noise_frac, size=len(idx_newest))
                cdxtop.loc[idx_newest, col] = cdxtop.loc[idx_newest, col].values * mult

    psm = _stage2_metric_loop_offline(BoS_top, cdxtop, nq=nq_stage2)
    psm_norm, _ = pbr.normalizeAndDropNA(psm)
    postBm, postNew = cdic.getPostDict()
    ws = {**{k: postBm[k]["w"] for k in postBm}, **{k: postNew[k]["w"] for k in postNew}}
    w = psm_norm.drop("source", axis=1)
    for col in w.columns:
        w[col] = psm_norm[col].values * ws.get(col, 1)
    psmdf = pd.concat([psm_norm[psm_norm.columns.difference(w.columns)], w], axis=1)
    return pbr.getAggScore(psmdf).head(topn_final)["source"].tolist()


def overlap_report(repro_top20, real_top20, universe_present):
    """Overlap of repro vs real top-20, with survivorship ceiling."""
    rset, realset = set(repro_top20), set(real_top20)
    inter = rset & realset
    present = [s for s in real_top20 if s in universe_present]
    ceiling = len(present)
    return {
        "overlap_n": len(inter),
        "overlap_pct": 100.0 * len(inter) / len(real_top20),
        "ceiling_n": ceiling,
        "ceiling_pct": 100.0 * ceiling / len(real_top20),
        "overlap_vs_ceiling_pct": (100.0 * len(inter) / ceiling) if ceiling else float("nan"),
        "matched": sorted(inter),
        "missed_present": sorted([s for s in present if s not in rset]),
        "missed_absent": sorted([s for s in real_top20 if s not in universe_present]),
        "extra": sorted([s for s in repro_top20 if s not in realset]),
    }
