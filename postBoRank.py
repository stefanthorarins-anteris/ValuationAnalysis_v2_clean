import os
import sys

import createDicts as cdic
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

import stage2_metrics as sm
import reporting_period as rp


# --------------------------------------------------------------------------- #
#  Stage-2 scorer -- postBoScoreRanking, split into single-responsibility
#  helpers.  The per-metric formulas live ONCE in stage2_metrics.py and are
#  shared with the offline reproduction (baseline_tools/stage2_pit.py); this
#  file owns the LIVE orchestration (input checks, the newest-first re-sort, the
#  live DCF fetch, normalisation/weighting/aggregation, and issuer-dedup).
# --------------------------------------------------------------------------- #
def postBoScoreRanking(bmtop, bstop, cdxtop, baseurl, api_key, period='quarter',
                       nq=16, as_of=None, weight_override=None, names=None,
                       dedup_issuers=True):
    # as_of : point-in-time date D (default None).  as_of=None reproduces the live
    # Stage-2 ranking BIT-FOR-BIT.  The parameter is threaded here so the PIT DCF/beta
    # engagement (computed point-in-time DcfToPrice + CycleHeat beta, design s2B/s2C)
    # has a live seam; the point-in-time DCF/beta substitution itself is a later
    # (registry-backed, Phase 3+) step and is NOT wired on this path yet.  With
    # as_of=None nothing below branches on it -> live behaviour unchanged.
    _assert_offline_dcf_is_score_neutral()
    print('Ranking the top 100 stocks, according to BoScore.')
    sys.stdout.flush()  # Ensure output is printed before progress bar

    _diagnose_inputs(bmtop, bstop, cdxtop)

    postBmRankingDict, postNewRankingDict = cdic.getPostDict()
    postScoreMetric_df = pd.DataFrame()
    postScoreMetric_df['source'] = bstop['source']
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postBmRankingDict.keys())], axis=1)
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postNewRankingDict.keys())], axis=1)

    # Build a stable weight mapping from the post dictionaries so we always have a
    # weight for each metric.  Per-cohort weight vector (carveOut.COHORT_WEIGHTS):
    # weight_override overrides the default weight for any metric it lists.
    # weight_override=None (the general/main pool) keeps the default vector.  A 0
    # weight zeroes that metric's AggScore contribution AND makes the weighted column
    # constant (0) -> neutral in rankOfRanks; it does NOT change cohort membership (a
    # row is dropped only if ALL metrics are NaN, upstream).
    weight_series = {**{k: postBmRankingDict[k]['w'] for k in postBmRankingDict},
                     **{k: postNewRankingDict[k]['w'] for k in postNewRankingDict}}
    if weight_override:
        weight_series = {**weight_series, **weight_override}

    cdxtop = _sort_cdx_newest_first(cdxtop)
    cdxtop['mcapQuants'] = sm.add_mcap_quants(cdxtop)

    # PER-SOURCE REPORTING FREQUENCY, computed ONCE for the pool.  Every windowed /
    # YoY Stage-2 metric below is parameterised by this source's rows-per-year so a
    # semi-annual filer is not scored on 2-year "YoY" growth and 2-year "TTM" windows
    # (audit C-1).  unknown -> quarterly, i.e. unchanged.
    freq_map = rp.frequency_by_source(cdxtop, verbose=True)

    # Note: Bulk endpoints require higher subscription tier, using individual API calls only
    dcf_bulk_dict = {}

    pbar = tqdm(total=len(bstop['source'].unique()))
    for tempcntr, ticker in enumerate(bstop['source']):
        tempcdx = cdxtop.loc[cdxtop['source'] == ticker]

        # DCF data (used for the DcfToPrice metric); live per-ticker fetch.
        dcf, dcf_from_bulk, resp_dcf_status, resp_dcf = _fetch_ticker_dcf(
            ticker, baseurl, api_key, dcf_bulk_dict)

        if tempcntr == 0:
            _diagnose_first_ticker_data(ticker, dcf, dcf_from_bulk, resp_dcf_status,
                                        resp_dcf, tempcdx)

        _compute_ticker_metrics(ticker, tempcdx, dcf, bstop, nq, tempcntr,
                                postBmRankingDict, postNewRankingDict, postScoreMetric_df,
                                rpy=rp.rows_per_year(freq_map, ticker))

        if tempcntr == 0:
            _diagnose_first_ticker_metrics(ticker, postScoreMetric_df)

        pbar.update(n=1)

    _diagnose_pre_normalize(postScoreMetric_df)

    # --- REVIEW-REFERENCE capture (READ-ONLY; must NOT perturb scoring) ----------
    # Snapshot the RAW per-ticker metrics BEFORE normalizeAndDropNA z-scores them IN
    # PLACE (and before its >4-std outlier drop below), so the human-review reference
    # artifacts (reviewReference.py) reflect true RAW values and the full pre-drop pool
    # membership.  This is a COPY only: it is returned in rankdic['postScoreMetric_raw']
    # and is NEVER read back into scoring / normalization / ranking.  Feeding cohort
    # means or percentiles derived from this back into the score would be cross-sectional
    # sector-neutralization, which is CEO-ratified OFF -- so the capture must stay a pure
    # side-channel.  Because postBoScoreRanking runs once per pool (general + each cohort
    # via postBo), this single line yields raw metrics for every pool automatically.
    postScoreMetric_raw = postScoreMetric_df.copy()

    # weight_series is passed so the outlier guard can EXEMPT zero-weight metrics: a
    # w=0 diagnostic column must not clamp (and, pre-fix, must not eject) a name that
    # contributes to no part of the AggScore.  This is the SAME weight_series -- incl.
    # any cohort weight_override -- that weights the columns below, so a metric the
    # cohort zeroed is exempt in that cohort's pool too.
    postScoreMetric_df, outlierlist = normalizeAndDropNA(postScoreMetric_df,
                                                         weight_series=weight_series)

    # Apply weights using the stable weight_series mapping; if a weight is missing,
    # default to 1.
    temp_normpsmdf_weighted = postScoreMetric_df.drop('source', axis=1)
    for col in temp_normpsmdf_weighted.columns:
        w = weight_series.get(col, 1)
        temp_normpsmdf_weighted[col] = postScoreMetric_df[col].values * w
    psmdf_normalized = pd.concat(
        [postScoreMetric_df[postScoreMetric_df.columns.difference(temp_normpsmdf_weighted.columns)],
         temp_normpsmdf_weighted], axis=1)

    postRank = getAggScore(psmdf_normalized)

    tmpcorr = np.corrcoef(list(postRank['BoScore'].values), list(postRank['AggScore'].values))
    BoAggCorr = tmpcorr[0, 1]

    postRank = getRankOfRanks(postRank)

    pbar.close()

    postRank_predupe = postRank.copy()
    postRank, issuer_dupes_dropped = _dedup_issuers_in_ranking(
        postRank, cdxtop, names, dedup_issuers)

    rankdic = {'postRank': postRank, 'postScoreMetric': postScoreMetric_df,
               'postScoreMetric_raw': postScoreMetric_raw,
               'psmdf_normalized': psmdf_normalized, 'BoAggCorr': BoAggCorr, 'outlierlist': outlierlist,
               'postRank_predupe': postRank_predupe, 'issuer_dupes_dropped': issuer_dupes_dropped}

    return rankdic


def _sort_cdx_newest_first(cdxtop):
    """Enforce NEWEST-first row order for the whole scorer (Stage-2 ORDERING FIX).

    Every metric indexes with .head(nq) / .iloc[0] / .iloc[0:4] / pct_change
    assuming the most-recent quarter is row 0.  data_quality.py sorts cdx
    OLDEST-first and nothing re-sorts it on the way here, so those reads would
    silently use the wrong end (stale windows, sign-flipped growth, time-reversed
    Piotroski).  Re-sort a COPY newest-first, robustly: dates are coerced to
    datetime (a naive string sort mis-orders mixed/malformed date strings).  We
    COPY because cdx_dftop100 is also stored in resdic and must not be mutated.
    Assert per-ticker row count and NaT count are unchanged.
    """
    cdxtop = cdxtop.copy()
    _n_before = cdxtop.groupby('source').size()
    cdxtop['date'] = pd.to_datetime(cdxtop['date'], errors='coerce')
    _nat_before = int(cdxtop['date'].isna().sum())
    cdxtop = cdxtop.sort_values(['source', 'date'], ascending=[True, False]).reset_index(drop=True)
    assert cdxtop.groupby('source').size().equals(_n_before), \
        "Stage-2 newest-first re-sort changed per-ticker row counts"
    assert int(cdxtop['date'].isna().sum()) == _nat_before, \
        "Stage-2 newest-first re-sort changed NaT count"
    return cdxtop


# OFFLINE SEAM for the DCF leg (added 2026-07-27).  Stage-2's ONLY network dependency is a
# per-ticker discounted-cash-flow call, one per name in the top-100 -- so re-scoring a SAVED
# panel through the DEPLOYED path could not be done without ~100 live FMP calls, which the
# work-machine rule explicitly discourages.  With this set, the call is skipped and the DCF
# frame comes back empty, exactly as it already does on an HTTP failure (a path production
# handles every run).
#
# PROVABLY SCORE-NEUTRAL, and ENFORCED: `DcfToPrice` carries w = 0.000 in the decisional
# vector, so its value cannot move AggScore by construction.  That is a fact about today's
# weights, not a law -- so if the weight is ever made non-zero, offline mode REFUSES rather
# than silently changing the score.  Default OFF: the live fetch path is unchanged.
OFFLINE_NO_DCF = os.environ.get('VA_OFFLINE_NO_DCF', '') == '1'


def _assert_offline_dcf_is_score_neutral():
    """Refuse offline scoring if DcfToPrice has acquired a weight."""
    if not OFFLINE_NO_DCF:
        return
    postBm, postNew = cdic.getPostDict()
    w = float({**postBm, **postNew}.get('DcfToPrice', {}).get('w', 0) or 0)
    if w != 0:
        raise SystemExit(
            "VA_OFFLINE_NO_DCF=1 but DcfToPrice now carries w=%r.  Offline mode skips the "
            "DCF fetch, which is only score-neutral while that weight is 0 -- refusing "
            "rather than emitting a silently different ranking." % w)
    print("!" * 78, flush=True)
    print("!!! VA_OFFLINE_NO_DCF=1 -- Stage-2 DCF fetch SKIPPED (no network).  DcfToPrice "
          "w=0.000,\n!!! so AggScore is unaffected; the DCF column is empty/NaN in any "
          "DISPLAY that reads it.", flush=True)
    print("!" * 78, flush=True)


def _fetch_ticker_dcf(ticker, baseurl, api_key, dcf_bulk_dict):
    """Fetch (or reuse bulk) DCF data for one ticker and return it as a normalised
    DataFrame.  Returns (dcf_df, dcf_from_bulk, resp_dcf_status, resp_dcf).
    """
    if OFFLINE_NO_DCF:
        return pd.DataFrame(), False, "offline-skipped", None
    dcf_from_bulk = ticker in dcf_bulk_dict
    resp_dcf = None
    if dcf_from_bulk:
        dcf_data = [dcf_bulk_dict[ticker]]
        resp_dcf_status = "bulk"
    else:
        # Fallback to individual API call
        resp_dcf = requests.get(f'{baseurl}v3/discounted-cash-flow/{ticker}?apikey={api_key}')
        resp_dcf_status = resp_dcf.status_code
        try:
            dcf_data = resp_dcf.json() if resp_dcf.status_code == 200 else []
        except:
            dcf_data = []

    # Convert bulk data (already in dict format) or API response to DataFrame
    dcf = pd.DataFrame.from_dict(dcf_data) if dcf_data and isinstance(dcf_data, list) else pd.DataFrame()

    # Normalize DCF column names - bulk CSV might have different names than JSON API
    if not dcf.empty:
        column_mapping = {}
        for col in dcf.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if col_lower == 'stockprice' or col_lower == 'stock_price':
                column_mapping[col] = 'Stock Price'
            elif col_lower == 'dcf':
                column_mapping[col] = 'dcf'
        if column_mapping:
            dcf = dcf.rename(columns=column_mapping)

    return dcf, dcf_from_bulk, resp_dcf_status, resp_dcf


def _compute_ticker_metrics(ticker, tempcdx, dcf, bstop, nq, tempcntr,
                            postBmRankingDict, postNewRankingDict, postScoreMetric_df,
                            rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Compute every Stage-2 metric for one ticker and write it into
    postScoreMetric_df.  All formulas come from the shared stage2_metrics module
    (kept in lockstep with the offline reproduction); this function only decides
    which column each result lands in.

    `rpy` is this TICKER's rows-per-year (4 quarterly / 2 semi-annual,
    reporting_period): it is threaded into every windowed or YoY metric so a
    semi-annual filer's windows span the same CALENDAR time as a quarterly peer's.
    rpy=4 is bit-identical to the previous behaviour.
    """
    def setv(col, val):
        postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, col] = val

    # ---- postBmRankingDict metrics ----
    for key1 in postBmRankingDict.keys():
        setv(key1, sm.postbm_metric(key1, postBmRankingDict[key1]['eqMet'], tempcdx, nq,
                                    rpy=rpy))

    # ---- postNewRankingDict metrics ----
    tempfcf = tempcdx.freeCashFlow
    tempshares = tempcdx.weightedAverageShsOut
    tempmcap = tempcdx.marketCap

    setv('freeCashFlowYield', sm.free_cash_flow_yield(tempfcf, tempmcap, nq, rpy=rpy))
    setv('freeCashFlowPerShareGrowth',
         sm.free_cash_flow_per_share_growth(tempfcf, tempshares, nq, rpy=rpy))
    setv('EPStoEPSmean', sm.eps_to_eps_mean(tempcdx, rpy=rpy))
    setv('marketCapRevQuants', tempcdx.mcapQuants.iloc[0])
    setv('tbVpRatio', sm.tbv_p_ratio(tempcdx, nq, rpy=rpy))
    setv('Altman-Z', sm.altman_z(tempcdx, rpy=rpy))
    setv('Piotroski', sm.piotroski(tempcdx, rpy=rpy))
    setv('priceGrowth', sm.price_growth(tempcdx, nq, rpy=rpy))
    # rpy MUST be passed: CycleHeat is a self-reference z-score, so its window length
    # decides the baseline. It was the only metric in this block taking neither nq nor
    # rpy (fix 2026-07-30); see stage2_metrics.CYCLEHEAT_BASE_NQ.
    setv('CycleHeat', sm.cycleheat(tempcdx, rpy=rpy))

    # DcfToPrice needs the live DCF frame; diagnostic-log missing columns for the
    # first ticker (matches the historical behaviour).
    if not dcf.empty and tempcntr == 0 and not (
            'dcf' in dcf.columns and any(c in dcf.columns for c in
                                         ('Stock Price', 'StockPrice', 'stock_price'))):
        print(f"  WARNING: DCF missing required columns. Available: {list(dcf.columns)}, "
              f"need: ['dcf', price_col]", flush=True)
    setv('DcfToPrice', sm.dcf_to_price(dcf, nq))

    # BoScore is a straight pass-through of the Stage-1 score (weight 0 in the live
    # vector).  Assigned as a Series to preserve the historical index-alignment.
    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, 'BoScore'] = \
        bstop.loc[bstop['source'] == ticker, 'score']


# --------------------------------------------------------------------------- #
#  Diagnostics (stdout only -- no effect on the emitted ranking)              #
# --------------------------------------------------------------------------- #
def _diagnose_inputs(bmtop, bstop, cdxtop):
    print("\n" + "=" * 60, flush=True)
    print("DIAGNOSTIC: Input dataframes check (BEFORE calculations)", flush=True)
    print("=" * 60, flush=True)

    if bmtop.empty:
        print("ERROR: bmtop (BoMetric top 100) is EMPTY!", flush=True)
    else:
        print(f"bmtop shape: {bmtop.shape} (rows, columns)", flush=True)
        print(f"bmtop unique sources: {bmtop['source'].nunique() if 'source' in bmtop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bmtop.columns:
            print(f"bmtop sample sources: {list(bmtop['source'].head(3).values)}", flush=True)
        print(f"bmtop columns (first 10): {list(bmtop.columns[:10])}", flush=True)
        numeric_cols = bmtop.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            nan_pct = (bmtop[numeric_cols].isna().sum() / len(bmtop) * 100).round(1)
            print(f"bmtop NaN percentage in numeric columns (first 5): {dict(nan_pct.head(5))}", flush=True)

    if bstop.empty:
        print("ERROR: bstop (BoScore top 100) is EMPTY!", flush=True)
    else:
        print(f"\nbstop shape: {bstop.shape} (rows, columns)", flush=True)
        print(f"bstop unique sources: {bstop['source'].nunique() if 'source' in bstop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bstop.columns:
            print(f"bstop sample sources: {list(bstop['source'].head(3).values)}", flush=True)
        if 'score' in bstop.columns:
            print(f"bstop score stats: min={bstop['score'].min():.4f}, max={bstop['score'].max():.4f}, mean={bstop['score'].mean():.4f}", flush=True)

    if cdxtop.empty:
        print("\nERROR: cdxtop (cdx top 100) is EMPTY!", flush=True)
    else:
        print(f"\ncdxtop shape: {cdxtop.shape} (rows, columns)", flush=True)
        print(f"cdxtop unique sources: {cdxtop['source'].nunique() if 'source' in cdxtop.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in cdxtop.columns:
            print(f"cdxtop sample sources: {list(cdxtop['source'].head(3).values)}", flush=True)
        print(f"cdxtop columns (first 10): {list(cdxtop.columns[:10])}", flush=True)
        required_cols = ['freeCashFlow', 'weightedAverageShsOut', 'marketCap', 'grahamNumber', 'price',
                         'tangibleBookValuePerShare', 'totalAssets', 'totalLiabilities', 'totalCurrentAssets',
                         'totalCurrentLiabilities', 'totalStockholdersEquity', 'operatingIncome', 'revenue',
                         'netIncome', 'netCashProvidedByOperatingActivities', 'longTermDebt', 'currentRatio',
                         'grossProfitMargin']
        missing_cols = [col for col in required_cols if col not in cdxtop.columns]
        if missing_cols:
            print(f"WARNING: cdxtop missing required columns: {missing_cols}", flush=True)
        else:
            print(f"cdxtop has all required columns: {required_cols}", flush=True)
        key_cols = [col for col in required_cols if col in cdxtop.columns]
        if key_cols:
            nan_pct = (cdxtop[key_cols].isna().sum() / len(cdxtop) * 100).round(1)
            print(f"cdxtop NaN percentage in key columns: {dict(nan_pct)}", flush=True)

    print("=" * 60 + "\n", flush=True)
    sys.stdout.flush()


def _diagnose_first_ticker_data(ticker, dcf, dcf_from_bulk, resp_dcf_status, resp_dcf, tempcdx):
    print(f"\nDIAGNOSTIC: First ticker ({ticker}) data:", flush=True)
    print(f"  DCF source: {'bulk' if dcf_from_bulk else 'individual'}, status: {resp_dcf_status}, empty: {dcf.empty}, shape: {dcf.shape if not dcf.empty else 'N/A'}", flush=True)
    if not dcf.empty:
        print(f"  DCF columns: {list(dcf.columns)}", flush=True)
    print(f"  tempcdx (fundamentals) empty: {tempcdx.empty}, shape: {tempcdx.shape if not tempcdx.empty else 'N/A'}", flush=True)
    if not tempcdx.empty:
        print(f"  tempcdx columns: {list(tempcdx.columns[:5])}...", flush=True)
        print(f"  tempcdx sample values (first row):", flush=True)
        print(f"    freeCashFlow: {tempcdx['freeCashFlow'].iloc[0] if 'freeCashFlow' in tempcdx.columns else 'N/A'}", flush=True)
        print(f"    marketCap: {tempcdx['marketCap'].iloc[0] if 'marketCap' in tempcdx.columns else 'N/A'}", flush=True)
        print(f"    operatingIncome: {tempcdx['operatingIncome'].iloc[0] if 'operatingIncome' in tempcdx.columns else 'N/A'}", flush=True)
    if not dcf_from_bulk and resp_dcf_status != 200:
        print(f"  DCF error: {resp_dcf.text[:100] if resp_dcf is not None else 'N/A'}", flush=True)
    print(f"  Note: Altman-Z and Piotroski calculated from tempcdx fundamentals", flush=True)


def _diagnose_first_ticker_metrics(ticker, postScoreMetric_df):
    sample_metrics = ['RoA', 'earnYield', 'grahamNumberToPrice', 'freeCashFlowYield', 'BoScore', 'Altman-Z', 'Piotroski', 'CycleHeat']
    print(f"\nDIAGNOSTIC: Sample metric values after calculation for {ticker}:", flush=True)
    for metric in sample_metrics:
        if metric in postScoreMetric_df.columns:
            val = postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, metric].values
            if len(val) > 0 and not pd.isna(val[0]):
                print(f"  {metric}: {val[0]}", flush=True)
            else:
                print(f"  {metric}: NOT CALCULATED (NaN)", flush=True)
        else:
            print(f"  {metric}: COLUMN NOT FOUND", flush=True)
    print(f"  Note: Altman-Z and Piotroski are now calculated from fundamentals (no API needed)", flush=True)


def _diagnose_pre_normalize(postScoreMetric_df):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: postScoreMetric_df statistics before normalizeAndDropNA")
    print("=" * 60)
    print(f"DataFrame shape: {postScoreMetric_df.shape} (rows, columns)")
    print(f"Total rows: {len(postScoreMetric_df)}")
    print(f"Columns: {list(postScoreMetric_df.columns)}")

    if not postScoreMetric_df.empty:
        metric_cols = [col for col in postScoreMetric_df.columns if col != 'source']
        print(f"\nMetric columns (excluding 'source'): {len(metric_cols)}")

        if len(metric_cols) > 0:
            numeric_df = postScoreMetric_df[metric_cols].apply(pd.to_numeric, errors='coerce')

            print("\nNaN statistics per column:")
            nan_counts = numeric_df.isna().sum()
            nan_pct = (nan_counts / len(numeric_df) * 100).round(2)
            for col in metric_cols:
                print(f"  {col}: {nan_counts[col]} NaN ({nan_pct[col]}%)")

            total_cells = len(numeric_df) * len(metric_cols)
            total_nan = numeric_df.isna().sum().sum()
            print(f"\nOverall: {total_nan}/{total_cells} NaN values ({total_nan/total_cells*100:.2f}%)")

            rows_all_nan = (numeric_df.isna().sum(axis=1) == len(metric_cols)).sum()
            print(f"Rows with ALL metrics NaN: {rows_all_nan}/{len(numeric_df)} ({rows_all_nan/len(numeric_df)*100:.2f}%)")

            rows_some_valid = (numeric_df.isna().sum(axis=1) < len(metric_cols)).sum()
            print(f"Rows with at least one valid metric: {rows_some_valid}/{len(numeric_df)} ({rows_some_valid/len(numeric_df)*100:.2f}%)")

            print("\nColumn statistics (for non-NaN values):")
            for col in metric_cols:
                col_data = numeric_df[col].dropna()
                if len(col_data) > 0:
                    print(f"  {col}: mean={col_data.mean():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}, count={len(col_data)}")
                else:
                    print(f"  {col}: ALL NaN (no valid values)")
        else:
            print("WARNING: No metric columns found!")
    else:
        print("WARNING: DataFrame is empty!")

    print("=" * 60 + "\n")


def _dedup_issuers_in_ranking(postRank, cdxtop, names, dedup_issuers):
    """Issuer-level de-dup of the EMITTED ranking (CEO standing principle: NO
    duplicate issuers in the deployed top-N).  postRank is AggScore-descending, and
    downstream emission (writeBoAggToCSV / createPresentation) takes head(N) off it,
    so collapsing same-issuer lines HERE makes the CEO-reviewed top-20 contain
    DISTINCT issuers.  We keep the HIGHEST-RANKED line per issuer and drop later
    same-issuer lines (share-classes / cross-listings, e.g. TFPM / TFPM.TO) -- the
    SAME rank-based rule and SAME fingerprint (carveOut.dedup_ranked /
    _issuer_components) the backtest harness (stage2_pit.reproduce_pit_top) uses.  So
    live and backtest agree on issuer IDENTITY and economic exposure (one slot per
    issuer) -- NOT necessarily on the specific surviving TICKER: on the carve-ON live
    path the upstream carve already collapsed each issuer to its mcap/sector-preferred
    line, whereas the backtest / carve-OFF path keeps the highest-RANKED line.  Both
    satisfy "distinct issuers".

    This changes ONLY which lines survive into the emitted ranking; no score, no sort
    order, and no other pick logic is touched.  For a LIVE run as_of is NOW, so the
    fingerprint reads CURRENT fundamentals -> merging same-issuer listings is correct
    with NO lookahead (the PIT-purity caveat applies only to backtest dedup at past D).
    If the carve-out already collapsed the universe to one line per issuer upstream,
    this is a safe no-op; it is load-bearing on the carve-off / carve-fallback path
    (and any carve-missed listing).

    Returns (postRank, issuer_dupes_dropped).
    """
    issuer_dupes_dropped = []
    if not dedup_issuers:
        return postRank, issuer_dupes_dropped
    try:
        import carveOut as _co
        ranked_srcs = postRank['source'].tolist()
        # AggScore in: an issuer's clone lines score EXACTLY equal, so which one the
        # sort happened to emit first was arbitrary -- resolve those ties by
        # investability rather than by sort stability (carveOut.dedup_ranked TIE-BREAK).
        _sc = postRank['AggScore'] if 'AggScore' in postRank.columns else None
        kept, issuer_dupes_dropped = _co.dedup_ranked(ranked_srcs, cdxtop,
                                                      names or {}, scores=_sc)
        if issuer_dupes_dropped:
            keptset = set(kept)
            postRank = postRank[postRank['source'].isin(keptset)].reset_index(drop=True)
            print("postBoRank issuer-dedup: collapsed %d same-issuer line(s) in the "
                  "ranking -> %s"
                  % (len(issuer_dupes_dropped),
                     ['%s->%s' % (d, k) for d, k in issuer_dupes_dropped]), flush=True)
    except Exception as _e:
        # LOUD FALLBACK (matches the carve-out banner at postBo.py:143-164). The
        # emission-time issuer-dedup IS the "no duplicate issuers in the top-20"
        # guarantee; if it fails we still ship (never crash the deliverable) but the
        # emitted top-20 may carry dual-listings / share-classes, so a single quiet
        # stdout line is a defect -- make the degradation IMPOSSIBLE to miss on BOTH
        # streams, exactly like the carve fallback.
        import traceback
        _banner = (
            "\n" + "!" * 78 + "\n"
            "!!! ISSUER-DEDUP DID NOT RUN -- EMITTED TOP-20 MAY CONTAIN DUAL-LISTINGS !!!\n"
            "!!! The ranking was NOT de-duplicated by issuer this run: expect possible !!!\n"
            "!!!   share-class / cross-listing DUPLICATES in the top-N (e.g. TFPM +     !!!\n"
            "!!!   TFPM.TO occupying two slots for one economic bet).                   !!!\n"
            "!!! Cause: %s: %s\n"
            "!!! DO NOT treat this top-20 as issuer-deduplicated.                       !!!\n"
            % (type(_e).__name__, _e)
            + "!" * 78 + "\n")
        print(_banner, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(_banner, flush=True)
        traceback.print_exc(file=sys.stdout)
    return postRank, issuer_dupes_dropped


WINSOR_SIGMA = 3.0        # winsorization threshold, in sigmas of the RAW column
# Safety bound on the mu/sigma <-> clip fixed-point loop.  Set well ABOVE what the real
# pools need: measured on the shipped 2026-07-17 columns, convergence takes up to 64
# passes (currentRatio 57, EPStoEPSmean 50, incomeQuality 64) -- a bound of 20 stopped
# just short of the fixed point and left max|z| = 3.000298 instead of <= 3.  The bound is
# NOT load-bearing for the result: the set of clipped cells stabilises by ~pass 20 and the
# remaining passes only settle sigma in the 6th decimal (verified: identical clipped-cell
# counts and identical AggScore ordering at 200 vs 500 passes).  Each pass is one
# vectorised op over ~100 rows, so the headroom is free.
WINSOR_MAX_PASSES = 200


WINSOR_Z_EPS = 1e-9       # slack on the max|z| <= WINSOR_SIGMA target test

# Columns that CANNOT have a fat tail because they are BOUNDED or DISCRETE by
# construction, and are therefore EXEMPT from winsorization entirely (review H3(c);
# this is the choice made, not just considered):
#   Piotroski            integer 0..9 (9 binary criteria)
#   CycleHeat            hard-capped to [-3, +3] in stage2_metrics.cycleheat_zscore
#   marketCapRevQuants   5 discrete values after fix 16 (-0.5, -1/6, 0, +1/6, +0.5)
# For a bounded discrete column a large |z| is a property of the SPLIT -- how many names
# sit on each level -- not of a heavy tail.  Piotroski 20 names at 7 and one at 2 puts
# the single WORST-F name at |z| = 4.36, and "winsorizing" that moves the worst F-score
# onto the best, destroying real structure to satisfy a threshold that was never meant
# for it.  Leaving them alone is strictly more honest than clipping them.
WINSOR_EXEMPT_BOUNDED = ('Piotroski', 'CycleHeat', 'marketCapRevQuants', 'mcapQuants')


# --------------------------------------------------------------------------- #
#  NORMALISATION METHOD  (A/B switch, 2026-07-27)                             #
# --------------------------------------------------------------------------- #
# 'zscore' : winsorize the RAW column, then (x - mu) / sigma.  THE SHIPPED PATH.
# 'rank'   : map the column to ranks, then through the inverse normal CDF.
#
# WHY 'rank' EXISTS.  Sigma-winsorization assumes an approximately symmetric, roughly
# Gaussian column; several of these metrics are strongly skewed, which the winsorizer
# reduces but cannot remove.  Two consequences the z-path cannot escape:
#   (1) CROSS-COLUMN INCOMPARABILITY.  A weight w only means "w units of AggScore per
#       sigma of THIS column", and a sigma of a skewed column is not a sigma of a
#       symmetric one, so the weight vector's components silently mean different things.
#       After a rank map every column is the SAME distribution by construction, so a
#       weight means one thing everywhere -- which is the precondition for the weight
#       vector to be re-fittable at all.
#   (2) THE MISSING-DATA REWARD (reviewer finding N1).  Both paths fill an unavailable
#       metric with 0, but 0 means different things: under z-scoring 0 is the winsorized
#       MEAN, which on the shipped 07-17 pool sat at the 52nd-65th percentile on 15 of 17
#       weighted columns -- so a name MISSING a metric scored ABOVE the typical name on
#       it, worth +0.1394 AggScore for full missingness against a 0.134 median-to-top-20
#       distance.  Under the rank map 0 is AT OR NEAR the median, so the same fillna(0) is
#       (near-)neutral and most of the reward vanishes without a special case.
#       PRECISELY (do NOT round this up to "0 is the median by construction" -- it is not):
#       the map centres on the median EXACTLY only when the column's observed values are
#       DISTINCT.  Ties displace the centre, because a tied group takes one averaged
#       plotting position instead of spanning several: `_rank_normal([1, 1, 2])` has mean
#       +0.0260, and on a 60/40 binary column the fill sits above ~60% of the pool.  This
#       is not a technicality -- it IS the entire measured residual.  Of the +0.0228 that
#       survives on the 07-17 pool, `marketCapRevQuants` alone contributes +0.0244: five
#       discrete levels, hence massive ties, hence a fill that is materially off-median.
#       Measured effect of the switch: full-missingness advantage +0.1616 -> +0.0228 (-86%),
#       and columns whose fill sits above their own median 14/18 -> 2/18.  A real and large
#       reduction; NOT an elimination.
#
# DEFAULT IS UNCHANGED.  'zscore' remains the production default: the deployed mu weights
# were tuned on the z-path, and a rank map changes what each weight MEANS, so switching
# the live default is a CEO decision that belongs WITH the weight re-fit, not before it.
# The switch exists so the two can be measured side by side on identical inputs.
NORM_ZSCORE = 'zscore'
NORM_RANK = 'rank'
NORM_METHOD_DEFAULT = NORM_ZSCORE

# Plotting position for the rank -> normal map.  Blom: p_i = (r_i - 3/8) / (n + 1/4).
# Chosen over Van der Waerden r/(n+1) because it is the near-unbiased normal-scores
# constant, i.e. E[z_i] is closest to the expected order statistic, so the resulting
# column's sd is closest to 1 at the pool sizes here (~100 names Stage-2, ~6.5k universe-
# wide).  It is a MONOTONE relabelling either way, so the choice cannot change the ORDER
# within a column -- only the spacing, and therefore how much a given rank gap is worth
# against another column's rank gap.
RANK_PLOT_A = 0.375


def _rank_normal(x):
    """Rank -> inverse-normal ('probit' / normal-scores) map of one metric column.

    * Ties get the AVERAGE rank, so tied names get the SAME score -- required: a
      discrete column (Piotroski, the market-cap quantile codes) is mostly ties, and
      breaking them by row order would inject pure noise into the ranking.
    * NaN keeps NaN and does NOT consume a rank slot, so the percentiles describe the
      OBSERVED population.  The caller's fillna(0) then lands a missing metric AT OR NEAR
      that population's median (finding N1) -- see the tie caveat below for when "near"
      is as good as it gets.
    * CENTRING, stated exactly.  With all values DISTINCT the ranks are symmetric about
      (n+1)/2, so the plotting positions are symmetric about 0.5 and the output has mean
      EXACTLY 0 and median exactly 0.  WITH TIES NEITHER HOLDS: a tied group collapses
      onto one averaged plotting position instead of spanning several, which displaces the
      centre.  `_rank_normal([1, 1, 2])` has mean +0.0260; a 60/40 binary column puts 0
      above ~60% of the pool, not 50%.  So "0 is the median" is a property of
      distinct-valued columns, NOT of this function -- and the discrete columns
      (`Piotroski` 0..9, `marketCapRevQuants` 5 levels) are exactly the ones where it
      fails.  On the 07-17 pool `marketCapRevQuants` alone accounts for +0.0244 of the
      +0.0228 residual missing-data advantage.  Do not restate this as "by construction".
    * sd is CLOSE to but not exactly 1 (finite-sample plotting position, and ties compress
      a column's spread).  That is deliberate and is NOT re-scaled: rescaling each column
      back to unit variance would hand a heavily-tied column the same spread as a
      fully-resolved one, i.e. it would re-weight the vector by tie structure.
    """
    from scipy.special import ndtri            # scipy is a declared dependency
    s = pd.to_numeric(x, errors='coerce')
    valid = s.notna()
    n = int(valid.sum())
    out = pd.Series(np.nan, index=s.index, dtype='float64')
    if n == 0:
        return out
    if n == 1:
        out[valid] = 0.0                        # a single observation IS the median
        return out
    r = s[valid].rank(method='average')
    p = (r - RANK_PLOT_A) / (n + 1.0 - 2.0 * RANK_PLOT_A)
    out[valid] = ndtri(p.to_numpy(dtype='float64'))
    return out


def _winsorize_raw(x, n_sigma=WINSOR_SIGMA, max_passes=WINSOR_MAX_PASSES):
    """Symmetric sigma-winsorization of a RAW metric column, iterated toward max|z| <=
    n_sigma.  Returns (series, n_cells_changed, n_passes, converged).

    Body of the loop is the specified two-pass move: compute mu/sigma, clip the raw
    values to mu +- n_sigma*sigma, recompute mu/sigma on the clipped column.  It is
    REPEATED because ONE pass is not enough, and that is measurable rather than
    theoretical: sigma of the first pass is itself inflated by the outlier being
    clipped, so mu1 +- 3*sigma1 lands far out in the tail, and after re-standardising on
    the (much tighter) clipped column the clipped value can sit at a LARGER |z| than
    before.  On the shipped 2026-07-17 pool a single two-pass left worst-column
    max|z| = 9.90 (EPStoEPSmean) -- WORSE than the +-4 z-clamp it replaced -- with
    sigma1/sigma2 up to 3.23.

    WHAT IS AND IS NOT GUARANTEED (review H3(d) -- the earlier "max|z| <= n_sigma BY
    CONSTRUCTION" claim was FALSE).  Iterating does NOT always reach the target: for a
    near-two-point column the clip ratio is constant, so the minority value decays
    GEOMETRICALLY toward the mode while its z-score -- which is scale-invariant -- never
    moves.  There is no interior fixed point.  Measured cases: 99x0 + 1x1e6 ends with the
    outlier at 4.9e-13 (1e-19 of input) and max|z| still 9.900; Piotroski 20x7/1x2 (n=21)
    burns 74 passes and stays at 4.364; CycleHeat 25x(-3)/2x(+1) (n=27) burns 194 of 200.
    The old exit test compounded this: `tol = 1e-12 * max(1.0, |mu|+sigma)` is an
    ABSOLUTE floor, so once a column's own scale fell below ~1e-12 the test passed
    vacuously and the loop exited OUTSIDE the threshold with no signal at all.

    So the contract is now explicit and two-branched:
      * converged=True  -> max|z| <= n_sigma + WINSOR_Z_EPS is VERIFIED on the returned
                           series (the target test is evaluated directly, not inferred
                           from a proxy).
      * converged=False -> the target is UNREACHABLE for this column's shape.  The
                           ORIGINAL raw values are returned UNCHANGED (never the
                           annihilated ones) and the caller logs it loudly.  A column
                           whose spread is structural is left at its natural z rather
                           than mangled.
    """
    orig = pd.to_numeric(x, errors='coerce')
    y = orig
    for p in range(max_passes):
        m, s = y.mean(), y.std()
        if not np.isfinite(m) or not np.isfinite(s) or s <= 0:
            # constant / all-NaN column: no outlier can exist, target trivially holds
            return orig, 0, p, True
        if float(((y - m) / s).abs().max()) <= n_sigma + WINSOR_Z_EPS:
            changed = int((~np.isclose(orig.to_numpy(dtype='float64'),
                                       y.to_numpy(dtype='float64'),
                                       rtol=0, atol=0, equal_nan=True)).sum())
            return y, changed, p, True
        y2 = y.clip(m - n_sigma * s, m + n_sigma * s)
        # NO-PROGRESS test: BIT-EXACT equality.  It must be exact, not a relative
        # tolerance -- the approach to the target is asymptotic, so a 1e-9 relative test
        # RACES the target test and loses: a lognormal(0,1.2) n=100 column reaches
        # max relative change 6.2e-10 at pass 18, one pass BEFORE max|z| first tests
        # <= 3, so a relative test declared that genuinely-winsorizable column
        # un-winsorizable and left its real fat tail alone.  Exact equality only fires
        # when the clip truly cannot move anything, which is what "no progress" means.
        if np.array_equal(y2.to_numpy(dtype='float64'), y.to_numpy(dtype='float64'),
                          equal_nan=True):
            return orig, 0, p + 1, False      # cannot make progress -> un-winsorizable
        y = y2
    return orig, 0, max_passes, False         # pass bound exhausted -> un-winsorizable


def normalizeAndDropNA(df, weight_series=None, winsor_sigma=WINSOR_SIGMA,
                       method=None, rank_bounded=True):
    """WINSORIZE each weighted RAW metric column at +-winsor_sigma, then
    cross-sectionally z-score; drop a row only when EVERY metric is NaN.

    method : NORM_ZSCORE (default, the shipped path -- everything below applies) or
             NORM_RANK (rank -> inverse-normal; see NORM_RANK's notes and _rank_normal).
             Under NORM_RANK the winsorizer is NOT run: a monotone relabelling of a column
             has the same ranks as the column, so clipping its tail cannot change the
             result.  The winsorizer is retained, not removed -- the z-path still uses it.
    rank_bounded : NORM_RANK only.  True (default) rank-maps the bounded/discrete columns
             (WINSOR_EXEMPT_BOUNDED) along with the rest.  Their WINSORIZATION exemption
             does not carry over, because it rests on a premise the rank map removes: they
             are exempt from clipping because a large |z| there is real structure that
             clipping would destroy, whereas ranking destroys nothing (it is
             order-preserving) and their spacing is ORDINAL anyway -- Piotroski 9 vs 8 is
             "one criterion better", not a measured distance, which is exactly what a
             normal-scores map assumes.  Leaving them un-mapped would also put a raw-scaled
             column beside 13 N(0,1) columns inside a frame every consumer downstream
             reads as normalised, i.e. it would re-introduce the cross-column
             incomparability the method exists to remove.  Set False to measure that
             alternative.

    OUTLIER HANDLING (audit H1/H2 fix 2026-07-19, upgraded 2026-07-25).  Originally
    this EJECTED any row with |z| > 4 in ANY metric column.  Three things were wrong:

      (a) ZERO-WEIGHT metrics could eject a name.  The mask ran over every column
          including the w=0 diagnostics (priceGrowth / DcfToPrice / BoScore), so a
          metric that contributes NOTHING to AggScore could delete a company from
          the ranking.  Proven on the shipped 2026-07-17 run: of the 10 names this
          function ejected, CART was ejected SOLELY on priceGrowth (w = 0.000).
      (b) EJECTION IS ADVERSE SELECTION.  |z| > 4 on a value/quality metric is
          usually the name being EXCEPTIONAL on that axis (highest FCF yield,
          fastest growth, safest balance sheet), i.e. exactly what the score is
          hunting.  Dropping it also silently shortened every top-100 deliverable
          to 90 rows (verified: postScoreMetric_raw = 100 rows, shipped
          postScoreMetric = 90).
      (c) CLAMPING THE Z IS ONLY HALF A FIX (the 2026-07-19 interim, now replaced).
          Clamping AFTER mu/sigma are computed leaves mu/sigma CONTAMINATED by the
          outlier, so every OTHER name's z stays deflated by the sigma the clipped
          value inflated -- currentRatio delivered sigma(z) ~ 0.5, i.e. about half its
          intended weight (audit H2).  And +-4 sigma is a don't-crash bound, not a
          winsorization threshold: a clean n~90 column's true max is |z| ~ 2.6-3.0, so
          a value clamped to 4 still sat above every legitimate name.

    NOW: the RAW column is winsorized at +-winsor_sigma (see _winsorize_raw), and
    mu/sigma for the z-score are computed on the WINSORIZED column -- so the outlier
    neither dominates the score nor distorts anyone else's z.  The NAME IS ALWAYS KEPT.

    WHAT IS GUARANTEED (corrected -- see _winsorize_raw; the earlier "BY CONSTRUCTION"
    wording was false): for every column the winsorizer REPORTS as converged, max|z| <=
    winsor_sigma is verified directly on the result.  A column whose shape makes that
    target unreachable is returned UNTOUCHED at its natural z and named loudly on stdout;
    it is NOT silently left part-clipped, and its raw values are NOT annihilated.

    THREE exemptions, all deliberate:
      * ZERO-WEIGHT columns -- a w=0 column can neither dominate the score nor eject
        anyone, so it stays the honest display-only diagnostic it is;
      * BOUNDED/DISCRETE columns (WINSOR_EXEMPT_BOUNDED) -- they cannot have a fat tail,
        so a large |z| there is real structure;
      * columns the target cannot be reached for (above).
    With no weight_series supplied (the offline baseline_tools callers) every non-bounded
    metric column is winsorized.

    INTERIM, by design.  Sigma-winsorization still assumes an approximately symmetric,
    roughly-Gaussian column.  Several of these metrics are strongly skewed and the
    weights were mu-tuned on the pipeline as it stood, so the principled end state is
    RANK-BASED (inverse-normal) normalization plus a weight RE-FIT -- both of which
    change the weight vector's meaning and therefore need a CEO decision.  This fix
    removes the demonstrable defects (contaminated mu/sigma, a bound that bounds
    nothing) without pre-empting that decision.

    NOT renormalized after clamping, deliberately: the production weights were
    tuned on this pipeline as-is, so rescaling z back to unit variance would
    silently re-weight the whole vector.

    Returns (frame, outlierlist).  outlierlist now holds ONLY the all-NaN rows
    that were genuinely dropped -- winsorized names are NOT outliers, they are in
    the ranking.
    """
    df.reset_index(inplace=True, drop=True)

    # Check if dataframe is empty or has no metric columns
    if df.empty:
        print("Warning: Input dataframe is empty.")
        return df, []

    # Replace inf values with nan (modern approach without inplace)
    metric_cols = [col for col in df.columns if col != 'source']

    if len(metric_cols) == 0:
        print("Warning: No metric columns found (only 'source' column present).")
        return df, []

    df_clean = df.copy()
    # Suppress the FutureWarning about downcasting in replace()
    with pd.option_context('future.no_silent_downcasting', True):
        for col in metric_cols:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

    # Drop rows only if ALL metric columns are NaN (completely invalid rows)
    # This is less aggressive than dropping rows with ANY NaN, preserving more data
    nan_counts = df_clean[metric_cols].isna().sum(axis=1)
    dropmask = nan_counts < len(metric_cols)  # Keep rows with at least one valid metric
    outlierlist = list(df_clean['source'][~dropmask].copy())
    dfnona = df_clean[dropmask].copy()

    # Guard: if all rows have NaN, return empty df with warning
    if dfnona.empty:
        print(f"Warning: All {len(df)} rows dropped due to NaN values (all metric columns were NaN). Returning empty dataframe.")
        return dfnona, list(df['source'])

    tempnum = dfnona.drop('source',axis=1).apply(pd.to_numeric, errors='coerce')

    if method is None:
        method = NORM_METHOD_DEFAULT
    if method not in (NORM_ZSCORE, NORM_RANK):
        raise ValueError("normalizeAndDropNA: method must be %r or %r, got %r"
                         % (NORM_ZSCORE, NORM_RANK, method))

    if method == NORM_RANK:
        # --- RANK -> INVERSE-NORMAL ---------------------------------------------------
        # No winsorization pass, because the map depends only on the ORDER within a column
        # and clipping cannot REORDER anything: every name the winsorizer would move keeps
        # its exact rank position relative to every other name.  (Precisely: clipping is
        # weakly monotone, so its only possible effect here is to MERGE the clipped tail
        # into one tied score -- it can never swap two names, and it cannot change any
        # unclipped name's score at all.  Verified in
        # test_rank_normalization.test_rank_normal_is_invariant_to_any_strictly_monotone_
        # transform + ..._clipping_only_merges_the_clipped_tail.)  The fat tail the
        # winsorizer exists to defuse cannot dominate a rank map in the first place, so
        # running it would only destroy raw values for no change in outcome.  Nothing is
        # silently skipped -- it is inapplicable.
        if rank_bounded:
            to_map = list(tempnum.columns)
        else:
            to_map = [c for c in tempnum.columns if c not in WINSOR_EXEMPT_BOUNDED]
        for col in to_map:
            tempnum[col] = _rank_normal(tempnum[col])
        # fillna(0) is the SAME line the z-path uses, and here 0 is the observed MEDIAN of
        # the column, so an unavailable metric is exactly neutral (finding N1).
        temp_normpsmdf = tempnum.fillna(0)
        skipped = [c for c in tempnum.columns if c not in to_map]
        print("normalizeAndDropNA[rank]: inverse-normal mapped %d column(s) over %d row(s)"
              "%s" % (len(to_map), len(tempnum),
                      ("; LEFT RAW (rank_bounded=False): %s" % skipped) if skipped else ""),
              flush=True)
        dfnona[temp_normpsmdf.columns] = temp_normpsmdf
        return dfnona.copy(), outlierlist

    # --- PASS 1: winsorize the WEIGHTED RAW columns at +-winsor_sigma -------------
    # Done BEFORE mu/sigma so the z-score below is computed on an UNCONTAMINATED
    # column. w=0 columns are left completely alone (display-only diagnostics).
    if weight_series is None:
        weighted = list(tempnum.columns)                # offline callers: guard all
    else:
        weighted = [c for c in tempnum.columns
                    if float(weight_series.get(c, 1) or 0) != 0]
    guarded = [c for c in weighted if c not in WINSOR_EXEMPT_BOUNDED]
    per_col_clipped, affected_rows, total_clipped = {}, pd.Series(False, index=tempnum.index), 0
    not_converged = {}
    for col in guarded:
        clipped, n_changed, n_passes, converged = _winsorize_raw(tempnum[col], winsor_sigma)
        if not converged:
            # LEFT AT ITS NATURAL Z, deliberately (see _winsorize_raw): the target is
            # unreachable for this column's shape, so clipping would annihilate the raw
            # value without moving a single z-score.
            s = tempnum[col]
            mz = float(((s - s.mean()) / s.std()).abs().max()) if s.std() else float('nan')
            not_converged[col] = (n_passes, mz)
            continue
        if n_changed:
            moved = ~np.isclose(tempnum[col].to_numpy(dtype='float64'),
                                clipped.to_numpy(dtype='float64'),
                                rtol=0, atol=0, equal_nan=True)
            affected_rows |= pd.Series(moved, index=tempnum.index)
            per_col_clipped[col] = (n_changed, n_passes)
            total_clipped += n_changed
        tempnum[col] = clipped
    if total_clipped:
        print(f"normalizeAndDropNA: winsorized {total_clipped} RAW metric cell(s) at "
              f"+-{winsor_sigma} sigma (names KEPT, mu/sigma recomputed after): "
              + ", ".join(f"{c}={n}(x{p} passes)" for c, (n, p) in per_col_clipped.items())
              + f" | affected names: {sorted(dfnona['source'][affected_rows.to_numpy()])}",
              flush=True)
    if not_converged:
        # LOUD, per column: a metric whose spread the winsorizer could not tame is a fact
        # the reader needs, not a silent internal outcome.
        print("!" * 78, flush=True)
        print("normalizeAndDropNA: WINSORIZATION NOT APPLIED to %d column(s) -- the "
              "max|z| <= %s target is UNREACHABLE for their shape (near-two-point /"
              " discrete), so they are LEFT AT THEIR NATURAL z:"
              % (len(not_converged), winsor_sigma), flush=True)
        for c, (p, mz) in not_converged.items():
            print("    %-28s passes=%-4d natural max|z|=%.4f  (raw values UNCHANGED)"
                  % (c, p, mz), flush=True)
        print("!" * 78, flush=True)
    exempt_bounded = [c for c in weighted if c in WINSOR_EXEMPT_BOUNDED]
    if exempt_bounded:
        print("normalizeAndDropNA: bounded/discrete metric(s) EXEMPT from winsorization "
              f"(cannot have a fat tail): {exempt_bounded}", flush=True)
    unguarded = [c for c in tempnum.columns if c not in weighted]
    if unguarded:
        print("normalizeAndDropNA: zero-weight metric(s) EXEMPT from winsorization "
              f"(display-only, cannot dominate or eject): {unguarded}", flush=True)

    # --- PASS 2: mu/sigma on the WINSORIZED columns, then z-score -----------------
    colmeans = tempnum.mean()
    colstds = tempnum.std()
    # Handle division by zero: if std is 0 or NaN, set normalized values to 0
    colstds = colstds.replace(0, np.nan).fillna(1)  # Avoid division by zero
    # subtract the mean and divide by the standard deviation
    temp_normpsmdf = (tempnum - colmeans) / colstds
    # Fill remaining NaN values with 0 (for columns that were all NaN)
    temp_normpsmdf = temp_normpsmdf.fillna(0)

    dfnona[temp_normpsmdf.columns] = temp_normpsmdf
    dfnonanorm = dfnona.copy()

    return dfnonanorm, outlierlist

def unweight_postrank_metrics(df, cols=None, verbose=False, label=''):
    """Recover the METRIC z-scale from a postRank-style frame: divide each metric column by
    its production weight.  Returns (new_df, kept_cols, dropped_zero_cols).

    WHY THIS IS A SHARED HELPER AND NOT INLINE ARITHMETIC (2026-07-30).  postRank's metric
    columns are `z x w`.  Any consumer that wants the metric must un-weight, and CONSUMERS THAT
    MUST AGREE WITH EACH OTHER ARE THE WHOLE PROBLEM: the OLS path fits coefficients in one
    function (`backtest_unified.run_top100_postrank_ols`) and applies them in another
    (`backtest_outputs.compute_ols_weighted_ranking`).  When only the FIT side was un-weighted,
    the two bases diverged for negative weights and the re-ranker inverted -- because
    `standardize(z x w) == sign(w) * standardize(z)`.  Before that, BOTH sides used `z x w` and
    the double negation made the result accidentally correct.  So a one-sided fix was worse
    than no fix, and the durable remedy is that both sides call the SAME function.

    `w = 0` columns are DROPPED, not divided: they are identically +-0.0 in postRank (the
    multiply annihilated them), so there is no information to recover and 0/0 is not a metric.
    Columns with no weight entry (e.g. `moatScore`, which is merged post-weighting and is
    already raw) are left untouched.
    """
    postBm, postNew = cdic.getPostDict()
    W = {**{k: float(postBm[k]['w']) for k in postBm},
         **{k: float(postNew[k]['w']) for k in postNew}}
    out = df.copy()
    if cols is None:
        cols = [c for c in out.columns if c in W]
    kept, dropped = [], []
    for c in cols:
        if c not in out.columns:
            continue
        w = W.get(c)
        if w is None:
            kept.append(c)          # not a weighted metric -- leave as-is
            continue
        if w == 0:
            dropped.append(c)
            continue
        out[c] = pd.to_numeric(out[c], errors='coerce') / w
        kept.append(c)
    if verbose:
        print('  %sun-weighted %d postRank metric column(s) to the metric z-scale (signs now '
              'match the metrics); dropped %d zero-weight column(s) %s'
              % (label, len(kept), len(dropped), dropped), flush=True)
    return out, kept, dropped


def getAggScore(df):
    #df['AggScore'] = np.nan
    cts = list(set(df.columns) - set(['source']))
    df['AggScore'] = df[cts].sum(axis=1)
    postRank = df
    postRank.sort_values(by='AggScore',ascending=False,inplace=True)
    postRank.reset_index(drop=True,inplace=True)

    return postRank

#  Columns that must never be ranked INSIDE the rank-of-ranks: AggScore is the weighted
#  SUM of every other column here, so ranking it alongside its own components counts the
#  whole score a second time (audit M1).
ROR_EXCLUDE = ['source', 'AggScore', 'rankOfRanks', 'rankOfRanks_diag']

#  DIAGNOSTIC name.  rankOfRanks orders NOTHING that ships (AggScore does), and it is
#  invariant to weight MAGNITUDE -- only weight SIGNS survive a per-column rank -- so it
#  is an EQUAL-WEIGHT alternative view, not a competing ranking.  It used to ship
#  unlabelled beside AggScore in three CSV families with a visibly different top-5
#  (2026-07-17: AggScore IMPP/RAVE/SYS1.L/INMD/AUDC vs rankOfRanks
#  IMPP/AJ91.DE/RFX.L/AEP.L/CAPD.L), which invites reading it as a second opinion it is
#  not entitled to be.  The `_diag` suffix is the label.
ROR_COLUMN = 'rankOfRanks_diag'


def getRankOfRanks(df):
    """Equal-weight rank-sum DIAGNOSTIC, emitted as ROR_COLUMN.

    Sums each name's per-metric rank and re-ranks the sum.  Because `rank()` discards
    magnitude, this weights every metric EQUALLY and keeps only the sign of the
    production weight -- deliberately a different lens from AggScore, never a substitute
    for it.  AggScore itself is EXCLUDED from the sum (audit M1 fix, 2026-07-19): it was
    previously included as a 22nd ranked column, i.e. the weighted sum of the other 21
    counted twice.  Verified against the shipped 2026-07-17 run: including it reproduced
    the shipped column bit-for-bit, and excluding it moves 80 of 90 names (max 11 rank
    positions).
    """
    postRankOfRanks = pd.DataFrame()
    for col in df.columns:
        if col not in ROR_EXCLUDE:
            postRankOfRanks[col + 'rank'] = df[col].rank(ascending=False,method='dense')

    cts = list(set(postRankOfRanks.columns) - set(['source']))
    df[ROR_COLUMN] = postRankOfRanks[cts].sum(1).rank(ascending=True,method='dense')

    return df



def postBoRankingPassFilter(df,mlist,lco,hco):
    pf = df[~df[df.columns.intersection(mlist)].lt(lco).any(axis=1)]
    pf = pf[~pf[pf.columns.intersection(mlist)].gt(hco).any(axis=1)]
    pf.reset_index(inplace=True, drop=True)

    return pf
