import sys

import createDicts as cdic
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

import stage2_metrics as sm


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
                                postBmRankingDict, postNewRankingDict, postScoreMetric_df)

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

    postScoreMetric_df, outlierlist = normalizeAndDropNA(postScoreMetric_df)

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


def _fetch_ticker_dcf(ticker, baseurl, api_key, dcf_bulk_dict):
    """Fetch (or reuse bulk) DCF data for one ticker and return it as a normalised
    DataFrame.  Returns (dcf_df, dcf_from_bulk, resp_dcf_status, resp_dcf).
    """
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
                            postBmRankingDict, postNewRankingDict, postScoreMetric_df):
    """Compute every Stage-2 metric for one ticker and write it into
    postScoreMetric_df.  All formulas come from the shared stage2_metrics module
    (kept in lockstep with the offline reproduction); this function only decides
    which column each result lands in.
    """
    def setv(col, val):
        postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, col] = val

    # ---- postBmRankingDict metrics ----
    for key1 in postBmRankingDict.keys():
        setv(key1, sm.postbm_metric(key1, postBmRankingDict[key1]['eqMet'], tempcdx, nq))

    # ---- postNewRankingDict metrics ----
    tempfcf = tempcdx.freeCashFlow
    tempshares = tempcdx.weightedAverageShsOut
    tempmcap = tempcdx.marketCap

    setv('freeCashFlowYield', sm.free_cash_flow_yield(tempfcf, tempmcap, nq))
    setv('freeCashFlowPerShareGrowth', sm.free_cash_flow_per_share_growth(tempfcf, tempshares, nq))
    setv('EPStoEPSmean', sm.eps_to_eps_mean(tempcdx))
    setv('marketCapRevQuants', tempcdx.mcapQuants.iloc[0])
    setv('tbVpRatio', sm.tbv_p_ratio(tempcdx, nq))
    setv('Altman-Z', sm.altman_z(tempcdx))
    setv('Piotroski', sm.piotroski(tempcdx))
    setv('priceGrowth', sm.price_growth(tempcdx, nq))
    setv('CycleHeat', sm.cycleheat(tempcdx))

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
        kept, issuer_dupes_dropped = _co.dedup_ranked(ranked_srcs, cdxtop, names or {})
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


def normalizeAndDropNA(df):
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
    # calculate the mean and standard deviation of each column (NaN values are skipped by default)
    colmeans = tempnum.mean()
    colstds = tempnum.std()
    # Handle division by zero: if std is 0 or NaN, set normalized values to 0
    colstds = colstds.replace(0, np.nan).fillna(1)  # Avoid division by zero
    # subtract the mean and divide by the standard deviation
    temp_normpsmdf = (tempnum - colmeans) / colstds
    # Fill remaining NaN values with 0 (for columns that were all NaN)
    temp_normpsmdf = temp_normpsmdf.fillna(0)
    dfnona[temp_normpsmdf.columns] = temp_normpsmdf
    mask = abs(temp_normpsmdf) > 4
    to_keep = (~mask).all(axis=1)  # Keep rows where ALL columns are within 4 std (stricter than original)
    dfnonanorm = dfnona[to_keep].copy()
    outlierlist = list(set(outlierlist + list(dfnona['source'][~to_keep])))

    # Guard: if filtering removed all rows, keep at least the top 20% (avoid empty result)
    if dfnonanorm.empty and len(dfnona) > 0:
        print(f"Warning: Outlier filtering (>4 std) dropped all {len(dfnona)} rows. Keeping top 20% by row count.")
        keep_count = max(1, len(dfnona) // 5)
        dfnonanorm = dfnona.head(keep_count).copy()

    return dfnonanorm, outlierlist

def getAggScore(df):
    #df['AggScore'] = np.nan
    cts = list(set(df.columns) - set(['source']))
    df['AggScore'] = df[cts].sum(axis=1)
    postRank = df
    postRank.sort_values(by='AggScore',ascending=False,inplace=True)
    postRank.reset_index(drop=True,inplace=True)

    return postRank

def getRankOfRanks(df):
    postRankOfRanks = pd.DataFrame()
    for col in df.columns:
        if col not in ['source']:
            postRankOfRanks[col + 'rank'] = df[col].rank(ascending=False,method='dense')

    cts = list(set(postRankOfRanks.columns) - set(['source']))
    df['rankOfRanks'] = postRankOfRanks[cts].sum(1).rank(ascending=True,method='dense')

    return df



def postBoRankingPassFilter(df,mlist,lco,hco):
    pf = df[~df[df.columns.intersection(mlist)].lt(lco).any(axis=1)]
    pf = pf[~pf[pf.columns.intersection(mlist)].gt(hco).any(axis=1)]
    pf.reset_index(inplace=True, drop=True)

    return pf
