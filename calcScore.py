import pandas as pd
import numpy as np
from tqdm import tqdm
import createDicts as cdic

def simpleScore_fromDict(bm_df,bm_ave,bm_da,n=8,as_of=None):
    """Stage-1 per-symbol scoring.

    as_of : point-in-time date D (default None).  as_of=None reproduces the live
    pipeline BIT-FOR-BIT: the PIT slice below is never entered, so every symbol is
    scored over its full panel exactly as today.  Only when a real D is supplied is
    each symbol's panel restricted to rows AVAILABLE on/before D (pit_slice, design
    L1/L4) BEFORE the head(n) scoring window -- so head(n) picks the correct
    as-of-D quarters instead of assuming "newest row == today".
    """
    print(f'Calculating scores for each stock symbol in BoMetric_df')

    # --- ORDERING INVARIANT (Stage-1): calcByTier's .head(n) scoring window
    # assumes each ticker's rows are NEWEST-first. On tonight's data BoMetric_df
    # arrives newest-first (verified 600/600 descending), but NOTHING on the live
    # path enforces it -- data_quality re-sorts only cdx_df, so this is an
    # incidental FMP ingestion order, not an invariant. Defensively re-sort a COPY
    # to newest-first: a no-op when already correct, a fix if the order ever drifts.
    # Dates coerced robustly (a naive string sort mis-orders mixed/malformed dates).
    # Mirrors stage2_pit._sort_newest_first and the Stage-2 re-sort in
    # postBoRank.postBoScoreRanking.
    if 'date' in bm_df.columns:
        bm_df = bm_df.copy()
        _n_before = bm_df.groupby('source').size()
        bm_df['date'] = pd.to_datetime(bm_df['date'], errors='coerce')
        bm_df = bm_df.sort_values(['source', 'date'], ascending=[True, False]).reset_index(drop=True)
        assert bm_df.groupby('source').size().equals(_n_before), \
            "Stage-1 newest-first re-sort changed per-ticker row counts"
    # test
    #    bm_df = BoMetric_df
    #    bm_da = BoMetric_dateAve
    #    bm_ave = datandmetricdic['BoMetric_ave']
    #   BoScore_df['date'] = BoMetric_dateAve.index
    #test
    dict_base, dict_mean, dict_diff, dict_unity, dict_special = cdic.getBaseMeanDiffUnitySpecialDicts()
    tbs_df = pd.DataFrame(columns=['score', 'source'])
    tbs_df['source'] = bm_df['source'].unique()
    pbar = tqdm(total=len(bm_df['source'].unique()))

    for ticker in bm_df['source'].unique():
        bmdf_tick = bm_df[bm_df['source'] == ticker]
        if as_of is not None:
            # PIT: keep only rows available on/before D, then score head(n) over them.
            # Never entered on a live run (as_of=None) -> live behaviour unchanged.
            import pit_slice as ps
            bmdf_tick = ps.slice_panel_as_of(bmdf_tick, D=as_of)
            if bmdf_tick.empty:
                continue
        tempscore = 0
        for key in dict_base:
            temp = calcByTier('base', dict_base[key]['Tier'], dict_base[key]['Sign'], bmdf_tick[key], bm_ave[key],key,n)
            tempscore = tempscore + temp
        for key in dict_mean:
            mkey = "m" + key[0].upper() + key[1:]
            temp = calcByTier('mean', dict_mean[key]['Tier'], dict_mean[key]['Sign'], bmdf_tick[mkey], bm_ave[mkey],key,n)
            tempscore = tempscore + temp
        for key in dict_diff:
            dkey = "d" + key[0].upper() + key[1:]
            temp = calcByTier('diff', dict_diff[key]['Tier'], dict_diff[key]['Sign'], bmdf_tick[dkey], bm_ave[dkey],key,n)
            tempscore = tempscore + temp
        for key in dict_unity:
            ukey = "u" + key[0].upper() + key[1:]
            temp = calcByTier('unity', dict_unity[key]['Tier'], dict_unity[key]['Sign'], bmdf_tick[ukey], bm_ave[ukey],key,n)
            tempscore = tempscore + temp
        for key in dict_special:
            temp = calcByTier('special', dict_special[key]['Tier'], dict_special[key]['Sign'], bmdf_tick[key], bm_ave[key],key,n)
            tempscore = tempscore + temp

        tbs_df.loc[tbs_df['source'] == ticker, 'score'] = tempscore
        pbar.update(n=1)

    pbar.close()
    tbs_df.sort_values('score', ascending=False,inplace=True)
    return tbs_df

def calcByTier(dict,Tier,Sign,metvec,avec,met,n):
    resvec = pd.DataFrame(columns=[met])
    w = 0
    if Tier == 'S':
        w = 1
    elif Tier == 'A':
        w = 0.75
    elif Tier == 'B':
        w = 0.5
    elif Tier == 'C':
        w = 0.3
    elif Tier == 'D':
        w = 0.1
    else:
        w = 0

    if dict == 'mean':
        testvec = metvec - avec
    elif dict == 'unity':
        testvec = metvec - 1
    else:
        testvec = metvec

    resvec[met] = [w if Sign * val > 0 else 0 for val in testvec]
    res = resvec[met].head(n).mean()

    return res


def getAves2(df):
    print('Getting average values')
    # Ensure 'date' exists and is datetime
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            pass

    # Work on numeric columns only for median calculations
    without_source = df.drop(columns=['source'], errors='ignore')
    # For full median across the dataset (numeric columns only)
    res_fullMean = without_source.select_dtypes(include=[float, int]).median(numeric_only=True)

    # Per-date medians (group by date) — use numeric columns only
    if 'date' in without_source.columns:
        res_withDates = without_source.groupby('date').median(numeric_only=True)
        res_withDates = res_withDates.iloc[::-1].reset_index()
    else:
        res_withDates = pd.DataFrame()

    colslost = set(df.columns) - set(res_fullMean.index)

    meandic = {'BoMetric_ave': res_fullMean, 'BoMetric_dateAve': res_withDates, 'colslost': colslost}
    return meandic

