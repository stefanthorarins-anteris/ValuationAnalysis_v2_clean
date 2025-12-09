import pandas as pd
import numpy as np
from tqdm import tqdm
import createDicts as cdic

def simpleScore_fromDict(bm_df,bm_ave,bm_da,n=8):
    print(f'Calculating scores for each stock symbol in BoMetric_df')
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

def getIQmean(df):
    dategrouped = df.groupby('date')
    # For each date, calculate the first and third quantiles (25th and 75th percentiles) for each column
    results = []
    for date, group in dategrouped:
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        mask = (group >= Q1) & (group <= Q3)
        df_middle_quantile = group[mask.all(axis=1)]
        mean = df_middle_quantile.mean()
        results.append({'date': date, 'mean': mean})
    result_df = pd.DataFrame(results)

    # Display the result
    print(mean)

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

def getAves_fuckedTTT(df):
    print('Getting average values')
    temp = df.drop(columns=['source'], axis=1)
    res_withDates = temp.groupby('date').mean()
    mc1 = set(df.columns) - set(res_withDates.columns)
    if len(mc1)>2:
        if 'source' in mc1:
            mc1.remove('source')
        if 'date' in mc1:
            mc1.remove('date')
        temp_mc1 = pd.DataFrame()
        temp_mc1['date'] = df['date'].unique()
        temp_mc1 = temp_mc1.sort_values(by='date').reset_index(drop=False)
        temp_mc1.drop('index',axis=1,inplace=True)
        temp_mc1 = temp_mc1.assign(**{string: None for string in mc1})
        res_withDates = res_withDates.assign(**{string: None for string in mc1})

        for col in mc1:
            for row in temp_mc1.itertuples():
                curdate = row.date
                indexes = df[df['date'] == curdate].index
                if len(indexes) < 0.33*df['source'].nunique():
                    if any(temp_mc1['date'] == curdate):
                        temp_mc1.drop(temp_mc1[temp_mc1['date'] == curdate].index, axis=0, inplace=True)
                    if curdate in res_withDates.index:
                        res_withDates.drop(res_withDates.loc[res_withDates.index == curdate].index, axis=0, inplace=True)
                else:
                    temp_date = df.iloc[indexes]
                    temp_mc1.loc[temp_mc1['date'] == curdate, col] = temp_date[col].median()
            res_withDates[col] = temp_mc1[col].values

    res_fullMean = df.median()
    for i in res_fullMean.index:
        if np.isnan(res_fullMean[i]):
            if np.isnan(res_withDates[i].mean()) == False:
                print(res_withDates[i])
                res_fullMean[i] = res_withDates[i].median()
            else:
                res_fullMean.dropna()

    colslost = set(df.columns) - set(res_fullMean.index)
    res_withDates = res_withDates.iloc[::-1]
#    temp = df.drop(columns=['source','date'], axis=1)
#    res_fullMean = res_fullMean.iloc[::-1]

    meandic = {'BoMetric_ave': res_fullMean, 'BoMetric_dateAve': res_withDates, 'colslost': colslost}
    return meandic
