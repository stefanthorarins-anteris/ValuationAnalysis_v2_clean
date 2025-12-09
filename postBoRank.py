import createDicts as cdic
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def postBoScoreRanking(bmtop,bstop,cdxtop,baseurl,api_key,period='quarter',nq=16):
    print('Ranking the top 100 stocks, according to BoScore.')
    #test
    #bmtop = BoM_dftop100
    #bstop = BoS_dftop100
    #cdxtop = cdx_dftop100
    #period='quarter'
    #nq = 12
    #baseurl = configdic['baseurl']
    #api_key = configdic['api_key']
    #test
    postBmRankingDict, postNewRankingDict = cdic.getPostDict()
    postScoreMetric_df = pd.DataFrame()
    postScoreMetric_df['source'] = bstop['source']
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postBmRankingDict.keys())], axis=1)
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postNewRankingDict.keys())], axis = 1)
    #postScoreMetric_df.drop(labels=['SalePerEmployee'],axis=1,inplace=True)
    #postScoreMetric_df['BoScore'] = bstop['score']*0.2
    postRanking_df = pd.DataFrame()
    weight_df = pd.DataFrame()
    # Build a stable weight mapping from the post dictionaries so we always have a weight for each metric
    postBmRankingDict_local, postNewRankingDict_local = cdic.getPostDict()
    weight_series = {**{k: postBmRankingDict_local[k]['w'] for k in postBmRankingDict_local},
                     **{k: postNewRankingDict_local[k]['w'] for k in postNewRankingDict_local}}
    #weight_df['BoScore'] = pd.Series(0.2)
    weightzerobool = False
    mcapAve = cdxtop.marketCap.mean()
    cdxtop['mcapQuants'] = (-1)*((pd.qcut(cdxtop['marketCap'], 4).cat.codes/(3) - 0.5))
    #mcapAve = 70310173993
    tempcntr = 0


    pbar = tqdm(total=len(bstop['source'].unique()))
    for ticker in bstop['source']:
        tempcdx = cdxtop.loc[cdxtop['source'] == ticker]
        tempfcf = tempcdx.freeCashFlow
        tempshares = tempcdx.weightedAverageShsOut
        tempmcap = tempcdx.marketCap
        tempmcapQuants = tempcdx.mcapQuants.iloc[0]
        #tempcr = tempcdx.currentRatio

        #resp_fr = requests.get(f'{baseurl}v3/ratios/{ticker}?period={period}&limit={nq}&apikey={api_key}')
        #resp_km = requests.get(f'{baseurl}v3/key-metrics/{ticker}?period={period}&limit={nq}&apikey={api_key}')
        resp_dcf = requests.get(f'{baseurl}v3/historical-discounted-cash-flow-statement/{ticker}?period={period}&limit={nq}&apikey={api_key}')
        #resp_emp = requests.get(f'{baseurl}v4/historical/employee_count?symbol={ticker}&apikey={api_key}')
        resp_fsc = requests.get(f'{baseurl}v4/score?symbol={ticker}&apikey={api_key}')
        resp_fg = requests.get(f'{baseurl}v3/financial-growth/{ticker}?limit={nq}&apikey={api_key}')
        #fr = pd.DataFrame.from_dict(resp_fr.json())
        #km = pd.DataFrame.from_dict(resp_km.json())
        #tempcr = fr['currentRatio']
        dcf = pd.DataFrame.from_dict(resp_dcf.json())
        #emp = pd.DataFrame.from_dict(resp_emp.json())
        fsc = pd.DataFrame.from_dict(resp_fsc.json())
        if not (fsc.empty or dcf.empty):
            fg = pd.DataFrame.from_dict(resp_fg.json())
            for key1 in postBmRankingDict.keys():
                met = postBmRankingDict[key1]['eqMet']
                weight = postBmRankingDict[key1]['w']
                temp = bmtop[bmtop['source']==ticker].head(nq)
                if key1 == 'grahamNumberToPrice':
                    tempgnprat = tempcdx['grahamNumber'] / tempcdx['price']
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempgnprat.head(nq).mean()
                elif key1 == 'bVpRatio':
                    tempbvtop = 1/tempcdx[met]
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempbvtop.head(nq).mean()
                elif key1 == 'revenueGrowth':
                    revPCTgr = tempcdx[met].pct_change(-4)
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = revPCTgr.head(nq).mean()
                else:
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempcdx[met].head(nq).mean()
                weight_df[key1] = pd.Series(weight)

            for key2 in postNewRankingDict.keys():
                weightzerobool = False
                #if key2 == 'FCFperShare':
                #    weight = postNewRankingDict[key2]['w']
                #    weight_df[key2] = pd.Series(weight)
                #    #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = fr['freeCashFlowPerShare'].head(nq).mean()*weight
                #    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempfcf/tempshares).head(nq).mean()*weight

                if key2 == 'freeCashFlowYield':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = km['freeCashFlowYield'].head(nq).mean()*weight
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempfcf / tempmcap).head(nq).mean()
                    #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = km.freeCashFlowYield.head(nq).mean() * weight

                if key2 == 'freeCashFlowPerShareGrowth':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    fcfps = tempfcf/tempshares
                    fcfpsgr = fcfps.pct_change(-4)
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = fcfpsgr.head(nq).mean()
                    #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = km.freeCashFlowYield.head(nq).mean() * weight

                if key2 == 'EPStoEPSmean':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    eps = tempcdx['netIncome'] / tempcdx['weightedAverageShsOut']
                    epsmean = eps.mean()
                    a = 0.4
                    tw = a*(1+(1-a) + (1-a)**2 + (1-a)**3)
                    if all(eps.iloc[0:4] > 0):
                        epstoepsmean = epsmean - (a/tw)*(eps.iloc[0] + eps.iloc[1]*(1-a) +
                                                         eps.iloc[2]*(1-a)**2 + eps.iloc[3]*(1-a)**3)
                    else:
                        epstoepsmean = 0

                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = epstoepsmean

                if key2 == 'DcfToPrice':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    temp = dcf['dcf'].head(nq).mean()
                    temp2 = dcf['price'][0]
                    temp_dcf = dcf['dcf']
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (temp/temp2)

                #if key2 == 'currentRatio':
                #    weight = postNewRankingDict[key2]['w']
                #    weight_df[key2] = pd.Series(weight)
                #    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = tempcr.head(nq).mean()*weight

                if key2 == 'marketCapRevQuants':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempmcap.head(nq).mean()/mcapAve - 1)*weight
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = tempmcapQuants

                if key2 == 'tbVpRatio':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    tbtp = tempcdx['tangibleBookValuePerShare']/tempcdx['price']
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] =\
                        tbtp.head(nq).mean()

                if key2 == 'Altman-Z':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] =\
                        fsc['altmanZScore'].head(nq).mean()

                if key2 == 'Piotroski':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] =\
                        fsc['piotroskiScore'].head(nq).mean()

                if key2 == 'BoScore':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = bstop.loc[
                        bstop['source'] == ticker, 'score']

                if key2 == 'priceGrowth':
                    weight = postNewRankingDict[key2]['w']
                    weight_df[key2] = pd.Series(weight)
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] =\
                        -dcf['price'].pct_change(-1).head(nq).mean()

            #    if key2 == 'SalePerEmployee':
            #        #temp = emp['employeeCount'].head(nq)
            #        #temp2 = inc['revenue'].head(nq)
            #        #spe = temp2/temp
            #        #dspe = spe - spe.shift(-1)

            #        #if any(np.isinf(dspe)) or any(np.isnan(dspe)):
            #            #weightzerobool = True

            #        #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = dspe.mean()

        tempcntr  = tempcntr + 1
        pbar.update(n=1)

    #normalize
    testpsm_df = postScoreMetric_df
    postScoreMetric_df, outlierlist = normalizeAndDropNA(postScoreMetric_df)

    temp_normpsmdf_weighted = postScoreMetric_df.drop('source', axis=1)
    # Apply weights using the stable weight_series mapping; if a weight is missing, default to 1
    for col in temp_normpsmdf_weighted.columns:
        w = weight_series.get(col, 1)
        temp_normpsmdf_weighted[col] = postScoreMetric_df[col].values * w
    #psmdf_normalized = pd.concat([postScoreMetric_df[postScoreMetric_df.columns.difference(tempnum.columns)], temp_normpsmdf_weighted], axis=1)
    psmdf_normalized = pd.concat(
        [postScoreMetric_df[postScoreMetric_df.columns.difference(temp_normpsmdf_weighted.columns)], temp_normpsmdf_weighted], axis=1)


    postRank = getAggScore(psmdf_normalized)

    tmpcorr = np.corrcoef(list(postRank['BoScore'].values), list(postRank['AggScore'].values))
    BoAggCorr = tmpcorr[0,1]

    postRank = getRankOfRanks(postRank)
    plotRank = postRank
    plotRank['rankOfRanks'] = plotRank['rankOfRanks']/10
    plotRank['AggScore'] = plotRank['AggScore']/10
    mlist = list(set(plotRank.columns) - set(['source']))
    plotRank = postBoRankingPassFilter(plotRank,mlist,5,5)

    #finalPostRank_df = getFinalRank(postRank)

    #roror = getRankOfRankOfRanks(finalPostRank_df)

    pbar.close()
    #rankdic = {'finalBoRank_df': finalPostRank_df, 'postRank': postRank, 'postRankOfRanks': postRankOfRanks,
    #           'psmdf_normalized': psmdf_normalized, 'BoAggCorr': BoAggCorr, 'outlierlist': outlierlist,
    #           'roror': roror}
    rankdic = {'postRank': postRank, 'postScoreMetric': postScoreMetric_df,
               'psmdf_normalized': psmdf_normalized, 'BoAggCorr': BoAggCorr, 'outlierlist': outlierlist}

    return rankdic

def normalizeAndDropNA(df):
    df.reset_index(inplace=True, drop=True)
    invdropmask = [False for i in range(len(df))]
    for col in df.columns:
        if col != 'source':
            tempcol = df[col].copy()
            # Replace inf values with nan (modern approach without inplace)
            tempcol = tempcol.replace([np.inf, -np.inf], np.nan)
            tempdm = tempcol.isna()
            invdropmask = [x | y for x, y in zip(invdropmask, tempdm)]
    dropmask = [not x for x in invdropmask]
    outlierlist = list(df['source'][invdropmask].copy())
    dfnona = df[dropmask].copy()
    
    # Guard: if all rows have NaN, return empty df with warning
    if dfnona.empty:
        print("Warning: All rows dropped due to NaN values. Returning empty dataframe.")
        return dfnona, list(df['source'])
    
    tempnum = dfnona.drop('source',axis=1).apply(pd.to_numeric, errors='coerce')
    # calculate the mean and standard deviation of each column
    colmeans = tempnum.mean()
    colstds = tempnum.std()
    # subtract the mean and divide by the standard deviation
    temp_normpsmdf = (tempnum - colmeans) / colstds
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

    #postRank = pd.DataFrame(columns=['source','AggScore'])
    #postRank['source'] = df['source']
    #postRank['BoScore'] = df['BoScore']
    #for i in range(0,len(postRank['source'])):
    #    rl = list(df.iloc[i,:])
    #    postRank.loc[postRank['source'] == rl[0], 'AggScore'] = sum(rl[1:])

    #postRank.dropna(inplace=True)
    #postRank.sort_values(by='AggScore',ascending=False,inplace=True)
    #postRank.reset_index(drop=True,inplace=True)

    return postRank

def getRankOfRanks(df):
    postRankOfRanks = pd.DataFrame()
    for col in df.columns:
        if col not in ['source']:
            postRankOfRanks[col + 'rank'] = df[col].rank(ascending=False,method='dense')

    cts = list(set(postRankOfRanks.columns) - set(['source']))
    df['rankOfRanks'] = postRankOfRanks[cts].sum(1).rank(ascending=True,method='dense')

    return df

def getRankOfRankOfRanks(df):
    roror = pd.DataFrame()
    roror['source'] = df['source']
    for col in df.columns:
        if col in ['rankOfRanks', 'AggScore']:
            roror[col + 'ror'] = df[col].rank(ascending=False,method='dense')

    roror['rankOfRanksOfRanks'] = roror.sum(1)
    roror['rankOfRanksOfRanks'] = roror['rankOfRanksOfRanks'].rank(ascending=True,method='dense')
    roror.sort_values(by='rankOfRanksOfRanks',inplace=True)

    return roror

def getFinalRank(pr_df,pror_df):
    tmpfpr_df = pd.DataFrame(columns=['source','AggScore'])
    tmpfpr_df['source'] = pr_df['source']
    tmpfpr_df['AggScore'] = pr_df['AggScore']
    finalPostRank_df = tmpfpr_df.merge(pror_df[['source','rankOfRanks','BoScorerank']], on='source',how='inner')

    return finalPostRank_df

def postBoRankingPassFilter(df,mlist,lco,hco):
    pf = df[~df[df.columns.intersection(mlist)].lt(lco).any(axis=1)]
    pf = pf[~pf[pf.columns.intersection(mlist)].gt(hco).any(axis=1)]
    pf.reset_index(inplace=True, drop=True)

    return pf