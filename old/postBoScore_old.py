import createDicts as cdic
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

def postBoScoreRanking(bmtop,bstop,cdxtop,baseurl,api_key,period='quarter',nq=9):
    #test
    bmtop = BoM_dftop100
    bstop = BoS_dftop100
    cdxtop = cdx_dftop100
    period='quarter'
    nq = 12
    #test
    postBmRankingDict, postNewRankingDict = cdic.getPostDict()
    postScoreMetric_df = pd.DataFrame()
    postScoreMetric_df['source'] = bstop['source']
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postBmRankingDict.keys())], axis=1)
    postScoreMetric_df = pd.concat([postScoreMetric_df, pd.DataFrame(columns=postNewRankingDict.keys())], axis = 1)
    #postScoreMetric_df.drop(labels=['SalePerEmployee'],axis=1,inplace=True)
    postScoreMetric_df['BoScore'] = bstop['score']*0.2
    postRanking_df = pd.DataFrame()
    weight_df = pd.DataFrame()
    weight_df['BoScore'] = pd.Series(0.2)
    weightzerobool = False
    mcapAve = cdxtop.marketCap.mean()
    cdxtop['mcapQuants'] = pd.qcut(cdxtop['marketCap'], 4, labels=[0.5, 0.25, 0, -0.5]).cat.codes
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

        resp_fr = requests.get(f'{baseurl}/ratios/{ticker}?period={period}&limit={nq}&apikey={api_key}')
        resp_dcf = requests.get(f'{baseurl}/historical-discounted-cash-flow-statement/{ticker}?period={period}&limit={nq}&apikey={api_key}')
        #resp_emp = requests.get(f'https://financialmodelingprep.com/api/v4/historical/employee_count?symbol={ticker}&apikey={api_key}')
        resp_fsc =  requests.get(f'https://financialmodelingprep.com/api/v4/score?symbol={ticker}&apikey={api_key}')
        fr = pd.DataFrame.from_dict(resp_fr.json())
        tempcr = fr['currentRatio']
        dcf = pd.DataFrame.from_dict(resp_dcf.json())
        #emp = pd.DataFrame.from_dict(resp_emp.json())
        fsc = pd.DataFrame.from_dict(resp_fsc.json())
        for key1 in postBmRankingDict.keys():
            met = postBmRankingDict[key1]['eqMet']
            weight = postBmRankingDict[key1]['w']
            temp = bmtop[bmtop['source']==ticker].head(nq)
            if key1 == 'grahamNumberToPrice':
                tempgnprat = tempcdx['grahamNumber'] / tempcdx['price']
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempgnprat.mean() * weight
            elif key1 == 'bVpRatio':
                tempbvtop = 1/tempcdx[met]
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempbvtop.mean() * weight
            else:
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key1] = tempcdx[met].mean()*weight
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
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempfcf / tempmcap).head(nq).mean() * weight
                #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = km.freeCashFlowYield.head(nq).mean() * weight

            if key2 == 'DcfToPrice':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                temp = dcf['dcf'].mean()
                temp2 = dcf['price'][0]
                temp_dcf = dcf['dcf']
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (temp/temp2)*weight

            #if key2 == 'currentRatio':
            #    weight = postNewRankingDict[key2]['w']
            #    weight_df[key2] = pd.Series(weight)
            #    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = tempcr.head(nq).mean()*weight

            if key2 == 'marketCapRevQuants':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                #postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = (tempmcap.head(nq).mean()/mcapAve - 1)*weight
                if tempmcapQuants == 0:
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = 1.5 * weight
                else:
                    postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = 1/tempmcapQuants * weight

            if key2 == 'Altman-Z':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = fsc['altmanZScore'].head(nq).mean()*weight

            if key2 == 'Piotroski':
                weight = postNewRankingDict[key2]['w']
                weight_df[key2] = pd.Series(weight)
                postScoreMetric_df.loc[postScoreMetric_df['source'] == ticker, key2] = fsc['piotroskiScore'].head(nq).mean()*weight

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
    tempnum = postScoreMetric_df.select_dtypes(include='number')
    # calculate the mean and standard deviation of each column
    colmeans = tempnum.mean()
    colstds = tempnum.std()
    # subtract the mean and divide by the standard deviation
    temp_normpsmdf = (tempnum - colmeans) / colstds
    # if you want to add the string column back
    psmdf_normalized = pd.concat([temp_normpsmdf, postScoreMetric_df[postScoreMetric_df.columns.difference(tempnum.columns)]], axis=1)

    pbar.close()
    postRank = pd.DataFrame(columns=['source','AggScore'])
    postRank['source'] = postScoreMetric_df['source']
    for i in range(0,len(postRank['source'])):
        rl = list(postScoreMetric_df.iloc[i,:])
        postRank.loc[postScoreMetric_df['source'] == rl[0],'AggScore'] = sum(rl[1:])

    #postRank.dropna(inplace=True)
    postRank.sort_values(by='AggScore',ascending=False,inplace=True)

    postRankOfRanks = pd.DataFrame()
    postRankOfRanks['source'] = postScoreMetric_df['source']
    for col in postScoreMetric_df.columns:
        if col not in ['pbRatio','marketCapRevQuants','source']:
            postRankOfRanks[col + 'rank'] = postScoreMetric_df[col].rank(ascending=False,method='dense')

    postRankOfRanks['rankOfRanks'] = postRankOfRanks.sum(1)
    postRankOfRanks['rankOfRanks'] = postRankOfRanks['rankOfRanks'].rank(ascending=False,method='dense')
    postRankOfRanks.sort_values(by='rankOfRanks',inplace=True)
    return postRank, postRankOfRanks, postScoreMetric_df






#see sector performance
#https://financialmodelingprep.com/api/v3/stock/sector-performance?apikey=YOUR_API_KEY

## Inspect Owner's Earnings

## Get price/intrinsicValue (priceFairValue from fincial ratios)

## Get historical dividends via
#https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/AAPL?apikey=YOUR_API_KEY

## see analyst estimates: https://financialmodelingprep.com/api/v3/analyst-estimates/AAPL?limit=30&apikey=YOUR_API_KEY

## get Stock price change from https://financialmodelingprep.com/api/v3/stock-price-change/AAPL?apikey=YOUR_API_KEY