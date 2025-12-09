import calcScore as cs
import postBoRank as pbr
import pandas as pd
import requests
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import numpy as np
import getData_gen as gdg

def postBoWrapper(dmdic):
    bmdf = dmdic['BoMetric_df']
    bmav = dmdic['BoMetric_ave']
    bmda = dmdic['BoMetric_dateAve']
    cdx_df = dmdic['cdx_df']
    n = dmdic['nrScorePeriods']
    BoScore_df = cs.simpleScore_fromDict(bmdf, bmav, bmda, n)
    BoS_dftop100 = BoScore_df.head(100)
    BoM_dftop100 = bmdf[bmdf['source'].isin(list(BoS_dftop100.source))].reset_index(drop=True)
    cdx_dftop100 = cdx_df[cdx_df['source'].isin(list(BoS_dftop100.source))].reset_index(drop=True)

    n= 16
    rankdic = pbr.postBoScoreRanking(BoM_dftop100, BoS_dftop100, cdx_dftop100, dmdic['baseurl'], dmdic['api_key'],
                                     dmdic['period'],n)

    metricList = ['earningsYield', 'grahamNumberToPrice', 'RoA', 'EPStoEPSmean', 'freeCashFlowYield', 'reveneGrowth']
    cutoff = 1.5
    psbrfilter = pbr.postBoRankingPassFilter(rankdic['postRank'],metricList,-cutoff,np.inf)

    regressMetricsOnROR(rankdic)

    resdic = {**rankdic, **{'BoS_dftop100': BoS_dftop100, 'BoM_dftop100': BoM_dftop100, 'cdx_dftop100': cdx_dftop100,
                          'BoScore_df': BoScore_df, 'psbrfilter': psbrfilter}}

    return resdic

def regressMetricsOnROR(rankdic):
    regressors = list(set(rankdic['postRank'].columns) - set(['rankOfRanks','AggScore','source']))
    regressant = ['rankOfRanks']
    df = rankdic['postRank']
    X = df[regressors]
    y = df[regressant]
    
    # Guard: skip regression if insufficient samples (need at least 1 sample to fit)
    if X.shape[0] < 1:
        print("Warning: Insufficient samples for regression (found 0 samples). Skipping regression.")
        return None
    
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    print("R-Squared:", r_squared)
    coef = model.coef_
    intercept = model.intercept_
    print("Coefficients:", coef)
    print("regressors:", regressors)
    print("Intercept:", intercept)
    print("coefreg:", tuple(zip(regressors, coef.tolist()[0])))
    return None

def writeResWrapper(resdic):
    ntopagg = resdic['ntopagg']
    ntopxlsx = resdic['ntopxlsx']
    fidag = datetime.today().strftime('%Y-%m-%d')
    fb_df = resdic['postRank']
    mscore = resdic['SLmeanMscore']
    cscore = resdic['SLmeanCscore']
    baseurl = resdic['baseurl']
    api_key = resdic['api_key']
    tickerfilter = resdic['tickerfilter']
    datasource = resdic['datasource']
    years = 6

    # create csv listing the ntopagg stocks
    fname_AggScoretop = f'AggScoreTop{ntopagg}-{fidag}_{datasource}_{tickerfilter}.csv'
    writeBoAggToCSV(fb_df, mscore, cscore, baseurl, api_key, ntopagg, fname_AggScoretop)

    # create presentation xlsx of ntopxlsx stocks
    fname_presentationtop= f'PresentationTop{ntopxlsx}-{fidag}_{datasource}_{tickerfilter}.xlsx'
    createPresentation(fb_df, mscore, cscore, baseurl, api_key, ntopxlsx, fname_presentationtop, years)

def writeBoAggToCSV(fb_df, mscore, cscore, baseurl, api_key, ntopagg, fname_AggScoretop):
    fbdf_tocsv = fb_df.head(ntopagg)
    symblist = list(fbdf_tocsv['source'])
    #BoComp_tocsv = pd.DataFrame(columns=['source','currentRatio','dividendYield','grahamNumberToPrice','price','beta',
    #                                    'sector','fmpRating','PEratio','M_score','C_score'])
    BoComp_tocsv = pd.DataFrame()
    BoComp_tocsv['source'] = symblist
    #quote_full = pd.DataFrame(requests.get(f'{baseurl}v3/quote/{symblist}?&apikey={api_key}').json())
    crVec = []
    dyVec = []
    GNtPVec = []
    priceVec = []
    margin = []
    dcf2p = []
    betaVec = []
    sectorVec = []
    ratingVec_fmp = []
    pEratioVec = []
    mscoreVec = []
    cscoreVec = []
    print(f'Writing top {ntopagg} stocks to .csv')
    pbar = tqdm(total=ntopagg)
    for row in BoComp_tocsv.itertuples():
        symb = row.source
        temp_resp_km = requests.get(f'{baseurl}v3/key-metrics/{symb}?period=quarter&limit=4&apikey={api_key}').json()
        temp_resp_fr = requests.get(f'{baseurl}v3/ratios/{symb}?period=quarter&limit=4&apikey={api_key}').json()
        temp_resp_pr = requests.get(f'{baseurl}v3/profile/{symb}?period=quarter&limit=4&apikey={api_key}').json()
        temp_resp_dcf = requests.get(f'{baseurl}v3/discounted-cash-flow/{symb}?apikey={api_key}').json()
        if type(temp_resp_fr[0]['currentRatio']) == int or type(temp_resp_fr[0]['currentRatio']) == float:
            crVec.append("{:.4f}".format(temp_resp_fr[0]['currentRatio']))
        else:
            crVec.append('NaN')
        if type(temp_resp_km[0]['dividendYield']) == int or type(temp_resp_km[0]['dividendYield']) == float:
            dyVec.append("{:.4f}".format(temp_resp_km[0]['dividendYield']*100))
        else:
            dyVec.append('NaN')
        gtp = (temp_resp_km[0]['grahamNumber']/temp_resp_pr[0]['price'])
        if type(gtp) == int or type(gtp) == float:
            GNtPVec.append("{:.4f}".format(gtp))
        else:
            GNtPVec.append('NaN')
        priceVec.append("{:.4f}".format(temp_resp_pr[0]['price']))
        betaVec.append("{:.4f}".format(temp_resp_pr[0]['beta']))
        sectorVec.append(temp_resp_pr[0]['sector'])
        perat = temp_resp_fr[0]['priceEarningsRatio']
        if type(perat) == int or type(perat) == float:
            pEratioVec.append("{:.4f}".format(perat))
        else:
            pEratioVec.append('NaN')
        temp_resp_rating = requests.get(f'{baseurl}/v3/rating/{symb}?apikey={api_key}').json()
        ratingVec_fmp.append(temp_resp_rating[0]['ratingRecommendation'])
        if not (mscore[mscore['source'] == symb]['M_Score_mean']).isna().item():
            mcurscore = mscore[mscore['source'] == symb]['M_Score_mean'].item().item()
            if type(mcurscore) == int or type(mcurscore) == float:
                mscoreVec.append("{:.4f}".format(mscore[mscore['source'] == symb]['M_Score_mean'].item()))
            else:
                mscoreVec.append('NaN')
        else:
            mscoreVec.append('NaN')
        if not (cscore[cscore['source'] == symb]['C_Score_mean']).isna().item():
            curcscore = cscore[cscore['source'] == symb]['C_Score_mean'].item().item()
            if type(curcscore) == int or type(curcscore) == float:
                cscoreVec.append("{:.4f}".format(cscore[cscore['source'] == symb]['C_Score_mean'].item()))
            else:
                cscoreVec.append('NaN')
        else:
            cscoreVec.append('NaN')

        if type(temp_resp_fr[0]['grossProfitMargin']) == int or type(temp_resp_fr[0]['grossProfitMargin']) == float:
            gpmsum= temp_resp_fr[0]['grossProfitMargin'] + temp_resp_fr[1]['grossProfitMargin'] + temp_resp_fr[2]['grossProfitMargin'] + temp_resp_fr[3]['grossProfitMargin']
            margin.append("{:.4f}".format(gpmsum*25))
        else:
            margin.append('NaN')

        if type(temp_resp_dcf[0]['dcf']) == int or type(temp_resp_dcf[0]['dcf']) == float:
            if type((temp_resp_dcf[0]['Stock Price'])) == int or type((temp_resp_dcf[0]['Stock Price'])) == float:
                dcf2p.append("{:.4f}".format(temp_resp_dcf[0]['dcf']/(temp_resp_dcf[0]['Stock Price'])))
            else:
                dcf2p.append('NaN')
        else:
            dcf2p.append('NaN')
        pbar.update(n=1)
    BoComp_tocsv['price'] = priceVec
    BoComp_tocsv['PE-ratio'] = pEratioVec
    BoComp_tocsv['beta'] = betaVec
    BoComp_tocsv['sector'] = sectorVec
    BoComp_tocsv['rating_fmp'] = ratingVec_fmp
    BoComp_tocsv['currentRatio'] = crVec
    BoComp_tocsv['dividendYield'] = dyVec
    BoComp_tocsv['GrahamNumberToPrice'] = GNtPVec
    BoComp_tocsv['GrossProfitMargin_ttm'] = margin
    BoComp_tocsv['DCF-to-Price'] = dcf2p
    BoComp_tocsv['M-Score'] = mscoreVec
    BoComp_tocsv['C-Score'] = cscoreVec
    BoComp_tocsv.to_csv(fname_AggScoretop)
    pbar.close()
    return None

def createPresentation(finalBoRank_df, mscore, cscore, baseurl, api_key, topn, fname, years):
    #test
    #fname = fname_spreadSheet
    #topn = 20
    #years = 10
    symblist = list(finalBoRank_df['source'].head(topn))
    #eyVec = []
    #quote_full = pd.DataFrame(requests.get(f'{baseurl}v3/quote/{symblist}?&apikey={api_key}').json())

    almago = datetime.today() - timedelta(weeks=5)
    if almago.day >= 5:
        lastlast_5th = almago.replace(day=5)
    else:
        lastlast_5th = (almago - timedelta(days=almago.day)).replace(day=5)
    ll5 = lastlast_5th.strftime('%Y-%d-%m')
    wb = openpyxl.Workbook()
    print(f'Writing top {topn} stocks to an .xlsx file for presentation')
    pbar = tqdm(total=topn)
    for symb in symblist[::-1]:
        km = pd.DataFrame(
            requests.get(f'{baseurl}v3/key-metrics/{symb}?period=annual&limit={years}&apikey={api_key}').json())
        fr = pd.DataFrame(
            requests.get(f'{baseurl}v3/ratios/{symb}?period=annual&limit={years}&apikey={api_key}').json())
        pr = requests.get(
            f'{baseurl}v3/profile/{symb}?apikey={api_key}').json()
        rating = requests.get(
            f'{baseurl}v3/rating/{symb}?apikey={api_key}').json()
        #target = requests.get(
        # f'https://financialmodelingprep.com/api/v4/price-target-consensus?symbol{symb}&apikey={api_key}').json()
        sp = requests.get(
            f'{baseurl}v4/stock_peers?symbol={symb}&apikey={api_key}').json()
        cf = pd.DataFrame(
            requests.get(f'{baseurl}v3/cash-flow-statement/{symb}?period=annual&limit={years}&apikey={api_key}').json())
        dcf = pd.DataFrame.from_dict(
            requests.get(f'{baseurl}v3/discounted-cash-flow/{symb}?apikey={api_key}').json())

        NYSEspe = requests.get(f'https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?date={ll5}&exchange=NYSE&apikey={api_key}').json()
        nspe_df = pd.DataFrame(NYSEspe)
        symb_df = pd.DataFrame(
            columns=['Symbol', 'Date', 'Earnings yield', 'PE-ratio', 'Price-to-book', 'Current ratio',
                     'Dividend yield', 'Price-to-fair value', 'Price'])
        symb_df['Symbol'] = fr['symbol']
        symb_df['Date'] = fr['date']
        symb_df['Earnings yield'] = (km.earningsYield * 100).apply(format_num)
        symb_df['PE-ratio'] = km.peRatio.apply(format_num)
        symb_df['Price-to-book'] = km.ptbRatio.apply(format_num)
        symb_df['Current ratio'] = km.currentRatio.apply(format_num)
        symb_df['Dividend yield'] = (km.dividendYield.fillna(0)*100).apply(format_num)
        symb_df['Price-to-fair value'] = fr.priceFairValue.apply(format_num)
        price = dcf['Stock Price'].apply(format_num)
        symb_df['Price'] = price
        symb_df['Graham number to price'] = (km.grahamNumber/dcf['Stock Price']).apply(format_num)

        fcf = cf.freeCashFlow
        shares = fcf/km.freeCashFlowPerShare
        #qdDCFperPrice = quickDCF(fcf,0.12,0,km.interestDebtPerShare*shares,shares,price)

        if symb not in wb.sheetnames:
            ws = wb.create_sheet(symb, 0)
            ws.title = symb
        else:
            ws = wb[symb]
        wb.active = ws

        for r in dataframe_to_rows(symb_df, index=False, header=True):
            ws.append(r)

        for cell in ws['A'] + ws[1]:
            cell.style = 'Pandas'

        bold_font = Font(bold=True)
        psdf_col = len(symb_df.columns)+2
        psdf_row = 1
        ws.cell(row=psdf_row, column=psdf_col).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col).value = 'Company'
        ws.cell(row=psdf_row+1, column=psdf_col).value = pr[0]['companyName']

        ws.cell(row=psdf_row, column=psdf_col + 1).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+1).value = 'beta'
        ws.cell(row=psdf_row+1, column=psdf_col+1).value = "{:.4f}".format(pr[0]['beta'])

        ws.cell(row=psdf_row, column=psdf_col + 2).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+2).value = 'Market Cap'
        ws.cell(row=psdf_row+1, column=psdf_col+2).value = "{:,.2f}".format(pr[0]['mktCap']/1000000) + " million"

        ws.cell(row=psdf_row, column=psdf_col + 3).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+3).value = 'Industry & Sector'
        ws.cell(row=psdf_row+1, column=psdf_col+3).value = pr[0]['industry']
        ws.cell(row=psdf_row+2, column=psdf_col+3).value = pr[0]['sector']

        ws.cell(row=psdf_row, column=psdf_col + 4).font = bold_font
        secpe = nspe_df[nspe_df['sector'] == pr[0]['sector']].pe.item()
        ws.cell(row=psdf_row, column=psdf_col+4).value = 'Sector Average PE-ratio'
        ws.cell(row=psdf_row+1, column=psdf_col + 4).value = str(round(float(secpe), 4))

        ws.cell(row=psdf_row + 5, column=psdf_col).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col).value = 'Rating Recommendation'
        ws.cell(row=psdf_row + 6, column=psdf_col).value = rating[0]['ratingRecommendation']

        ws.cell(row=psdf_row + 5, column=psdf_col+2).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col+2).value = '(QD?) DCF per price'
        ws.cell(row=psdf_row + 6, column=psdf_col+2).value = str(round(float(dcf['dcf']/dcf['Stock Price']), 4))
        #ws.cell(row=psdf_row + 6, column=psdf_col+2).value = qdDCFperPrice

        ws.cell(row=psdf_row+5, column=psdf_col + 4).font = bold_font
        peerslist = sp[0]['peersList']
        ws.cell(row=psdf_row + 5, column=psdf_col + 4).value = 'List of peers'
        for peer in peerslist:
            ws.cell(row=psdf_row +  6 + peerslist.index(peer), column=psdf_col + 4).value = peer

        resize_columns(ws)

        pbar.update(n=1)


    wb.save(fname)
    wb.close()
    pbar.close()

    return None

# Function to resize columns in an Excel sheet
def resize_columns(ws):
    for column_cells in ws.iter_cols():
        length = max(len(str(cell.value)) for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length

def format_num(x):
    return "{:.4f}".format(x)

def moatIdentifier(symblist, cdx_df, n=20):
    #moatdf = pd.DataFrame(columns=['source','moatScore'])
    #for symb in symbollist:
    # calculate FCFyield (>5-10%), Gross Margin (GrossProfit/Revenue>30%)
    # Calculate sales/Assets (>0.75)
    # Calculate RoE (>15%) and RoA(>10%) and ROIC (>15%)
    # High earnings even when market earnings are down (beta < 1?)
    # Calculate SG&A/GrossProfit (<15%) [sale, general and administrative expenses]
    # Calculate Depreciation/GrossProfit (<10%)
    # Calculate InterestExpenses/OperatingIncome (<15%)
    # Calculate NetMargin [NetIncome/Revenue] (>20%)
    # Calculate Capex/NetIncome (<25%)
    # Calculate TotalLiabilities/ShareholderEquity (<0.8)
    #moatdf = pd.DataFrame(columns=['source','moatScore','FCFyield','GrossMargin','RevtoASS','RoE','RoA','ROIC','SGAtoGP','DeptoGP',
    #                               'InteresttoOI', 'NetMargin','CapExtoEarnings','TLtoEquity'])
    moatdf = pd.DataFrame(columns=['source','moatScore','FCFyield','GrossMargin','RevtoASS','RoE','RoA','ROIC','SGAtoGP','DeptoGP',
                                   'NetMargin','CapExtoEarnings','TLtoEquity'])
    nan_dict = {'source': np.nan,  'moatScore': np.nan, 'FCFyield': np.nan, 'GrossMargin': np.nan, 'RevtoASS': np.nan,
                'RoE': np.nan, 'RoA': np.nan, 'ROIC': np.nan, 'SGAtoGP': np.nan, 'DeptoGP': np.nan,
                'NetMargin': np.nan, 'CapExtoEarnings': np.nan, 'TLtoEquity': np.nan}
    tempdf_orig = moatdf
    tempdf_orig = tempdf_orig.append(nan_dict, ignore_index=True)
    for symb in symblist:
        tempdf = tempdf_orig
        cdx_temp = cdx_df[cdx_df['source'] == symb]
        tempdf['source'] = symb
        fcfmask = cdx_temp['pfcfRatio'] != 0
        fcfyield_filter = cdx_temp['pfcfRatio'][fcfmask]
        tempdf['FCFyield'] = (1/fcfyield_filter).head(n).mean()-0.1
        tempdf['GrossMargin'] = cdx_temp['grossProfitMargin'].head(n).mean()-0.3
        tempdf['RevtoASS'] = (cdx_temp['revenue']/cdx_temp['totalAssets']).head(n).mean()-0.75
        tempdf['RoE'] = cdx_temp['returnOnEquity'].head(n).mean()-0.15
        tempdf['RoA'] = cdx_temp['returnOnAssets'].head(n).mean()-0.1
        tempdf['ROIC'] = cdx_temp['returnOnCapitalEmployed'].head(n).mean() - 0.15
        gpmask = cdx_temp['grossProfit'] != 0
        gp_filter = cdx_temp['grossProfit'][gpmask]
        tempdf['SGAtoGP'] = 0.15-(cdx_temp['sellingGeneralAndAdministrativeExpenses'][gpmask]/gp_filter).head(n).mean()
        tempdf['DeptoGP'] = 0.1 - (cdx_temp['depreciationAndAmortization'][gpmask]/gp_filter).head(n).mean()
        #tempdf['InteresttoOI'] = 0.15 - (cdx_df['interestExpense']/cdx_df['operatingIncome']).head(n).mean()
        tempdf['NetMargin'] = cdx_temp['netProfitMargin'].head(n).mean() - 0.2
        nimask = cdx_temp['netIncomePerShare'] != 0
        ni_filter = cdx_temp['netIncomePerShare'][nimask]
        tempdf['CapExtoEarnings'] = 0.2 - (cdx_temp['capexPerShare'][nimask]/ni_filter).head(n).mean()
        tempdf['TLtoEquity'] = 0.8 - (cdx_temp['totalLiabilities']/cdx_temp['totalStockholdersEquity']).head(n).mean()
        numeric_df = tempdf.select_dtypes(include='number')
        mask = numeric_df > 0
        tempdf['moatScore'] = mask.sum(axis=1)

        moatdf = pd.concat([moatdf,tempdf]).reset_index(drop=True)

    moatdf.sort_values(by='moatScore', ascending=False, inplace=True)

    return moatdf


#CLEAN THIS UP
def findHighestOfEachSector(resdic):
    #sectorlist = resdic['sectorlist']
    #sectorlist =  ['all', 'Unspecified', 'Basic Materials', 'Healthcare', 'Financial Services',
    #              'Energy', 'Consumer Cyclical', 'Consumer Defensive', 'Industrials',
    #              'Communication Services', 'Technology', 'Real Estate', 'Utilities','Biotechnology']
    #sectorlist.remove('all')
    #sectorlist.remove('Unspecified')
    sectordic = pd.read_pickle('sectorsdic_fmp.pickle')
    sectorlist = list(sectordic.keys())
    baseurl = resdic['baseurl']
    api_key = resdic['api_key']
    sectorsfound = []
    sectorsnotfound = sectorlist
    bsdf = resdic['BoScore_df'].reset_index(drop=True)
    bsdf['score'] = bsdf['score'].astype(float)
    bshs = pd.DataFrame(columns=[bsdf.columns])
    bshs = pd.DataFrame(columns=['source', 'score', 'sector'])
    symblist = bsdf['source']

    for sector in sectorlist:
        sslist = sectordic[sector]
        if len(bsdf.loc[bsdf['source'].isin(sslist), 'score']) > 0:
            highest_rowid = bsdf.loc[bsdf['source'].isin(sslist), 'score'].idxmax()
            newrow = {'source': bsdf.loc[highest_rowid]['source'], 'score': bsdf.loc[highest_rowid]['score'], 'sector': sector}
            tempdf = pd.DataFrame([newrow])
            bshs = pd.concat([bshs, tempdf], ignore_index=True)
    bshs = bshs.reset_index(drop=True)

    #bshs['sector'] = sectorlist

    for i in range(0,len(symblist)-1):
        symb = symblist.iloc[i]
        temp_resp_pr = gdg.safe_get(f'{baseurl}v3/profile/{symb}?period=quarter&limit=4&apikey={api_key}')
        if temp_resp_pr is None or len(temp_resp_pr) == 0:
            continue
        sector = temp_resp_pr[0]['sector']
        if sector not in sectorsfound:
            if len(sector) > 0:
                sectorsfound.append(sector)
                bshs = pd.concat([bshs, bsdf[bsdf['source']==symb]], ignore_index=True)
                sectorsnotfound.remove(sector)
            if len(sectorsnotfound) == 0:
                break
        if i%100 == 0 and i > 0:
            print(f'iteration {i}')
            print(f'Sectors not found: {sectorsnotfound}')
            if i > 1000:
                break

    resdic = {**resdic, **{'BoScore_highsectors': bshs}}

    return resdic

#see sector performance
#https://financialmodelingprep.com/api/v3/stock/sector-performance?apikey=YOUR_API_KEY
#https://financialmodelingprep.com/api/v3/historical-sectors-performance?limit=50&apikey={api_key}

## Inspect Owner's Earnings

## Get historical dividends via
#https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/AAPL?apikey=YOUR_API_KEY

## see analyst estimates: https://financialmodelingprep.com/api/v3/analyst-estimates/AAPL?limit=30&apikey=YOUR_API_KEY
