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
import warnings

# Suppress FutureWarning about DataFrame concatenation with empty/all-NA entries
warnings.filterwarnings('ignore', message='.*concatenation with empty or all-NA entries.*')

def postBoWrapper(dmdic):
    import sys
    import numpy as np
    
    # Diagnostic: Check input data BEFORE any calculations
    print("\n" + "="*60, flush=True)
    print("DIAGNOSTIC: postBoWrapper input data check (BEFORE score calculation)", flush=True)
    print("="*60, flush=True)
    
    bmdf = dmdic['BoMetric_df']
    bmav = dmdic.get('BoMetric_ave', {})
    bmda = dmdic.get('BoMetric_dateAve', pd.DataFrame())
    cdx_df = dmdic.get('cdx_df', pd.DataFrame())
    n = dmdic.get('nrScorePeriods', 8)
    
    # Check BoMetric_df
    if bmdf.empty:
        print("ERROR: BoMetric_df is EMPTY!", flush=True)
    else:
        print(f"BoMetric_df shape: {bmdf.shape} (rows, columns)", flush=True)
        print(f"BoMetric_df unique sources: {bmdf['source'].nunique() if 'source' in bmdf.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in bmdf.columns:
            first_source = bmdf['source'].iloc[0]
            print(f"BoMetric_df first source: {first_source}", flush=True)
            # Show sample rows for first source
            first_source_data = bmdf[bmdf['source'] == first_source].head(3)
            print(f"Sample rows for {first_source} (first 3):", flush=True)
            numeric_cols_sample = first_source_data.select_dtypes(include=[np.number]).columns[:5]
            if len(numeric_cols_sample) > 0:
                print(first_source_data[['date', 'source'] + list(numeric_cols_sample)].to_string(), flush=True)
        # Check for NaN in numeric columns
        numeric_cols = bmdf.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            nan_pct = (bmdf[numeric_cols].isna().sum() / len(bmdf) * 100).round(1)
            print(f"\nBoMetric_df NaN percentage in numeric columns (first 5):", flush=True)
            for col, pct in list(nan_pct.head(5).items()):
                print(f"  {col}: {pct}%", flush=True)
    
    # Check cdx_df
    if cdx_df.empty:
        print("\nERROR: cdx_df is EMPTY!", flush=True)
    else:
        print(f"\ncdx_df shape: {cdx_df.shape} (rows, columns)", flush=True)
        print(f"cdx_df unique sources: {cdx_df['source'].nunique() if 'source' in cdx_df.columns else 'NO SOURCE COLUMN'}", flush=True)
        if 'source' in cdx_df.columns:
            first_source = cdx_df['source'].iloc[0]
            print(f"cdx_df first source: {first_source}", flush=True)
            # Show sample rows for first source
            first_source_data = cdx_df[cdx_df['source'] == first_source].head(3)
            print(f"Sample rows for {first_source} (first 3):", flush=True)
            key_cols = ['date', 'source', 'marketCap', 'freeCashFlow', 'price', 'totalAssets']
            available_cols = [col for col in key_cols if col in first_source_data.columns]
            if len(available_cols) > 0:
                print(first_source_data[available_cols].to_string(), flush=True)
    
    # Check BoMetric_ave (could be dict, Series, or DataFrame)
    if bmav is None:
        print("\nWARNING: BoMetric_ave is None!", flush=True)
    elif isinstance(bmav, pd.DataFrame):
        if bmav.empty:
            print("\nWARNING: BoMetric_ave DataFrame is empty!", flush=True)
        else:
            print(f"\nBoMetric_ave DataFrame shape: {bmav.shape}", flush=True)
            print(f"BoMetric_ave columns: {list(bmav.columns[:5])}", flush=True)
    elif isinstance(bmav, pd.Series):
        if bmav.empty:
            print("\nWARNING: BoMetric_ave Series is empty!", flush=True)
        else:
            print(f"\nBoMetric_ave Series length: {len(bmav)}", flush=True)
            print(f"BoMetric_ave sample values (first 10):", flush=True)
            for idx, val in bmav.head(10).items():
                print(f"  {idx}: {val}", flush=True)
    elif isinstance(bmav, dict):
        if not bmav:
            print("\nWARNING: BoMetric_ave dict is empty!", flush=True)
        else:
            print(f"\nBoMetric_ave dict has {len(bmav)} keys", flush=True)
            print(f"BoMetric_ave sample values (first 10):", flush=True)
            for key, val in list(bmav.items())[:10]:
                print(f"  {key}: {val}", flush=True)
    else:
        print(f"\nBoMetric_ave type: {type(bmav)}", flush=True)
    
    print("="*60 + "\n", flush=True)
    sys.stdout.flush()
    
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
    # Note: Bulk endpoints require higher subscription tier, using individual API calls only
    profile_bulk_dict = {}
    rating_bulk_dict = {}
    dcf_bulk_dict = {}
    
    print(f'Writing top {ntopagg} stocks to .csv')
    pbar = tqdm(total=ntopagg)
    for row in BoComp_tocsv.itertuples():
        symb = row.source
        temp_resp_km = requests.get(f'{baseurl}v3/key-metrics/{symb}?period=quarter&limit=4&apikey={api_key}').json()
        temp_resp_fr = requests.get(f'{baseurl}v3/ratios/{symb}?period=quarter&limit=4&apikey={api_key}').json()
        
        # Use bulk data for profile, rating, and DCF if available, otherwise fallback to individual calls
        if symb in profile_bulk_dict:
            temp_resp_pr = [profile_bulk_dict[symb]]  # Convert dict to list format
        else:
            temp_resp_pr = requests.get(f'{baseurl}v3/profile/{symb}?apikey={api_key}').json()
        
        if symb in dcf_bulk_dict:
            temp_resp_dcf = [dcf_bulk_dict[symb]]  # Convert dict to list format
        else:
            temp_resp_dcf_raw = requests.get(f'{baseurl}v3/discounted-cash-flow/{symb}?apikey={api_key}')
            temp_resp_dcf = temp_resp_dcf_raw.json()
        
        # Diagnostic: Check what the DCF API actually returns (only for first ticker)
        if len(crVec) == 0:  # Only print for first ticker to avoid spam
            print(f"\nDEBUG: DCF API response for {symb}:")
            print(f"  Status code: {temp_resp_dcf_raw.status_code}")
            print(f"  Response type: {type(temp_resp_dcf)}")
            if isinstance(temp_resp_dcf, list):
                print(f"  Response length: {len(temp_resp_dcf)}")
                if len(temp_resp_dcf) > 0:
                    print(f"  First element type: {type(temp_resp_dcf[0])}")
                    if isinstance(temp_resp_dcf[0], dict):
                        print(f"  First element keys: {list(temp_resp_dcf[0].keys())}")
            elif isinstance(temp_resp_dcf, dict):
                print(f"  Dict keys: {list(temp_resp_dcf.keys())}")
                print(f"  Dict content: {temp_resp_dcf}")
            else:
                print(f"  Response content: {temp_resp_dcf}")
        
        # Handle case where API returns a dict instead of a list (API might have changed)
        if isinstance(temp_resp_dcf, dict):
            # If it's a dict, try to convert to list format or extract error
            if 'Error Message' in temp_resp_dcf or 'error' in str(temp_resp_dcf).lower():
                temp_resp_dcf = []  # Treat as empty
            else:
                # Try to wrap in list if it's a single DCF object
                temp_resp_dcf = [temp_resp_dcf] if temp_resp_dcf else []
        
        # Check if API responses are empty before accessing
        # Check currentRatio
        if len(temp_resp_fr) == 0 or 'currentRatio' not in temp_resp_fr[0]:
            crVec.append('NaN')
        elif type(temp_resp_fr[0]['currentRatio']) == int or type(temp_resp_fr[0]['currentRatio']) == float:
            crVec.append("{:.4f}".format(temp_resp_fr[0]['currentRatio']))
        else:
            crVec.append('NaN')
            
        # Check dividendYield
        if len(temp_resp_km) == 0 or 'dividendYield' not in temp_resp_km[0]:
            dyVec.append('NaN')
        elif type(temp_resp_km[0]['dividendYield']) == int or type(temp_resp_km[0]['dividendYield']) == float:
            dyVec.append("{:.4f}".format(temp_resp_km[0]['dividendYield']*100))
        else:
            dyVec.append('NaN')
            
        # Check grahamNumberToPrice
        if len(temp_resp_km) == 0 or len(temp_resp_pr) == 0:
            GNtPVec.append('NaN')
        elif 'grahamNumber' not in temp_resp_km[0] or 'price' not in temp_resp_pr[0]:
            GNtPVec.append('NaN')
        elif temp_resp_km[0]['grahamNumber'] is None or temp_resp_pr[0]['price'] is None:
            GNtPVec.append('NaN')
        else:
            gtp = (temp_resp_km[0]['grahamNumber']/temp_resp_pr[0]['price'])
            if type(gtp) == int or type(gtp) == float:
                GNtPVec.append("{:.4f}".format(gtp))
            else:
                GNtPVec.append('NaN')
                
        # Check price
        if len(temp_resp_pr) == 0 or 'price' not in temp_resp_pr[0]:
            priceVec.append('NaN')
        else:
            priceVec.append("{:.4f}".format(temp_resp_pr[0]['price']))
            
        # Check beta
        if len(temp_resp_pr) == 0 or 'beta' not in temp_resp_pr[0]:
            betaVec.append('NaN')
        else:
            betaVec.append("{:.4f}".format(temp_resp_pr[0]['beta']))
            
        # Check sector
        if len(temp_resp_pr) == 0 or 'sector' not in temp_resp_pr[0]:
            sectorVec.append('NaN')
        else:
            sectorVec.append(temp_resp_pr[0]['sector'])
            
        # Check priceEarningsRatio
        if len(temp_resp_fr) == 0 or 'priceEarningsRatio' not in temp_resp_fr[0]:
            pEratioVec.append('NaN')
        else:
            perat = temp_resp_fr[0]['priceEarningsRatio']
            if type(perat) == int or type(perat) == float:
                pEratioVec.append("{:.4f}".format(perat))
            else:
                pEratioVec.append('NaN')
                
        # Check rating
        # Use bulk rating data if available, otherwise fallback to individual call
        if symb in rating_bulk_dict:
            temp_resp_rating = [rating_bulk_dict[symb]]
        else:
            temp_resp_rating = requests.get(f'{baseurl}v3/rating/{symb}?apikey={api_key}').json()
        if len(temp_resp_rating) == 0 or 'ratingRecommendation' not in temp_resp_rating[0]:
            ratingVec_fmp.append('NaN')
        else:
            ratingVec_fmp.append(temp_resp_rating[0]['ratingRecommendation'])
            
        # Check M_Score
        if not (mscore[mscore['source'] == symb]['M_Score_mean']).isna().item():
            mcurscore = mscore[mscore['source'] == symb]['M_Score_mean'].item().item()
            if type(mcurscore) == int or type(mcurscore) == float:
                mscoreVec.append("{:.4f}".format(mscore[mscore['source'] == symb]['M_Score_mean'].item()))
            else:
                mscoreVec.append('NaN')
        else:
            mscoreVec.append('NaN')
            
        # Check C_Score
        if not (cscore[cscore['source'] == symb]['C_Score_mean']).isna().item():
            curcscore = cscore[cscore['source'] == symb]['C_Score_mean'].item().item()
            if type(curcscore) == int or type(curcscore) == float:
                cscoreVec.append("{:.4f}".format(cscore[cscore['source'] == symb]['C_Score_mean'].item()))
            else:
                cscoreVec.append('NaN')
        else:
            cscoreVec.append('NaN')

        # Check grossProfitMargin (needs 4 periods)
        if len(temp_resp_fr) == 0 or 'grossProfitMargin' not in temp_resp_fr[0]:
            margin.append('NaN')
        elif len(temp_resp_fr) < 4:
            margin.append('NaN')
        elif type(temp_resp_fr[0]['grossProfitMargin']) == int or type(temp_resp_fr[0]['grossProfitMargin']) == float:
            gpmsum= temp_resp_fr[0]['grossProfitMargin'] + temp_resp_fr[1]['grossProfitMargin'] + temp_resp_fr[2]['grossProfitMargin'] + temp_resp_fr[3]['grossProfitMargin']
            margin.append("{:.4f}".format(gpmsum*25))
        else:
            margin.append('NaN')

        # Check DCF to Price
        if len(temp_resp_dcf) == 0:
            dcf2p.append('NaN')
        elif 'dcf' not in temp_resp_dcf[0]:
            dcf2p.append('NaN')
        elif temp_resp_dcf[0]['dcf'] is None:
            dcf2p.append('NaN')
        elif type(temp_resp_dcf[0]['dcf']) == int or type(temp_resp_dcf[0]['dcf']) == float:
            if 'Stock Price' not in temp_resp_dcf[0]:
                dcf2p.append('NaN')
            elif temp_resp_dcf[0]['Stock Price'] is None:
                dcf2p.append('NaN')
            elif type(temp_resp_dcf[0]['Stock Price']) == int or type(temp_resp_dcf[0]['Stock Price']) == float:
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
    # Add CycleHeat from postRank data (already calculated in postBoRank)
    if 'CycleHeat' in fbdf_tocsv.columns:
        BoComp_tocsv['CycleHeat'] = fbdf_tocsv['CycleHeat'].values
    # Add moatScore from postRank data (merged from moatIdentifier)
    if 'moatScore' in fbdf_tocsv.columns:
        BoComp_tocsv['moatScore'] = fbdf_tocsv['moatScore'].values
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
        dcf_resp = requests.get(f'{baseurl}v3/discounted-cash-flow/{symb}?apikey={api_key}').json()
        dcf = pd.DataFrame.from_dict(dcf_resp) if dcf_resp else pd.DataFrame()
        
        # Check if DCF has required columns, use fallback if empty
        dcf_has_data = not dcf.empty and 'Stock Price' in dcf.columns and 'dcf' in dcf.columns

        NYSEspe = requests.get(f'https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?date={ll5}&exchange=NYSE&apikey={api_key}').json()
        nspe_df = pd.DataFrame(NYSEspe) if isinstance(NYSEspe, list) and len(NYSEspe) > 0 else pd.DataFrame()
        nspe_has_data = not nspe_df.empty and 'sector' in nspe_df.columns and 'pe' in nspe_df.columns
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
        # Handle empty DCF data (common for non-US stocks)
        if dcf_has_data:
            price = dcf['Stock Price'].apply(format_num)
            symb_df['Price'] = price
            symb_df['Graham number to price'] = (km.grahamNumber/dcf['Stock Price']).apply(format_num)
        else:
            symb_df['Price'] = 'N/A'
            symb_df['Graham number to price'] = 'N/A'

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
        
        # Check if profile data is available
        pr_has_data = pr and len(pr) > 0 and isinstance(pr[0], dict)
        
        ws.cell(row=psdf_row, column=psdf_col).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col).value = 'Company'
        ws.cell(row=psdf_row+1, column=psdf_col).value = pr[0].get('companyName', 'N/A') if pr_has_data else 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 1).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+1).value = 'beta'
        if pr_has_data and pr[0].get('beta') is not None:
            ws.cell(row=psdf_row+1, column=psdf_col+1).value = "{:.4f}".format(pr[0]['beta'])
        else:
            ws.cell(row=psdf_row+1, column=psdf_col+1).value = 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 2).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+2).value = 'Market Cap'
        if pr_has_data and pr[0].get('mktCap') is not None:
            ws.cell(row=psdf_row+1, column=psdf_col+2).value = "{:,.2f}".format(pr[0]['mktCap']/1000000) + " million"
        else:
            ws.cell(row=psdf_row+1, column=psdf_col+2).value = 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 3).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+3).value = 'Industry & Sector'
        if pr_has_data and 'industry' in pr[0]:
            ws.cell(row=psdf_row+1, column=psdf_col+3).value = pr[0]['industry']
            ws.cell(row=psdf_row+2, column=psdf_col+3).value = pr[0].get('sector', 'N/A')
        else:
            ws.cell(row=psdf_row+1, column=psdf_col+3).value = 'N/A'
            ws.cell(row=psdf_row+2, column=psdf_col+3).value = 'N/A'

        ws.cell(row=psdf_row, column=psdf_col + 4).font = bold_font
        ws.cell(row=psdf_row, column=psdf_col+4).value = 'Sector Average PE-ratio'
        # Handle sector PE ratio - may not be available on all subscriptions
        if nspe_has_data and pr_has_data and 'sector' in pr[0]:
            sector_match = nspe_df[nspe_df['sector'] == pr[0]['sector']]
            if not sector_match.empty:
                secpe = sector_match.pe.iloc[0]
                ws.cell(row=psdf_row+1, column=psdf_col + 4).value = str(round(float(secpe), 4))
            else:
                ws.cell(row=psdf_row+1, column=psdf_col + 4).value = 'N/A'
        else:
            ws.cell(row=psdf_row+1, column=psdf_col + 4).value = 'N/A'

        ws.cell(row=psdf_row + 5, column=psdf_col).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col).value = 'Rating Recommendation'
        # Handle empty rating response
        if rating and len(rating) > 0 and 'ratingRecommendation' in rating[0]:
            ws.cell(row=psdf_row + 6, column=psdf_col).value = rating[0]['ratingRecommendation']
        else:
            ws.cell(row=psdf_row + 6, column=psdf_col).value = 'N/A'

        ws.cell(row=psdf_row + 5, column=psdf_col+2).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col+2).value = '(QD?) DCF per price'
        if dcf_has_data:
            ws.cell(row=psdf_row + 6, column=psdf_col+2).value = str(round(float(dcf['dcf']/dcf['Stock Price']), 4))
        else:
            ws.cell(row=psdf_row + 6, column=psdf_col+2).value = 'N/A'
        #ws.cell(row=psdf_row + 6, column=psdf_col+2).value = qdDCFperPrice

        ws.cell(row=psdf_row+5, column=psdf_col + 4).font = bold_font
        ws.cell(row=psdf_row + 5, column=psdf_col + 4).value = 'List of peers'
        # Handle empty stock peers response
        if sp and len(sp) > 0 and isinstance(sp[0], dict) and 'peersList' in sp[0]:
            peerslist = sp[0]['peersList']
            for i, peer in enumerate(peerslist):
                ws.cell(row=psdf_row + 6 + i, column=psdf_col + 4).value = peer
        else:
            ws.cell(row=psdf_row + 6, column=psdf_col + 4).value = 'N/A'

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
    tempdf_orig = pd.concat([moatdf, pd.DataFrame([nan_dict])], ignore_index=True)
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

        moatdf = pd.concat([moatdf, tempdf]).reset_index(drop=True)

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

    # Removed second loop that was searching for sectors not in sectordic
    # The first loop (lines 388-394) already finds the highest-scoring stock for each sector
    # that exists in the current dataset. The second loop was inefficiently searching through
    # all symbols for sectors that don't exist due to sector mapping/consolidation.

    resdic = {**resdic, **{'BoScore_highsectors': bshs}}

    return resdic

#see sector performance
#https://financialmodelingprep.com/api/v3/stock/sector-performance?apikey=YOUR_API_KEY
#https://financialmodelingprep.com/api/v3/historical-sectors-performance?limit=50&apikey={api_key}

## Inspect Owner's Earnings

## Get historical dividends via
#https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/AAPL?apikey=YOUR_API_KEY

## see analyst estimates: https://financialmodelingprep.com/api/v3/analyst-estimates/AAPL?limit=30&apikey=YOUR_API_KEY
